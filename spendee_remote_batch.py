"""Single-user Spendee MCP for Render: read, create and carefully edit transactions.

Upload this file to the root of your fork of https://github.com/gibsn/mcp-spendee.
This is an unofficial integration. OAuth uses durable encrypted Redis storage.
Editing uses Firestore's REST commit API and is
NOT proven against a live Spendee account: first test a disposable transaction.

Installation (Render build):
    pip install -e . 'mcp>=1.30,<2' 'fastmcp>=2.13,<3' 'py-key-value-aio[redis]' cryptography
Start command: python spendee_remote_persistent.py

Batch tools: prepare_transaction_batch, commit_approved_batch, get_batch_status.
Batch state is encrypted in the existing Upstash TLS Redis.

Required Render secrets: GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET,
PUBLIC_BASE_URL, JWT_SIGNING_KEY, GITHUB_ALLOWED_LOGIN, REDIS_URL,
SPENDEE_EMAIL, SPENDEE_PASSWORD.
Other settings: SPENDEE_TIMEZONE=Asia/Singapore,
SPENDEE_GLOBAL_CURRENCY=SGD, PYTHON_VERSION=3.12.11.
"""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import hmac
import json
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL
from zoneinfo import ZoneInfo

import anyio
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token
from cryptography.fernet import Fernet
from key_value.aio.stores.redis import RedisStore
from redis.asyncio import Redis
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from spendee.firestore import _decode_document, _encode_value

from mcp_spendee.client import ResourceId, SpendeeGateway, TransactionType
from mcp_spendee.config import Settings


def env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required setting: {name}")
    return value


# Keep FastMCP's OAuth client registrations / upstream tokens in a durable
# external store. Render Free's filesystem is erased on every spin-down.
# A cryptographic key is deterministically derived from JWT_SIGNING_KEY;
# rotating that setting invalidates the previous encrypted OAuth records.
_fernet_key = base64.urlsafe_b64encode(
    hashlib.sha256(("spendee-mcp-oauth-redis-v1:" + env("JWT_SIGNING_KEY")).encode()).digest()
)
# Construct the Redis client with redis-py's URL parser. This is important:
# some py-key-value releases lose the TLS scheme if RedisStore(url=...) is used.
# Upstash's TCP endpoint requires TLS (rediss://), not plaintext (redis://).
_redis_url = env("REDIS_URL")
if not _redis_url.startswith("rediss://"):
    raise RuntimeError("REDIS_URL must be the Upstash TCP URL beginning with rediss://")
_redis_client = Redis.from_url(
    _redis_url,
    decode_responses=True,
    socket_connect_timeout=10,
    socket_timeout=20,
)
_oauth_storage = FernetEncryptionWrapper(
    key_value=RedisStore(client=_redis_client),
    fernet=Fernet(_fernet_key),
)
_auth = GitHubProvider(
    client_id=env("GITHUB_CLIENT_ID"),
    client_secret=env("GITHUB_CLIENT_SECRET"),
    base_url=env("PUBLIC_BASE_URL").rstrip("/"),
    jwt_signing_key=env("JWT_SIGNING_KEY"),
    client_storage=_oauth_storage,
)
mcp = FastMCP(name="Spendee — private", auth=_auth)
_gateway = SpendeeGateway(Settings.from_env())


def require_owner() -> None:
    """Every tool is allowed only for the configured GitHub identity."""
    token = get_access_token()
    claims = token.claims if token is not None else {}
    login = str(claims.get("login") or "")
    if not login or login.casefold() != env("GITHUB_ALLOWED_LOGIN").casefold():
        raise PermissionError("Access denied for this GitHub identity")


@mcp.tool()
async def spendee_status() -> dict[str, object]:
    """Check connection settings without exposing secret values."""
    require_owner()
    return Settings.from_env().status()


@mcp.tool()
async def list_wallets() -> list[dict[str, Any]]:
    """List wallet names, currencies, and IDs for transactions."""
    require_owner()
    return await anyio.to_thread.run_sync(_gateway.list_wallets)


@mcp.tool()
async def list_labels() -> list[dict[str, str]]:
    """List Spendee labels; labels must exist before they can be attached."""
    require_owner()
    return await anyio.to_thread.run_sync(_gateway.list_labels)


@mcp.tool()
async def list_categories(
    wallet_id: ResourceId | None = None,
    category_type: TransactionType | None = None,
) -> list[dict[str, Any]]:
    """List categories and their IDs."""
    require_owner()
    return await anyio.to_thread.run_sync(
        lambda: _gateway.list_categories(wallet_id=wallet_id, category_type=category_type)
    )


@mcp.tool()
async def list_transactions(
    wallet_id: ResourceId | None = None,
    offset: int = 0,
    limit: int | None = 100,
    include_labels: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Read Spendee transactions. Dates are UTC days/offset-aware timestamps."""
    require_owner()
    return await anyio.to_thread.run_sync(
        lambda: _gateway.list_transactions(
            wallet_id=wallet_id,
            offset=offset,
            limit=limit,
            include_labels=include_labels,
            date_from=date_from,
            date_to=date_to,
        )
    )


@mcp.tool()
async def create_transaction(
    wallet_id: ResourceId,
    category_id: ResourceId,
    amount: float,
    transaction_type: TransactionType,
    note: str | None = None,
    labels: list[str] | None = None,
    occurred_at: str | None = None,
    currency: str | None = None,
    exchange_rate: float | None = None,
    allow_duplicate: bool = False,
    confirm: bool = False,
    request_id: str | None = None,
    preview_token: str | None = None,
) -> dict[str, Any]:
    """Preview/create an income or expense with explicit two-turn approval.

    ALWAYS first call confirm=false, show the preview, and STOP.
    ONLY after the user explicitly approves the exact preview in a separate
    turn, call confirm=true with the issued preview_token and a unique
    request_id (UUID). A cryptographic token binds the approved preview to
    its amount, date, labels and wallet. Tokens expire after 15 minutes.
    The underlying gateway also checks for existing exact duplicates.
    Never enable allow_duplicate unless user explicitly approves two
    genuinely separate identical transactions.
    """
    require_owner()
    if not confirm:
        if allow_duplicate:
            raise ValueError("allow_duplicate is not available in previews; discuss duplicates first")
        result = await anyio.to_thread.run_sync(
            lambda: _gateway.create_transaction(
                wallet_id=wallet_id,
                category_id=category_id,
                amount=amount,
                transaction_type=transaction_type,
                note=note,
                labels=labels,
                occurred_at=occurred_at,
                currency=currency,
                exchange_rate=exchange_rate,
                confirm=False,
            )
        )
        transaction = result["transaction"]
        claim = {
            "v": 1,
            "kind": "create",
            "transaction": transaction,
            "wallet_id": str(wallet_id),
            "category_id": str(category_id),
            "allow_duplicate": False,
            "expires_at": int(time.time()) + 900,
        }
        return {
            **result,
            "preview_token": _seal_preview(claim),
            "expires_in_minutes": 15,
            "requires_explicit_user_confirmation": True,
        }

    if not request_id:
        raise ValueError("REQUEST_ID_REQUIRED: Confirmed creation needs a unique request_id (UUID); no write attempted")
    authorization = _unseal_preview(preview_token, expected_kind="create")
    if (str(wallet_id) != authorization.get("wallet_id")
            or str(category_id) != authorization.get("category_id")):
        raise PreviewTokenError(
            "PREVIEW_INPUT_MISMATCH",
            "The confirmed wallet/category differs from the approved creation preview. Preview again; no write attempted.",
        )
    if allow_duplicate:
        # Refuse to toggle deduplication bypass after the preview.
        raise ValueError("Cannot bypass duplicate protection through this tool")
    frozen = authorization["transaction"]
    # A missing date/rate on confirmation means reuse the exact preview.
    frozen_date = frozen["occurred_at"]
    if occurred_at is not None:
        supplied = _utc_timestamp(occurred_at)[0]
        preview_local = dt.datetime.fromisoformat(frozen_date)
        if preview_local.tzinfo is None:
            preview_local = preview_local.replace(tzinfo=ZoneInfo(Settings.from_env().timezone))
        previewed = preview_local.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        if dt.datetime.fromisoformat(supplied.replace("Z", "+00:00")) != dt.datetime.fromisoformat(previewed.replace("Z", "+00:00")):
            raise PreviewTokenError("PREVIEW_INPUT_MISMATCH", "Transaction date changed after the creation preview. Preview again; no write attempted.")
    frozen_currency = frozen.get("foreign_currency") or frozen.get("currency")
    if currency is not None and currency.upper() != frozen_currency:
        raise PreviewTokenError("PREVIEW_INPUT_MISMATCH", "Currency changed after the creation preview. Preview again; no write attempted.")
    fixed_rate = frozen.get("foreign_rate") if frozen.get("foreign_currency") else None
    if exchange_rate is not None and fixed_rate is not None:
        if Decimal(str(exchange_rate)) != Decimal(str(fixed_rate)):
            raise PreviewTokenError("PREVIEW_INPUT_MISMATCH", "Exchange rate changed after the creation preview. Preview again; no write attempted.")
    if exchange_rate is not None and fixed_rate is None:
        raise PreviewTokenError("PREVIEW_INPUT_MISMATCH", "Exchange rate changed after the creation preview. Preview again; no write attempted.")

    preview = await anyio.to_thread.run_sync(
        lambda: _gateway.create_transaction(
            wallet_id=wallet_id,
            category_id=category_id,
            amount=amount,
            transaction_type=transaction_type,
            note=note,
            labels=labels,
            occurred_at=frozen_date,
            currency=frozen_currency,
            exchange_rate=fixed_rate,
            confirm=False,
        )
    )
    if preview["transaction"] != frozen:
        raise PreviewTokenError("PREVIEW_INPUT_MISMATCH", "The generated transaction does not match the approved creation preview. Preview again; no write attempted.")
    return await anyio.to_thread.run_sync(
        lambda: _gateway.create_transaction(
            wallet_id=wallet_id,
            category_id=category_id,
            amount=amount,
            transaction_type=transaction_type,
            note=note,
            labels=labels,
            occurred_at=frozen_date,
            currency=frozen_currency,
            exchange_rate=fixed_rate,
            confirm=True,
            request_id=request_id,
            allow_duplicate=False,
        )
    )


# ---- Carefully scoped existing-transaction edit support -------------------
# Editing intentionally does NOT support deletion, wallet moves, transfers,
# recurring transactions, or direct foreign-currency amount changes.
# Firestore is Spendee's private backend; test changes on a disposable record.


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64url(value: str) -> bytes:
    """Decode token segments strictly; reject invalid base64url characters."""
    if not value or not isinstance(value, str):
        raise ValueError("Empty token segment")
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


class PreviewTokenError(ValueError):
    """Safe, actionable preview errors with stable machine-readable codes.

    This is deliberately a ValueError so existing MCP error propagation works.
    Never include the token, token body, signing key, or other secrets in errors.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _seal_preview(payload: dict[str, Any]) -> str:
    body = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    sig = hmac.new(env("JWT_SIGNING_KEY").encode(), body.encode(), hashlib.sha256).digest()
    return body + "." + _b64url(sig)


def _unseal_preview(token: str | None, *, expected_kind: str | None = None) -> dict[str, Any]:
    """Validate a preview and report *why* it cannot be confirmed.

    A signed token can be trusted only after HMAC verification. Historical edit
    previews lacked `kind`, so recognize them during their 15-minute lifetime.
    """
    if not token:
        raise PreviewTokenError(
            "PREVIEW_TOKEN_REQUIRED",
            "A previously issued preview_token is required. Preview the operation before confirming; no write attempted.",
        )
    if not isinstance(token, str) or len(token) > 65536 or token.count(".") != 1:
        raise PreviewTokenError(
            "PREVIEW_TOKEN_MALFORMED",
            "The preview token has an invalid format. Generate a new preview; no write attempted.",
        )

    encoded_body, encoded_signature = token.split(".", 1)
    try:
        body = _unb64url(encoded_body)
        signature = _unb64url(encoded_signature)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise PreviewTokenError(
            "PREVIEW_TOKEN_MALFORMED",
            "The preview token contains invalid encoding. Generate a new preview; no write attempted.",
        ) from exc
    if len(signature) != hashlib.sha256().digest_size:
        raise PreviewTokenError(
            "PREVIEW_TOKEN_MALFORMED",
            "The preview token signature has an invalid length. Generate a new preview; no write attempted.",
        )

    expected_signature = hmac.new(
        env("JWT_SIGNING_KEY").encode(), encoded_body.encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise PreviewTokenError(
            "PREVIEW_BAD_SIGNATURE",
            "This preview does not match the server's signing key. It may have been changed or issued before a key rotation. Generate a new preview; no write attempted.",
        )

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreviewTokenError(
            "PREVIEW_TOKEN_MALFORMED",
            "The signed preview payload is not valid JSON. Generate a new preview; no write attempted.",
        ) from exc
    if not isinstance(payload, dict):
        raise PreviewTokenError(
            "PREVIEW_TOKEN_MALFORMED",
            "The preview payload has an invalid structure. Generate a new preview; no write attempted.",
        )
    if type(payload.get("v")) is not int or payload["v"] != 1:
        raise PreviewTokenError(
            "PREVIEW_UNSUPPORTED_VERSION",
            "This preview was generated by an incompatible server version. Generate a new preview; no write attempted.",
        )
    expires_at = payload.get("expires_at")
    if type(expires_at) is not int or expires_at <= 0:
        raise PreviewTokenError(
            "PREVIEW_TOKEN_MALFORMED",
            "The preview has no valid expiry timestamp. Generate a new preview; no write attempted.",
        )
    if expires_at <= time.time():
        raise PreviewTokenError(
            "PREVIEW_EXPIRED",
            "The preview expired before confirmation. Generate a new preview and obtain approval again; no write attempted.",
        )
    if expected_kind:
        actual_kind = payload.get("kind")
        legacy_edit = (
            expected_kind == "edit"
            and actual_kind is None
            and isinstance(payload.get("transaction_id"), str)
            and isinstance(payload.get("changes"), dict)
        )
        if actual_kind != expected_kind and not legacy_edit:
            raise PreviewTokenError(
                "PREVIEW_WRONG_OPERATION",
                f"This token is not a valid {expected_kind} preview. Generate the correct preview; no write attempted.",
            )
    return payload


def _validate_edit_preview(authorized: dict[str, Any], claim: dict[str, Any]) -> None:
    """Explain stale records separately from changed edit inputs."""
    if (authorized.get("wallet_id") != claim["wallet_id"]
            or authorized.get("transaction_id") != claim["transaction_id"]):
        raise PreviewTokenError(
            "PREVIEW_INPUT_MISMATCH",
            "The target wallet or transaction differs from the approved edit preview. Preview again; no write attempted.",
        )
    if authorized.get("updated_at") != claim["updated_at"]:
        raise PreviewTokenError(
            "PREVIEW_STALE",
            "This transaction changed in Spendee after the preview. Read its latest details before creating a new edit preview; no write attempted.",
        )
    if authorized.get("changes") != claim["changes"]:
        raise PreviewTokenError(
            "PREVIEW_INPUT_MISMATCH",
            "The requested edits differ from the approved preview. Preview the exact changes again; no write attempted.",
        )


def _utc_timestamp(value: str) -> tuple[str, str, int]:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise ValueError("occurred_at must be ISO 8601, e.g. 2026-11-14T15:30:00+08:00") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("occurred_at must have a UTC offset, e.g. +08:00")
    timezone_name = Settings.from_env().timezone
    local = parsed.astimezone(ZoneInfo(timezone_name))
    offset = int(local.utcoffset().total_seconds())
    utc = parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return utc, timezone_name, offset


def _positive_decimal(value: float) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("amount must be a finite positive number") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("amount must be a finite positive number")
    if number.as_tuple().exponent < -2 and number != number.quantize(Decimal("0.01")):
        raise ValueError("amount must not have more than two decimal places")
    return number


def _fetch_transaction(api: Any, wallet_id: ResourceId, transaction_id: str):
    """Read exactly one document in the user's wallet; never look up by note."""
    try:
        canonical_id = str(UUID(transaction_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("transaction_id must be the UUID returned by list_transactions") from exc
    wallet = api._resolve_firestore_wallet(wallet_id)
    user_id = api.firestore_user_id
    wallet_uuid = wallet["id"]
    name = (
        f"{api._firestore_documents_name}/users/{user_id}/wallets/"
        f"{wallet_uuid}/transactions/{canonical_id}"
    )
    url = f"{api._firestore_documents_url}/users/{user_id}/wallets/{wallet_uuid}/transactions/{canonical_id}"
    raw = api._firestore_request("GET", url).json()
    if raw.get("name") != name or not raw.get("updateTime"):
        raise RuntimeError("Unexpected transaction document; refusing to update")
    doc = _decode_document(raw)
    if doc.get("type") != "regular":
        raise ValueError("Only regular manual transactions can be edited")
    return wallet, raw, doc


def _build_edit(
    api: Any,
    wallet: dict[str, Any],
    current: dict[str, Any],
    *,
    amount: float | None,
    transaction_type: TransactionType | None,
    category_id: ResourceId | None,
    note: str | None,
    occurred_at: str | None,
    labels: list[str] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    changes: dict[str, Any] = {}
    firestore_fields: dict[str, Any] = {}
    before: dict[str, Any] = {
        "amount": current.get("amount"),
        "category": current.get("category"),
        "note": current.get("note"),
        "made_at": current.get("madeAt"),
        "currency": wallet.get("currency"),
    }
    old_amount = Decimal(str(current["amount"]))
    old_kind = "expense" if old_amount < 0 else "income"
    next_kind = transaction_type or old_kind
    if next_kind not in ("income", "expense"):
        raise ValueError("transaction_type must be expense or income")
    if next_kind != old_kind and category_id is None:
        raise ValueError("Changing expense/income requires a matching new category_id")
    new_amount = _positive_decimal(amount) if amount is not None else abs(old_amount)
    signed_amount = -new_amount if next_kind == "expense" else new_amount
    if signed_amount != old_amount:
        if current.get("customCurrencyValue"):
            raise ValueError("Editing foreign-currency amounts is not supported; use the Spendee app")
        currency = str(wallet.get("currency") or "").upper()
        if currency != "SGD":
            raise ValueError("Amount edits in this version are limited to SGD wallets")
        units_per_usd = Decimal(str(api._usd_exchange_rate(currency)))
        if units_per_usd <= 0:
            raise ValueError("Invalid exchange rate from Spendee")
        formatted = api._decimal_text(signed_amount)
        changes["amount"] = formatted
        firestore_fields["amount"] = _encode_value(formatted)
        firestore_fields["usdValue"] = _encode_value({
            "amount": api._decimal_text(signed_amount / units_per_usd),
            "exchangeRate": api._decimal_text(units_per_usd),
        })

    if category_id is not None:
        category = api._resolve_firestore_category(category_id)
        if category.get("type") != next_kind:
            raise ValueError("The category type must match expense/income")
        if category.get("state") not in (None, "active"):
            raise ValueError("Category is not active")
        if category["id"] != current.get("category"):
            changes["category_id"] = category_id
            changes["firestore_category_id"] = category["id"]
            firestore_fields["category"] = _encode_value(category["id"])

    if note is not None:
        if len(note) > 1000:
            raise ValueError("note exceeds 1000 characters")
        if note != (current.get("note") or ""):
            changes["note"] = note
            firestore_fields["note"] = _encode_value(note)

    if occurred_at is not None:
        timestamp, timezone_name, offset = _utc_timestamp(occurred_at)
        # Normalize different ISO spellings for the same instant.
        old_timestamp = current.get("madeAt")
        if old_timestamp:
            old_timestamp = dt.datetime.fromisoformat(old_timestamp.replace("Z", "+00:00"))
            new_timestamp = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            changed_time = old_timestamp != new_timestamp
        else:
            changed_time = True
        if changed_time:
            changes["occurred_at"] = timestamp
            firestore_fields["madeAt"] = {"timestampValue": timestamp}
            firestore_fields["madeAtTimezone"] = _encode_value(timezone_name)
            firestore_fields["madeAtTimezoneOffset"] = _encode_value(offset)

    if labels is not None:
        resolved = api._resolve_label_names(labels)
        target_labels = sorted(resolved.values(), key=str.casefold)
        actual = api.get_transaction_labels(wallet["id"], current["_id"])
        before["labels"] = actual
        if target_labels != sorted(actual, key=str.casefold):
            changes["labels"] = target_labels

    return changes, firestore_fields, before


def _edit_transaction_sync(
    wallet_id: ResourceId,
    transaction_id: str,
    *,
    amount: float | None,
    transaction_type: TransactionType | None,
    category_id: ResourceId | None,
    note: str | None,
    occurred_at: str | None,
    labels: list[str] | None,
    confirm: bool,
    preview_token: str | None,
) -> dict[str, Any]:
    # The upstream gateway owns the shared Spendee session and its lock.
    with _gateway._locked("edit_transaction"):
        api = _gateway._get_api()
        wallet, raw, doc = _fetch_transaction(api, wallet_id, transaction_id)
        changes, firestore_fields, before = _build_edit(
            api, wallet, doc,
            amount=amount,
            transaction_type=transaction_type,
            category_id=category_id,
            note=note,
            occurred_at=occurred_at,
            labels=labels,
        )
        if not changes:
            return {"status": "no_change", "transaction_id": transaction_id, "current": before}

        claim = {
            "v": 1,
            "wallet_id": str(wallet_id),
            "transaction_id": str(UUID(transaction_id)),
            "updated_at": raw["updateTime"],
            "changes": changes,
        }
        if not confirm:
            claim["expires_at"] = int(time.time()) + 15 * 60
            return {
                "status": "preview",
                "wallet_id": wallet_id,
                "transaction_id": transaction_id,
                "before": before,
                "proposed_changes": changes,
                "expires_in_minutes": 15,
                "preview_token": _seal_preview(claim),
                "requires_explicit_user_confirmation": True,
            }

        authorized = _unseal_preview(preview_token, expected_kind="edit")
        _validate_edit_preview(authorized, claim)

        if firestore_fields:
            # Update only changed fields; compare-and-swap protects against a
            # concurrent modification from the Spendee app or another client.
            write = {
                "update": {"name": raw["name"], "fields": firestore_fields},
                "updateMask": {"fieldPaths": sorted(firestore_fields)},
                "currentDocument": {"updateTime": raw["updateTime"]},
                "updateTransforms": [{
                    "fieldPath": "updatedAt",
                    "setToServerValue": "REQUEST_TIME",
                }],
            }
            api._firestore_request(
                "POST", f"{api._firestore_documents_url}:commit", json={"writes": [write]}
            )
            # Check the changed fields with normalized numbers/timestamps:
            # Firestore may rewrite "45" as "45.0" or add fractional seconds.
            url = f"{api._firestore_documents_url}/" + raw["name"].split("/documents/", 1)[1]
            after = _decode_document(api._firestore_request("GET", url).json())
            from spendee.firestore import _decode_value
            for key in ("amount", "note", "category", "madeAt", "madeAtTimezone", "madeAtTimezoneOffset", "usdValue"):
                if key not in firestore_fields:
                    continue
                proposed = _decode_value(firestore_fields[key])
                received = after.get(key)
                if key == "amount":
                    equal = Decimal(str(received)) == Decimal(str(proposed))
                elif key == "madeAt":
                    equal = (dt.datetime.fromisoformat(str(received).replace("Z", "+00:00"))
                             == dt.datetime.fromisoformat(str(proposed).replace("Z", "+00:00")))
                elif key == "usdValue":
                    equal = (isinstance(received, dict)
                             and Decimal(str(received.get("amount"))) == Decimal(str(proposed.get("amount")))
                             and Decimal(str(received.get("exchangeRate"))) == Decimal(str(proposed.get("exchangeRate"))))
                else:
                    equal = received == proposed
                if not equal:
                    raise RuntimeError("Update may have applied, but readback differed; inspect Spendee")

        result = {"status": "updated", "transaction_id": transaction_id, "changes": changes}
        if "labels" in changes:
            try:
                label_result = api.set_transaction_labels(
                    wallet["id"], str(UUID(transaction_id)), changes["labels"]
                )
                result["labels_result"] = label_result
            except Exception:
                # Field updates may have succeeded, so never retry blindly.
                result["status"] = "partial_success_labels_failed"
                result["warning"] = (
                    "Other changes may have saved, but labels failed. "
                    "Inspect the transaction and preview again before retrying."
                )
        return result


@mcp.tool()
async def update_transaction(
    wallet_id: ResourceId,
    transaction_id: str,
    amount: float | None = None,
    transaction_type: TransactionType | None = None,
    category_id: ResourceId | None = None,
    note: str | None = None,
    occurred_at: str | None = None,
    labels: list[str] | None = None,
    confirm: bool = False,
    preview_token: str | None = None,
) -> dict[str, Any]:
    """Preview/edit a regular Spendee transaction by exact UUID.

    First call with confirm=false, display the proposed changes, and STOP.
    Call confirm=true only AFTER the user explicitly approves this exact
    preview in a separate chat turn. Include unchanged inputs AND the returned
    preview_token. Preview expires in 15 minutes and rejects stale data.

    Supports SGD amount, expense/income type, category, note, timestamp and
    existing labels. None means 'leave unchanged'; labels=[] clears labels;
    note='' clears the note. Dates require ISO 8601 UTC offset (+08:00).
    Does not support deletion, wallet moves, transfers, or editing foreign
    currency amounts. UNOFFICIAL private API: test on a disposable entry.
    """
    require_owner()
    return await anyio.to_thread.run_sync(
        lambda: _edit_transaction_sync(
            wallet_id=wallet_id,
            transaction_id=transaction_id,
            amount=amount,
            transaction_type=transaction_type,
            category_id=category_id,
            note=note,
            occurred_at=occurred_at,
            labels=labels,
            confirm=confirm,
            preview_token=preview_token,
        )
    )


# ---- Approval-bound, persistent batch creation -----------------------------
# A prepared batch is immutable. The assistant must show the exact preview and
# wait for the user's explicit approval in a NEW conversation turn before it
# may call commit_approved_batch. Neither preparation nor status checks write
# any transactions. Redis stores encrypted operation state for safe recovery.

_BATCH_PREFIX = "spendee:mcp:batch:v1:"
_BATCH_APPROVAL_SECONDS = 24 * 60 * 60
_BATCH_RETENTION_SECONDS = 30 * 24 * 60 * 60
_BATCH_MAX_ITEMS = 50
_BATCH_PAGE_SIZE = 100
_BATCH_MAX_PAGES = 30
_ALLOWED_PAYMENT_LABELS = {
    "PayLah", "UOB", "MariBank", "Citibank Rewards", "Citibank PremMiles", "Trust",
}
_BATCH_CRYPT = Fernet(base64.urlsafe_b64encode(
    hashlib.sha256(("spendee-mcp-batch-storage-v1:" + env("JWT_SIGNING_KEY")).encode()).digest()
))


def _batch_key(batch_id: str) -> str:
    try:
        return _BATCH_PREFIX + str(UUID(batch_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("BATCH_ID_INVALID: batch_id must be the UUID returned by preparation") from exc


async def _batch_save(batch: dict[str, Any]) -> None:
    payload = json.dumps(batch, sort_keys=True, separators=(",", ":")).encode()
    encrypted = _BATCH_CRYPT.encrypt(payload).decode("ascii")
    ttl = (_BATCH_APPROVAL_SECONDS if batch["status"] == "prepared"
           else _BATCH_RETENTION_SECONDS)
    await _redis_client.set(_batch_key(batch["batch_id"]), encrypted, ex=ttl)


async def _batch_load(batch_id: str) -> dict[str, Any]:
    encrypted = await _redis_client.get(_batch_key(batch_id))
    if encrypted is None:
        raise ValueError("BATCH_NOT_FOUND: batch not found or expired. Prepare a new batch; no write attempted")
    try:
        loaded = json.loads(_BATCH_CRYPT.decrypt(encrypted.encode("ascii")))
    except Exception as exc:
        raise ValueError("BATCH_UNREADABLE: stored batch cannot be decrypted; check key rotation, do not retry blindly") from exc
    if not isinstance(loaded, dict) or loaded.get("batch_id") != str(UUID(batch_id)):
        raise ValueError("BATCH_UNREADABLE: stored batch is corrupt; no write attempted")
    return loaded


def _batch_normalize(raw: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Item {index}: expected a transaction object")
    allowed = {"category_id", "amount", "transaction_type", "note", "labels", "date", "occurred_at", "currency"}
    if set(raw) - allowed:
        raise ValueError(f"Item {index}: unexpected fields: {sorted(set(raw) - allowed)}")
    category_id = raw.get("category_id")
    if isinstance(category_id, bool) or not isinstance(category_id, (int, str)) or not str(category_id).strip():
        raise ValueError(f"Item {index}: valid existing category_id required")
    if raw.get("transaction_type", "expense") != "expense":
        raise ValueError(f"Item {index}: batch tool is for outgoing expenses only")
    amount = _positive_decimal(raw.get("amount"))
    labels = raw.get("labels", [])
    if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
        raise ValueError(f"Item {index}: labels must be a list of existing Spendee label names")
    if len(labels) != len(set(labels)) or any(x not in (_ALLOWED_PAYMENT_LABELS | {"For Split"}) for x in labels):
        raise ValueError(f"Item {index}: labels must use the approved Spendee spelling, without repetitions")
    payments = set(labels) & _ALLOWED_PAYMENT_LABELS
    if len(payments) > 1 or ("For Split" in labels and len(payments) != 1):
        raise ValueError(f"Item {index}: at most one payment method, optionally followed by For Split")
    note = raw.get("note", "")
    if not isinstance(note, str) or not note.strip() or len(note) > 1000:
        raise ValueError(f"Item {index}: non-empty note must be no more than 1000 characters")
    currency = raw.get("currency", "SGD")
    if not isinstance(currency, str) or currency.upper() != "SGD":
        raise ValueError(f"Item {index}: use the final SGD charge; foreign-currency batches are not supported")
    date = raw.get("date")
    occurred_at = raw.get("occurred_at")
    if date is None and occurred_at is None:
        raise ValueError(f"Item {index}: date (YYYY-MM-DD) or offset-aware occurred_at is required")
    local_tz = ZoneInfo(Settings.from_env().timezone)
    if date is not None:
        if not isinstance(date, str) or len(date) != 10 or date != dt.date.fromisoformat(date).isoformat():
            raise ValueError(f"Item {index}: invalid YYYY-MM-DD date")
    if occurred_at is not None:
        _utc_timestamp(occurred_at)  # Reject naive timestamps before saving.
        parsed = dt.datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        parsed_local = parsed.astimezone(local_tz)
        if date is not None and date != parsed_local.date().isoformat():
            raise ValueError(f"Item {index}: date and occurred_at refer to different Singapore dates")
        date = parsed_local.date().isoformat()
        stamp = parsed_local.isoformat(timespec="seconds")
    else:
        stamp = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(12, 0), tzinfo=local_tz).isoformat()
    return {
        "index": index, "category_id": category_id, "amount": format(amount, ".2f"),
        "transaction_type": "expense", "note": note, "labels": labels,
        "date": date, "occurred_at": stamp, "currency": "SGD",
    }


def _batch_args(wallet_id: ResourceId, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "wallet_id": wallet_id, "category_id": item["category_id"],
        "amount": float(item["amount"]), "transaction_type": "expense",
        "note": item["note"], "labels": item["labels"],
        "occurred_at": item["occurred_at"], "currency": item["currency"],
    }


def _batch_fingerprint(wallet_id: ResourceId, items: list[dict[str, Any]]) -> str:
    payload = {"wallet_id": str(wallet_id), "items": items}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _batch_record_id(record: dict[str, Any]) -> str | None:
    value = record.get("uuid") or record.get("id") or record.get("transaction_id")
    if value is not None and str(value).strip():
        return str(value)
    return None


def _batch_result_id(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    direct = _batch_record_id(result)
    if direct:
        return direct
    for field in ("transaction", "created_transaction", "result"):
        if isinstance(result.get(field), dict):
            value = _batch_record_id(result[field])
            if value:
                return value
    return None


def _batch_local_date(record: dict[str, Any]) -> str | None:
    stamp = record.get("start_date") or record.get("occurred_at") or record.get("madeAt")
    if not isinstance(stamp, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(ZoneInfo(Settings.from_env().timezone)).date().isoformat()
    except ValueError:
        return None


def _batch_match(item: dict[str, Any], record: dict[str, Any]) -> str | None:
    """Return exact/possible/None. Never claim exact if a field cannot be checked."""
    if record.get("type") != "expense" or record.get("status", "active") != "active":
        return None
    if _batch_local_date(record) != item["date"]:
        return None
    try:
        amount = Decimal(str(record["amount"]))
        if abs(amount) != Decimal(item["amount"]):
            return None
    except (KeyError, InvalidOperation, TypeError):
        return None
    same_note = isinstance(record.get("note"), str) and record["note"].strip().casefold() == item["note"].strip().casefold()
    same_category = record.get("category_id") is not None and str(record["category_id"]) == str(item["category_id"])
    raw_labels = record.get("labels")
    known_labels = isinstance(raw_labels, list)
    same_labels = known_labels and {str(x).casefold() for x in raw_labels} == {x.casefold() for x in item["labels"]}
    same_currency = not record.get("foreign_currency")  # SGD-wallet records generally have null foreign_currency.
    if same_note and same_category and same_labels and same_currency:
        return "exact"
    if same_note:
        return "possible"  # A match with different labels/category may be a miscategorized prior expense.
    return None  # Different descriptions/merchants are not automatically duplicates.


def _batch_month_bounds(dates: list[str]) -> list[tuple[str, str]]:
    # One narrow, paginated query per affected calendar month, widened by a day
    # at either end because Spendee stores instants in UTC.
    months = sorted({s[:7] for s in dates})
    ranges = []
    for month in months:
        first = dt.date.fromisoformat(month + "-01")
        next_month = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        ranges.append(((first - dt.timedelta(days=1)).isoformat(),
                       (next_month + dt.timedelta(days=1)).isoformat()))
    return ranges


def _batch_fetch_existing(wallet_id: ResourceId, dates: list[str]) -> list[dict[str, Any]]:
    """Fail closed on any incomplete existing-transaction check."""
    records: dict[str, dict[str, Any]] = {}
    for date_from, date_to in _batch_month_bounds(dates):
        for page in range(_BATCH_MAX_PAGES):
            chunk = _gateway.list_transactions(
                wallet_id=wallet_id, offset=page * _BATCH_PAGE_SIZE,
                limit=_BATCH_PAGE_SIZE, include_labels=True,
                date_from=date_from, date_to=date_to,
            )
            if not isinstance(chunk, list):
                raise RuntimeError("BATCH_DUPLICATE_CHECK_FAILED: expected a list of Spendee transactions")
            for n, record in enumerate(chunk):
                if not isinstance(record, dict):
                    raise RuntimeError("BATCH_DUPLICATE_CHECK_FAILED: unexpected transaction record")
                key = _batch_record_id(record) or f"{date_from}:{page}:{n}"
                records[key] = record
            if len(chunk) < _BATCH_PAGE_SIZE:
                break
        else:
            raise RuntimeError("BATCH_DUPLICATE_CHECK_INCOMPLETE: too many transactions; narrow the batch dates")
    return list(records.values())


def _batch_public(batch: dict[str, Any]) -> dict[str, Any]:
    results = batch["results"]
    statuses = {status: sum(r["status"] == status for r in results)
                for status in {r["status"] for r in results}}
    verified_total = sum((Decimal(batch["items"][p]["amount"])
                          for p, r in enumerate(results) if r["status"] == "verified"), Decimal("0"))
    unverified_total = sum((Decimal(batch["items"][p]["amount"])
                            for p, r in enumerate(results)
                            if r["status"] in ("unverified", "in_flight", "created_unverified")), Decimal("0"))
    return {
        "status": batch["status"], "batch_id": batch["batch_id"],
        "wallet_id": batch["wallet_id"], "fingerprint": batch["fingerprint"],
        "count": len(batch["items"]), "total_sgd": batch["total_sgd"],
        "verified_count": statuses.get("verified", 0),
        "verified_total_sgd": format(verified_total, ".2f"),
        "unverified_count": (statuses.get("unverified", 0)
                             + statuses.get("in_flight", 0) + statuses.get("created_unverified", 0)),
        "unverified_total_sgd": format(unverified_total, ".2f"),
        "items": [{**{k: item[k] for k in ("index", "date", "amount", "category_id", "note", "labels", "currency")},
                   "status": r["status"], "transaction_id": r.get("transaction_id"),
                   "error_code": r.get("error_code")}
                  for item, r in zip(batch["items"], results)],
        "status_counts": statuses, "excluded_duplicates": batch["excluded_duplicates"],
        "needs_clarification": batch["needs_clarification"],
        "note": "Unverified and in-flight writes must be reconciled against Spendee before retrying.",
    }


@mcp.tool()
async def prepare_transaction_batch(
    wallet_id: ResourceId,
    transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """READ-ONLY: prepare up to 50 SGD expenses for ONE explicit batch approval.

    Each entry: {category_id, amount, date:'YYYY-MM-DD' OR
    occurred_at:'ISO8601+08:00', note, labels:[], currency:'SGD'}.
    Labels may be empty for user-approved fund allocations. Outgoing expenses
    only. No new categories or labels are created. Checks Spendee for duplicates;
    exact matches are excluded and ambiguous matches held for clarification.

    ALWAYS show the returned ready_to_log table, wallet, count, total, excluded
    duplicates, and clarification items, then STOP and wait for explicit approval
    in a SEPARATE user turn. Do not call commit_approved_batch in this turn.
    This tool only reads Spendee and saves encrypted preparation data in Redis.
    """
    require_owner()
    if isinstance(wallet_id, bool) or not isinstance(wallet_id, (int, str)) or not str(wallet_id).strip():
        raise ValueError("A valid, previously confirmed wallet_id is required")
    if not isinstance(transactions, list) or not 1 <= len(transactions) <= _BATCH_MAX_ITEMS:
        raise ValueError(f"Provide between 1 and {_BATCH_MAX_ITEMS} transactions")
    items = [_batch_normalize(x, i + 1) for i, x in enumerate(transactions)]
    existing = await anyio.to_thread.run_sync(
        lambda: _batch_fetch_existing(wallet_id, [x["date"] for x in items])
    )
    duplicates: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for item in items:
        signature = _batch_fingerprint(wallet_id, [{k: v for k, v in item.items() if k != "index"}])
        if signature in seen:
            uncertain.append({"index": item["index"], "transaction": item,
                              "reason": f"Indistinguishable from another batch entry (row {seen[signature]})"})
            continue
        seen[signature] = item["index"]
        exact = [r for r in existing if _batch_match(item, r) == "exact"]
        possible = [r for r in existing if _batch_match(item, r) == "possible"]
        if exact:
            duplicates.append({"index": item["index"], "transaction": item,
                               "existing_transaction_ids": [_batch_record_id(r) for r in exact]})
        elif possible:
            uncertain.append({"index": item["index"], "transaction": item,
                              "reason": "Potential existing Spendee match with incomplete/different details",
                              "existing_transaction_ids": [_batch_record_id(r) for r in possible]})
        else:
            ready.append(item)
    ready.sort(key=lambda item: (item["date"], item["occurred_at"], item["index"]))
    if not ready:
        return {"status": "nothing_to_prepare", "wallet_id": wallet_id,
                "count": 0, "total_sgd": "0.00", "ready_to_log": [],
                "excluded_duplicates": duplicates, "needs_clarification": uncertain,
                "requires_explicit_user_confirmation": False}
    gateway_previews = []
    for item in ready:
        result = await anyio.to_thread.run_sync(
            lambda item=item: _gateway.create_transaction(**_batch_args(wallet_id, item), confirm=False)
        )
        if not isinstance(result, dict) or not isinstance(result.get("transaction"), dict):
            raise RuntimeError(f"BATCH_PREFLIGHT_FAILED: no valid creation preview for row {item['index']}")
        if result.get("status") in ("duplicate", "already_exists"):
            raise RuntimeError(f"BATCH_PREFLIGHT_DUPLICATE: row {item['index']} was rejected as duplicate")
        gateway_previews.append(result["transaction"])
    batch_id = str(uuid4())
    fingerprint = _batch_fingerprint(wallet_id, ready)
    expires = int(time.time()) + _BATCH_APPROVAL_SECONDS
    approval_claim = {"v": 1, "kind": "batch", "batch_id": batch_id,
                      "wallet_id": str(wallet_id), "fingerprint": fingerprint,
                      "expires_at": expires}
    batch = {
        "v": 1, "batch_id": batch_id, "fingerprint": fingerprint,
        "wallet_id": wallet_id, "status": "prepared", "created_at": int(time.time()),
        "expires_at": expires, "items": ready, "gateway_previews": gateway_previews,
        "baseline_ids": sorted({id for r in existing if (id := _batch_record_id(r))}),
        "results": [{"index": x["index"], "status": "not_attempted"} for x in ready],
        "total_sgd": format(sum((Decimal(x["amount"]) for x in ready), Decimal("0")), ".2f"),
        "excluded_duplicates": duplicates, "needs_clarification": uncertain,
    }
    await _batch_save(batch)
    return {
        "status": "prepared", "batch_id": batch_id,
        "approval_token": _seal_preview(approval_claim),
        "fingerprint": fingerprint, "expires_in_hours": 24,
        "wallet_id": wallet_id, "count": len(ready), "total_sgd": batch["total_sgd"],
        "ready_to_log": ready, "excluded_duplicates": duplicates,
        "needs_clarification": uncertain,
        "requires_explicit_user_confirmation": True,
        "instruction": "Show the full approved batch to the user and STOP; commit only after a separate explicit user approval.",
    }


@mcp.tool()
async def commit_approved_batch(batch_id: str, approval_token: str) -> dict[str, Any]:
    """WRITE: commit the exact prepared batch, ONLY after the user explicitly
    approves the displayed batch in a separate conversation turn.

    Use the opaque approval_token from prepare_transaction_batch; no second
    preview is needed. The token is bound to the immutable batch fingerprint,
    wallet, and 24-hour expiry. Never call during the prepare turn or invent
    an approval. This is not atomic: the server journals every item before
    writing, never auto-retries uncertain writes, and verifies via Spendee readback.
    """
    require_owner()
    authorized = _unseal_preview(approval_token, expected_kind="batch")
    batch = await _batch_load(batch_id)
    if (authorized.get("batch_id") != batch["batch_id"]
            or authorized.get("fingerprint") != batch["fingerprint"]
            or authorized.get("wallet_id") != str(batch["wallet_id"])
            or authorized.get("expires_at") != batch["expires_at"]
            or _batch_fingerprint(batch["wallet_id"], batch["items"]) != batch["fingerprint"]):
        raise PreviewTokenError("PREVIEW_INPUT_MISMATCH", "Batch contents or approval changed; no write attempted")
    if batch["status"] != "prepared":
        return _batch_public(batch)  # Never execute the same approved batch twice.
    lock_key = _batch_key(batch_id) + ":lock"
    lock_value = str(uuid4())
    locked = await _redis_client.set(lock_key, lock_value, ex=3600, nx=True)
    if not locked:
        return {"status": "batch_busy", "batch_id": batch_id,
                "instruction": "Use get_batch_status; do not resend a creation request."}
    try:
        # Re-read under lock: a concurrent caller may have finished the batch.
        batch = await _batch_load(batch_id)
        if batch["status"] != "prepared":
            return _batch_public(batch)
        batch["status"] = "processing"
        await _batch_save(batch)
        try:
            existing = await anyio.to_thread.run_sync(
                lambda: _batch_fetch_existing(batch["wallet_id"], [x["date"] for x in batch["items"]])
            )
        except Exception:
            batch["status"] = "needs_reconciliation"
            batch["failure_code"] = "BATCH_PREWRITE_READ_FAILED"
            await _batch_save(batch)
            return _batch_public(batch)

        for position, item in enumerate(batch["items"]):
            row = batch["results"][position]
            matches = [r for r in existing if _batch_match(item, r) in ("exact", "possible")]
            if matches:
                row["status"] = "skipped_existing_duplicate" if all(_batch_match(item, r) == "exact" for r in matches) else "needs_clarification"
                row["existing_transaction_ids"] = [_batch_record_id(r) for r in matches]
                await _batch_save(batch)
                continue
            try:
                preview = await anyio.to_thread.run_sync(
                    lambda item=item: _gateway.create_transaction(**_batch_args(batch["wallet_id"], item), confirm=False)
                )
                if not isinstance(preview, dict) or preview.get("transaction") != batch["gateway_previews"][position]:
                    row["status"] = "blocked_changed"
                    row["error_code"] = "BATCH_PREVIEW_CHANGED"
                    await _batch_save(batch)
                    continue
            except Exception:
                row["status"] = "blocked_changed"
                row["error_code"] = "BATCH_PREFLIGHT_FAILED"
                await _batch_save(batch)
                continue

            # Durable in-flight record BEFORE sending the write. A timeout
            # after this point is ambiguous and must never trigger blind retry.
            request_id = str(uuid5(NAMESPACE_URL, f"spendee-batch:{batch_id}:{item['index']}"))
            row.update({"status": "in_flight", "request_id": request_id})
            await _batch_save(batch)
            try:
                response = await anyio.to_thread.run_sync(
                    lambda item=item, request_id=request_id: _gateway.create_transaction(
                        **_batch_args(batch["wallet_id"], item), confirm=True,
                        request_id=request_id, allow_duplicate=False,
                    )
                )
                if not isinstance(response, dict) or response.get("status") in ("error", "failed"):
                    raise RuntimeError("Gateway did not confirm transaction creation")
                row.update({"status": "created_unverified", "transaction_id": _batch_result_id(response)})
                await _batch_save(batch)
            except Exception:
                row.update({"status": "unverified", "error_code": "BATCH_WRITE_OUTCOME_UNKNOWN"})
                batch["status"] = "needs_reconciliation"
                await _batch_save(batch)
                break

        # Read back successful writes. Read-only verification must not retry
        # the create action or turn an ambiguous response into a success.
        created = [(p, item, batch["results"][p]) for p, item in enumerate(batch["items"])
                   if batch["results"][p]["status"] == "created_unverified"]
        if created:
            try:
                observed = await anyio.to_thread.run_sync(
                    lambda: _batch_fetch_existing(batch["wallet_id"], [item["date"] for _, item, _ in created])
                )
                baseline = set(batch["baseline_ids"])
                used_ids: set[str] = set()
                for _, item, row in created:
                    candidates = [r for r in observed if _batch_match(item, r) == "exact"
                                  and _batch_record_id(r) not in baseline
                                  and _batch_record_id(r) not in used_ids]
                    if row.get("transaction_id"):
                        candidates = [r for r in candidates if row["transaction_id"] in
                                      (str(r.get("id")), str(r.get("uuid")))]
                    if len(candidates) == 1:
                        verified_id = _batch_record_id(candidates[0])
                        row.update({"status": "verified", "transaction_id": verified_id})
                        used_ids.add(verified_id)
                    else:
                        row.update({"status": "unverified", "error_code": "BATCH_READBACK_MISMATCH"})
                    await _batch_save(batch)
            except Exception:
                for _, _, row in created:
                    if row["status"] == "created_unverified":
                        row.update({"status": "unverified", "error_code": "BATCH_READBACK_FAILED"})
                await _batch_save(batch)
        statuses = {r["status"] for r in batch["results"]}
        if "unverified" in statuses or "not_attempted" in statuses or "in_flight" in statuses:
            batch["status"] = "needs_reconciliation"
        elif statuses == {"verified"}:
            batch["status"] = "completed"
        else:
            batch["status"] = "completed_with_exclusions"
        await _batch_save(batch)
        return _batch_public(batch)
    finally:
        # Release only our lock; never delete a newer owner's lock.
        try:
            await _redis_client.eval(
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "return redis.call('DEL', KEYS[1]) else return 0 end",
                1, lock_key, lock_value,
            )
        except Exception:
            # A lock cleanup outage must not hide a persisted write outcome.
            # Redis will expire the lock; retrieve the batch via get_batch_status.
            pass


@mcp.tool()
async def get_batch_status(batch_id: str) -> dict[str, Any]:
    """READ-ONLY: inspect persisted batch outcomes. This never initiates or
    retries a Spendee write, including when a past request was interrupted."""
    require_owner()
    batch = await _batch_load(batch_id)
    result = _batch_public(batch)
    if batch["status"] == "processing":
        lock_exists = await _redis_client.get(_batch_key(batch_id) + ":lock")
        if not lock_exists:
            # A crashed worker may have sent a write before its reply was saved.
            for row in batch["results"]:
                if row["status"] == "in_flight":
                    row["status"] = "unverified"
                    row["error_code"] = "BATCH_INTERRUPTED_IN_FLIGHT"
            batch["status"] = "needs_reconciliation"
            await _batch_save(batch)
            result = _batch_public(batch)
        else:
            result["instruction"] = "Batch is still processing. Recheck status; do not resend writes."
    if batch["status"] == "prepared" and batch["expires_at"] > time.time():
        result["ready_to_log"] = batch["items"]
        result["approval_token"] = _seal_preview({
            "v": 1, "kind": "batch", "batch_id": batch["batch_id"],
            "wallet_id": str(batch["wallet_id"]), "fingerprint": batch["fingerprint"],
            "expires_at": batch["expires_at"],
        })
        result["instruction"] = "Display the batch and wait for explicit user approval in a separate turn before committing."
    return result



if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
    )
