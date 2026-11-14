"""Single-user Spendee MCP for Render: read, create and carefully edit transactions.

Upload this file to the root of your fork of https://github.com/gibsn/mcp-spendee.
This is an unofficial integration. OAuth uses durable encrypted Redis storage.
Editing uses Firestore's REST commit API and is
NOT proven against a live Spendee account: first test a disposable transaction.

Installation (Render build):
    pip install -e . 'mcp>=1.30,<2' 'fastmcp>=2.13,<3' 'py-key-value-aio[redis]' cryptography
Start command: python spendee_remote_persistent.py

Required Render secrets: GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET,
PUBLIC_BASE_URL, JWT_SIGNING_KEY, GITHUB_ALLOWED_LOGIN, REDIS_URL,
SPENDEE_EMAIL, SPENDEE_PASSWORD.
Other settings: SPENDEE_TIMEZONE=Asia/Singapore,
SPENDEE_GLOBAL_CURRENCY=SGD, PYTHON_VERSION=3.12.11.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import anyio
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token
from cryptography.fernet import Fernet
from key_value.aio.stores.redis import RedisStore
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
_oauth_storage = FernetEncryptionWrapper(
    key_value=RedisStore(url=env("REDIS_URL")),
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

    if not request_id or not preview_token:
        raise ValueError("Confirmed creation requires the preview_token and a unique request_id")
    authorization = _unseal_preview(preview_token)
    if authorization.get("kind") != "create":
        raise ValueError("Wrong preview type; preview a creation first")
    if (str(wallet_id) != authorization.get("wallet_id")
            or str(category_id) != authorization.get("category_id")):
        raise ValueError("Wallet/category changed; preview again")
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
            raise ValueError("Date changed; preview again")
    frozen_currency = frozen.get("foreign_currency") or frozen.get("currency")
    if currency is not None and currency.upper() != frozen_currency:
        raise ValueError("Currency changed; preview again")
    fixed_rate = frozen.get("foreign_rate") if frozen.get("foreign_currency") else None
    if exchange_rate is not None and fixed_rate is not None:
        if Decimal(str(exchange_rate)) != Decimal(str(fixed_rate)):
            raise ValueError("Exchange rate changed; preview again")
    if exchange_rate is not None and fixed_rate is None:
        raise ValueError("Exchange rate changed; preview again")

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
        raise ValueError("Transaction details changed; preview again")
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
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _seal_preview(payload: dict[str, Any]) -> str:
    body = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    sig = hmac.new(env("JWT_SIGNING_KEY").encode(), body.encode(), hashlib.sha256).digest()
    return body + "." + _b64url(sig)


def _unseal_preview(token: str) -> dict[str, Any]:
    try:
        body, signature = token.split(".", 1)
        actual = _unb64url(signature)
        expected = hmac.new(env("JWT_SIGNING_KEY").encode(), body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(actual, expected):
            raise ValueError("Invalid preview signature")
        payload = json.loads(_unb64url(body))
        if payload.get("v") != 1 or payload.get("expires_at", 0) <= time.time():
            raise ValueError("Preview expired; preview again")
        return payload
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or expired edit preview; preview the change again") from exc


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

        if not preview_token:
            raise ValueError("Preview first and provide its preview_token to confirm")
        authorized = _unseal_preview(preview_token)
        expected = dict(claim)
        expected["expires_at"] = authorized.get("expires_at")
        if authorized != expected:
            raise ValueError("Transaction changed or preview does not match; preview again")

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


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
    )
