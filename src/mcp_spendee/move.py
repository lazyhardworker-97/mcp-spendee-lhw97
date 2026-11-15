"""Move a regular transaction and its label relations between Firestore wallets."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

from spendee import SpendeeFirestoreError
from spendee.firestore import _decode_document

from mcp_spendee.diagnostics import event, resource_tag

_DOCUMENT_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
# Restrict raw relocation to the known regular transaction shape. Other fields
# can carry references that must be updated by the application that owns them.
_TRANSACTION_FIELDS = {
    "amount",
    "author",
    "category",
    "createdAt",
    "customCurrencyValue",
    "madeAt",
    "madeAtTimezone",
    "madeAtTimezoneOffset",
    "modelVersion",
    "note",
    "path",
    "status",
    "type",
    "updatedAt",
    "usdValue",
}
_LABEL_FIELDS = {"author", "createdAt", "label", "modelVersion", "path", "updatedAt"}


def _get(api: Any, path: str) -> dict[str, Any] | None:
    try:
        return api._firestore_request("GET", f"{api._firestore_documents_url}/{path}").json()
    except SpendeeFirestoreError as exc:
        # The pinned Spendee client can drop the HTTP response on its exception.
        if getattr(exc.response, "status_code", None) == 404 or str(exc).startswith(
            "Firestore request failed with HTTP 404:"
        ):
            return None
        raise


def _labels(api: Any, path: str) -> list[dict[str, Any]]:
    url = f"{api._firestore_documents_url}/{path}/transactionLabels"
    documents = []
    params: dict[str, Any] = {"pageSize": 100}
    while True:
        page = api._firestore_request("GET", url, params=params).json()
        documents.extend(page.get("documents", []))
        if not (token := page.get("nextPageToken")):
            return sorted(documents, key=lambda item: item["name"])
        params = {**params, "pageToken": token}


def _wallet(api: Any, user: str, wallets: list[dict[str, Any]], requested: int | str) -> dict:
    matches = [w for w in wallets if requested in (w["id"], w.get("legacy_id"))]
    if len(matches) != 1:
        raise ValueError("Wallet ID must identify exactly one existing wallet")
    wallet_id = matches[0]["id"]
    if not _DOCUMENT_ID.fullmatch(wallet_id):
        raise ValueError("Unsupported wallet ID")
    raw = _get(api, f"users/{user}/wallets/{wallet_id}")
    if raw is None:
        raise ValueError("Wallet no longer exists")
    wallet = _decode_document(raw)
    if (
        wallet.get("type") != "cash"
        or wallet.get("status") != "active"
        or wallet.get("path") != {"user": user, "wallet": wallet_id}
        or not wallet.get("currency")
    ):
        raise ValueError("Moves require active manual wallets owned by the current user")
    return wallet


def _read(
    api: Any, path: str, user: str, wallet: str, tx_id: str, raw: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transaction = _decode_document(raw)
    expected = {"user": user, "wallet": wallet, "transaction": tx_id}
    if (
        transaction.get("type") != "regular"
        or transaction.get("modelVersion") != 1
        or transaction.get("author") != user
        or transaction.get("status") not in (None, "active")
        or transaction.get("path") != expected
        or not transaction.get("category")
        or not transaction.get("madeAt")
        or raw["fields"].keys() - _TRANSACTION_FIELDS
    ):
        raise ValueError("Only regular transactions owned by the current user are supported")
    try:
        amount = Decimal(str(transaction.get("amount")))
    except InvalidOperation as exc:
        raise ValueError("Transaction has an invalid amount") from exc
    if not amount.is_finite() or not amount:
        raise ValueError("Transaction has an invalid amount")

    documents = [raw, *_labels(api, path)]
    if len(documents) > 200:
        raise ValueError("Too many labels to move safely in a single commit")
    for doc in documents[1:]:
        label = _decode_document(doc)
        if (
            doc["fields"].keys() - _LABEL_FIELDS
            or label.get("author") != user
            or label.get("modelVersion") != 1
            or not label.get("label")
            or not _DOCUMENT_ID.fullmatch(label["_id"])
            or label.get("path") != {**expected, "transactionLabel": label["_id"]}
        ):
            raise ValueError("Transaction has an unsupported label relation")
    return documents, transaction


def _normalized(docs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            doc["name"].rsplit("/", 1)[-1],
            {
                k: v
                for k, v in doc["fields"].items()
                if k != "path" and not (index == 0 and k == "updatedAt")
            },
        )
        for index, doc in enumerate(docs)
    ]


def _writes(
    docs: list[dict[str, Any]], source_name: str, target_name: str, target_id: str
) -> list[dict[str, Any]]:
    writes = []
    for doc in docs:
        fields = deepcopy(doc["fields"])
        fields["path"]["mapValue"]["fields"]["wallet"] = {"stringValue": target_id}
        write: dict[str, Any] = {
            "update": {
                "name": target_name + doc["name"].removeprefix(source_name),
                "fields": fields,
            },
            "currentDocument": {"exists": False},
        }
        if doc["name"] == source_name:
            fields.pop("updatedAt", None)
            write["updateTransforms"] = [
                {"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"}
            ]
        writes.append(write)
    for doc in reversed(docs):
        writes.append({"delete": doc["name"], "currentDocument": {"updateTime": doc["updateTime"]}})
    return writes


def move_firestore_transaction(
    api: Any,
    source_wallet_id: int | str,
    target_wallet_id: int | str,
    transaction_id: str,
    preview_token: str | None,
) -> dict[str, Any]:
    if not _DOCUMENT_ID.fullmatch(transaction_id):
        raise ValueError("Use the transaction ID from list_transactions")
    user = api.firestore_user_id
    wallets = api.list_firestore_wallets()
    source = _wallet(api, user, wallets, source_wallet_id)
    target = _wallet(api, user, wallets, target_wallet_id)
    if source["_id"] == target["_id"]:
        raise ValueError("Source and destination must be different wallets")
    if source["currency"] != target["currency"]:
        raise ValueError("Wallet currencies must match to preserve transaction amounts")
    source_path = f"users/{user}/wallets/{source['_id']}/transactions/{transaction_id}"
    target_path = f"users/{user}/wallets/{target['_id']}/transactions/{transaction_id}"
    source_raw = _get(api, source_path)
    target_raw = _get(api, target_path)
    if source_raw is not None and target_raw is not None:
        raise ValueError("Destination already has a transaction with this ID; nothing moved")
    if source_raw is None and (target_raw is None or preview_token is None):
        raise ValueError("Transaction not found in the source wallet")
    if source_raw is not None:
        docs, tx = _read(api, source_path, user, source["_id"], transaction_id, source_raw)
    else:
        docs, tx = _read(api, target_path, user, target["_id"], transaction_id, target_raw)

    if source_raw is not None and _labels(api, target_path):
        raise ValueError("Destination already has label relations for this ID; nothing moved")

    normalized = _normalized(docs)
    token = hashlib.sha256(
        json.dumps(
            [
                source_path,
                target_path,
                source.get("name"),
                target.get("name"),
                source["currency"],
                normalized,
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    result = {
        "transaction_id": transaction_id,
        "source_wallet_name": source.get("name"),
        "target_wallet_name": target.get("name"),
        "wallet_currency": source["currency"],
        "amount": tx["amount"],
        "occurred_at": tx["madeAt"],
        "category_id": tx["category"],
        "note": tx.get("note"),
        "foreign_values": tx.get("customCurrencyValue"),
        "label_count": len(docs) - 1,
    }
    if preview_token is None:
        return {"status": "preview", "preview_token": token, **result}
    if token != preview_token:
        raise ValueError("Transaction or its labels changed; review a new preview")
    if source_raw is None:
        if _labels(api, source_path):
            raise ValueError("Source still has label relations; inspect both wallets")
        return {"status": "already_moved", **result}

    event("move_commit", source=resource_tag(source["_id"]), target=resource_tag(target["_id"]))
    writes = _writes(
        docs, source_raw["name"], f"{api._firestore_documents_name}/{target_path}", target["_id"]
    )
    api._firestore_request(
        "POST", f"{api._firestore_documents_url}:commit", json={"writes": writes}
    )
    moved = _get(api, target_path)
    if moved is None or _get(api, source_path) is not None:
        raise ValueError("Move could not be verified; retry with the same preview token")
    moved_docs, _ = _read(api, target_path, user, target["_id"], transaction_id, moved)
    if _normalized(moved_docs) != normalized or _labels(api, source_path):
        raise ValueError(
            "Move readback differed from preview; inspect both wallets before retrying"
        )
    return {"status": "moved", **result}
