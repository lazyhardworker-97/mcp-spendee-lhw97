from copy import deepcopy

import pytest
from spendee.firestore import _encode_value

from mcp_spendee.move import move_firestore_transaction


class FakeFirestore:
    firestore_user_id = "user-id"
    _firestore_documents_url = "https://firestore.invalid/documents"
    _firestore_documents_name = "projects/test/databases/(default)/documents"

    def __init__(self):
        self.docs = {}
        self.commits = []
        for wallet in ("source", "target"):
            self.add(
                f"users/user-id/wallets/{wallet}",
                {
                    "name": wallet,
                    "path": {"user": "user-id", "wallet": wallet},
                    "type": "cash",
                    "status": "active",
                    "currency": "RUB",
                },
            )
        self.source = "users/user-id/wallets/source/transactions/expense-1"
        self.target = self.source.replace("/source/", "/target/")
        self.label = "/transactionLabels/label-1"
        path = {"user": "user-id", "wallet": "source", "transaction": "expense-1"}
        self.add(
            self.source,
            {
                "type": "regular",
                "modelVersion": 1,
                "author": "user-id",
                "path": path,
                "amount": "-145.80",
                "note": "Test",
                "category": "category-id",
                "customCurrencyValue": {
                    "currency": "VND",
                    "amount": "-48000",
                    "exchangeRate": "0.0030375",
                },
            },
        )
        self.docs[self.source]["fields"]["madeAt"] = {"timestampValue": "2026-11-09T10:00:00Z"}
        self.add(
            self.source + self.label,
            {
                "label": "food",
                "author": "user-id",
                "modelVersion": 1,
                "path": {**path, "transactionLabel": "label-1"},
            },
        )

        self.docs[self.source + self.label]["fields"]["createdAt"] = {
            "timestampValue": "2026-11-09T10:00:00Z"
        }

    def add(self, path, data):
        self.docs[path] = {
            "name": f"{self._firestore_documents_name}/{path}",
            "fields": {k: _encode_value(v) for k, v in data.items()},
            "updateTime": "2026-11-15T00:00:00Z",
        }

    def list_firestore_wallets(self):
        return [{"id": "source", "legacy_id": 123}, {"id": "target"}]

    def _firestore_request(self, method, url, **kwargs):
        from spendee import SpendeeFirestoreError

        path = url.removeprefix(self._firestore_documents_url).lstrip("/")
        if method == "GET":
            if path.endswith("/transactionLabels"):
                result = {
                    "documents": [v for k, v in self.docs.items() if k.rsplit("/", 1)[0] == path]
                }
            elif path in self.docs:
                result = self.docs[path]
            else:
                raise SpendeeFirestoreError("Firestore request failed with HTTP 404: Missing")
        else:
            assert path == ":commit"
            self.commits.append(deepcopy(kwargs["json"]["writes"]))
            for write in self.commits[-1]:
                name = write.get("delete") or write["update"]["name"]
                key = name.removeprefix(self._firestore_documents_name + "/")
                if "delete" in write:
                    assert write["currentDocument"] == {"updateTime": self.docs[key]["updateTime"]}
                    del self.docs[key]
                else:
                    assert write["currentDocument"] == {"exists": False}
                    assert key not in self.docs
                    fields = write["update"]["fields"]
                    if "/transactionLabels/" in key:
                        assert "createdAt" not in fields
                        assert write["updateTransforms"] == [
                            {"fieldPath": "createdAt", "setToServerValue": "REQUEST_TIME"}
                        ]
                        fields = {**fields, "createdAt": {"timestampValue": "2026-11-15T10:00:00Z"}}
                    self.docs[key] = {
                        **write["update"],
                        "fields": fields,
                        "updateTime": "2026-11-15T10:00:00Z",
                    }

            result = {}
        return type("Response", (), {"json": lambda _: deepcopy(result)})()

    def move(self, token=None):
        return move_firestore_transaction(self, 123, "target", "expense-1", token)


def test_move_preserves_original_and_labels_and_retry_does_not_write_again():
    api = FakeFirestore()
    original = deepcopy(api.docs[api.source]["fields"])
    preview = api.move()
    assert preview["status"] == "preview"
    assert preview["amount"] == "-145.80"
    assert not api.commits
    assert api.move(preview["preview_token"])["status"] == "moved"
    assert api.source not in api.docs and api.source + api.label not in api.docs
    moved = api.docs[api.target]["fields"]
    assert moved["path"]["mapValue"]["fields"]["wallet"] == {"stringValue": "target"}
    assert {k: v for k, v in moved.items() if k != "path"} == {
        k: v for k, v in original.items() if k != "path"
    }
    assert api.docs[api.target + api.label]["fields"]["label"] == {"stringValue": "food"}
    assert api.move(preview["preview_token"])["status"] == "already_moved"
    assert len(api.commits) == 1
    assert len(api.commits[0]) == 4


@pytest.mark.parametrize(
    "change", ["edit", "collision", "currency", "unsupported", "orphaned_label"]
)
def test_move_refuses_stale_preview_or_unsupported_data(change):
    api = FakeFirestore()
    token = api.move()["preview_token"]
    if change == "edit":
        api.docs[api.source]["fields"]["note"] = {"stringValue": "new note"}
    elif change == "collision":
        api.add(api.target, {})
    elif change == "currency":
        api.docs["users/user-id/wallets/target"]["fields"]["currency"] = {"stringValue": "USD"}
    elif change == "orphaned_label":
        api.add(api.target + api.label, {})
    else:
        api.docs[api.source]["fields"]["type"] = {"stringValue": "transfer"}
    with pytest.raises(ValueError):
        api.move(token)
    assert not api.commits
