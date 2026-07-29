"""
Local JSON document store standing in for Firestore during local dev.

Deliberately Firestore-shaped (collection/document, get/set/list/update) so
swapping this for `firebase_admin.firestore.client()` in Phase 4 (production
Firebase deploy) only touches this module — nothing in api/main.py's call
sites needs to change.

Not for production use: no concurrency control, no auth, no indexing. Local
testing only.
"""
import json
import os
import threading

STORE_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "live", "store")

_lock = threading.RLock()  # update() calls set() while holding the lock — must be reentrant


class Store:
    def __init__(self, root=STORE_ROOT):
        self.root = root

    def _collection_dir(self, collection):
        d = os.path.join(self.root, collection)
        os.makedirs(d, exist_ok=True)
        return d

    def _doc_path(self, collection, doc_id):
        return os.path.join(self._collection_dir(collection), f"{doc_id}.json")

    def set(self, collection, doc_id, data: dict):
        with _lock:
            with open(self._doc_path(collection, doc_id), "w") as f:
                json.dump(data, f, indent=2, default=str)
        return data

    def get(self, collection, doc_id):
        path = self._doc_path(collection, doc_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def update(self, collection, doc_id, patch: dict):
        with _lock:
            existing = self.get(collection, doc_id) or {}
            existing.update(patch)
            return self.set(collection, doc_id, existing)

    def list(self, collection):
        d = self._collection_dir(collection)
        docs = []
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".json"):
                with open(os.path.join(d, fname)) as f:
                    docs.append(json.load(f))
        return docs

    def exists(self, collection, doc_id):
        return os.path.exists(self._doc_path(collection, doc_id))


store = Store()
