import os
from typing import Callable, Dict, List
import json
from morpho.util import compute_hash
from morpho.handler import BaseHandler, handlers
from morpho.adapters import get_adapter
from morpho.database import get_client
from pathlib import Path


class LibraryIndexer:
    def __init__(self, config):
        self.library: Dict[str, Dict] = dict()
        self.config = config
        self.embedder = get_adapter(config=config, role="embedder")
        self.db_client = get_client(config=config)

    def get_hash_list(self) -> List[str]:
        # Outputs hash list of all documents
        return list(self.library.keys())

    def get_by_hash(self, hash: str) -> Dict:
        # Given hash, get document
        return self.library[hash]

    def load(self, metadata_path):
        """Loads files"""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Path not found: {metadata_path}")

        metadata_path = Path(metadata_path)
        for filename in os.listdir(metadata_path):
            if filename.endswith(".json"):
                file_path = metadata_path / filename

                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                data_type = metadata["data_type"]
                handler: BaseHandler = handlers[data_type]
                metadata: Dict = handler.serialize(metadata)
                hash = compute_hash(repr(metadata))
                self.library[hash] = metadata

    def filter(self, condition: Dict[str, Callable]):
        # Given condition, filter documents
        all_docs = self.get_hash_list()
        if len(all_docs) == 0:
            print("[Library] Filtering empty library")
        result = dict()
        for k in all_docs:
            metadata: Dict = self.get_by_hash(k)
            selected = all(f(metadata[c]) for c, f in condition.items())
            if selected:
                result[k] = metadata
        print(f"[Library] All: {len(self.library)} Filtered: {len(result)}")
        self.library = result

    def embed(self):
        "Generates vectors, and saves to ChromaDB."
        if not self.library:
            print("[indexer.py] Libaray empty")
        for hash, metadata in self.library.items():
            embedding = self.embedder.embed_document(metadata=metadata)
            self.db_client.upsert(
                hash=hash, embedding=embedding, metadata=metadata)

    def query(self, metadata: Dict, n_results: int):
        # Given metadata, do semantic search
        if self.db_client.get_total_count() == 0:
            print("[indexer.py] DB empty")
        embedding = self.embedder.embed_document(metadata=metadata)
        return self.db_client.query(embedding=embedding, n_results=n_results)

    def hash_to_embedding(self, hash: str) -> List:
        result = self.db_client.collection.get(
            ids=[hash], include=["embeddings"])
        if len(["ids"]) == 0 or len(["ids"]) > 1:
            print("hash matches none or multiple documents in DB")
            return None
        else:
            assert result["embeddings"].shape[0] == 1
            return result["embeddings"][0].tolist()

    def get_by_hash(self, hash: str) -> Dict:
        # Given hash, get document
        return self.library[hash]
