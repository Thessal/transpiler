import os
from typing import Dict
import json
from morpho.util import compute_hash
from morpho.handler import handlers
from morpho.adapters import get_adapter
from morpho.database import get_client
from pathlib import Path


class LibraryIndexer:
    def __init__(self, config):
        self.library: Dict[str, str] = dict()
        self.config = config
        self.embedder = get_adapter(config=config, role="embedder")
        self.db_client = get_client(config=config)
    
    def get_by_hash(self, hash:str):
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
                handler = handlers[data_type]
                metadata = handler.serialize(metadata)
                hash = compute_hash(repr(metadata))
                self.library[hash] = metadata

    def embed(self):
        "Generates vectors, and saves to ChromaDB."
        if not self.library:
            print("[indexer.py] Libaray empty")
        for hash, metadata in self.library.items():
            embedding = self.embedder.embed_document(metadata=metadata)
            self.db_client.upsert(
                hash=hash, embedding=embedding, metadata=metadata)

    def query(self, metadata: Dict, n_results: int):
        if self.db_client.get_total_count() == 0:
            print("[indexer.py] DB empty")
        embedding = self.embedder.embed_document(metadata=metadata)
        return self.db_client.query(embedding=embedding, n_results=n_results)
