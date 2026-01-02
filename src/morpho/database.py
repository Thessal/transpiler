from typing import Any, Dict, List
import chromadb


class DatabaseClient:
    def upsert(self, hash: str, embedding: List[float], metadata: Any):
        raise NotImplementedError

    def query(self, embedding: List[float], n_results: int) -> Any:
        raise NotImplementedError

    def get_total_count(self) -> int:
        raise NotImplementedError


class ChromaClient:
    def __init__(self, persist_directory=None, collection_name=None):
        # Initialize ChromaDB (Persistent Storage)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Create or load a collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Optimization for BGE-M3
        )

    def upsert(self, hash: str, embedding: List[float], metadata: Dict):
        # Store in ChromaDB
        # We store the content so we can retrieve it later without the text file
        self.collection.upsert(
            ids=[hash],
            embeddings=[embedding],
            metadatas=[{k: v for k, v in metadata.items() if k != "serialized"}],
            documents=[metadata["serialized"]]
        )

    def query(self, embedding: List[float], n_results=3):
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            # include=["embeddings", "documents", "metadatas", "distances"]
        )
        return results

    def get_total_count(self) -> int:
        """
        Returns the total number of documents/segments indexed in the collection.
        In statistical terms, this represents the total 'N' of your dataset.
        """
        try:
            return self.collection.count()
        except Exception as e:
            # Error Prevention: Catching potential connection or persistence errors
            print(f"Error retrieving count from ChromaDB: {e}")
            return 0


def get_client(config: Dict) -> DatabaseClient:
    _config = config["database"]
    provider = _config.get("provider", "chroma").lower()
    cfg = _config.get(provider, {})
    if provider == "chroma":
        return ChromaClient(**cfg)
    else:
        raise ValueError(f"Unknown provider: {provider}")
