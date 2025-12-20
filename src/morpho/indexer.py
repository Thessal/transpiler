import os
import json
import ollama
import chromadb
from chromadb.config import Settings


class DocumentIndexer:
    def __init__(self, config_path="config.json"):
        # Load external configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)["indexer"]

        # Initialize Ollama Client
        self.ollama_client = ollama.Client(host=self.config['ollama']['host'])
        self.model = self.config['ollama']['model']

        # Initialize ChromaDB (Persistent Storage)
        self.chroma_client = chromadb.PersistentClient(
            path=self.config['chroma']['persist_directory']
        )

        # Create or load a collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config['chroma']['collection_name'],
            metadata={"hnsw:space": "cosine"}  # Optimization for BGE-M3
        )

    def process_directory(self, directory_path):
        """Loads txt files, generates vectors, and saves to ChromaDB."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Path not found: {directory_path}")

        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                self._index_file(file_path, filename)

    def _index_file(self, file_path, filename):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return

            # Generate embedding via Ollama
            response = self.ollama_client.embed(
                model=self.model,
                input=content
            )
            embedding = response['embeddings'][0]

            # Store in ChromaDB
            # We store the content so we can retrieve it later without the text file
            self.collection.upsert(
                ids=[filename],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"source": file_path}]
            )
            print(f"Indexed: {filename}")
            print(f"Embedding: {embedding[0]}...{embedding[-1]}")

        except Exception as e:
            print(f"Error indexing {filename}: {str(e)}")

    def semantic_search(self, query, n_results=3):
        """Find the most similar documents to a query string."""
        query_embedding = self.ollama_client.embed(
            model=self.model,
            input=query
        )['embeddings'][0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results


if __name__ == "__main__":
    # Initialize the system
    indexer = DocumentIndexer("config.json")

    # 1. Indexing phase
    indexer.process_directory("./data_folder")

    # 2. Retrieval phase (Example search)
    print("\n--- Performing Semantic Search ---")
    matches = indexer.semantic_search(
        "interest rate volatility in emerging markets")
    for i, doc in enumerate(matches['documents'][0]):
        print(f"Match {i+1}: {matches['ids'][0][i]}")
