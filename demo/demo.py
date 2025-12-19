import sys
sys.path.insert(0,'../src')

import morpho

# Initialize the system
indexer = morpho.DocumentIndexer("demo_config.json")

# 1. Indexing phase
indexer.process_directory("../docs/alpha")

# 2. Retrieval phase (Example search)
print("\n--- Performing Semantic Search ---")
matches = indexer.semantic_search("convergence and divergence")
for i, doc in enumerate(matches['documents'][0]):
    print(f"Match {i+1}: {matches['ids'][0][i]}")