import sys
sys.path.insert(0,'../src')

# import morpho

# # Initialize the system
# indexer = morpho.DocumentIndexer("demo_config.json")

# # 1. Indexing phase
# indexer.process_directory("../docs/alpha")

# # 2. Retrieval phase (Example search)
# print("\n--- Performing Semantic Search ---")
# matches = indexer.semantic_search("convergence and divergence")
# for i, doc in enumerate(matches['documents'][0]):
#     print(f"Match {i+1}: {matches['ids'][0][i]}")


from morpho import StrategyWorkflow

workflow = StrategyWorkflow("demo_config.json")

inputs = {
    "idea_file": "alpha/momentum.txt",
    "spec_file": "language/spec.txt",
    "doc_files": ["alpha/reversion.txt"],  # List of filenames
    "func_files": ["operators/ts_delay.txt", "operators/ts_mean.txt"]     # List of filenames
}

workflow.run(**inputs)

