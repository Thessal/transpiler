from morpho import StrategyWorkflow
import sys
sys.path.insert(0, '../src')

# Indexing need to be intergrated into the workflow
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


workflow = StrategyWorkflow("demo_config.json")

inputs = {
    "idea_file": "../demo/demo_input.txt",
    "spec_file": "language/spec.txt",
    "doc_files": ["alpha/momentum.txt", "alpha/reversion.txt"],
    "func_files": ["operators/ts_delay.txt", "operators/ts_mean.txt"]
}

workflow.run(**inputs)
