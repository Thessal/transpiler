import pytest
from morpho import load_json, LibraryIndexer

def library_load():
    config = load_json("./tests/demo_config.json")
    library = LibraryIndexer(config)
    library.load(metadata_path="./tests/test_data")
    return library

def library_embed(library:LibraryIndexer):
    library.embed()

def library_query(library:LibraryIndexer):
    # test_key = list(library.library.keys())[0]
    test_key = "53da9fe95ea0b097d41bf55a42253b4bb7a2f20ba8f4801e07364b216b2d079f"
    metadata = library.get_by_hash(test_key)
    result = library.query(metadata, n_results=1)
    assert result["ids"][0][0] == test_key
    assert (result["distances"][0][0] < 1e-3)

def test_library():
    library = library_load()
    library_embed(library)
    assert library.db_client.get_total_count() == 4
    library_query(library)

if __name__ == "__main__":
    pytest.main()