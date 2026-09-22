from rag_bench.store import VectorStore


def test_add_and_query_returns_relevant_chunks():
    store = VectorStore(collection_name="test_collection")
    store.add(
        ids=["a", "b", "c"],
        documents=[
            "The Eiffel Tower is in Paris, France.",
            "Mount Everest is the tallest mountain on Earth.",
            "Python is a popular programming language.",
        ],
    )

    results = store.query("Where is the Eiffel Tower?", top_k=1)

    assert len(results) == 1
    assert "Eiffel Tower" in results[0].document
    assert results[0].id == "a"
