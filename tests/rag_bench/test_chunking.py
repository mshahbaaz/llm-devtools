from rag_bench.chunking import chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))  # 100 words
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert len(first_words) == 20
    # last 5 words of chunk 1 should equal first 5 words of chunk 2 (the overlap)
    assert first_words[-5:] == second_words[:5]


def test_chunk_text_handles_short_text():
    chunks = chunk_text("just five words here now", chunk_size=20, overlap=5)
    assert chunks == ["just five words here now"]
