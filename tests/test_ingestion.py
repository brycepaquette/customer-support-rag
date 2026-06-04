from customer_support_rag.ingestion import Document, chunk_document


def test_short_document():
    doc = Document(
        doc_id="test1",
        title="Short Document",
        text="This is a short document.",
        source_url="http://example.com/short",
        word_count=5,
    )
    chunks = chunk_document(doc, chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test1_chunk0"
    assert chunks[0].text == "This is a short document."


def test_chunks_have_unique_ids():
    doc = Document(
        doc_id="test2",
        title="Test Document",
        text=" ".join(f"word{i}" for i in range(1000)),
        source_url="http://example.com/test",
        word_count=1000,
    )
    chunks = chunk_document(doc, chunk_size=200, overlap=50)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    assert len(chunk_ids) == len(chunks), "Chunk IDs are not unique"


def test_empty_document_returns_no_chunks():
    doc = Document(
        doc_id="test3",
        title="Empty Document",
        text="",
        source_url="http://example.com/empty",
        word_count=0,
    )
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert len(chunks) == 0, "Empty document should return no chunks"
