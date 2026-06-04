import sys
from pathlib import Path

from bs4 import BeautifulSoup

from customer_support_rag.models import Chunk, Document


def strip_html_boilerplate(html: str) -> str:
    """Remove navigation, footer, sidebars, scripts, styles from HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted tags
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Get main content (usually in article, main, or content div)
    content = soup.find(["article", "main", "div.content"])
    if content is None:
        content = soup.body if soup.body else soup

    # Extract text with some structure preservation
    text = content.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def extract_title(html: str) -> str:
    """Extract title from HTML document."""
    soup = BeautifulSoup(html, "lxml")

    # Try various title sources in order of preference
    if soup.title and soup.title.string:
        title_str: str = soup.title.string.strip()
        return title_str

    h1 = soup.find("h1")
    if h1:
        h1_text: str = h1.get_text(strip=True)
        return h1_text

    return "Untitled"


def load_document(path: Path, source_url: str) -> Document:
    """Load a single HTML file and return a Document."""
    raw = path.read_text(encoding="utf-8")
    text = strip_html_boilerplate(raw)

    return Document(
        doc_id=path.stem,
        title=extract_title(raw),
        text=text,
        source_url=source_url,
        word_count=len(text.split()),
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load all HTML documents from a directory and return as list of Documents."""
    documents = []

    for html_file in sorted(corpus_dir.glob("*.html")):
        doc_id = html_file.stem
        # Reconstruct the URL from the filename
        # filename format: "celonis_apis_auth.html" -> "https://developer.celonis.com/celonis-apis/auth"
        url_path = "/" + doc_id.replace("_", "/")
        source_url = f"https://developer.celonis.com{url_path}"

        try:
            doc = load_document(html_file, source_url)
            documents.append(doc)
        except Exception as e:
            print(f"Failed to load {html_file}: {e}")

    return documents


def print_corpus_stats(documents: list[Document]) -> None:
    """Print summary statistics about the corpus."""
    if not documents:
        print("No documents loaded.")
        return

    total_words = sum(doc.word_count for doc in documents)
    avg_words = total_words / len(documents)

    print(f"\n{'=' * 60}")
    print("Corpus Statistics")
    print(f"{'=' * 60}")
    print(f"Total documents:      {len(documents)}")
    print(f"Total word count:     {total_words:,}")
    print(f"Average words/doc:    {avg_words:.1f}")
    print(f"Min words in a doc:   {min(doc.word_count for doc in documents)}")
    print(f"Max words in a doc:   {max(doc.word_count for doc in documents)}")
    print(f"{'=' * 60}\n")


def chunk_document(
    document: Document, chunk_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    """Split a Document into overlapping Chunks."""
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    words = document.text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_chunk{chunk_id}",
                text=chunk_text,
                source=document.source_url,
                chunk_index=chunk_id,
                token_count=int((end - start) / 0.75),
            )
        )
        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def chunk_corpus(
    documents: list[Document], chunk_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    """Chunk all documents in the corpus."""
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks


if __name__ == "__main__":
    corpus_dir = Path("./corpus")
    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    docs = load_corpus(corpus_dir)
    print_corpus_stats(docs)
