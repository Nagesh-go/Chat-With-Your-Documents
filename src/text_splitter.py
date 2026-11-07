import os
import sys
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ── Recommended chunk settings for general-purpose document Q&A ──
CHUNK_SIZE    = 1000  # Maximum characters per chunk
CHUNK_OVERLAP = 200   # Characters shared between consecutive chunks
def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.
    Metadata (source, page) from each parent Document is preserved
    and inherited by every child chunk it produces.
    Args:
        documents (List[Document]): Raw documents from PyPDFLoader.
    Returns:
        List[Document]: A flat list of chunk Documents with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,           # Counts by characters (not tokens)
        add_start_index=True,          # Adds char start index to metadata
    )
    chunks = splitter.split_documents(documents)
    return chunks
def inspect_chunks(chunks: List[Document], show_content: bool = True):
    """
    Prints a detailed diagnostic view of the chunked documents.
    Args:
        chunks       : The list of chunk Documents to inspect.
        show_content : Whether to print each chunk's text content.
    """
    print("\n" + "=" * 65)
    print("CHUNK INSPECTION REPORT")
    print("=" * 65)
    print(f"Total chunks produced : {len(chunks)}")
    print(f"Chunk size setting    : {CHUNK_SIZE} characters")
    print(f"Overlap setting       : {CHUNK_OVERLAP} characters\n")
    char_counts = [len(c.page_content) for c in chunks]
    print(f"Smallest chunk : {min(char_counts)} characters")
    print(f"Largest chunk  : {max(char_counts)} characters")
    print(f"Average chunk  : {sum(char_counts) // len(char_counts)} characters")
    if show_content:
        print("\n" + "-" * 65)
        for idx, chunk in enumerate(chunks):
            page_num     = chunk.metadata.get("page", 0)
            source       = chunk.metadata.get("source", "Unknown")
            start_index  = chunk.metadata.get("start_index", "N/A")
            char_count   = len(chunk.page_content)
            print(f"\n[Chunk {idx + 1:>3}]")
            print(f"  Source      : {os.path.basename(source)}")
            print(f"  Page        : {page_num + 1} (0-indexed: {page_num})")
            print(f"  Start Index : {start_index}")
            print(f"  Length      : {char_count} characters")
            print(f"  Content Preview:")
            print(f"  {'-'*55}")
            # Show first 200 characters of the chunk
            preview = chunk.page_content.strip()[:200].replace("\n", " ")
            print(f"  {preview}")
            print(f"  {'-'*55}")
if __name__ == "__main__":
    # ── Import document loader from the same src/ package ──
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    pdf_path = os.path.join("data", "sample.pdf")
    if not os.path.exists(pdf_path):
        print(f"[ERROR] '{pdf_path}' not found. Run 'python create_sample_pdf.py' first.")
        sys.exit(1)
    # Step 1: Load the PDF
    print("[STEP 1] Loading PDF documents...")
    documents = load_single_pdf(pdf_path)
    print(f"         Loaded {len(documents)} page(s).\n")
    # Step 2: Split into chunks
    print("[STEP 2] Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"         Produced {len(chunks)} chunk(s).\n")
    # Step 3: Inspect the results
    inspect_chunks(chunks)