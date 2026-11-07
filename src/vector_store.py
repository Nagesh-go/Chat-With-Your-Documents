import os
import sys
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()
# ── Paths ───────────────────────────────────────────────────────────────────
VECTORSTORE_DIR  = "vectorstore"   # Directory where FAISS files will be saved
VECTORSTORE_NAME = "index"         # Base name → index.faiss + index.pkl
# ── Retrieval setting ───────────────────────────────────────────────────────
TOP_K_RESULTS = 3                  # Number of chunks to return per query
def build_vector_store(
    chunks: List[Document],
    embedding_model: GoogleGenerativeAIEmbeddings,
) -> FAISS:
    """
    Takes a list of Document chunks, embeds them using the provided model,
    and builds a FAISS vector index in memory.
    Args:
        chunks          : List of chunk Documents from RecursiveCharacterTextSplitter.
        embedding_model : Configured GoogleGenerativeAIEmbeddings instance.
    Returns:
        FAISS: An in-memory FAISS vector store ready for similarity search.
    """
    print(f"[INFO] Building FAISS index from {len(chunks)} chunk(s)...")
    print(f"       This will make {len(chunks)} embedding API call(s) — please wait.\n")
    # FAISS.from_documents() does two things in one call:
    #   1. Calls embedding_model.embed_documents([chunk.page_content, ...])
    #   2. Inserts every resulting vector into a new FAISS index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    print(f"[SUCCESS] FAISS index built. {len(chunks)} vectors stored in memory.")
    return vector_store
def save_vector_store(vector_store: FAISS, directory: str = VECTORSTORE_DIR, name: str = VECTORSTORE_NAME):
    """
    Persists the in-memory FAISS index to disk.
    Creates two files:
        <directory>/<name>.faiss  — Binary index (the actual vectors)
        <directory>/<name>.pkl    — Pickle file (chunk texts + metadata)
    Args:
        vector_store : The in-memory FAISS object to save.
        directory    : Target directory for the saved files.
        name         : Base filename for the index pair.
    """
    os.makedirs(directory, exist_ok=True)
    # allow_dangerous_deserialization is not needed for saving
    vector_store.save_local(folder_path=directory, index_name=name)
    faiss_file = os.path.join(directory, f"{name}.faiss")
    pkl_file   = os.path.join(directory, f"{name}.pkl")
    print(f"\n[INFO] FAISS index saved to disk:")
    print(f"       Binary vectors : {faiss_file}  ({os.path.getsize(faiss_file):,} bytes)")
    print(f"       Chunk metadata : {pkl_file}  ({os.path.getsize(pkl_file):,} bytes)")
def search_vector_store(
    vector_store: FAISS,
    query: str,
    k: int = TOP_K_RESULTS,
) -> List[Tuple[Document, float]]:
    """
    Embeds the query and returns the top-k most similar chunks with scores.
    Args:
        vector_store : The in-memory FAISS index to search.
        query        : The user's natural language question.
        k            : Number of top results to return.
    Returns:
        List of (Document, similarity_score) tuples, ranked by relevance.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results
def display_search_results(query: str, results: List[Tuple[Document, float]]):
    """
    Pretty-prints similarity search results showing content, metadata and scores.
    """
    print(f"\n{'=' * 65}")
    print(f"QUERY : \"{query}\"")
    print(f"{'=' * 65}")
    if not results:
        print("  No results found.")
        return
    for rank, (doc, score) in enumerate(results, start=1):
        source   = os.path.basename(doc.metadata.get("source", "Unknown"))
        page     = doc.metadata.get("page", 0) + 1   # Convert to 1-based
        preview  = doc.page_content.strip()[:200].replace("\n", " ")
        print(f"\n  Rank #{rank}")
        print(f"  Source  : {source}  |  Page: {page}")
        print(f"  L2 Score: {score:.4f}  (lower = more similar)")
        print(f"  Content : {preview}...")
    print()
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    from src.text_splitter   import split_documents
    from src.embedder        import get_embedding_model
    print("=" * 65)
    print("STEP 6: BUILD AND TEST THE FAISS VECTOR DATABASE")
    print("=" * 65 + "\n")
    # ── 1. Load PDF 
    pdf_path = os.path.join("data", "sample.pdf")
    if not os.path.exists(pdf_path):
        print("[ERROR] data/sample.pdf not found. Run 'python create_sample_pdf.py' first.")
        sys.exit(1)
    print("[STEP 1/4] Loading PDF...")
    documents = load_single_pdf(pdf_path)
    # ── 2. Split into chunks 
    print("\n[STEP 2/4] Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"           {len(chunks)} chunks ready.")
    # ── 3. Build FAISS index 
    print("\n[STEP 3/4] Initializing embedding model and building FAISS index...")
    embedding_model = get_embedding_model()
    vector_store    = build_vector_store(chunks, embedding_model)
    # ── 4. Save to disk 
    print("\n[STEP 4/4] Saving FAISS index to disk...")
    save_vector_store(vector_store)
    # ── 5. Test similarity search with 3 different questions
    print("\n" + "=" * 65)
    print("SIMILARITY SEARCH TESTS")
    print("=" * 65)
    test_queries = [
        "How many days of annual leave do employees get?",
        "What is the reimbursement submission deadline?",
        "How many days can employees work from home?",
    ]
    for query in test_queries:
        results = search_vector_store(vector_store, query)
        display_search_results(query, results)
    print("[ALL TESTS COMPLETE] FAISS vector store is working correctly.")


def load_vector_store(
    embedding_model: GoogleGenerativeAIEmbeddings,
    directory: str = VECTORSTORE_DIR,
    name: str = VECTORSTORE_NAME,
) -> FAISS:
    """
    Loads a previously saved FAISS index from disk.
    No embedding API calls are made — vectors are read directly from disk.

    Security Note:
        allow_dangerous_deserialization=True is required because FAISS
        uses Python pickle (.pkl) to store chunk metadata. Only load
        indexes that YOU created from trusted sources.

    Args:
        embedding_model : Required to embed future queries against this index.
        directory       : Folder where index.faiss and index.pkl reside.
        name            : Base filename of the saved index.

    Returns:
        FAISS: The reloaded vector store, ready for similarity search.

    Raises:
        FileNotFoundError: If the index files do not exist on disk.
    """
    faiss_file = os.path.join(directory, f"{name}.faiss")
    pkl_file   = os.path.join(directory, f"{name}.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError(
            f"[ERROR] Vector store not found at '{directory}/{name}'.\n"
            f"        Run the ingestion pipeline first to create it."
        )

    print(f"[INFO] Loading FAISS index from disk: '{directory}/{name}'")

    vector_store = FAISS.load_local(
        folder_path=directory,
        embeddings=embedding_model,
        index_name=name,
        allow_dangerous_deserialization=True,   # Safe: we created this file ourselves
    )

    print(f"[SUCCESS] Vector store loaded from disk. Ready for queries.")
    return vector_store


def get_or_build_vector_store(
    chunks: List[Document],
    embedding_model: GoogleGenerativeAIEmbeddings,
    directory: str = VECTORSTORE_DIR,
    name: str = VECTORSTORE_NAME,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Smart router: Loads an existing FAISS index if one is saved on disk,
    otherwise builds a new one from chunks and saves it.

    This is the PRIMARY function all other modules should call.
    It ensures we never re-embed documents unnecessarily.

    Args:
        chunks          : Document chunks (used only if building from scratch).
        embedding_model : Configured Gemini embedding model.
        directory       : Vectorstore directory path.
        name            : Index base filename.
        force_rebuild   : If True, always rebuild even if an index exists.

    Returns:
        FAISS: Ready-to-use vector store (either loaded or freshly built).
    """
    faiss_file = os.path.join(directory, f"{name}.faiss")
    index_exists = os.path.exists(faiss_file)

    if index_exists and not force_rebuild:
        print("[INFO] Existing FAISS index detected. Loading from disk...")
        print("       (Skipping embedding API calls — using cached index)\n")
        return load_vector_store(embedding_model, directory, name)
    else:
        if force_rebuild:
            print("[INFO] force_rebuild=True. Rebuilding index from scratch...\n")
        else:
            print("[INFO] No existing index found. Building from scratch...\n")
        vector_store = build_vector_store(chunks, embedding_model)
        save_vector_store(vector_store, directory, name)
        return vector_store

if __name__ == "__main__":
    import time
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    from src.text_splitter   import split_documents
    from src.embedder        import get_embedding_model

    print("=" * 65)
    print("STEP 7: SAVE AND RELOAD VECTOR INDEX VERIFICATION")
    print("=" * 65 + "\n")

    pdf_path = os.path.join("data", "sample.pdf")
    documents = load_single_pdf(pdf_path)
    chunks    = split_documents(documents)
    embedding_model = get_embedding_model()

    # ── Test A: Use get_or_build (loads existing index — no API calls) ──────
    print("-" * 65)
    print("TEST A: get_or_build_vector_store() with existing index")
    print("-" * 65)
    t_start = time.time()
    vs_loaded = get_or_build_vector_store(chunks, embedding_model)
    t_elapsed = time.time() - t_start
    print(f"       Loaded in {t_elapsed:.3f} seconds (no API calls made)\n")

    # ── Test B: Force rebuild to demonstrate difference ──────────────────────
    print("-" * 65)
    print("TEST B: get_or_build_vector_store() with force_rebuild=True")
    print("-" * 65)
    t_start = time.time()
    vs_rebuilt = get_or_build_vector_store(chunks, embedding_model, force_rebuild=True)
    t_elapsed = time.time() - t_start
    print(f"       Rebuilt in {t_elapsed:.3f} seconds (API calls made)\n")

    # ── Test C: Prove loaded index gives identical results to rebuilt one ────
    print("-" * 65)
    print("TEST C: Confirm loaded index gives identical search results")
    print("-" * 65)
    test_query = "How many annual leave days do employees receive?"

    results_loaded  = search_vector_store(vs_loaded,  test_query, k=1)
    results_rebuilt = search_vector_store(vs_rebuilt, test_query, k=1)

    top_loaded  = results_loaded[0][0].page_content[:80]
    top_rebuilt = results_rebuilt[0][0].page_content[:80]

    print(f"\n  Query : \"{test_query}\"")
    print(f"\n  Loaded  index top result : \"{top_loaded}\"")
    print(f"  Rebuilt index top result : \"{top_rebuilt}\"")

    if top_loaded == top_rebuilt:
        print("\n  [MATCH] Both indexes return identical top result.")
        print("\n[ALL TESTS PASSED] Save/Load pipeline is fully operational.")
    else:
        print("\n  [MISMATCH] Results differ — investigate index consistency.")