"""
Retrieval Module
Wraps the FAISS vector store in a LangChain retriever,
and provides a formatted context assembler for LLM prompt injection.
"""
import os
import sys
from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

# ── Retrieval configuration ─────────────────────────────────────────────────
SEARCH_TYPE = "similarity"   # Options: "similarity" | "mmr" | "similarity_score_threshold"
TOP_K       = 4              # Number of chunks to retrieve per query


def build_retriever(vector_store: FAISS) -> VectorStoreRetriever:
    """
    Converts a FAISS vector store into a LangChain retriever object.
    The retriever exposes a standard .invoke(query) interface that all
    LangChain chains use internally.

    Args:
        vector_store : An in-memory FAISS object (built or loaded).

    Returns:
        VectorStoreRetriever: A configured retriever ready for chaining.
    """
    retriever = vector_store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K},
    )

    print(f"[INFO] Retriever configured.")
    print(f"       Search type : {SEARCH_TYPE}")
    print(f"       Top-K       : {TOP_K} chunks per query\n")

    return retriever


def retrieve_context(retriever: VectorStoreRetriever, query: str) -> List[Document]:
    """
    Retrieves the top-K most relevant document chunks for a given query.

    Args:
        retriever : The configured VectorStoreRetriever.
        query     : The user's natural language question.

    Returns:
        List[Document]: Retrieved chunks sorted by relevance (most relevant first).
    """
    docs = retriever.invoke(query)
    return docs


def format_context_for_prompt(docs: List[Document]) -> str:
    """
    Formats a list of retrieved Document chunks into a clean, structured
    context string ready to be injected into an LLM prompt.

    Each chunk is labeled with its source file and page number so the LLM
    can reference them when generating citations in its answer.

    Args:
        docs : Retrieved Document chunks.

    Returns:
        str: A formatted multi-line context string.
    """
    if not docs:
        return "No relevant context found in the documents."

    context_parts = []

    for idx, doc in enumerate(docs, start=1):
        source   = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1   # Convert to 1-based
        content  = doc.page_content.strip()

        context_parts.append(
            f"[Source {idx}: {source}, Page {page_num}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def display_retrieval_results(query: str, docs: List[Document]):
    """
    Diagnostic display of retrieval results with metadata.
    """
    print(f"\n{'=' * 65}")
    print(f"QUERY       : \"{query}\"")
    print(f"RETRIEVED   : {len(docs)} chunk(s)")
    print(f"{'=' * 65}")

    for idx, doc in enumerate(docs, start=1):
        source   = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        length   = len(doc.page_content)
        preview  = doc.page_content.strip()[:180].replace("\n", " ")

        print(f"\n  Chunk #{idx}")
        print(f"  Source  : {source}  |  Page: {page_num}  |  Length: {length} chars")
        print(f"  Preview : {preview}...")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    from src.text_splitter   import split_documents
    from src.embedder        import get_embedding_model
    from src.vector_store    import get_or_build_vector_store

    print("=" * 65)
    print("STEP 8: DOCUMENT RETRIEVAL SYSTEM TEST")
    print("=" * 65 + "\n")

    # ── 1. Load pipeline components ──────────────────────────────────────────
    pdf_path  = os.path.join("data", "sample.pdf")
    documents = load_single_pdf(pdf_path)
    chunks    = split_documents(documents)
    embedding_model = get_embedding_model()

    # ── 2. Load the FAISS index (uses cached index — no API calls) ───────────
    vector_store = get_or_build_vector_store(chunks, embedding_model)

    # ── 3. Build the retriever ───────────────────────────────────────────────
    retriever = build_retriever(vector_store)

    # ── 4. Run retrieval tests ───────────────────────────────────────────────
    test_queries = [
        "How many annual leave days do employees receive?",
        "What is the deadline for submitting expense reimbursements?",
        "How many days per week can employees work remotely?",
        "What happens to unused leave at the end of the year?",
    ]

    print("-" * 65)
    print("RETRIEVAL RESULTS")
    print("-" * 65)

    for query in test_queries:
        docs = retrieve_context(retriever, query)
        display_retrieval_results(query, docs)

    # ── 5. Show formatted context as it will appear in LLM prompt ───────────
    print("\n\n" + "=" * 65)
    print("FORMATTED CONTEXT (as injected into LLM prompt)")
    print("=" * 65)
    sample_query = "How many days can employees carry over unused leave?"
    sample_docs  = retrieve_context(retriever, sample_query)
    formatted    = format_context_for_prompt(sample_docs)

    print(f"\nQuery: \"{sample_query}\"\n")
    print("Context that will be passed to Groq LLM:")
    print("-" * 65)
    print(formatted)
    print("-" * 65)
    print("\n[SUCCESS] Retrieval system is fully operational.")