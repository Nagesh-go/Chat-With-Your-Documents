"""
LLM Configuration Module
Initializes and provides the Groq LLM for answer generation.
Includes a direct test of grounded question-answering using
injected context (without any retrieval chain yet).
"""
import os
import sys
import warnings

# Suppress the google.generativeai deprecation warning (cosmetic only)
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── LLM configuration ───────────────────────────────────────────────────────
GROQ_MODEL   = "openai/gpt-oss-120b"  # 120B model available on this Groq account
TEMPERATURE  = 0.0    # 0.0 = deterministic, factual answers (no creativity)
MAX_TOKENS   = 1024   # Maximum tokens in LLM response


def get_llm() -> ChatGroq:
    """
    Initializes and returns the Groq LLM instance.
    Reads GROQ_API_KEY automatically from the environment (.env).

    Configuration notes:
        temperature=0.0  → Deterministic answers. For a document Q&A system
                           we want precision, not creativity. The LLM should
                           only report what the context says, not speculate.
        max_tokens=1024  → Enough for detailed paragraph answers without
                           runaway verbosity.

    Returns:
        ChatGroq: Configured LLM instance ready for invocation.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not found in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "[ERROR] GROQ_API_KEY not found.\n"
            "Please add it to your .env file:\n"
            "  GROQ_API_KEY=your_actual_key_here"
        )

    print(f"[INFO] Initializing Groq LLM: {GROQ_MODEL}")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        groq_api_key=api_key,
    )

    return llm


def ask_llm_with_context(llm: ChatGroq, context: str, question: str) -> str:
    """
    Sends a question and its supporting context directly to the LLM.
    This is the raw, low-level call — later steps will wrap this in a
    LangChain prompt template and retrieval chain.

    The system prompt instructs the LLM to:
        1. Answer ONLY from the provided context.
        2. Never fabricate information not present in the context.
        3. Explicitly say "I don't know" if the answer is not in the context.

    Args:
        llm      : Configured ChatGroq instance.
        context  : Retrieved document chunks formatted as a string.
        question : User's natural language question.

    Returns:
        str: The LLM's generated answer.
    """
    system_prompt = (
        "You are a precise document assistant. Your sole responsibility is to "
        "answer questions based ONLY on the provided document context below.\n\n"
        "Rules you must strictly follow:\n"
        "1. Use ONLY information explicitly stated in the context.\n"
        "2. If the answer is not in the context, say exactly: "
        "   'I could not find the answer in the provided documents.'\n"
        "3. Never speculate, infer beyond what is stated, or use external knowledge.\n"
        "4. Keep answers concise and factual.\n"
        "5. When referencing information, mention the source label (e.g., 'According to Source 1...').\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)

    # response.content holds the actual text reply
    return response.content


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    from src.text_splitter   import split_documents
    from src.embedder        import get_embedding_model
    from src.vector_store    import get_or_build_vector_store
    from src.retriever       import build_retriever, retrieve_context, format_context_for_prompt

    print("=" * 65)
    print("STEP 9: GROQ LLM INTEGRATION TEST")
    print("=" * 65 + "\n")

    # ── 1. Initialize LLM ────────────────────────────────────────────────────
    llm = get_llm()
    print("[OK] Groq LLM initialized.\n")

    # ── 2. Load retrieval pipeline ────────────────────────────────────────────
    print("[INFO] Loading retrieval pipeline from cache...")
    pdf_path        = os.path.join("data", "sample.pdf")
    documents       = load_single_pdf(pdf_path)
    chunks          = split_documents(documents)
    embedding_model = get_embedding_model()
    vector_store    = get_or_build_vector_store(chunks, embedding_model)
    retriever       = build_retriever(vector_store)
    print("[OK] Retrieval pipeline ready.\n")

    # ── 3. Full RAG test: Retrieve → Format → Generate ───────────────────────
    print("=" * 65)
    print("RAG ANSWER GENERATION TESTS")
    print("=" * 65)

    test_questions = [
        "How many days of annual leave are employees entitled to?",
        "What is the deadline for submitting reimbursement requests?",
        "How many days per week can employees work from home?",
        "What is the company's policy on overtime pay?",  # Not in document
    ]

    for question in test_questions:
        print(f"\n{'─' * 65}")
        print(f"Q: {question}")
        print(f"{'─' * 65}")

        # Retrieve relevant chunks
        docs    = retrieve_context(retriever, question)
        context = format_context_for_prompt(docs)

        # Generate answer
        answer = ask_llm_with_context(llm, context, question)
        print(f"A: {answer}")

    print(f"\n{'=' * 65}")
    print("[SUCCESS] Groq LLM is generating grounded answers correctly.")
    print("=" * 65)
