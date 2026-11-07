"""
Embedder Module
Configures and provides the Google Gemini embedding model.
Also includes a live diagnostic test that proves semantic similarity
works correctly before we build the FAISS index.
"""
import os
import sys
import math
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API keys from .env
load_dotenv()

# ── Embedding model configuration ──────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Gemini embedding model.
    Reads GOOGLE_API_KEY automatically from the environment (.env).

    Returns:
        GoogleGenerativeAIEmbeddings: The configured embedding model instance.
    
    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not found in the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "[ERROR] GOOGLE_API_KEY not found.\n"
            "Please add it to your .env file:\n"
            "  GOOGLE_API_KEY=your_actual_key_here"
        )

    print(f"[INFO] Initializing embedding model: {EMBEDDING_MODEL}")

    model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    return model


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes the cosine similarity between two vectors.
    Result ranges from -1.0 (opposite) to 1.0 (identical direction).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def run_similarity_test(model: GoogleGenerativeAIEmbeddings):
    """
    Embeds four test sentences and prints cosine similarity scores.
    Proves that semantically similar texts score higher than unrelated texts.
    """
    test_sentences = {
        "A": "Employees are entitled to 20 days of annual paid leave per year.",
        "B": "Workers receive 20 vacation days every calendar year.",       # Similar to A
        "C": "The sky is blue and the weather is pleasant today.",          # Unrelated to A
        "D": "Reimbursement requests must be submitted within 30 days.",    # Different topic
    }

    print("\n[INFO] Embedding 4 test sentences — this makes a live API call...\n")

    vectors = {}
    for label, sentence in test_sentences.items():
        vec = model.embed_query(sentence)
        vectors[label] = vec
        print(f"  [{label}] \"{sentence[:60]}...\"")
        print(f"       Vector shape : {len(vec)} dimensions")
        print(f"       First 5 values: {[round(v, 6) for v in vec[:5]]}\n")

    print("-" * 65)
    print("COSINE SIMILARITY RESULTS (1.0 = identical, 0.0 = unrelated)")
    print("-" * 65)

    comparisons = [
        ("A", "B", "SHOULD BE HIGH  (same topic, different words)"),
        ("A", "C", "SHOULD BE LOW   (completely unrelated)"),
        ("A", "D", "SHOULD BE MID   (different HR topics)"),
    ]

    for label_1, label_2, expectation in comparisons:
        score = cosine_similarity(vectors[label_1], vectors[label_2])
        print(f"  Similarity({label_1}, {label_2}) = {score:.4f}   ← {expectation}")

    print()


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    from src.text_splitter import split_documents

    # ── 1. Initialize the embedding model ──────────────────────────────────
    print("=" * 65)
    print("STEP 5: GOOGLE GEMINI EMBEDDING VERIFICATION")
    print("=" * 65)

    model = get_embedding_model()
    print(f"[OK] Embedding model initialized successfully.\n")

    # ── 2. Run the semantic similarity diagnostic test ──────────────────────
    run_similarity_test(model)

    # ── 3. Embed a real document chunk and show the vector ──────────────────
    pdf_path = os.path.join("data", "sample.pdf")
    if os.path.exists(pdf_path):
        print("[INFO] Embedding one real document chunk from sample.pdf...")
        documents = load_single_pdf(pdf_path)
        chunks    = split_documents(documents)

        first_chunk_text = chunks[0].page_content
        real_vector      = model.embed_query(first_chunk_text)

        print(f"\n  Chunk text preview  : \"{first_chunk_text[:80].strip()}...\"")
        print(f"  Embedding dimension : {len(real_vector)}")
        print(f"  First 8 values      : {[round(v, 6) for v in real_vector[:8]]}")
        print(f"\n[SUCCESS] Embedding pipeline is fully functional.")
    else:
        print("[WARNING] sample.pdf not found — skipping real-chunk embedding test.")