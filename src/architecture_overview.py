"""
RAG Architecture Walkthrough Simulation
Demonstrtates the lifecycle of data moving through Ingestion,
Vector Indexing, Retrieval, and Prompt Augmentation.
"""

def simulate_rag_lifecycle():
    print("=" * 60)
    print("STEP 1: INGESTION - Raw Document Loading")
    print("=" * 60)
    raw_document = {
        "source": "data/sample_policy.pdf",
        "page": 1,
        "text": (
            "Annual leave policy: Full-time employees receive 20 days of paid "
            "annual leave per calendar year. Leave must be approved by the department "
            "manager at least two weeks in advance. Unused leave up to 5 days can be "
            "carried over to the next year."
        )
    }
    print(f"Loaded document from: {raw_document['source']}, Page: {raw_document['page']}")
    print(f"Content Length: {len(raw_document['text'])} characters\n")
    print("=" * 60)
    print("STEP 2: CHUNKING - Splitting Text with Overlap")
    print("=" * 60)
    # Simulated chunks with character boundaries
    chunk_1 = raw_document['text'][:140]
    chunk_2 = raw_document['text'][100:230]  # Overlapping slice
    chunks = [
        {"chunk_id": 1, "text": chunk_1, "metadata": {"page": 1, "source": raw_document['source']}},
        {"chunk_id": 2, "text": chunk_2, "metadata": {"page": 1, "source": raw_document['source']}}
    ]
    for c in chunks:
        print(f"[Chunk {c['chunk_id']}] -> \"{c['text']}\"")
        print(f"       Metadata: {c['metadata']}")
    print()
    print("=" * 60)
    print("STEP 3: EMBEDDING - Converting Text to Vectors")
    print("=" * 60)
    # Gemini embeddings produce 768-dimensional dense vectors
    simulated_vector = [0.0245, -0.0912, 0.4512, "... 765 more dimensions ...", 0.1189]
    print(f"Text Chunk 1 converted to Vector Representation:")
    print(f"Shape: 768 floating-point numbers -> {simulated_vector}\n")
    print("=" * 60)
    print("STEP 4: RETRIEVAL - User Query to Context Matching")
    print("=" * 60)
    user_query = "How many annual leave days do I get?"
    print(f"User Query: \"{user_query}\"")
    print("Embedding user query...")
    print("Running FAISS nearest-neighbor similarity search...")
    # FAISS selects Chunk 1 as the closest semantic match
    retrieved_chunk = chunks[0]
    print(f"[Retrieved Top-1 Match]: \"{retrieved_chunk['text']}\"\n")
    print("=" * 60)
    print("STEP 5: AUGMENTATION - Injecting Context into Prompt")
    print("=" * 60)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Do not make up facts outside the context.\n\n"
        f"CONTEXT:\n{retrieved_chunk['text']} (Source: {retrieved_chunk['metadata']['source']}, Page: {retrieved_chunk['metadata']['page']})\n\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )
    print("Final Augmented Prompt sent to Groq LLM:")
    print("-" * 50)
    print(system_prompt)
    print("-" * 50 + "\n")
    print("=" * 60)
    print("STEP 6: GENERATION - Grounded Output with Citations")
    print("=" * 60)
    simulated_llm_response = (
        "Full-time employees receive 20 days of paid annual leave per calendar year. "
        "[Source: data/sample_policy.pdf, Page 1]"
    )
    print(f"LLM Response:\n{simulated_llm_response}\n")
if __name__ == "__main__":
    simulate_rag_lifecycle()