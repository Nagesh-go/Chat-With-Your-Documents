"""
Prompt Template Module
Defines the structured ChatPromptTemplate for document Q&A and
assembles the complete LCEL chain: prompt | llm | output_parser.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ── QA Prompt Template ──────────────────────────────────────────────────────
# This template has two roles:
#   "system" : Standing instructions that shape the LLM's behaviour.
#   "human"  : The actual question from the user at runtime.
#
# Two input variables are required at invocation time:
#   {context}  : The formatted retrieved document chunks (filled by retriever).
#   {question} : The user's natural language question (filled at runtime).

QA_SYSTEM_PROMPT = """You are a precise and helpful document assistant. \
Your job is to answer the user's question using ONLY the provided document context.

Follow these rules strictly:
1. Base your answer EXCLUSIVELY on the information in the CONTEXT section below.
2. If the answer cannot be found in the context, respond with:
   "I could not find the answer in the provided documents."
3. Never use outside knowledge, never speculate, never hallucinate facts.
4. Cite your source using the label provided (e.g., "According to Source 1...").
5. Keep answers clear, concise, and factual.
6. If multiple sources support the answer, reference all relevant ones.

CONTEXT:
{context}"""

QA_HUMAN_PROMPT = "{question}"


def build_qa_prompt() -> ChatPromptTemplate:
    """
    Constructs and returns the QA ChatPromptTemplate.
    The template expects two input variables: {context} and {question}.

    Returns:
        ChatPromptTemplate: Ready-to-use prompt template.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", QA_HUMAN_PROMPT),
    ])
    return prompt


def build_qa_chain(llm: ChatGroq) -> any:
    """
    Assembles the complete LCEL QA chain:
        ChatPromptTemplate | ChatGroq | StrOutputParser

    The StrOutputParser automatically extracts the plain text string
    from the AIMessage object returned by the LLM — so the chain
    outputs a clean str rather than AIMessage(content='...').

    Args:
        llm : Configured ChatGroq instance.

    Returns:
        Runnable: An LCEL chain invokable with {"context": ..., "question": ...}
    """
    prompt = build_qa_prompt()
    output_parser = StrOutputParser()

    # The pipe operator | chains the components left-to-right:
    #   prompt formats the messages →
    #   llm generates a response (AIMessage) →
    #   output_parser extracts the .content string
    chain = prompt | llm | output_parser

    return chain


def run_qa_chain(chain, context: str, question: str) -> str:
    """
    Invokes the LCEL chain with a context string and a question.

    Args:
        chain    : The assembled LCEL QA chain.
        context  : Formatted retrieved document chunks.
        question : User's natural language question.

    Returns:
        str: The LLM's plain text answer.
    """
    answer = chain.invoke({
        "context": context,
        "question": question,
    })
    return answer


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader  import load_single_pdf
    from src.text_splitter    import split_documents
    from src.embedder         import get_embedding_model
    from src.vector_store     import get_or_build_vector_store
    from src.retriever        import build_retriever, retrieve_context, format_context_for_prompt
    from src.llm              import get_llm

    print("=" * 65)
    print("STEP 10: QA PROMPT TEMPLATE & LCEL CHAIN TEST")
    print("=" * 65 + "\n")

    # ── 1. Build all pipeline components ─────────────────────────────────────
    print("[INFO] Initializing pipeline components...")
    llm             = get_llm()
    pdf_path        = os.path.join("data", "sample.pdf")
    documents       = load_single_pdf(pdf_path)
    chunks          = split_documents(documents)
    embedding_model = get_embedding_model()
    vector_store    = get_or_build_vector_store(chunks, embedding_model)
    retriever       = build_retriever(vector_store)
    print("[OK] All components ready.\n")

    # ── 2. Inspect the prompt template structure ──────────────────────────────
    print("=" * 65)
    print("PROMPT TEMPLATE STRUCTURE")
    print("=" * 65)
    prompt = build_qa_prompt()
    print(f"Template type    : {type(prompt).__name__}")
    print(f"Number of messages: {len(prompt.messages)}")
    print(f"Input variables  : {prompt.input_variables}")

    print("\nFormatted prompt preview (with sample values):")
    print("-" * 65)
    sample_messages = prompt.format_messages(
        context="[Source 1: sample.pdf, Page 1]\nEmployees receive 20 days annual leave.",
        question="How many leave days do employees get?"
    )
    for msg in sample_messages:
        role = type(msg).__name__.replace("Message", "")
        print(f"[{role}]:\n{msg.content[:300]}\n")
    print("-" * 65)

    # ── 3. Build the LCEL chain ───────────────────────────────────────────────
    print("\n[INFO] Building LCEL chain: prompt | llm | StrOutputParser")
    chain = build_qa_chain(llm)
    print(f"[OK] Chain type: {type(chain).__name__}\n")

    # ── 4. Run test questions through the chain ───────────────────────────────
    print("=" * 65)
    print("LCEL CHAIN QA TESTS")
    print("=" * 65)

    test_questions = [
        "How many days of annual leave are employees entitled to?",
        "Can unused leave be carried over and if so, how many days?",
        "What are the core working hours for remote employees?",
        "Does the company provide dental insurance?",   # Not in document
    ]

    for question in test_questions:
        print(f"\n{'─' * 65}")
        print(f"Q: {question}")
        print(f"{'─' * 65}")

        docs    = retrieve_context(retriever, question)
        context = format_context_for_prompt(docs)
        answer  = run_qa_chain(chain, context, question)

        print(f"A: {answer}")

    print(f"\n{'=' * 65}")
    print(f"[OK] Output type is: {type(answer).__name__}  ← Must be 'str', not 'AIMessage'")
    print("[SUCCESS] LCEL QA chain is working correctly.")
    print("=" * 65)