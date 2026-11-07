"""
Conversation Module
Implements conversational RAG with chat history and question condensation.
Maintains a running list of (human, ai) message pairs across chat turns.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS


# ── Condenser Prompt ────────────────────────────────────────────────────────
# This prompt is used to rewrite a vague follow-up question into a
# complete, self-contained question using the conversation history.
# It is called BEFORE retrieval, not after.

CONDENSER_SYSTEM_PROMPT = """You are a question reformulator for a document Q&A system.

Given a conversation history and the user's latest follow-up question, \
rewrite the question into a single, complete, standalone question \
that can be understood WITHOUT reading the conversation history.

Rules:
- Resolve all pronouns (it, they, this, that, those) to their actual referents.
- Keep all specific details from the original question.
- Do NOT answer the question — only rewrite it.
- If the question is already self-contained, return it unchanged.
- Output ONLY the rewritten question, nothing else."""

CONDENSER_HUMAN_PROMPT = """Conversation history:
{chat_history}

Latest user question: {question}

Standalone question:"""


# ── QA Prompt with History ──────────────────────────────────────────────────
# This extends the prompt from Step 10 to include conversation history
# so the LLM can reference prior turns when composing its answer.

QA_SYSTEM_WITH_HISTORY = """You are a precise and helpful document assistant.
Answer the user's question using ONLY the provided document context.

Rules:
1. Use ONLY information explicitly stated in the CONTEXT section.
2. If the answer is not in the context, say exactly:
   "I could not find the answer in the provided documents."
3. Never speculate or use external knowledge.
4. Cite sources using their labels (e.g., "According to Source 1...").
5. Be concise and factual.

CONTEXT:
{context}"""


def build_condenser_chain(llm: ChatGroq):
    """
    Builds the question condensation chain.
    Input  : {chat_history: List[BaseMessage], question: str}
    Output : str (the standalone rewritten question)
    """
    condenser_prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSER_SYSTEM_PROMPT),
        ("human",  CONDENSER_HUMAN_PROMPT),
    ])
    return condenser_prompt | llm | StrOutputParser()


def build_qa_chain_with_history(llm: ChatGroq):
    """
    Builds the QA chain that is aware of conversation history.
    Input  : {context: str, question: str, chat_history: List[BaseMessage]}
    Output : str (the LLM's answer)
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_WITH_HISTORY),
        MessagesPlaceholder(variable_name="chat_history"),  # Injects prior turns
        ("human", "{question}"),
    ])
    return qa_prompt | llm | StrOutputParser()


def format_history_for_display(chat_history: List[BaseMessage]) -> str:
    """Formats the chat history list into a readable string for the condenser."""
    if not chat_history:
        return "(no prior conversation)"
    lines = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            lines.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


class ConversationalRAG:
    """
    A stateful conversational RAG session.
    Maintains chat history across multiple question-answer turns.
    
    Usage:
        session = ConversationalRAG(llm, retriever)
        answer  = session.chat("How many leave days do employees get?")
        answer  = session.chat("Can any of it be carried over?")  # Follow-up works!
        session.clear_history()  # Reset for a new conversation
    """

    def __init__(self, llm: ChatGroq, retriever):
        self.llm              = llm
        self.retriever        = retriever
        self.chat_history: List[BaseMessage] = []
        self.condenser_chain  = build_condenser_chain(llm)
        self.qa_chain         = build_qa_chain_with_history(llm)

    def chat(self, user_question: str, verbose: bool = True) -> str:
        """
        Processes one conversational turn:
        1. Condenses the question using history (if history exists).
        2. Retrieves relevant chunks using the condensed question.
        3. Generates an answer using the QA chain + history.
        4. Appends both the original question and answer to history.

        Args:
            user_question : The raw question from the user.
            verbose       : Whether to print intermediate steps.

        Returns:
            str: The LLM's answer.
        """
        # ── Step 1: Condense the question ───────────────────────────────────
        if self.chat_history:
            history_text = format_history_for_display(self.chat_history)

            if verbose:
                print(f"\n  [Condensing] Rewriting follow-up question...")

            standalone_question = self.condenser_chain.invoke({
                "chat_history": history_text,
                "question": user_question,
            })

            if verbose:
                print(f"  [Original ] \"{user_question}\"")
                print(f"  [Rewritten] \"{standalone_question}\"")
        else:
            # First turn — no history to condense against
            standalone_question = user_question
            if verbose:
                print(f"\n  [First turn — no condensation needed]")

        # ── Step 2: Retrieve relevant chunks ────────────────────────────────
        docs = self.retriever.invoke(standalone_question)
        context = self._format_context(docs)

        if verbose:
            sources = set(
                f"{os.path.basename(d.metadata.get('source','?'))} p.{d.metadata.get('page',0)+1}"
                for d in docs
            )
            print(f"  [Retrieved] {len(docs)} chunks from: {', '.join(sources)}")

        # ── Step 3: Generate answer with history ─────────────────────────────
        answer = self.qa_chain.invoke({
            "context": context,
            "question": user_question,         # Original question (not condensed)
            "chat_history": self.chat_history, # Full message history
        })

        # ── Step 4: Update history ───────────────────────────────────────────
        self.chat_history.append(HumanMessage(content=user_question))
        self.chat_history.append(AIMessage(content=answer))

        return answer

    def _format_context(self, docs) -> str:
        """Formats retrieved Document chunks into labeled context string."""
        if not docs:
            return "No relevant context found in the documents."
        parts = []
        for idx, doc in enumerate(docs, 1):
            source   = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_num = doc.metadata.get("page", 0) + 1
            parts.append(f"[Source {idx}: {source}, Page {page_num}]\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    def clear_history(self):
        """Resets the conversation — starts a fresh session."""
        self.chat_history = []
        print("[INFO] Chat history cleared. Starting new conversation.")

    def print_history(self):
        """Prints the full conversation history."""
        print("\n" + "=" * 65)
        print("CONVERSATION HISTORY")
        print("=" * 65)
        if not self.chat_history:
            print("  (empty)")
            return
        for i, msg in enumerate(self.chat_history, 1):
            role = "You" if isinstance(msg, HumanMessage) else "Bot"
            print(f"\n  [{i}] {role}: {msg.content}")
        print()


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.document_loader import load_single_pdf
    from src.text_splitter   import split_documents
    from src.embedder        import get_embedding_model
    from src.vector_store    import get_or_build_vector_store
    from src.retriever       import build_retriever
    from src.llm             import get_llm

    print("=" * 65)
    print("STEP 11: CONVERSATIONAL RAG WITH CHAT HISTORY")
    print("=" * 65 + "\n")

    # ── Initialize all components ─────────────────────────────────────────────
    print("[INFO] Initializing pipeline...")
    llm             = get_llm()
    documents       = load_single_pdf(os.path.join("data", "sample.pdf"))
    chunks          = split_documents(documents)
    embedding_model = get_embedding_model()
    vector_store    = get_or_build_vector_store(chunks, embedding_model)
    retriever       = build_retriever(vector_store)
    print("[OK] Pipeline ready.\n")

    # ── Create a conversational session ───────────────────────────────────────
    session = ConversationalRAG(llm, retriever)

    # ── Simulate a multi-turn conversation ────────────────────────────────────
    print("=" * 65)
    print("MULTI-TURN CONVERSATION SIMULATION")
    print("=" * 65)

    conversation = [
        "How many days of annual leave are employees entitled to?",
        "Can any of it be carried over to the next year?",         # "it" = annual leave
        "How much notice is required before taking leave?",
        "What about working from home — how many days is that?",
        "And what are the core hours for that?",                   # "that" = remote work
    ]

    for question in conversation:
        print(f"\n{'─' * 65}")
        print(f"USER: {question}")
        print(f"{'─' * 65}")
        answer = session.chat(question, verbose=True)
        print(f"\nBOT: {answer}")

    # ── Print the full history ─────────────────────────────────────────────────
    session.print_history()

    print("[SUCCESS] Conversational RAG with chat history is working correctly.")