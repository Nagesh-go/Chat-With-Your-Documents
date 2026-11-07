"""
Question Condenser Module
Handles standalone question generation from conversational follow-ups.
Includes HistoryManager for windowed history to prevent token overflow.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import List, Tuple
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ── Configuration ────────────────────────────────────────────────────────────
HISTORY_WINDOW_SIZE = 6    # Max number of messages (3 exchanges) sent to condenser
                            # Each exchange = 1 HumanMessage + 1 AIMessage = 2 messages


# ── Condenser Prompt (production-grade with few-shot examples) ───────────────
CONDENSER_SYSTEM = """You are a question reformulator for a document Q&A system.

Given a conversation history and the user's latest follow-up question, \
rewrite the question into a single, complete, standalone question \
that can be understood WITHOUT reading the conversation history.

RULES:
- Resolve ALL pronouns (it, they, this, that, those, them, its) to their referents.
- Include all implied context from the conversation.
- Do NOT answer the question — ONLY rewrite it.
- If the question is already fully self-contained, return it EXACTLY as written.
- Output ONLY the rewritten question. No explanations, no preamble.

EXAMPLES:

Example 1:
History:
  Human: How many annual leave days do employees get?
  AI: Employees get 20 days of annual leave per year.
Follow-up: Can any of it be carried over?
Standalone: Can any annual leave days be carried over to the following year?

Example 2:
History:
  Human: What is the remote work policy?
  AI: Employees may work remotely up to 3 days per week.
Follow-up: And what are the core hours for that?
Standalone: What are the core working hours for remote employees?

Example 3:
History:
  Human: What is maternity leave entitlement?
  AI: Female employees get 26 weeks of paid maternity leave.
Follow-up: Does that apply to adoptions too?
Standalone: Does the 26-week paid maternity leave also apply to adopted children?"""

CONDENSER_HUMAN = """Conversation history:
{chat_history}

Follow-up question: {question}
Standalone question:"""


@dataclass
class HistoryManager:
    """
    Manages the conversation history with windowing to prevent token overflow.

    Attributes:
        full_history  : Complete list of all messages in the conversation.
        window_size   : Max number of messages passed to the condenser.
    """
    full_history: List[BaseMessage] = field(default_factory=list)
    window_size: int = HISTORY_WINDOW_SIZE

    def add_exchange(self, question: str, answer: str):
        """Appends one Q&A exchange (2 messages) to the full history."""
        self.full_history.append(HumanMessage(content=question))
        self.full_history.append(AIMessage(content=answer))

    def get_windowed_history(self) -> List[BaseMessage]:
        """
        Returns only the most recent `window_size` messages.
        This is what gets sent to the condenser — not the full history.
        """
        return self.full_history[-self.window_size:]

    def get_full_history(self) -> List[BaseMessage]:
        """Returns the complete untruncated history (used by the QA chain)."""
        return self.full_history

    def format_for_condenser(self) -> str:
        """
        Formats the windowed history as a readable string for the condenser prompt.
        The condenser uses a plain text version (not raw message objects).
        """
        windowed = self.get_windowed_history()
        if not windowed:
            return "(no prior conversation)"
        lines = []
        for msg in windowed:
            if isinstance(msg, HumanMessage):
                lines.append(f"  Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Truncate very long AI answers to 300 chars for the condenser
                # We only need the gist for pronoun resolution, not the full answer
                content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                lines.append(f"  AI: {content}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        """Returns True if no exchanges have been recorded yet."""
        return len(self.full_history) == 0

    def turn_count(self) -> int:
        """Returns the number of completed Q&A exchanges."""
        return len(self.full_history) // 2

    def clear(self):
        """Resets the history."""
        self.full_history = []

    def summary(self) -> str:
        """Returns a one-line status string."""
        total   = len(self.full_history)
        windowed = min(total, self.window_size)
        return (f"History: {self.turn_count()} turn(s) | "
                f"{total} messages total | "
                f"Sending {windowed} to condenser")


def build_condenser_chain(llm: ChatGroq):
    """
    Builds the production-grade question condensation LCEL chain.
    Input  : {"chat_history": str, "question": str}
    Output : str (the standalone rewritten question)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSER_SYSTEM),
        ("human",  CONDENSER_HUMAN),
    ])
    return prompt | llm | StrOutputParser()


def condense_question(
    condenser_chain,
    history_manager: HistoryManager,
    user_question: str,
    verbose: bool = True,
) -> str:
    """
    Condenses a follow-up question into a standalone question.
    Skips the LLM call entirely if no history exists (first question).

    Args:
        condenser_chain : The LCEL condensation chain.
        history_manager : The HistoryManager holding conversation state.
        user_question   : The raw user question (may be vague).
        verbose         : Whether to print condensation details.

    Returns:
        str: A standalone question suitable for FAISS retrieval.
    """
    # First question — skip condensation, no API call needed
    if history_manager.is_empty():
        if verbose:
            print(f"  [Turn 1] No history — using question as-is.")
        return user_question

    history_text = history_manager.format_for_condenser()

    if verbose:
        print(f"  {history_manager.summary()}")
        print(f"  [Original ] \"{user_question}\"")

    standalone = condenser_chain.invoke({
        "chat_history": history_text,
        "question": user_question,
    })

    # Strip any accidental surrounding quotes the LLM may add
    standalone = standalone.strip().strip('"').strip("'")

    if verbose:
        changed = standalone.lower() != user_question.lower()
        status  = "REWRITTEN" if changed else "UNCHANGED (already standalone)"
        print(f"  [Rewritten] \"{standalone}\"  ← {status}")

    return standalone


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.llm import get_llm

    print("=" * 65)
    print("STEP 12: QUESTION CONDENSATION TESTS")
    print("=" * 65 + "\n")

    llm              = get_llm()
    condenser_chain  = build_condenser_chain(llm)
    history          = HistoryManager(window_size=HISTORY_WINDOW_SIZE)

    print("[OK] Condenser chain built.\n")

    # ── Define 6 condensation test scenarios ─────────────────────────────────
    scenarios: List[Tuple[str, str, str]] = [
        # (prior_question, prior_answer, follow_up)
        (
            "How many annual leave days do employees get?",
            "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
            "Can any of it be carried over?"
        ),
        (
            "Can any of it be carried over?",
            "Unused leave of up to 5 days may be carried over to the following year.",
            "Who approves the carryover?"
        ),
        (
            "What is the remote work policy?",
            "Employees may work remotely up to 3 days per week subject to manager approval.",
            "What are the core hours for that?"
        ),
        (
            "What are the core hours for remote work?",
            "Core working hours of 10:00 AM to 4:00 PM must be maintained regardless of location.",
            "Does that apply on public holidays too?"
        ),
        (
            "What is the maternity leave entitlement?",
            "Female employees are entitled to 26 weeks of paid maternity leave.",
            "What is the reimbursement deadline?"   # Completely new topic — already standalone
        ),
        (
            "What is the reimbursement deadline?",
            "Reimbursements must be submitted within 30 days of incurring the expense.",
            "Can you explain the entire company leave policy?"  # Broad — no pronouns
        ),
    ]

    print("=" * 65)
    print("RUNNING 6 CONDENSATION SCENARIOS")
    print("=" * 65)

    for idx, (prior_q, prior_a, follow_up) in enumerate(scenarios, start=1):
        print(f"\n{'─' * 65}")
        print(f"SCENARIO {idx}")
        print(f"{'─' * 65}")

        # Seed the history with the prior exchange
        history.add_exchange(prior_q, prior_a)

        standalone = condense_question(
            condenser_chain, history, follow_up, verbose=True
        )

    # ── Show final history state ──────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"FINAL HISTORY STATE")
    print(f"{'=' * 65}")
    print(f"  Total turns recorded : {history.turn_count()}")
    print(f"  Window size setting  : {HISTORY_WINDOW_SIZE} messages")
    print(f"  Messages in window   : {len(history.get_windowed_history())}")
    print(f"\n[SUCCESS] Question condensation module is fully operational.")