from __future__ import annotations

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.llm_factory import create_llm
from config import ChatAgentConfig
from core.checkpointer import get_checkpointer
from core.session_state import LineItem, ParsedBill, SessionState
from core.settlement import Settlement

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are SplitPro, a friendly assistant that helps groups split restaurant bills fairly.

## Your Job
Guide the user through these steps in order:
1. Parse any bills they paste — call add_bill for each one
2. Confirm who the participants are — call set_participants
3. For each bill, work through items asking who had what — call assign_items or mark_items_unassigned
4. Confirm who paid for each bill — call set_payer
5. When the user says they're done or asks for the final split — call calculate_split

The user can add more bills at any point. Always be ready to call add_bill again.

## Rules
- Call tools proactively the moment you have the information. Do NOT say "let me know when ready."
- When assigning items, if the user says "Alice and Bob shared the pasta", use assigned_to=["Alice","Bob"] and shared=true.
- When a bill item has qty > 1 and the user specifies how many units each person had (e.g., "Alice had 1 out of 4 burgers"), use qty_per_person={{"Alice": 1}} in the assignment. Unspecified units are split equally among the remaining assigned_to.
- Fractional units are allowed (e.g., "Alice had 25% of one burger" with item qty=4 → qty_per_person={{"Alice": 0.25}}).
- Fuzzy-match item names: if the user says "the chicken thing", match to the closest item name.
- Tax and tip are NOT assigned via assign_items — they are handled proportionally by calculate_split automatically.
- For lump-sum bills with no line items, add one item called "Total" with the full amount and mark it as shared.
- Keep replies concise. Confirm tool actions briefly and move the conversation forward.
- After calling add_bill, immediately ask for participants if not set, or move to item assignment if participants are known.

## Current Session State
{state_summary}
"""


def _build_tools(state: SessionState) -> list:
    """Build the six tools as closures over the shared SessionState instance."""

    @tool
    def add_bill(
        raw_text: str,
        description: str,
        items: list[dict],
        tax: float,
        tip: float,
    ) -> str:
        """
        Parse a bill pasted by the user and add it to the session.
        Call this whenever the user submits new bill text.

        Args:
            raw_text: The exact text the user pasted.
            description: A short human-readable label, e.g. "Dinner at Spice Garden".
            items: List of line items. Each must have "name" (str) and "price" (float, total for all units).
                   Optionally include "qty" (int, default 1) for items ordered in multiple units.
                   Do NOT include tax or tip lines here — pass them separately.
            tax: Tax amount from the bill. Use 0.0 if none.
            tip: Tip or service charge from the bill. Use 0.0 if none.
        """
        bill_id = state.next_bill_id()
        parsed_items = [
            LineItem(name=it["name"], price=float(it["price"]), qty=int(it.get("qty", 1)))
            for it in items
        ]
        bill = ParsedBill(
            bill_id=bill_id,
            raw_text=raw_text,
            description=description,
            items=parsed_items,
            tax=tax,
            tip=tip,
        )
        state.bills.append(bill)
        return (
            f"Added {bill_id}: '{description}' with {len(parsed_items)} item(s), "
            f"subtotal=${bill.subtotal():.2f}, tax=${tax:.2f}, tip=${tip:.2f}, "
            f"total=${bill.total():.2f}"
        )

    @tool
    def set_participants(names: list[str]) -> str:
        """
        Set the full list of participants for this session.
        Call this after the user tells you who is splitting the bills.

        Args:
            names: List of participant names, e.g. ["Alice", "Bob", "Carol"].
        """
        state.participants = [n.strip().title() for n in names if n.strip()]
        return f"Participants set: {', '.join(state.participants)}"

    @tool
    def assign_items(bill_id: str, assignments: list[dict]) -> str:
        """
        Assign one or more items on a bill to specific participants.
        Call this as the user tells you who had what. Partial calls are fine.

        Args:
            bill_id: The bill identifier, e.g. "bill_1".
            assignments: List of assignment objects. Each must have:
                - item_name (str): Item name as it appears in the bill.
                - assigned_to (list[str]): Participant names who had this item (equal split of full qty).
                - shared (bool): True if cost splits equally among assigned_to.
                - qty_per_person (dict, optional): Maps participant name -> number of units they consumed
                  (can be fractional, e.g. 0.25 for a quarter unit). Use this for qty-based splits.
                  Unallocated qty is split equally among the remaining assigned_to.
                  Example: item qty=4, qty_per_person={"Alice": 1, "Bob": 3} => Alice pays 1/4, Bob pays 3/4.
        """
        bill = state.get_bill(bill_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found. Known bills: {[b.bill_id for b in state.bills]}"

        results = []
        for a in assignments:
            item_name = a.get("item_name", "")
            assigned_to = [n.strip().title() for n in a.get("assigned_to", [])]
            shared = bool(a.get("shared", len(assigned_to) > 1))
            qty_per_person_raw: dict = a.get("qty_per_person", {})

            # Exact match first, then partial
            item = next((i for i in bill.items if i.name.lower() == item_name.lower()), None)
            if item is None:
                item = next((i for i in bill.items if item_name.lower() in i.name.lower()), None)
            if item is None:
                all_names = [i.name for i in bill.items]
                results.append(f"'{item_name}' not found. Available: {all_names}")
                continue

            if qty_per_person_raw:
                # Qty-based assignment
                qty_per_person = {n.strip().title(): float(q) for n, q in qty_per_person_raw.items()}
                unknown = [p for p in qty_per_person if p not in state.participants]
                if unknown:
                    results.append(
                        f"Unknown participant(s) {unknown} for '{item.name}'. "
                        f"Known: {state.participants}"
                    )
                    continue

                allocated_qty = sum(qty_per_person.values())
                if allocated_qty > item.qty + 1e-9:
                    results.append(
                        f"'{item.name}': allocated qty {allocated_qty} exceeds item qty {item.qty}."
                    )
                    continue

                item.qty_allocations = qty_per_person
                # assigned_to = explicitly listed people + those in qty_per_person
                all_assigned = list(qty_per_person.keys())
                for p in assigned_to:
                    if p not in all_assigned:
                        all_assigned.append(p)
                item.assigned_to = all_assigned
                item.shared = False
                item.unassigned = False

                remaining_qty = item.qty - allocated_qty
                detail = ", ".join(f"{p}:{q}u" for p, q in qty_per_person.items())
                if remaining_qty > 1e-9:
                    remainder_people = [p for p in assigned_to if p not in qty_per_person]
                    detail += f" | {remaining_qty:.2f} units split among {remainder_people or 'all'}"
                results.append(f"'{item.name}' (qty-based) -> {detail}")
            else:
                # Standard equal-split assignment
                unknown = [p for p in assigned_to if p not in state.participants]
                if unknown:
                    results.append(
                        f"Unknown participant(s) {unknown} for '{item.name}'. "
                        f"Known: {state.participants}"
                    )
                    continue

                item.assigned_to = assigned_to
                item.shared = shared
                item.unassigned = False
                item.qty_allocations = {}
                tag = " (shared)" if shared else ""
                results.append(f"'{item.name}' -> {', '.join(assigned_to)}{tag}")

        remaining = bill.unassigned_items()
        summary = "; ".join(results)
        if remaining:
            summary += f" | Still unassigned: {[i.name for i in remaining]}"
        else:
            summary += f" | All items assigned for {bill_id}."
        return summary

    @tool
    def set_payer(bill_id: str, paid_by: str) -> str:
        """
        Record who paid for a specific bill.
        Call this when the user says who covered the tab.

        Args:
            bill_id: The bill identifier.
            paid_by: Name of the participant who paid.
        """
        bill = state.get_bill(bill_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found."

        paid_by_normalized = paid_by.strip().title()
        if paid_by_normalized not in state.participants:
            return (
                f"Error: '{paid_by_normalized}' is not in the participant list. "
                f"Known participants: {state.participants}"
            )

        bill.paid_by = paid_by_normalized
        return f"Recorded: {paid_by_normalized} paid for {bill_id} ('{bill.description}')"

    @tool
    def mark_items_unassigned(bill_id: str, item_names: list[str]) -> str:
        """
        Mark specific items to be split equally among ALL participants.
        Use this when the user says "just split X evenly" or "don't worry about Y".

        Args:
            bill_id: The bill identifier.
            item_names: List of item names to split equally among everyone.
        """
        bill = state.get_bill(bill_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found."
        if not state.participants:
            return "Error: set participants first before marking items as shared."

        updated = []
        for name in item_names:
            item = next((i for i in bill.items if i.name.lower() == name.lower()), None)
            if item is None:
                item = next((i for i in bill.items if name.lower() in i.name.lower()), None)
            if item:
                item.assigned_to = list(state.participants)
                item.shared = True
                item.unassigned = True
                updated.append(item.name)

        if not updated:
            return f"No matching items found for: {item_names}"
        return f"Marked for equal split among all: {', '.join(updated)}"

    @tool
    def calculate_split() -> str:
        """
        Compute the final settlement using all bills and assignments.
        Call this when the user says they are done or asks for the final result.
        Tax and tip are automatically distributed proportionally.
        Any unassigned items are split equally among all participants.
        """
        if not state.participants:
            return "Error: no participants set."
        if not state.bills:
            return "Error: no bills added yet."

        missing_payers = [b.bill_id for b in state.bills if b.paid_by is None]
        if missing_payers:
            return f"Error: payer not set for {missing_payers}. Please set payers first."

        balances, warnings = Settlement.compute_balances(state.participants, state.bills)
        settlements = Settlement.generate_settlements(balances)
        report = Settlement.format_report(balances, settlements)

        if warnings:
            warning_text = "Note: " + "; ".join(warnings) + "\n\n"
            report = warning_text + report

        state.finalized = True
        return report

    return [add_bill, set_participants, assign_items, set_payer, mark_items_unassigned, calculate_split]


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


class ChatAgent:
    """Conversational bill-splitting agent — a small LangGraph graph
    (agent node -> tools node, looping until the model stops calling tools),
    checkpointed by session_id via core/checkpointer.py. Conversation
    continuity (the equivalent of the old hand-rolled message_history) is
    handled entirely by the checkpointer, keyed by thread_id = session_id —
    nothing to manually serialize/restore here anymore.
    """

    def __init__(self, session_id: str, config: ChatAgentConfig = None):
        self.session_id = session_id
        self._config = config or ChatAgentConfig()
        self._max_iterations = self._config.max_iterations
        self.state = SessionState()
        self._rebuild()

    def _rebuild(self) -> None:
        """(Re)build tools/llm/graph bound to the current self.state.

        Must run whenever self.state is replaced (see set_state()) — the
        tool closures and the graph's ToolNode both need to close over the
        same SessionState instance, so they're rebuilt together.
        """
        self._tools = _build_tools(self.state)
        llm = create_llm(self._config)
        self._llm_with_tools = llm.bind_tools(self._tools)

        graph = StateGraph(GraphState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self._tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        self._graph = graph.compile(checkpointer=get_checkpointer())

    def _agent_node(self, graph_state: GraphState) -> dict:
        """Render the system prompt fresh from live state on every call —
        same dynamic-prompt behavior as the original hand-rolled loop."""
        system_message = SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(state_summary=self.state.state_summary())
        )
        response = self._llm_with_tools.invoke([system_message] + graph_state["messages"])
        return {"messages": [response]}

    def set_state(self, state: SessionState) -> None:
        """Replace the bill-splitting state (e.g. when restoring a session
        from the DB) and rebuild the tools/graph to match."""
        self.state = state
        self._rebuild()

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the final response.

        The checkpointer transparently loads prior messages for this
        thread_id and persists the new ones — no manual history handling.
        """
        config = {
            "configurable": {"thread_id": self.session_id},
            # Each old "iteration" was one LLM call; the graph takes ~2 steps
            # per iteration (agent + tools), so double the budget plus slack.
            "recursion_limit": self._max_iterations * 2 + 1,
        }
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=user_message)]}, config=config
            )
        except GraphRecursionError:
            return "I'm having trouble processing that. Could you try rephrasing?"
        return result["messages"][-1].content
