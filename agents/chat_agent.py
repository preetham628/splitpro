from __future__ import annotations

from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.llm_factory import create_llm
from config import ChatAgentConfig
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


_RECEIPT_PREFIX = "I've scanned a receipt image."
_NON_BILL_PREFIX = "The user uploaded an image. It is NOT a bill"


def _serialize_history(history: list) -> list[dict]:
    out = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            out.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            out.append({
                "type": "ai",
                "content": msg.content,
                "tool_calls": msg.tool_calls or [],
                "additional_kwargs": msg.additional_kwargs or {},
            })
        elif isinstance(msg, ToolMessage):
            out.append({
                "type": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
    return out


def _deserialize_history(data: list[dict]) -> list:
    out = []
    for d in data:
        t = d.get("type")
        if t == "human":
            out.append(HumanMessage(content=d["content"]))
        elif t == "ai":
            out.append(AIMessage(
                content=d.get("content", ""),
                tool_calls=d.get("tool_calls", []),
                additional_kwargs=d.get("additional_kwargs", {}),
            ))
        elif t == "tool":
            out.append(ToolMessage(
                content=d["content"],
                tool_call_id=d["tool_call_id"],
            ))
    return out


def _extract_text(content) -> str:
    """Normalize AIMessage content — can be a str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return ""


def derive_display_messages(history: list) -> list[dict]:
    """Convert internal LangChain message history to frontend-renderable bubbles."""
    out = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            if content.startswith(_RECEIPT_PREFIX):
                out.append({"role": "user", "content": "[Receipt image uploaded]"})
            elif content.startswith(_NON_BILL_PREFIX):
                out.append({"role": "user", "content": "[Image uploaded — not a receipt]"})
            else:
                out.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            text = _extract_text(msg.content)
            if text:
                out.append({"role": "agent", "content": text})
    return out


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

        global_balances: dict[str, float] = defaultdict(float)
        warnings = []

        for bill in state.bills:
            person_subtotal: dict[str, float] = defaultdict(float)

            for item in bill.items:
                if item.qty_allocations:
                    # Qty-based: pay unit_price * units consumed
                    unit_price = item.unit_price
                    allocated_qty = sum(item.qty_allocations.values())
                    for person, pqty in item.qty_allocations.items():
                        person_subtotal[person] += unit_price * pqty

                    # Unallocated remainder goes equally to assigned_to minus those with explicit qtys
                    remaining_qty = item.qty - allocated_qty
                    if remaining_qty > 1e-9:
                        remainder_people = [p for p in item.assigned_to if p not in item.qty_allocations]
                        if not remainder_people:
                            remainder_people = state.participants
                        share = unit_price * remaining_qty / len(remainder_people)
                        for person in remainder_people:
                            person_subtotal[person] += share
                else:
                    # Fall back to all participants if still unassigned
                    recipients = item.assigned_to if item.assigned_to else state.participants
                    if not item.assigned_to:
                        warnings.append(f"'{item.name}' in {bill.bill_id} had no assignment — split equally.")
                    share = item.price / len(recipients)
                    for person in recipients:
                        person_subtotal[person] += share

            # Proportionally distribute tax + tip
            bill_subtotal = sum(person_subtotal.values())
            combined_extra = bill.tax + bill.tip
            if combined_extra > 0:
                if bill_subtotal > 0:
                    for person in list(person_subtotal.keys()):
                        proportion = person_subtotal[person] / bill_subtotal
                        person_subtotal[person] += proportion * combined_extra
                else:
                    equal_share = combined_extra / len(state.participants)
                    for person in state.participants:
                        person_subtotal[person] += equal_share

            # Accumulate into global balances
            bill_total = sum(person_subtotal.values())
            global_balances[bill.paid_by] += bill_total
            for person, amount in person_subtotal.items():
                global_balances[person] -= amount

        balances = dict(global_balances)
        settlements = Settlement.generate_settlements(balances)
        report = Settlement.format_report(balances, settlements)

        if warnings:
            warning_text = "Note: " + "; ".join(warnings) + "\n\n"
            report = warning_text + report

        state.finalized = True
        return report

    return [add_bill, set_participants, assign_items, set_payer, mark_items_unassigned, calculate_split]


class ChatAgent:
    """Conversational bill-splitting agent powered by LangChain tool calling."""

    def __init__(self, config: ChatAgentConfig = None):
        cfg = config or ChatAgentConfig()
        self._max_iterations = cfg.max_iterations

        self.state = SessionState()
        self.message_history: list[Any] = []

        self._tools = _build_tools(self.state)
        self.tool_map = {t.name: t for t in self._tools}

        llm = create_llm(cfg)
        self.llm_with_tools = llm.bind_tools(self._tools)

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "history": _serialize_history(self.message_history),
        }

    @classmethod
    def from_dict(cls, data: dict, config: ChatAgentConfig) -> "ChatAgent":
        agent = cls(config=config)
        # Replace the blank state with the restored one and rebuild tool closures
        agent.state = SessionState.from_dict(data.get("state", {}))
        agent._tools = _build_tools(agent.state)
        agent.tool_map = {t.name: t for t in agent._tools}
        llm = create_llm(config)
        agent.llm_with_tools = llm.bind_tools(agent._tools)
        agent.message_history = _deserialize_history(data.get("history", []))
        return agent

    def _system_message(self) -> SystemMessage:
        content = SYSTEM_PROMPT_TEMPLATE.format(state_summary=self.state.state_summary())
        return SystemMessage(content=content)

    def chat(self, user_message: str) -> str:
        """Process one user turn, executing tools as needed, and return the final response."""
        self.message_history.append(HumanMessage(content=user_message))

        for _ in range(self._max_iterations):
            messages = [self._system_message()] + self.message_history
            ai_response = self.llm_with_tools.invoke(messages)

            if not ai_response.tool_calls:
                self.message_history.append(ai_response)
                return ai_response.content

            # Execute all tool calls and collect results
            self.message_history.append(ai_response)
            for tc in ai_response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = tc["id"]

                if tool_name not in self.tool_map:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = self.tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Error executing {tool_name}: {e}"

                self.message_history.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call_id)
                )

        return "I'm having trouble processing that. Could you try rephrasing?"
