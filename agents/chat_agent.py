from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.llm_factory import create_llm
from config import ChatAgentConfig
from core import database as db
from core.checkpointer import get_checkpointer
from core.session_state import LineItem, ParsedBill, SessionState
from core.settlement import Settlement

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are SplitPro, a friendly assistant that helps groups split restaurant bills fairly.

## Group chat
This is a shared group chat — several named participants may message you within the same
session, not just one person. Incoming messages may be tagged with the speaker's name in
brackets, e.g. "[Alice] we had pizza" — use that tag to track who said what, and answer
questions like "what did Bob say" or "did Alice already pay" accurately.

## Approvals
Every new or changed expense — a new bill, or an edit to its items, split, or payer — is
staged as a proposal and needs an admin to approve it before it affects anyone's balance.
Whenever add_bill, assign_items, set_payer, or mark_items_unassigned creates or updates a
proposal, say so explicitly (e.g. "proposed, awaiting admin approval") — never say an
expense was simply "added" or "updated" as if it were already final. If someone disputes
something that was already approved, just call the same tools as usual; corrections are
routed to a new proposal automatically and need no different handling from you.

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


def _bill_to_payload(bill: ParsedBill) -> dict:
    """Convert a ParsedBill into the plain-dict shape expense_proposals.payload
    is stored as (bill_id, description, raw_text, items[], tax, tip, paid_by) —
    the same shape create_proposal()/update_proposal_payload() expect and
    decide_proposal() materializes back into bills/bill_items."""
    return {
        "bill_id": bill.bill_id,
        "description": bill.description,
        "raw_text": bill.raw_text,
        "items": [
            {
                "name": item.name,
                "price": item.price,
                "qty": item.qty,
                "assigned_to": list(item.assigned_to),
                "shared": item.shared,
                "unassigned": item.unassigned,
                "qty_allocations": dict(item.qty_allocations),
            }
            for item in bill.items
        ],
        "tax": bill.tax,
        "tip": bill.tip,
        "paid_by": bill.paid_by,
    }


def _payload_to_bill(payload: dict) -> ParsedBill:
    """The inverse of _bill_to_payload() — rebuild a ParsedBill from a
    proposal payload dict so edits can reuse the exact same item-matching /
    assignment logic the tools already use against a live SessionState bill,
    whether the payload came from a pending proposal or a snapshot of an
    already-approved bill."""
    items = [
        LineItem(
            name=i["name"],
            price=i["price"],
            qty=i.get("qty", 1),
            assigned_to=list(i.get("assigned_to", [])),
            shared=i.get("shared", False),
            unassigned=i.get("unassigned", False),
            qty_allocations=dict(i.get("qty_allocations", {})),
        )
        for i in payload.get("items", [])
    ]
    return ParsedBill(
        bill_id=payload["bill_id"],
        raw_text=payload.get("raw_text", ""),
        description=payload.get("description", ""),
        items=items,
        tax=payload.get("tax", 0.0),
        tip=payload.get("tip", 0.0),
        paid_by=payload.get("paid_by"),
    )


def _find_pending_proposal(session_id: str, bill_id: str) -> Optional[dict]:
    """Find a still-pending proposal targeting this app-level bill_id, if any."""
    for proposal in db.list_pending_proposals(session_id):
        if proposal["payload"].get("bill_id") == bill_id:
            return proposal
    return None


def _load_bill_for_edit(
    state: SessionState, session_id: str, bill_id: str, user_id: Optional[int]
):
    """Resolve bill_id to an editable ParsedBill, plus enough context to
    persist the edit afterward. Returns (mode, bill, ref):

      - ("legacy", bill, None): no DB-backed speaker context (the CLI path,
        user_id is None) — bill is the live object in state.bills; the
        caller mutates it directly and there's nothing further to persist.
      - ("proposal", bill, proposal_id): a pending proposal already targets
        this bill_id — bill is rebuilt fresh from its payload; the caller
        must db.update_proposal_payload(proposal_id, ...) after mutating.
      - ("approved", bill, row_id): bill_id matches an already-approved bill
        with no pending proposal — bill is a COPY (state.bills stays
        untouched); the caller must propose the mutated payload as a
        correction via db.create_proposal(..., supersedes_bill_id=row_id).
      - (None, None, None): bill_id doesn't match anything.
    """
    if user_id is None:
        return "legacy", state.get_bill(bill_id), None

    proposal = _find_pending_proposal(session_id, bill_id)
    if proposal is not None:
        return "proposal", _payload_to_bill(proposal["payload"]), proposal["id"]

    approved = state.get_bill(bill_id)
    if approved is not None:
        row_id = db.get_bill_row_id(session_id, bill_id)
        return "approved", _payload_to_bill(_bill_to_payload(approved)), row_id

    return None, None, None


def _persist_edit(
    session_id: str,
    user_id: Optional[int],
    mode: str,
    bill: ParsedBill,
    ref,
    changed: bool,
) -> str:
    """Write an edited bill back to the right place per _load_bill_for_edit's
    mode, and return a suffix to append to the tool's result message. A no-op
    (empty suffix) for the legacy path, or when nothing actually changed."""
    if not changed or mode == "legacy":
        return ""
    if mode == "proposal":
        db.update_proposal_payload(ref, _bill_to_payload(bill))
        return " (Updated proposal — still awaiting admin approval.)"
    if mode == "approved":
        db.create_proposal(
            session_id,
            proposed_by=user_id,
            payload=_bill_to_payload(bill),
            supersedes_bill_id=ref,
        )
        return (
            " This bill was already approved — the change is proposed as a "
            "correction and needs admin approval before it takes effect."
        )
    return ""


def _apply_assign_items(
    bill: ParsedBill, assignments: list[dict], participants: list[str]
) -> tuple[bool, str]:
    """Shared mutation logic for the assign_items tool — operates on any
    ParsedBill (live, or a temporary one rebuilt from a proposal payload).
    Returns (changed, summary)."""
    results = []
    changed = False
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
            unknown = [p for p in qty_per_person if p not in participants]
            if unknown:
                results.append(
                    f"Unknown participant(s) {unknown} for '{item.name}'. "
                    f"Known: {participants}"
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
            changed = True

            remaining_qty = item.qty - allocated_qty
            detail = ", ".join(f"{p}:{q}u" for p, q in qty_per_person.items())
            if remaining_qty > 1e-9:
                remainder_people = [p for p in assigned_to if p not in qty_per_person]
                detail += f" | {remaining_qty:.2f} units split among {remainder_people or 'all'}"
            results.append(f"'{item.name}' (qty-based) -> {detail}")
        else:
            # Standard equal-split assignment
            unknown = [p for p in assigned_to if p not in participants]
            if unknown:
                results.append(
                    f"Unknown participant(s) {unknown} for '{item.name}'. "
                    f"Known: {participants}"
                )
                continue

            item.assigned_to = assigned_to
            item.shared = shared
            item.unassigned = False
            item.qty_allocations = {}
            changed = True
            tag = " (shared)" if shared else ""
            results.append(f"'{item.name}' -> {', '.join(assigned_to)}{tag}")

    remaining = bill.unassigned_items()
    summary = "; ".join(results)
    if remaining:
        summary += f" | Still unassigned: {[i.name for i in remaining]}"
    else:
        summary += f" | All items assigned for {bill.bill_id}."
    return changed, summary


def _apply_set_payer(bill: ParsedBill, paid_by: str, participants: list[str]) -> tuple[bool, str]:
    """Shared mutation logic for the set_payer tool."""
    paid_by_normalized = paid_by.strip().title()
    if paid_by_normalized not in participants:
        return False, (
            f"Error: '{paid_by_normalized}' is not in the participant list. "
            f"Known participants: {participants}"
        )

    bill.paid_by = paid_by_normalized
    return True, f"Recorded: {paid_by_normalized} paid for {bill.bill_id} ('{bill.description}')"


def _apply_mark_unassigned(
    bill: ParsedBill, item_names: list[str], participants: list[str]
) -> tuple[bool, str]:
    """Shared mutation logic for the mark_items_unassigned tool."""
    updated = []
    for name in item_names:
        item = next((i for i in bill.items if i.name.lower() == name.lower()), None)
        if item is None:
            item = next((i for i in bill.items if name.lower() in i.name.lower()), None)
        if item:
            item.assigned_to = list(participants)
            item.shared = True
            item.unassigned = True
            updated.append(item.name)

    if not updated:
        return False, f"No matching items found for: {item_names}"
    return True, f"Marked for equal split among all: {', '.join(updated)}"


def _build_tools(state: SessionState, session_id: str, get_current_user_id) -> list:
    """Build the six tools as closures over the shared SessionState instance,
    plus session_id and get_current_user_id (a zero-arg callable returning
    the current turn's speaker_user_id, or None). A callable rather than a
    fixed value because the tools are only rebuilt when the agent's state is
    replaced (_rebuild()), while the speaker can change on every chat() call —
    ChatAgent.chat() updates the value it returns before invoking the graph.

    get_current_user_id() returning None means there's no DB-backed speaker
    for this turn (the CLI path, main.py, never passes one) — in that case
    every tool falls back to mutating state.bills directly, exactly as
    before this proposal-based rework. Otherwise, new or changed expenses go
    through core/database.py's expense_proposals staging table and only ever
    land in state.bills once an admin approves them.
    """

    @tool
    def add_bill(
        raw_text: str,
        description: str,
        items: list[dict],
        tax: float,
        tip: float,
    ) -> str:
        """
        Parse a bill pasted by the user and propose it for this session.
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
        item_dicts = [
            {
                "name": it["name"],
                "price": float(it["price"]),
                "qty": int(it.get("qty", 1)),
                "assigned_to": [],
                "shared": False,
                "unassigned": False,
                "qty_allocations": {},
            }
            for it in items
        ]
        subtotal = sum(d["price"] for d in item_dicts)

        user_id = get_current_user_id()
        if user_id is None:
            bill = ParsedBill(
                bill_id=bill_id,
                raw_text=raw_text,
                description=description,
                items=[LineItem(name=d["name"], price=d["price"], qty=d["qty"]) for d in item_dicts],
                tax=tax,
                tip=tip,
            )
            state.bills.append(bill)
            return (
                f"Added {bill_id}: '{description}' with {len(item_dicts)} item(s), "
                f"subtotal=${bill.subtotal():.2f}, tax=${tax:.2f}, tip=${tip:.2f}, "
                f"total=${bill.total():.2f}"
            )

        payload = {
            "bill_id": bill_id,
            "description": description,
            "raw_text": raw_text,
            "items": item_dicts,
            "tax": tax,
            "tip": tip,
            "paid_by": None,
        }
        db.create_proposal(session_id, proposed_by=user_id, payload=payload)
        return (
            f"Proposed {bill_id}: '{description}' with {len(item_dicts)} item(s), "
            f"subtotal=${subtotal:.2f}, tax=${tax:.2f}, tip=${tip:.2f}, "
            f"total=${subtotal + tax + tip:.2f} — awaiting admin approval before it affects balances."
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
        Works whether the bill is still awaiting approval or was already
        approved — the change itself always needs (re-)approval either way.

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
        user_id = get_current_user_id()
        mode, bill, ref = _load_bill_for_edit(state, session_id, bill_id, user_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found. Known bills: {[b.bill_id for b in state.bills]}"

        changed, summary = _apply_assign_items(bill, assignments, state.participants)
        return summary + _persist_edit(session_id, user_id, mode, bill, ref, changed)

    @tool
    def set_payer(bill_id: str, paid_by: str) -> str:
        """
        Record who paid for a specific bill.
        Call this when the user says who covered the tab.

        Args:
            bill_id: The bill identifier.
            paid_by: Name of the participant who paid.
        """
        user_id = get_current_user_id()
        mode, bill, ref = _load_bill_for_edit(state, session_id, bill_id, user_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found."

        changed, message = _apply_set_payer(bill, paid_by, state.participants)
        return message + _persist_edit(session_id, user_id, mode, bill, ref, changed)

    @tool
    def mark_items_unassigned(bill_id: str, item_names: list[str]) -> str:
        """
        Mark specific items to be split equally among ALL participants.
        Use this when the user says "just split X evenly" or "don't worry about Y".

        Args:
            bill_id: The bill identifier.
            item_names: List of item names to split equally among everyone.
        """
        if not state.participants:
            return "Error: set participants first before marking items as shared."

        user_id = get_current_user_id()
        mode, bill, ref = _load_bill_for_edit(state, session_id, bill_id, user_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found."

        changed, message = _apply_mark_unassigned(bill, item_names, state.participants)
        return message + _persist_edit(session_id, user_id, mode, bill, ref, changed)

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
        self._current_user_id: Optional[int] = None
        self._rebuild()

    def _rebuild(self) -> None:
        """(Re)build tools/llm/graph bound to the current self.state.

        Must run whenever self.state is replaced (see set_state()) — the
        tool closures and the graph's ToolNode both need to close over the
        same SessionState instance, so they're rebuilt together.
        """
        self._tools = _build_tools(self.state, self.session_id, lambda: self._current_user_id)
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

    def chat(
        self,
        user_message: str,
        speaker_name: Optional[str] = None,
        speaker_user_id: Optional[int] = None,
    ) -> str:
        """Process one user turn and return the final response.

        The checkpointer transparently loads prior messages for this
        thread_id and persists the new ones — no manual history handling.

        speaker_name/speaker_user_id identify who's talking in a shared
        group session: speaker_user_id drives whether the tools stage new
        or changed expenses as proposals awaiting admin approval (see
        _build_tools), and speaker_name is surfaced to the model by
        prefixing the message content — e.g. "[Alice] we had pizza" —
        chosen over LangChain's HumanMessage(name=...) field since it isn't
        reliably surfaced to the underlying model across all three
        providers (OpenAI/Anthropic/Google) this app supports, and mixing
        that field for one provider with a prefix for others would make
        attribution behavior inconsistent depending on which provider a
        session happens to use.

        Both parameters default to None, and the CLI (main.py) never passes
        them — with speaker_name=None the message content is unchanged from
        before this parameter existed, and with speaker_user_id=None every
        tool takes its original, non-proposal code path (see _build_tools),
        so CLI behavior is unaffected.
        """
        self._current_user_id = speaker_user_id
        content = f"[{speaker_name}] {user_message}" if speaker_name else user_message

        config = {
            "configurable": {"thread_id": self.session_id},
            # Each old "iteration" was one LLM call; the graph takes ~2 steps
            # per iteration (agent + tools), so double the budget plus slack.
            "recursion_limit": self._max_iterations * 2 + 1,
        }
        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=content)]}, config=config
            )
        except GraphRecursionError:
            return "I'm having trouble processing that. Could you try rephrasing?"
        return result["messages"][-1].content
