from __future__ import annotations

from typing import Annotated, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agents.llm_factory import create_llm
from config import ChatAgentConfig
from core import database as db
from core.checkpointer import get_checkpointer
from core.session_state import LineItem, ParsedBill, SessionState, validate_contribution_map
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
5. Call calculate_split any time to show the group the current settlement — it's always
   computed fresh from the latest approved data, so there's no separate "finalize" step.
   Call it whenever the user asks "who owes what" or "show me the split", even repeatedly
   as more bills/corrections get approved — each call just reflects whatever is approved
   as of that moment.

The user can add more bills at any point. Always be ready to call add_bill again.

## Rules
- Call tools proactively the moment you have the information. Do NOT say "let me know when ready."
- When assigning items, if the user says "Alice and Bob shared the pasta", use assigned_to=["Alice","Bob"] and shared=true.
- assign_items supports three allocation modes for a split that isn't equal — qty_per_person (units), percentage_per_person (0-100), or amount_per_person (dollars) — pick whichever matches how the user described it. See the assign_items tool description for worked examples of each. Anyone in assigned_to not given an explicit share splits whatever's left equally.
- When a bill item has qty > 1 and the user specifies how many units each person had (e.g., "Alice had 1 out of 4 burgers"), use qty_per_person={{"Alice": 1}} in the assignment. Unspecified units are split equally among the remaining assigned_to.
- Fractional units are allowed (e.g., "Alice had 25% of one burger" with item qty=4 → qty_per_person={{"Alice": 0.25}}).
- set_payer supports multiple payers on one bill, each by exact amount, percentage (0-100) of the bill total, or equal share of the rest — see the set_payer tool description for worked examples.
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
                "cost_allocations": dict(item.cost_allocations),
            }
            for item in bill.items
        ],
        "tax": bill.tax,
        "tip": bill.tip,
        "paid_by": dict(bill.paid_by),
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
            cost_allocations=dict(i.get("cost_allocations", {})),
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
        paid_by=dict(payload.get("paid_by") or {}),
    )


def _find_pending_proposal(session_id: str, bill_id: str) -> Optional[dict]:
    """Find a still-pending proposal targeting this app-level bill_id, if any."""
    for proposal in db.list_pending_proposals(session_id):
        if proposal["payload"].get("bill_id") == bill_id:
            return proposal
    return None


def _speaker_user_id(config: RunnableConfig) -> Optional[int]:
    """Pull the current turn's speaker_user_id out of the graph's per-invocation
    config. This is read fresh from `config` on every tool/node call — not
    from any attribute on ChatAgent — because a cached ChatAgent instance
    (server.py keeps one per session_id, reused across requests) can have
    two turns from two different speakers in flight at once; a value stored
    on `self` would be a shared-mutable-state race between them. `config` is
    built fresh per chat() call and threaded through by LangGraph itself, so
    each invocation only ever sees its own speaker_user_id.
    """
    return (config.get("configurable") or {}).get("speaker_user_id")


def _known_bill_ids(state: SessionState, session_id: str, user_id: Optional[int]) -> list[str]:
    """Bill ids a 'bill not found' error can honestly point to: approved
    bills plus, in DB-backed mode, still-pending proposals — otherwise a
    bill the model just proposed (and was told about via the tool's own
    return value) would vanish from this list on the very next tool call
    that references it, since it's not in state.bills yet.

    A pending correction proposal reuses its target bill's app-level
    bill_id (that's how it's matched back up on approval — see
    decide_proposal() in core/database.py), so the same id can legitimately
    appear in both state.bills and the pending list at once; dict.fromkeys
    dedupes while preserving order, since a caller-facing list showing the
    same id twice would just be confusing, not more informative.
    """
    ids = [b.bill_id for b in state.bills]
    if user_id is not None:
        ids += [p["payload"]["bill_id"] for p in db.list_pending_proposals(session_id)]
    return list(dict.fromkeys(ids))


def _pending_proposals_summary(session_id: str) -> str:
    """Render still-pending proposals for inclusion in the system prompt, so
    the model doesn't lose track of a bill it just proposed — state_summary()
    only ever reflects state.bills (approved data), so without this a bill
    proposed this turn looks like it doesn't exist come the next turn's
    prompt, even though the model's own previous tool result said otherwise."""
    pending = db.list_pending_proposals(session_id)
    if not pending:
        return ""

    lines = ["PENDING PROPOSALS (awaiting admin approval — not yet reflected in BILLS above):"]
    for p in pending:
        payload = p["payload"]
        tag = " [correction to an approved bill]" if p.get("supersedes_bill_id") else ""
        lines.append(f"  [{payload['bill_id']}] {payload.get('description', '')}{tag}")
    return "\n".join(lines)


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


def _resolve_contribution_entries(
    explicit: Dict[str, float],
    is_percentage: bool,
    equal_share_names: list,
    total: float,
) -> tuple:
    """Shared math for set_payer and assign_items: turn a mix of
    explicit-amount/explicit-percentage entries plus a list of
    "whatever's left, split evenly" names into a canonical
    name -> dollar-amount map summing to `total`.

    `explicit` is name -> amount (if is_percentage is False) or name ->
    percentage 0-100 (if is_percentage is True). Returns (resolved_map,
    error) — error is a human-readable string (percentage out of range, or
    explicit entries alone already exceed the total) when resolution isn't
    possible; resolved_map is None in that case.
    """
    if is_percentage:
        for name, pct in explicit.items():
            if not (0 <= pct <= 100):
                return None, f"percentage for '{name}' must be between 0 and 100, got {pct}."
        resolved = {name: total * pct / 100.0 for name, pct in explicit.items()}
    else:
        resolved = dict(explicit)

    explicit_total = sum(resolved.values())
    if explicit_total > total + 0.01:
        return None, (
            f"the explicit amounts/percentages given already add up to ${explicit_total:.2f}, "
            f"which exceeds the ${total:.2f} total."
        )

    if equal_share_names:
        remainder = total - explicit_total
        share = remainder / len(equal_share_names)
        for name in equal_share_names:
            resolved[name] = resolved.get(name, 0.0) + share

    return resolved, None


def _apply_assign_items(
    bill: ParsedBill, assignments: list[dict], participants: list[str]
) -> tuple[bool, str]:
    """Shared mutation logic for the assign_items tool — operates on any
    ParsedBill (live, or a temporary one rebuilt from a proposal payload).
    Returns (changed, summary).

    Each assignment picks at most one allocation mode — qty_per_person,
    percentage_per_person, or amount_per_person — which is resolved here,
    together with any remaining `assigned_to` names (equal split of
    whatever's left), into the item's canonical cost_allocations dollar
    map. No allocation dict at all keeps the original equal-split-of-the-
    full-price-among-assigned_to behavior (empty cost_allocations)."""
    results = []
    changed = False
    for a in assignments:
        item_name = a.get("item_name", "")
        assigned_to = [n.strip().title() for n in a.get("assigned_to", [])]
        shared = bool(a.get("shared", len(assigned_to) > 1))
        qty_per_person_raw: dict = a.get("qty_per_person", {})
        percentage_per_person_raw: dict = a.get("percentage_per_person", {})
        amount_per_person_raw: dict = a.get("amount_per_person", {})

        modes_given = sum(bool(m) for m in (qty_per_person_raw, percentage_per_person_raw, amount_per_person_raw))
        if modes_given > 1:
            results.append(
                f"'{item_name}': specify only one of qty_per_person, percentage_per_person, "
                f"or amount_per_person per item."
            )
            continue

        # Exact match first, then partial
        item = next((i for i in bill.items if i.name.lower() == item_name.lower()), None)
        if item is None:
            item = next((i for i in bill.items if item_name.lower() in i.name.lower()), None)
        if item is None:
            all_names = [i.name for i in bill.items]
            results.append(f"'{item_name}' not found. Available: {all_names}")
            continue

        if qty_per_person_raw:
            # Qty-based assignment: resolve units to dollars via unit_price,
            # then fall through to the same dollar-remainder-split math.
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

            all_assigned = list(qty_per_person.keys())
            for p in assigned_to:
                if p not in all_assigned:
                    all_assigned.append(p)
            remainder_people = [p for p in all_assigned if p not in qty_per_person]

            unit_price = item.unit_price
            explicit_amounts = {name: unit_price * q for name, q in qty_per_person.items()}
            cost_allocations, error = _resolve_contribution_entries(
                explicit_amounts, is_percentage=False, equal_share_names=remainder_people, total=item.price
            )
            if error:
                results.append(f"'{item.name}': {error}")
                continue
            if not validate_contribution_map(cost_allocations, item.price):
                results.append(
                    f"'{item.name}': resolved qty allocations don't add up to the item price "
                    f"(${item.price:.2f})."
                )
                continue

            item.cost_allocations = cost_allocations
            item.assigned_to = all_assigned
            item.shared = False
            item.unassigned = False
            changed = True

            remaining_qty = item.qty - allocated_qty
            detail = ", ".join(f"{p}:{q}u" for p, q in qty_per_person.items())
            if remaining_qty > 1e-9:
                detail += f" | {remaining_qty:.2f} unit(s) split among {remainder_people or 'all'}"
            results.append(f"'{item.name}' (qty-based) -> {detail}")

        elif percentage_per_person_raw or amount_per_person_raw:
            is_percentage = bool(percentage_per_person_raw)
            raw = percentage_per_person_raw if is_percentage else amount_per_person_raw
            explicit = {n.strip().title(): float(v) for n, v in raw.items()}
            unknown = [p for p in explicit if p not in participants]
            if unknown:
                results.append(
                    f"Unknown participant(s) {unknown} for '{item.name}'. "
                    f"Known: {participants}"
                )
                continue

            all_assigned = list(explicit.keys())
            for p in assigned_to:
                if p not in all_assigned:
                    all_assigned.append(p)
            remainder_people = [p for p in all_assigned if p not in explicit]

            cost_allocations, error = _resolve_contribution_entries(
                explicit, is_percentage=is_percentage, equal_share_names=remainder_people, total=item.price
            )
            if error:
                results.append(f"'{item.name}': {error}")
                continue
            if not validate_contribution_map(cost_allocations, item.price):
                kind = "percentages" if is_percentage else "amounts"
                alloc_str = ", ".join(f"{n}: ${v:.2f}" for n, v in cost_allocations.items())
                results.append(
                    f"'{item.name}': resolved {kind} ({alloc_str}) sum to "
                    f"${sum(cost_allocations.values()):.2f}, which doesn't match the item "
                    f"price of ${item.price:.2f}."
                )
                continue

            item.cost_allocations = cost_allocations
            item.assigned_to = all_assigned
            item.shared = False
            item.unassigned = False
            changed = True

            kind_label = "percentage-based" if is_percentage else "amount-based"
            unit = "%" if is_percentage else "$"
            detail = ", ".join(f"{p}:{v}{unit}" for p, v in explicit.items())
            if remainder_people:
                detail += f" | remainder split among {remainder_people}"
            results.append(f"'{item.name}' ({kind_label}) -> {detail}")
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
            item.cost_allocations = {}
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


def _apply_set_payer(bill: ParsedBill, payers: list[dict], participants: list[str]) -> tuple[bool, str]:
    """Shared mutation logic for the set_payer tool. Resolves a mix of
    explicit-amount, explicit-percentage, and equal-share-of-remainder payer
    entries into the bill's canonical paid_by dollar map, validating it sums
    to the bill total before storing anything."""
    explicit_amounts: Dict[str, float] = {}
    explicit_percentages: Dict[str, float] = {}
    equal_share_names = []
    seen = set()

    for entry in payers:
        name = str(entry.get("name", "")).strip().title()
        if not name:
            return False, "Error: each payer entry needs a 'name'."
        if name in seen:
            return False, f"Error: duplicate payer entry for '{name}'."
        seen.add(name)
        if name not in participants:
            return False, (
                f"Error: '{name}' is not in the participant list. "
                f"Known participants: {participants}"
            )

        has_amount = entry.get("amount") is not None
        has_pct = entry.get("percentage") is not None
        if has_amount and has_pct:
            return False, f"Error: payer entry for '{name}' specifies both 'amount' and 'percentage' — use only one."

        if has_amount:
            explicit_amounts[name] = float(entry["amount"])
        elif has_pct:
            explicit_percentages[name] = float(entry["percentage"])
        else:
            equal_share_names.append(name)

    total = bill.total()

    # Resolve percentage entries to dollars first, then treat them exactly
    # like explicit amounts for the remainder-splitting math.
    if explicit_percentages:
        for name, pct in explicit_percentages.items():
            if not (0 <= pct <= 100):
                return False, f"Error: percentage for '{name}' must be between 0 and 100, got {pct}."
        explicit_amounts.update({name: total * pct / 100.0 for name, pct in explicit_percentages.items()})

    resolved, error = _resolve_contribution_entries(
        explicit_amounts, is_percentage=False, equal_share_names=equal_share_names, total=total
    )
    if error:
        return False, f"Error: {error}"

    if not validate_contribution_map(resolved, total):
        amounts_str = ", ".join(f"{n}: ${a:.2f}" for n, a in resolved.items())
        return False, (
            f"Error: resolved payer amounts ({amounts_str}) sum to ${sum(resolved.values()):.2f}, "
            f"which doesn't match {bill.bill_id}'s total of ${total:.2f}. Adjust and try again."
        )

    bill.paid_by = resolved
    paid_by_str = ", ".join(f"{n}: ${a:.2f}" for n, a in resolved.items())
    return True, f"Recorded payer(s) for {bill.bill_id} ('{bill.description}'): {paid_by_str}"


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


def _build_tools(state: SessionState, session_id: str) -> list:
    """Build the six tools as closures over the shared SessionState instance
    and session_id. Each tool that needs to know who's speaking declares an
    extra `config: RunnableConfig` parameter — LangChain's tool machinery
    detects that annotation, injects the graph's actual per-invocation
    RunnableConfig at call time, and hides the parameter from the schema the
    LLM sees, so it never shows up as a model-fillable argument. See
    _speaker_user_id() for why this is a config lookup and not a value
    closed over here or stored on ChatAgent.

    _speaker_user_id(config) returning None means there's no DB-backed
    speaker for this turn (the CLI path, main.py, never passes one) — in
    that case every tool falls back to mutating state.bills directly,
    exactly as before this proposal-based rework. Otherwise, new or changed
    expenses go through core/database.py's expense_proposals staging table
    and only ever land in state.bills once an admin approves them.
    """

    @tool
    def add_bill(
        raw_text: str,
        description: str,
        items: list[dict],
        tax: float,
        tip: float,
        config: RunnableConfig,
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
                "cost_allocations": {},
            }
            for it in items
        ]
        subtotal = sum(d["price"] for d in item_dicts)

        user_id = _speaker_user_id(config)
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
            "paid_by": {},
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
    def assign_items(bill_id: str, assignments: list[dict], config: RunnableConfig) -> str:
        """
        Assign one or more items on a bill to specific participants.
        Call this as the user tells you who had what. Partial calls are fine.
        Works whether the bill is still awaiting approval or was already
        approved — the change itself always needs (re-)approval either way.

        Args:
            bill_id: The bill identifier, e.g. "bill_1".
            assignments: List of assignment objects. Each must have:
                - item_name (str): Item name as it appears in the bill.
                - assigned_to (list[str]): Everyone who had this item, including anyone also
                  named in qty_per_person/percentage_per_person/amount_per_person below.
                - shared (bool): True if cost splits equally among assigned_to (only meaningful
                  when none of the three allocation dicts below are used).

              Pick AT MOST ONE of the following three allocation modes per item — they are
              mutually exclusive with each other and with plain equal-split assigned_to. In all
              three, anyone listed in assigned_to but NOT given an explicit share splits
              whatever's left of the item price equally among themselves.

                - qty_per_person (dict[str, float], optional): participant name -> number of
                  units they consumed (fractional units allowed, e.g. 0.25 for a quarter unit).
                  Resolves to dollars via the item's price / qty.
                  Example: item "Burgers" has qty=4, price=$40 ($10/unit). "Alice had 1 burger,
                  Bob had 3" ->
                    assigned_to=["Alice", "Bob"], qty_per_person={"Alice": 1, "Bob": 3}
                  (Alice pays $10, Bob pays $30.)

                - percentage_per_person (dict[str, float], optional): participant name ->
                  percentage (0-100, NOT 0-1) of this item's price.
                  Example: item "Beer" costs $21, 8 people shared it, Alice had more than her
                  share. "Alice had 30% of the beer, the other 7 split the rest equally" ->
                    assigned_to=["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Gina", "Hank"],
                    percentage_per_person={"Alice": 30}
                  (Alice pays $6.30; the other 7 split the remaining $14.70 -> $2.10 each.)

                - amount_per_person (dict[str, float], optional): participant name -> exact
                  dollar amount of this item's price they owe.
                  Example: item "Cake" costs $18. "Alice put in $12 for the cake, Bob covers
                  the rest" ->
                    assigned_to=["Alice", "Bob"], amount_per_person={"Alice": 12.0}
                  (Alice pays $12, Bob pays the remaining $6.)

              Omitting all three (just assigned_to + shared) keeps the original behavior: the
              full item price splits equally among everyone in assigned_to.

              Whichever mode is used, the resolved per-person dollar amounts must add up to the
              item's price (within a small rounding tolerance) — if they don't (e.g. percentages
              or amounts that overshoot), this call returns an error instead of assigning
              anything for that item.
        """
        user_id = _speaker_user_id(config)
        mode, bill, ref = _load_bill_for_edit(state, session_id, bill_id, user_id)
        if bill is None:
            known = _known_bill_ids(state, session_id, user_id)
            return f"Error: bill '{bill_id}' not found. Known bills: {known}"

        changed, summary = _apply_assign_items(bill, assignments, state.participants)
        return summary + _persist_edit(session_id, user_id, mode, bill, ref, changed)

    @tool
    def set_payer(bill_id: str, payers: list[dict], config: RunnableConfig) -> str:
        """
        Record who paid for a specific bill. Supports a single payer or multiple payers
        splitting the payment itself (as opposed to splitting who owes what for the items).

        Call this when the user says who covered the tab.

        Args:
            bill_id: The bill identifier.
            payers: List of payer entries, each a dict with a "name" (str) and AT MOST ONE of:
                - "amount" (float): the exact dollar amount this person paid.
                - "percentage" (float, 0-100, NOT 0-1): the percentage of the bill's total
                  (subtotal + tax + tip) this person paid.
                - neither key: this person's payment is an equal share of whatever's left of
                  the bill total after the explicit amount/percentage entries above are
                  subtracted out.

              Examples (bill total = $100):
                - "Alice paid for everything":
                  payers=[{"name": "Alice"}]
                - "Alice paid $60, Bob paid $40":
                  payers=[{"name": "Alice", "amount": 60.0}, {"name": "Bob", "amount": 40.0}]
                - "Alice paid 73%, Bob paid the rest":
                  payers=[{"name": "Alice", "percentage": 73.0}, {"name": "Bob"}]
                - "Alice, Bob, and Carol split paying for it evenly":
                  payers=[{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]

              The resolved amounts must add up to the bill's total within a small rounding
              tolerance — if they don't, this call returns an error instead of recording
              anything.
        """
        user_id = _speaker_user_id(config)
        mode, bill, ref = _load_bill_for_edit(state, session_id, bill_id, user_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found."

        changed, message = _apply_set_payer(bill, payers, state.participants)
        return message + _persist_edit(session_id, user_id, mode, bill, ref, changed)

    @tool
    def mark_items_unassigned(bill_id: str, item_names: list[str], config: RunnableConfig) -> str:
        """
        Mark specific items to be split equally among ALL participants.
        Use this when the user says "just split X evenly" or "don't worry about Y".

        Args:
            bill_id: The bill identifier.
            item_names: List of item names to split equally among everyone.
        """
        # Bill-not-found is checked before participants-not-set — same order
        # as pre-proposal-rework code; swapping it would change which error
        # a caller sees for e.g. an unknown bill_id with no participants set
        # yet, which the CLI path is supposed to reproduce byte-for-byte.
        user_id = _speaker_user_id(config)
        mode, bill, ref = _load_bill_for_edit(state, session_id, bill_id, user_id)
        if bill is None:
            return f"Error: bill '{bill_id}' not found."

        if not state.participants:
            return "Error: set participants first before marking items as shared."

        changed, message = _apply_mark_unassigned(bill, item_names, state.participants)
        return message + _persist_edit(session_id, user_id, mode, bill, ref, changed)

    @tool
    def calculate_split() -> str:
        """
        Show the group the current settlement — who owes whom, based on all approved
        bills/assignments/payers so far. A pure report: it reads state.bills and never
        changes anything, so call it as often as you like (whenever the user asks "who
        owes what" or "show me the split"), not just once at the end. There's no
        "finalize" step — the numbers simply reflect whatever's been approved by the
        time you call it, and will differ on a later call if more bills or corrections
        get approved in between.
        Tax and tip are automatically distributed proportionally.
        Any unassigned items are split equally among all participants.
        """
        if not state.participants:
            return "Error: no participants set."
        if not state.bills:
            return "Error: no bills added yet."

        missing_payers = [b.bill_id for b in state.bills if not b.paid_by]
        if missing_payers:
            return f"Error: payer not set for {missing_payers}. Please set payers first."

        balances, warnings = Settlement.compute_balances(state.participants, state.bills)
        settlements = Settlement.generate_settlements(balances)
        report = Settlement.format_report(balances, settlements)

        if warnings:
            warning_text = "Note: " + "; ".join(warnings) + "\n\n"
            report = warning_text + report

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
        self._tools = _build_tools(self.state, self.session_id)
        llm = create_llm(self._config)
        self._llm_with_tools = llm.bind_tools(self._tools)

        graph = StateGraph(GraphState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self._tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        self._graph = graph.compile(checkpointer=get_checkpointer())

    def _agent_node(self, graph_state: GraphState, config: RunnableConfig) -> dict:
        """Render the system prompt fresh from live state on every call —
        same dynamic-prompt behavior as the original hand-rolled loop, plus
        still-pending proposals when this turn has a DB-backed speaker (see
        _pending_proposals_summary) so the model doesn't lose track of a
        bill it just proposed before an admin has approved it."""
        summary = self.state.state_summary()
        if _speaker_user_id(config) is not None:
            pending_summary = _pending_proposals_summary(self.session_id)
            if pending_summary:
                summary = f"{summary}\n\n{pending_summary}"

        system_message = SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(state_summary=summary)
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
        _build_tools/_speaker_user_id), and speaker_name is surfaced to the
        model by prefixing the message content — e.g. "[Alice] we had
        pizza" — chosen over LangChain's HumanMessage(name=...) field since
        it isn't reliably surfaced to the underlying model across all three
        providers (OpenAI/Anthropic/Google) this app supports, and mixing
        that field for one provider with a prefix for others would make
        attribution behavior inconsistent depending on which provider a
        session happens to use.

        speaker_user_id is threaded through as part of this call's own
        `config` dict (configurable.speaker_user_id) rather than stored on
        `self` — a cached ChatAgent (server.py keeps one per session_id,
        reused across requests) can have concurrent chat() calls in flight
        for different speakers, and an attribute set here would be a
        shared-mutable-state race between them. `config` is local to this
        invocation, so each call's tools/agent-node only ever see their own
        speaker, however chat() calls for the same session interleave.

        Both parameters default to None, and the CLI (main.py) never passes
        them — with speaker_name=None the message content is unchanged from
        before this parameter existed, and with speaker_user_id=None every
        tool takes its original, non-proposal code path (see _build_tools),
        so CLI behavior is unaffected.
        """
        content = f"[{speaker_name}] {user_message}" if speaker_name else user_message

        config = {
            "configurable": {
                "thread_id": self.session_id,
                "speaker_user_id": speaker_user_id,
            },
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
