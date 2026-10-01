from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LineItem:
    """A single line item parsed from a bill."""
    name: str
    price: float
    qty: int = 1                                             # number of units ordered
    assigned_to: List[str] = field(default_factory=list)
    shared: bool = False      # True = split equally among assigned_to
    unassigned: bool = False  # True = intentionally split among all participants
    # person -> dollar amount of this item's price they're responsible for.
    # Percentage/equal-split/"N of M units" are input conveniences that
    # resolve to these dollar amounts at the tool-call layer — this is
    # always the canonical, already-resolved shape.
    cost_allocations: Dict[str, float] = field(default_factory=dict)

    @property
    def unit_price(self) -> float:
        """Price per unit — for display only; allocation math uses
        cost_allocations (dollar amounts) directly, not unit counts."""
        return self.price / self.qty if self.qty > 0 else self.price


@dataclass
class ParsedBill:
    """One bill submitted by the user. A session can have multiple bills."""
    bill_id: str
    raw_text: str
    description: str
    items: List[LineItem] = field(default_factory=list)
    tax: float = 0.0
    tip: float = 0.0
    # person -> dollar amount they paid toward this bill. Supports multiple
    # payers on one bill (e.g. {"Alice": 1800.0, "Sumit": 1200.0}). Empty
    # dict means no payer recorded yet.
    paid_by: Dict[str, float] = field(default_factory=dict)

    def subtotal(self) -> float:
        """Sum of all item prices (before tax/tip)."""
        return sum(item.price for item in self.items)

    def total(self) -> float:
        return self.subtotal() + self.tax + self.tip

    def unassigned_items(self) -> List[LineItem]:
        return [i for i in self.items if not i.assigned_to and not i.unassigned]


@dataclass
class SessionState:
    """Single source of truth for one SplitPro chat session."""
    participants: List[str] = field(default_factory=list)
    bills: List[ParsedBill] = field(default_factory=list)
    finalized: bool = False

    def next_bill_id(self) -> str:
        """A bill_id that's never reused within this session's lifetime.

        Deliberately not len(self.bills) + 1 — that scheme reissues the same
        id after a bill is removed and a new one added (e.g. once an
        edit/delete-bill capability exists), which would let a stale,
        still-pending expense_proposals row silently target the wrong bill
        on approval (proposals are matched by (session_id, bill_id) — see
        decide_proposal() in core/database.py).
        """
        return f"bill_{uuid.uuid4().hex[:8]}"

    def get_bill(self, bill_id: str) -> Optional[ParsedBill]:
        for b in self.bills:
            if b.bill_id == bill_id:
                return b
        return None

    def all_bills_ready(self) -> bool:
        """True when every bill has a payer and no unassigned items."""
        if not self.bills:
            return False
        return all(
            bool(b.paid_by) and len(b.unassigned_items()) == 0
            for b in self.bills
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        bills = []
        for b in d.get("bills", []):
            items = [
                LineItem(
                    name=i["name"],
                    price=i["price"],
                    qty=i.get("qty", 1),
                    assigned_to=i.get("assigned_to", []),
                    shared=i.get("shared", False),
                    unassigned=i.get("unassigned", False),
                    cost_allocations=i.get("cost_allocations", {}),
                )
                for i in b.get("items", [])
            ]
            bills.append(ParsedBill(
                bill_id=b["bill_id"],
                raw_text=b.get("raw_text", ""),
                description=b.get("description", ""),
                items=items,
                tax=b.get("tax", 0.0),
                tip=b.get("tip", 0.0),
                paid_by=b.get("paid_by") or {},
            ))
        return cls(
            participants=d.get("participants", []),
            bills=bills,
            finalized=d.get("finalized", False),
        )

    def state_summary(self) -> str:
        """Compact text representation injected into every system prompt."""
        lines = []
        participants_str = ", ".join(self.participants) if self.participants else "not set yet"
        lines.append(f"PARTICIPANTS: {participants_str}")
        lines.append(f"BILLS ({len(self.bills)} total):")

        if not self.bills:
            lines.append("  (none yet)")
        else:
            for b in self.bills:
                if b.paid_by:
                    paid_by_str = "{" + ", ".join(
                        f"{person}: ${amount:.2f}" for person, amount in b.paid_by.items()
                    ) + "}"
                else:
                    paid_by_str = "not set"
                lines.append(
                    f"  [{b.bill_id}] {b.description} | "
                    f"paid_by={paid_by_str} | "
                    f"subtotal=${b.subtotal():.2f} | tax=${b.tax:.2f} | tip=${b.tip:.2f}"
                )
                for item in b.items:
                    qty_str = f" x{item.qty}" if item.qty > 1 else ""
                    if item.assigned_to:
                        assignment = ", ".join(item.assigned_to)
                        tag = " (shared)" if item.shared else ""
                    else:
                        assignment = "UNASSIGNED"
                        tag = ""
                    lines.append(f"    - {item.name}{qty_str}: ${item.price:.2f} -> {assignment}{tag}")
                    if item.cost_allocations:
                        for person, amount in item.cost_allocations.items():
                            lines.append(f"      {person}: ${amount:.2f}")

                pending = b.unassigned_items()
                if pending:
                    lines.append(f"    NEEDS ASSIGNMENT: {', '.join(i.name for i in pending)}")

        return "\n".join(lines)


MAX_CONTRIBUTION_TOLERANCE = 0.50  # dollars; see validate_contribution_map()


def validate_contribution_map(amounts: Dict[str, float], total: float, epsilon: float = 0.01) -> bool:
    """True if a person -> dollar-amount contribution map (paid_by or a
    LineItem's cost_allocations) sums to `total` within tolerance.

    `epsilon` is a per-person cent-rounding allowance, not a flat tolerance:
    the accepted discrepancy is `epsilon * max(len(amounts), 1)`, capped at
    `MAX_CONTRIBUTION_TOLERANCE`. A flat tolerance tight enough to catch real
    mistakes on a single contribution (e.g. a typo) is too tight for a
    legitimate N-way split, since dividing a total evenly to the cent is
    frequently impossible — a basic 3-way equal split of $100.00 is
    {33.33, 33.33, 33.33}, which sums to $99.99, a $0.01 discrepancy that
    only grows with more people. Scaling by the number of contributors keeps
    a 1-2 person map's tolerance tight while still accepting the unavoidable
    rounding remainder on a larger split.

    The cap exists because the scaling is otherwise unbounded: at ~100
    contributors, an uncapped tolerance (0.01 * 100 = $1.00) would let a
    genuine $1.00 error in one person's contribution slip through as
    "valid". This app's realistic group sizes (2-20 people) never approach
    the cap — true per-cent rounding error for an even split is at most
    about 0.005 per person, so the cap (0.50) comfortably covers up to
    ~100 people's worth of legitimate rounding while still catching a
    dollar-scale mistake at that same size.

    A reusable building block for the tool-call layer's input validation
    (e.g. rejecting a set_payer/assign_item call whose percentages or
    absolute amounts don't actually add up to the bill/item total) — not
    called from anywhere in this module itself yet.
    """
    tolerance = min(epsilon * max(len(amounts), 1), MAX_CONTRIBUTION_TOLERANCE)
    return abs(sum(amounts.values()) - total) <= tolerance
