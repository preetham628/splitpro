from __future__ import annotations
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
    qty_allocations: Dict[str, float] = field(default_factory=dict)  # person -> fractional units

    @property
    def unit_price(self) -> float:
        return self.price / self.qty if self.qty > 0 else self.price

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "price": self.price,
            "qty": self.qty,
            "assigned_to": self.assigned_to,
            "shared": self.shared,
            "unassigned": self.unassigned,
            "qty_allocations": self.qty_allocations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LineItem":
        return cls(
            name=d["name"],
            price=d["price"],
            qty=d.get("qty", 1),
            assigned_to=d.get("assigned_to", []),
            shared=d.get("shared", False),
            unassigned=d.get("unassigned", False),
            qty_allocations=d.get("qty_allocations", {}),
        )


@dataclass
class ParsedBill:
    """One bill submitted by the user. A session can have multiple bills."""
    bill_id: str
    raw_text: str
    description: str
    items: List[LineItem] = field(default_factory=list)
    tax: float = 0.0
    tip: float = 0.0
    paid_by: Optional[str] = None

    def subtotal(self) -> float:
        """Sum of all item prices (before tax/tip)."""
        return sum(item.price for item in self.items)

    def total(self) -> float:
        return self.subtotal() + self.tax + self.tip

    def unassigned_items(self) -> List[LineItem]:
        return [i for i in self.items if not i.assigned_to and not i.unassigned]

    def to_dict(self) -> dict:
        return {
            "bill_id": self.bill_id,
            "raw_text": self.raw_text,
            "description": self.description,
            "items": [i.to_dict() for i in self.items],
            "tax": self.tax,
            "tip": self.tip,
            "paid_by": self.paid_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedBill":
        bill = cls(
            bill_id=d["bill_id"],
            raw_text=d.get("raw_text", ""),
            description=d["description"],
            tax=d.get("tax", 0.0),
            tip=d.get("tip", 0.0),
            paid_by=d.get("paid_by"),
        )
        bill.items = [LineItem.from_dict(i) for i in d.get("items", [])]
        return bill


@dataclass
class SessionState:
    """Single source of truth for one SplitPro chat session."""
    participants: List[str] = field(default_factory=list)
    bills: List[ParsedBill] = field(default_factory=list)
    finalized: bool = False

    def next_bill_id(self) -> str:
        return f"bill_{len(self.bills) + 1}"

    def get_bill(self, bill_id: str) -> Optional[ParsedBill]:
        for b in self.bills:
            if b.bill_id == bill_id:
                return b
        return None

    def to_dict(self) -> dict:
        return {
            "participants": self.participants,
            "bills": [b.to_dict() for b in self.bills],
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        s = cls(
            participants=d.get("participants", []),
            finalized=d.get("finalized", False),
        )
        s.bills = [ParsedBill.from_dict(b) for b in d.get("bills", [])]
        return s

    def all_bills_ready(self) -> bool:
        """True when every bill has a payer and no unassigned items."""
        if not self.bills:
            return False
        return all(
            b.paid_by is not None and len(b.unassigned_items()) == 0
            for b in self.bills
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
                lines.append(
                    f"  [{b.bill_id}] {b.description} | "
                    f"paid_by={b.paid_by or 'not set'} | "
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
                    if item.qty_allocations:
                        for person, pqty in item.qty_allocations.items():
                            lines.append(f"      {person}: {pqty} unit(s) = ${item.unit_price * pqty:.2f}")

                pending = b.unassigned_items()
                if pending:
                    lines.append(f"    NEEDS ASSIGNMENT: {', '.join(i.name for i in pending)}")

        return "\n".join(lines)
