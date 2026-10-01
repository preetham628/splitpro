"""Tests for core/session_state.py's id generation.

Only next_bill_id() is covered here — see tests/test_database.py for the
data-layer consequences of bill_id collisions (decide_proposal matches
proposals to bills by (session_id, bill_id)).
"""

from core.session_state import LineItem, ParsedBill, SessionState, validate_contribution_map


def test_next_bill_id_is_never_reused():
    """Must not repeat even after bills are removed from the session — a
    reused bill_id would let a stale expense_proposals row silently target
    the wrong bill on approval. Not tied to len(state.bills), so this holds
    regardless of additions/removals.
    """
    state = SessionState()
    ids = {state.next_bill_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(i.startswith("bill_") for i in ids)


def test_next_bill_id_survives_shrinking_bill_list():
    state = SessionState()
    first = state.next_bill_id()
    state.bills.append(ParsedBill(bill_id=first, raw_text="", description=""))
    state.bills.clear()  # simulate a future "remove bill" capability
    second = state.next_bill_id()
    assert second != first


# ---------- Multi-payer / flexible-split model ----------

def test_parsed_bill_paid_by_defaults_to_empty_dict_not_none():
    """paid_by's default changed from None to an empty dict (empty dict is
    now the "not set yet" sentinel) — all_bills_ready() relies on this via
    bool(b.paid_by)."""
    bill = ParsedBill(bill_id="bill_1", raw_text="", description="")
    assert bill.paid_by == {}

    state = SessionState(participants=["Alice"], bills=[bill])
    assert state.all_bills_ready() is False  # no payer recorded yet

    bill.paid_by = {"Alice": 10.0}
    assert state.all_bills_ready() is True


def test_line_item_cost_allocations_default_and_unit_price():
    item = LineItem(name="Burger", price=12.0, qty=2)
    assert item.cost_allocations == {}
    assert item.unit_price == 6.0  # display-only; allocation math uses cost_allocations


def test_from_dict_round_trips_paid_by_and_cost_allocations():
    state = SessionState.from_dict({
        "participants": ["Alice", "Sumit"],
        "bills": [{
            "bill_id": "bill_1",
            "description": "Dinner",
            "raw_text": "raw",
            "tax": 1.0,
            "tip": 2.0,
            "paid_by": {"Alice": 1800.0, "Sumit": 1200.0},
            "items": [{
                "name": "Burger", "price": 12.0, "qty": 2,
                "assigned_to": ["Alice", "Sumit"],
                "cost_allocations": {"Alice": 8.0, "Sumit": 4.0},
            }],
        }],
        "finalized": False,
    })
    bill = state.bills[0]
    assert bill.paid_by == {"Alice": 1800.0, "Sumit": 1200.0}
    assert bill.items[0].cost_allocations == {"Alice": 8.0, "Sumit": 4.0}


def test_from_dict_defaults_missing_paid_by_to_empty_dict():
    state = SessionState.from_dict({
        "participants": [],
        "bills": [{"bill_id": "bill_1", "description": "", "raw_text": ""}],
    })
    assert state.bills[0].paid_by == {}


def test_state_summary_renders_multi_payer_map():
    state = SessionState(
        participants=["Alice", "Sumit"],
        bills=[ParsedBill(
            bill_id="bill_1", raw_text="", description="Dinner",
            items=[LineItem(
                name="Burger", price=12.0, qty=2, assigned_to=["Alice", "Sumit"],
                cost_allocations={"Alice": 8.0, "Sumit": 4.0},
            )],
            tax=1.0, tip=2.0,
            paid_by={"Alice": 1800.0, "Sumit": 1200.0},
        )],
    )
    summary = state.state_summary()
    assert "paid_by={Alice: $1800.00, Sumit: $1200.00}" in summary
    assert "Alice: $8.00" in summary
    assert "Sumit: $4.00" in summary


def test_state_summary_shows_not_set_for_empty_paid_by():
    state = SessionState(
        participants=["Alice"],
        bills=[ParsedBill(bill_id="bill_1", raw_text="", description="Drinks")],
    )
    assert "paid_by=not set" in state.state_summary()


def test_validate_contribution_map():
    assert validate_contribution_map({"Alice": 1800.0, "Sumit": 1200.0}, 3000.0) is True
    assert validate_contribution_map({"Alice": 10.0}, 10.0005) is True
    assert validate_contribution_map({"Alice": 10.0}, 11.0) is False
    assert validate_contribution_map({}, 0.0) is True
