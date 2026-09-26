"""Tests for core/session_state.py's id generation.

Only next_bill_id() is covered here — see tests/test_database.py for the
data-layer consequences of bill_id collisions (decide_proposal matches
proposals to bills by (session_id, bill_id)).
"""

from core.session_state import ParsedBill, SessionState


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
