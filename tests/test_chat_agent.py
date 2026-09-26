"""Tests for agents/chat_agent.py's proposal-based tools.

These exercise the tool closures built by _build_tools() directly (each
LangChain @tool wraps a plain function reachable via .invoke({...})), rather
than going through ChatAgent.chat() and a real LLM — no network access or
API key needed, and it isolates exactly the new DB-aware behavior this
module adds: add_bill/assign_items/set_payer/mark_items_unassigned staging
proposals instead of mutating SessionState.bills directly, and routing edits
to an already-approved bill through a superseding correction proposal.

Each test gets its own fresh scratch SQLite file (tmp_path), same pattern as
tests/test_database.py.
"""

import pytest

from agents.chat_agent import _build_tools
from core import database as db
from core.session_state import SessionState


@pytest.fixture
def fresh_db(tmp_path):
    path = str(tmp_path / "test.db")
    db.create_db(path)
    db.init_db(path)
    return path


def make_user(email, google_id):
    return db.upsert_user(google_id=google_id, email=email, name=email.split("@")[0], avatar_url="")


def make_session_with_members(session_id, admin_id, *member_ids):
    db.create_session(session_id, admin_id, "Test Session")
    db.add_session_member(session_id, admin_id, role="admin")
    for uid in member_ids:
        db.add_session_member(session_id, uid, role="member")


def build_tools_for(state, session_id, user_id_holder):
    """user_id_holder is a single-item list used as mutable storage so the
    returned get_current_user_id() reflects whatever the test sets it to
    afterward, the same way ChatAgent stores self._current_user_id."""
    return _build_tools(state, session_id, lambda: user_id_holder[0])


def one_item_bill(price=30.0):
    return {
        "raw_text": "Pizza $30",
        "description": "Pizza night",
        "items": [{"name": "Pizza", "price": price, "qty": 1}],
        "tax": 0.0,
        "tip": 0.0,
    }


# ---------- add_bill ----------

def test_add_bill_stages_a_pending_proposal_not_state_bill(fresh_db):
    admin = make_user("admin@example.com", "g-admin")
    alice = make_user("alice@example.com", "g-alice")
    session_id = "sess-add-1"
    make_session_with_members(session_id, admin["id"], alice["id"])

    state = SessionState()
    state.participants = ["Alice", "Bob"]
    user_id = [alice["id"]]
    add_bill, *_rest = build_tools_for(state, session_id, user_id)

    msg = add_bill.invoke(one_item_bill())

    assert "awaiting admin approval" in msg
    assert "Added " not in msg  # must not claim it's final
    assert state.bills == []

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["proposed_by"] == alice["id"]
    assert pending[0]["payload"]["description"] == "Pizza night"
    assert pending[0]["payload"]["paid_by"] is None


def test_add_bill_legacy_path_when_no_current_user(fresh_db):
    """No speaker/user context (the CLI's shape) — behaves exactly like the
    pre-proposal implementation: appends straight to state.bills, no DB call."""
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    user_id = [None]
    add_bill, *_rest = build_tools_for(state, "unused-session-id", user_id)

    msg = add_bill.invoke(one_item_bill())

    assert msg.startswith("Added bill_")
    assert len(state.bills) == 1
    assert state.bills[0].description == "Pizza night"


# ---------- assign_items / set_payer / mark_items_unassigned on a pending proposal ----------

def test_assign_items_updates_pending_proposal_in_place(fresh_db):
    admin = make_user("admin@example.com", "g-admin2")
    alice = make_user("alice@example.com", "g-alice2")
    bob = make_user("bob@example.com", "g-bob2")
    session_id = "sess-assign-1"
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    state = SessionState()
    state.participants = ["Alice", "Bob"]
    user_id = [alice["id"]]
    add_bill, _set_p, assign_items, set_payer, _mark, _calc = build_tools_for(state, session_id, user_id)

    add_bill.invoke(one_item_bill())
    bill_id = db.list_pending_proposals(session_id)[0]["payload"]["bill_id"]

    user_id[0] = bob["id"]  # a different participant edits the same pending proposal
    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice", "Bob"], "shared": True}],
    })

    assert "still awaiting admin approval" in msg
    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1  # updated in place, not a second proposal
    assert pending[0]["payload"]["items"][0]["assigned_to"] == ["Alice", "Bob"]

    msg = set_payer.invoke({"bill_id": bill_id, "paid_by": "alice"})
    assert "still awaiting admin approval" in msg
    assert db.list_pending_proposals(session_id)[0]["payload"]["paid_by"] == "Alice"


def test_assign_items_unknown_bill_id_error_message(fresh_db):
    admin = make_user("admin@example.com", "g-admin3")
    session_id = "sess-assign-2"
    make_session_with_members(session_id, admin["id"])

    state = SessionState()
    user_id = [admin["id"]]
    _add, _set_p, assign_items, _payer, _mark, _calc = build_tools_for(state, session_id, user_id)

    msg = assign_items.invoke({
        "bill_id": "bill_does_not_exist",
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice"]}],
    })
    assert "not found" in msg
    assert db.list_pending_proposals(session_id) == []


# ---------- Corrections to an already-approved bill ----------

def test_correction_to_approved_bill_creates_superseding_proposal(fresh_db):
    admin = make_user("admin@example.com", "g-admin4")
    alice = make_user("alice@example.com", "g-alice4")
    bob = make_user("bob@example.com", "g-bob4")
    session_id = "sess-correction-1"
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    state = SessionState()
    state.participants = ["Alice", "Bob"]
    db.save_session_state(session_id, state, [])  # persist participants, like server.py every turn

    user_id = [alice["id"]]
    add_bill, *_rest = build_tools_for(state, session_id, user_id)
    add_bill.invoke(one_item_bill())

    proposal = db.list_pending_proposals(session_id)[0]
    bill_id = proposal["payload"]["bill_id"]
    db.decide_proposal(proposal["id"], admin["id"], "approved")
    approved_row_id = db.get_bill_row_id(session_id, bill_id)

    # Reload state the way server.py's _get_agent()/set_state() would after approval.
    approved_state = SessionState.from_dict(db.load_session_state(session_id))
    assert approved_state.bills[0].paid_by is None

    user_id2 = [bob["id"]]
    _add2, _set_p2, _assign2, set_payer2, _mark2, _calc2 = build_tools_for(
        approved_state, session_id, user_id2
    )
    msg = set_payer2.invoke({"bill_id": bill_id, "paid_by": "Alice"})

    assert "already approved" in msg
    assert "correction" in msg

    # The original approved bill must stay untouched until the correction itself is approved.
    still_untouched = SessionState.from_dict(db.load_session_state(session_id))
    assert still_untouched.bills[0].paid_by is None
    assert len(still_untouched.bills) == 1

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["supersedes_bill_id"] == approved_row_id
    assert pending[0]["payload"]["paid_by"] == "Alice"
    assert pending[0]["proposed_by"] == bob["id"]

    # Approving the correction updates the SAME bill row, not a duplicate.
    db.decide_proposal(pending[0]["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    assert len(final_state.bills) == 1
    assert final_state.bills[0].bill_id == bill_id
    assert final_state.bills[0].paid_by == "Alice"


def test_mark_items_unassigned_requires_participants_before_touching_db(fresh_db):
    admin = make_user("admin@example.com", "g-admin5")
    session_id = "sess-mark-1"
    make_session_with_members(session_id, admin["id"])

    state = SessionState()  # no participants set
    user_id = [admin["id"]]
    add_bill, _set_p, _assign, _payer, mark_items_unassigned, _calc = build_tools_for(
        state, session_id, user_id
    )
    add_bill.invoke(one_item_bill())
    bill_id = db.list_pending_proposals(session_id)[0]["payload"]["bill_id"]

    msg = mark_items_unassigned.invoke({"bill_id": bill_id, "item_names": ["Pizza"]})
    assert "set participants first" in msg
    # No pending-proposal mutation should have happened.
    assert db.list_pending_proposals(session_id)[0]["payload"]["items"][0]["unassigned"] is False
