"""Tests for agents/chat_agent.py's proposal-based tools.

These exercise the tool closures built by _build_tools() directly (each
LangChain @tool wraps a plain function reachable via .invoke({...})), rather
than going through ChatAgent.chat() and a real LLM — no network access or
API key needed, and it isolates exactly the new DB-aware behavior this
module adds: add_bill/assign_items/set_payer/mark_items_unassigned staging
proposals instead of mutating SessionState.bills directly, and routing edits
to an already-approved bill through a superseding correction proposal.

Per-turn speaker identity is passed as a LangChain RunnableConfig
(`config={"configurable": {"speaker_user_id": ...}}`) rather than any state
on the ChatAgent instance — that's what lets two concurrent chat() calls for
the same session use different speakers without racing each other (see
test_concurrent_speakers_do_not_leak_across_chat_calls below, which forces
real thread interleaving through ChatAgent.chat() itself — the same pattern
tests/test_database.py uses for its own concurrency tests).

Each test gets its own fresh scratch SQLite file (tmp_path), same pattern as
tests/test_database.py.
"""

import threading
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.chat_agent import ChatAgent, _build_tools, _pending_proposals_summary
from config import ChatAgentConfig
from core import checkpointer, database as db
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


def as_speaker(user_id):
    """The RunnableConfig shape ChatAgent.chat() threads through the graph —
    tool calls made with a different one below never share state with each
    other, unlike the old self._current_user_id approach they replace."""
    return {"configurable": {"speaker_user_id": user_id}}


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
    add_bill, *_rest = _build_tools(state, session_id)

    msg = add_bill.invoke(one_item_bill(), config=as_speaker(alice["id"]))

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
    add_bill, *_rest = _build_tools(state, "unused-session-id")

    msg = add_bill.invoke(one_item_bill())  # no config at all, same as a plain tool call

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
    add_bill, _set_p, assign_items, set_payer, _mark, _calc = _build_tools(state, session_id)

    add_bill.invoke(one_item_bill(), config=as_speaker(alice["id"]))
    bill_id = db.list_pending_proposals(session_id)[0]["payload"]["bill_id"]

    # A different participant edits the same still-pending proposal.
    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice", "Bob"], "shared": True}],
    }, config=as_speaker(bob["id"]))

    assert "still awaiting admin approval" in msg
    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1  # updated in place, not a second proposal
    assert pending[0]["payload"]["items"][0]["assigned_to"] == ["Alice", "Bob"]

    msg = set_payer.invoke(
        {"bill_id": bill_id, "paid_by": "alice"}, config=as_speaker(bob["id"])
    )
    assert "still awaiting admin approval" in msg
    assert db.list_pending_proposals(session_id)[0]["payload"]["paid_by"] == "Alice"


def test_assign_items_unknown_bill_id_error_lists_pending_proposals(fresh_db):
    """The 'not found' error's known-bills list must include still-pending
    proposals, not just state.bills — otherwise a bill the model just
    proposed looks nonexistent on the very next tool call that references it."""
    admin = make_user("admin@example.com", "g-admin3")
    session_id = "sess-assign-2"
    make_session_with_members(session_id, admin["id"])

    state = SessionState()
    add_bill, _set_p, assign_items, _payer, _mark, _calc = _build_tools(state, session_id)
    add_bill.invoke(one_item_bill(), config=as_speaker(admin["id"]))
    real_bill_id = db.list_pending_proposals(session_id)[0]["payload"]["bill_id"]

    msg = assign_items.invoke({
        "bill_id": "bill_does_not_exist",
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice"]}],
    }, config=as_speaker(admin["id"]))

    assert "not found" in msg
    assert real_bill_id in msg
    assert len(db.list_pending_proposals(session_id)) == 1  # untouched, no phantom write


def test_assign_items_unknown_bill_id_legacy_path(fresh_db):
    state = SessionState()
    _add, _set_p, assign_items, _payer, _mark, _calc = _build_tools(state, "unused-session-id")

    msg = assign_items.invoke({
        "bill_id": "bill_does_not_exist",
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice"]}],
    })
    assert msg == "Error: bill 'bill_does_not_exist' not found. Known bills: []"


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

    add_bill, *_rest = _build_tools(state, session_id)
    add_bill.invoke(one_item_bill(), config=as_speaker(alice["id"]))

    proposal = db.list_pending_proposals(session_id)[0]
    bill_id = proposal["payload"]["bill_id"]
    db.decide_proposal(proposal["id"], admin["id"], "approved")
    approved_row_id = db.get_bill_row_id(session_id, bill_id)

    # Reload state the way server.py's _get_agent()/set_state() would after approval.
    approved_state = SessionState.from_dict(db.load_session_state(session_id))
    assert approved_state.bills[0].paid_by is None

    _add2, _set_p2, _assign2, set_payer2, _mark2, _calc2 = _build_tools(approved_state, session_id)
    msg = set_payer2.invoke(
        {"bill_id": bill_id, "paid_by": "Alice"}, config=as_speaker(bob["id"])
    )

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


def test_known_bill_ids_dedupes_a_correction_sharing_its_approved_bills_id(fresh_db):
    """A pending correction proposal reuses its target bill's app-level
    bill_id (see decide_proposal() in core/database.py) — so once one
    exists, that id is legitimately in both state.bills and the pending
    list at once. The 'Known bills' error list must show it once, not twice."""
    admin = make_user("admin@example.com", "g-admin9")
    alice = make_user("alice@example.com", "g-alice9")
    bob = make_user("bob@example.com", "g-bob9")
    session_id = "sess-dedupe-1"
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    state = SessionState()
    state.participants = ["Alice", "Bob"]
    db.save_session_state(session_id, state, [])

    add_bill, *_rest = _build_tools(state, session_id)
    add_bill.invoke(one_item_bill(), config=as_speaker(alice["id"]))
    proposal = db.list_pending_proposals(session_id)[0]
    bill_id = proposal["payload"]["bill_id"]
    db.decide_proposal(proposal["id"], admin["id"], "approved")

    approved_state = SessionState.from_dict(db.load_session_state(session_id))
    _add2, _set_p2, assign_items2, set_payer2, _mark2, _calc2 = _build_tools(approved_state, session_id)

    # Propose a correction (same bill_id) so it now exists in BOTH
    # approved_state.bills and the pending-proposals list simultaneously.
    set_payer2.invoke({"bill_id": bill_id, "paid_by": "Alice"}, config=as_speaker(bob["id"]))
    assert len(db.list_pending_proposals(session_id)) == 1

    msg = assign_items2.invoke({
        "bill_id": "bill_does_not_exist",
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice"]}],
    }, config=as_speaker(bob["id"]))

    assert msg.count(bill_id) == 1, f"expected '{bill_id}' exactly once in: {msg}"


# ---------- mark_items_unassigned check ordering ----------

def test_mark_items_unassigned_reports_bill_not_found_before_participants_check(fresh_db):
    """Pre-proposal-rework code checked bill-not-found before
    participants-not-set; that order must survive the DB-aware rework
    (and be exactly what the legacy/CLI path still does)."""
    admin = make_user("admin@example.com", "g-admin5")
    session_id = "sess-mark-1"
    make_session_with_members(session_id, admin["id"])

    state = SessionState()  # no participants set
    _add, _set_p, _assign, _payer, mark_items_unassigned, _calc = _build_tools(state, session_id)

    msg = mark_items_unassigned.invoke(
        {"bill_id": "bill_does_not_exist", "item_names": ["Pizza"]},
        config=as_speaker(admin["id"]),
    )
    assert msg == "Error: bill 'bill_does_not_exist' not found."


def test_mark_items_unassigned_requires_participants_once_bill_exists(fresh_db):
    admin = make_user("admin@example.com", "g-admin6")
    session_id = "sess-mark-2"
    make_session_with_members(session_id, admin["id"])

    state = SessionState()  # no participants set
    add_bill, _set_p, _assign, _payer, mark_items_unassigned, _calc = _build_tools(state, session_id)
    add_bill.invoke(one_item_bill(), config=as_speaker(admin["id"]))
    bill_id = db.list_pending_proposals(session_id)[0]["payload"]["bill_id"]

    msg = mark_items_unassigned.invoke(
        {"bill_id": bill_id, "item_names": ["Pizza"]}, config=as_speaker(admin["id"])
    )
    assert "set participants first" in msg
    # No pending-proposal mutation should have happened.
    assert db.list_pending_proposals(session_id)[0]["payload"]["items"][0]["unassigned"] is False


# ---------- Concurrency safety ----------

class _StallingToolCallLLM:
    """Stand-in chat model — no live API calls, no API key needed — that
    reproduces the exact race window the old self._current_user_id design
    lived in: Alice's chat() call reaches the model, then genuinely BLOCKS
    (a real thread, parked on a real Event) for as long as a slow network
    round-trip would take, while Bob's entire chat() turn — including his
    own tool execution — runs to completion on the very same cached
    ChatAgent instance. Only once Bob is fully done does Alice's call
    proceed to actually run her tool. Under the old design this is exactly
    when self._current_user_id would have already been overwritten to
    bob_id by Bob's chat() call; under the config-threaded fix, Alice's own
    speaker_user_id was captured into her own call's config before the
    model was ever invoked, so it can't have been touched by Bob's turn.

    Distinguishes turns by the "[Alice]"/"[Bob]" prefix ChatAgent.chat()
    puts on message content, and by whether the latest message is the
    original HumanMessage (need a tool call) or the ToolMessage that comes
    back after the tool ran (turn is done) — mirroring how a real
    tool-calling model would behave across the two agent-node visits per turn.
    """

    def __init__(self):
        self.alice_llm_called = threading.Event()
        self.bob_turn_done = threading.Event()

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            if last.content.startswith("[Alice]"):
                self.alice_llm_called.set()
                assert self.bob_turn_done.wait(timeout=5), "Bob's turn never completed"
                return AIMessage(content="", tool_calls=[{
                    "name": "add_bill",
                    "id": "call-alice-1",
                    "args": {
                        "raw_text": "r", "description": "Alice's bill",
                        "items": [{"name": "Pizza", "price": 10.0}], "tax": 0.0, "tip": 0.0,
                    },
                }])
            if last.content.startswith("[Bob]"):
                assert self.alice_llm_called.wait(timeout=5), "Alice's LLM call never started"
                return AIMessage(content="", tool_calls=[{
                    "name": "add_bill",
                    "id": "call-bob-1",
                    "args": {
                        "raw_text": "r", "description": "Bob's bill",
                        "items": [{"name": "Soda", "price": 5.0}], "tax": 0.0, "tip": 0.0,
                    },
                }])
        return AIMessage(content="ok")  # second agent-node visit, after the tool ran: end the turn


def test_concurrent_speakers_do_not_leak_across_chat_calls(fresh_db, monkeypatch):
    """Regression test for the self._current_user_id race: two real threads
    call ChatAgent.chat() concurrently on the SAME cached agent/session —
    the actual object and code path the original bug lived in — with
    Alice's model call forced to stall until Bob's whole turn has finished.
    A reintroduction of shared mutable per-turn identity on `self` would
    misattribute Alice's proposal to Bob here; the config-threaded fix does not.
    """
    admin = make_user("admin@example.com", "g-admin7")
    alice = make_user("alice@example.com", "g-alice7")
    bob = make_user("bob@example.com", "g-bob7")
    session_id = str(uuid.uuid4())
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    fake_llm = _StallingToolCallLLM()
    monkeypatch.setattr("agents.chat_agent.create_llm", lambda config: fake_llm)
    checkpointer.init_checkpointer(":memory:")

    agent = ChatAgent(session_id=session_id, config=ChatAgentConfig())

    results = {}

    def alice_call():
        results["alice"] = agent.chat(
            "we had pizza", speaker_name="Alice", speaker_user_id=alice["id"]
        )

    def bob_call():
        results["bob"] = agent.chat(
            "we had soda", speaker_name="Bob", speaker_user_id=bob["id"]
        )
        fake_llm.bob_turn_done.set()

    t_alice = threading.Thread(target=alice_call)
    t_bob = threading.Thread(target=bob_call)
    t_alice.start()
    t_bob.start()
    t_alice.join(timeout=10)
    t_bob.join(timeout=10)

    assert not t_alice.is_alive(), "Alice's chat() call never returned"
    assert not t_bob.is_alive(), "Bob's chat() call never returned"

    # The fake LLM's canned final reply ("ok") isn't what's under test here —
    # what matters is which user each proposal actually landed under.
    pending = db.list_pending_proposals(session_id)
    by_description = {p["payload"]["description"]: p for p in pending}
    assert by_description["Alice's bill"]["proposed_by"] == alice["id"]
    assert by_description["Bob's bill"]["proposed_by"] == bob["id"]


# ---------- Pending-proposal visibility in the system prompt ----------

def test_pending_proposals_summary_lists_proposed_and_correction_bills(fresh_db):
    admin = make_user("admin@example.com", "g-admin8")
    alice = make_user("alice@example.com", "g-alice8")
    session_id = "sess-summary-1"
    make_session_with_members(session_id, admin["id"], alice["id"])

    assert _pending_proposals_summary(session_id) == ""

    state = SessionState()
    add_bill, *_rest = _build_tools(state, session_id)
    add_bill.invoke(one_item_bill(), config=as_speaker(alice["id"]))

    summary = _pending_proposals_summary(session_id)
    assert "PENDING PROPOSALS" in summary
    assert "Pizza night" in summary
    assert "[correction" not in summary  # a fresh proposal, not a correction
