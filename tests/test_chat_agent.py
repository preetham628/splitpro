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

import sqlite3
import threading
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.chat_agent import ChatAgent, _build_tools, _pending_proposals_summary
from config import ChatAgentConfig
from core import checkpointer, database as db
from core.session_state import LineItem, ParsedBill, SessionState
from core.settlement import Settlement


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
    assert pending[0]["payload"]["paid_by"] == {}


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
    add_bill, _set_p, assign_items, set_payer, _mark, _rename, _add_item, _rm_item, _rm_bill, _calc = _build_tools(state, session_id)

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
        {"bill_id": bill_id, "payers": [{"name": "alice"}]}, config=as_speaker(bob["id"])
    )
    assert "still awaiting admin approval" in msg
    assert db.list_pending_proposals(session_id)[0]["payload"]["paid_by"] == {"Alice": 30.0}


def test_assign_items_unknown_bill_id_error_lists_pending_proposals(fresh_db):
    """The 'not found' error's known-bills list must include still-pending
    proposals, not just state.bills — otherwise a bill the model just
    proposed looks nonexistent on the very next tool call that references it."""
    admin = make_user("admin@example.com", "g-admin3")
    session_id = "sess-assign-2"
    make_session_with_members(session_id, admin["id"])

    state = SessionState()
    add_bill, _set_p, assign_items, _payer, _mark, _rename, _add_item, _rm_item, _rm_bill, _calc = _build_tools(state, session_id)
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
    _add, _set_p, assign_items, _payer, _mark, _rename, _add_item, _rm_item, _rm_bill, _calc = _build_tools(state, "unused-session-id")

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
    assert approved_state.bills[0].paid_by == {}

    _add2, _set_p2, _assign2, set_payer2, _mark2, _rename2, _add_item2, _rm_item2, _rm_bill2, _calc2 = _build_tools(approved_state, session_id)
    msg = set_payer2.invoke(
        {"bill_id": bill_id, "payers": [{"name": "Alice"}]}, config=as_speaker(bob["id"])
    )

    assert "already approved" in msg
    assert "correction" in msg

    # The original approved bill must stay untouched until the correction itself is approved.
    still_untouched = SessionState.from_dict(db.load_session_state(session_id))
    assert still_untouched.bills[0].paid_by == {}
    assert len(still_untouched.bills) == 1

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["supersedes_bill_id"] == approved_row_id
    assert pending[0]["payload"]["paid_by"] == {"Alice": 30.0}
    assert pending[0]["proposed_by"] == bob["id"]

    # Approving the correction updates the SAME bill row, not a duplicate.
    db.decide_proposal(pending[0]["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    assert len(final_state.bills) == 1
    assert final_state.bills[0].bill_id == bill_id
    assert final_state.bills[0].paid_by == {"Alice": 30.0}


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
    _add2, _set_p2, assign_items2, set_payer2, _mark2, _rename2, _add_item2, _rm_item2, _rm_bill2, _calc2 = _build_tools(approved_state, session_id)

    # Propose a correction (same bill_id) so it now exists in BOTH
    # approved_state.bills and the pending-proposals list simultaneously.
    set_payer2.invoke({"bill_id": bill_id, "payers": [{"name": "Alice"}]}, config=as_speaker(bob["id"]))
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
    _add, _set_p, _assign, _payer, mark_items_unassigned, _rename, _add_item, _rm_item, _rm_bill, _calc = _build_tools(state, session_id)

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
    add_bill, _set_p, _assign, _payer, mark_items_unassigned, _rename, _add_item, _rm_item, _rm_bill, _calc = _build_tools(state, session_id)
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


# ---------- set_payer: multi-mode payer resolution (task7) ----------
#
# All of the following use the legacy (no speaker/DB) path, which mutates
# state.bills directly — the resolution math itself (in _apply_set_payer /
# _resolve_contribution_entries) doesn't depend on the proposal-staging
# plumbing covered above.

def _legacy_tools(participants, bill_kwargs=None):
    """A SessionState with one participant list and one freshly-added bill,
    plus its tools built on the legacy (no DB speaker) path."""
    state = SessionState()
    state.participants = participants
    (
        add_bill, set_participants, assign_items, set_payer, mark_items_unassigned,
        _rename, _add_item, _rm_item, _rm_bill, calculate_split,
    ) = _build_tools(
        state, "unused-session-id"
    )
    add_bill.invoke(bill_kwargs or one_item_bill())
    return state, assign_items, set_payer, calculate_split


def test_set_payer_single_implicit_payer_gets_full_total():
    state, _assign, set_payer, _calc = _legacy_tools(["Alice", "Bob"])
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({"bill_id": bill_id, "payers": [{"name": "alice"}]})

    assert "Recorded payer" in msg
    assert state.bills[0].paid_by == {"Alice": 30.0}


def test_set_payer_percentage_plus_equal_share_remainder():
    """'Alice paid 73%, Bob paid the rest' — a percentage entry plus one
    bare equal-share entry absorbing 100% - 73% = 27% of the total."""
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({
        "bill_id": bill_id,
        "payers": [{"name": "Alice", "percentage": 73.0}, {"name": "Bob"}],
    })

    assert "Recorded payer" in msg
    assert state.bills[0].paid_by["Alice"] == pytest.approx(73.0)
    assert state.bills[0].paid_by["Bob"] == pytest.approx(27.0)


def test_set_payer_multiple_explicit_amounts():
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    set_payer.invoke({
        "bill_id": bill_id,
        "payers": [{"name": "Alice", "amount": 60.0}, {"name": "Bob", "amount": 40.0}],
    })

    assert state.bills[0].paid_by == {"Alice": 60.0, "Bob": 40.0}


def test_set_payer_rejects_amounts_that_dont_sum_to_total():
    """validate_contribution_map must reject a caller-provided split that
    doesn't sum to the right total — and the bill's paid_by must stay
    untouched (empty) rather than silently storing the bad split."""
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({
        "bill_id": bill_id,
        "payers": [{"name": "Alice", "amount": 999.0}, {"name": "Bob", "amount": 1.0}],
    })

    assert "Error" in msg
    assert state.bills[0].paid_by == {}


def test_set_payer_rejects_unknown_participant():
    state, _assign, set_payer, _calc = _legacy_tools(["Alice", "Bob"])
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({"bill_id": bill_id, "payers": [{"name": "Zoe"}]})

    assert "Error" in msg
    assert "not in the participant list" in msg
    assert state.bills[0].paid_by == {}


def test_set_payer_accepts_legitimate_n_way_rounding_split():
    """Three people each paying $33.34 on a $100 bill sums to $100.02 — a
    legitimate N-way rounding remainder validate_contribution_map's own
    scaled tolerance accepts. A flat epsilon pre-check in
    _resolve_contribution_entries used to reject this before
    validate_contribution_map ever got a say; it must not anymore."""
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob", "Carol"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({
        "bill_id": bill_id,
        "payers": [
            {"name": "Alice", "amount": 33.34},
            {"name": "Bob", "amount": 33.34},
            {"name": "Carol", "amount": 33.34},
        ],
    })

    assert "Error" not in msg
    assert state.bills[0].paid_by == {"Alice": 33.34, "Bob": 33.34, "Carol": 33.34}


def test_set_payer_rejects_negative_amount():
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({
        "bill_id": bill_id,
        "payers": [{"name": "Alice", "amount": 150.0}, {"name": "Bob", "amount": -50.0}],
    })

    assert "Error" in msg
    assert "non-negative" in msg
    assert state.bills[0].paid_by == {}


def test_set_payer_rejects_overshoot_that_only_appears_after_remainder_resolution():
    """A single overshooting explicit amount plus a bare equal-share entry
    passes the raw non-negative check (1000.0 is itself non-negative) and
    validate_contribution_map's sum check (the remainder is defined as
    total - explicit_total, so the final map always sums to the total by
    construction, regardless of sign) — only visible once the remainder is
    actually distributed and comes out negative. Must still be rejected,
    and paid_by must stay untouched rather than storing
    {"Alice": 1000.0, "Bob": -900.0}."""
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({
        "bill_id": bill_id,
        "payers": [{"name": "Alice", "amount": 1000.0}, {"name": "Bob"}],
    })

    assert "Error" in msg
    assert "negative" in msg
    assert state.bills[0].paid_by == {}


def test_set_payer_rejects_percentage_overshoot_that_only_appears_after_remainder_resolution():
    """Same class of bug as above, via two individually-in-range (0-100)
    percentage entries whose sum still exceeds 100%."""
    state, _assign, set_payer, _calc = _legacy_tools(
        ["Alice", "Bob", "Carol"], one_item_bill(price=100.0)
    )
    bill_id = state.bills[0].bill_id

    msg = set_payer.invoke({
        "bill_id": bill_id,
        "payers": [
            {"name": "Alice", "percentage": 80},
            {"name": "Bob", "percentage": 90},
            {"name": "Carol"},
        ],
    })

    assert "Error" in msg
    assert state.bills[0].paid_by == {}


# ---------- assign_items: percentage_per_person / amount_per_person modes ----------

def test_assign_items_percentage_mode_splits_remainder_equally():
    """'Alice had 30% of the beer, the other 3 split the rest equally'."""
    state, assign_items, _set_payer, _calc = _legacy_tools(
        ["Alice", "Bob", "Carol", "Dan"],
        {
            "raw_text": "Beer $21",
            "description": "Bar tab",
            "items": [{"name": "Beer", "price": 21.0, "qty": 1}],
            "tax": 0.0,
            "tip": 0.0,
        },
    )
    bill_id = state.bills[0].bill_id

    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Beer",
            "assigned_to": ["Alice", "Bob", "Carol", "Dan"],
            "percentage_per_person": {"Alice": 30},
        }],
    })

    assert "percentage-based" in msg
    item = state.bills[0].items[0]
    assert item.cost_allocations["Alice"] == pytest.approx(6.3)
    assert item.cost_allocations["Bob"] == pytest.approx(4.9)
    assert item.cost_allocations["Carol"] == pytest.approx(4.9)
    assert item.cost_allocations["Dan"] == pytest.approx(4.9)
    assert sum(item.cost_allocations.values()) == pytest.approx(21.0)


def test_assign_items_amount_mode_splits_remainder_equally():
    """'Alice put in $12 for the cake, Bob covers the rest'."""
    state, assign_items, _set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"],
        {
            "raw_text": "Cake $18",
            "description": "Dessert",
            "items": [{"name": "Cake", "price": 18.0, "qty": 1}],
            "tax": 0.0,
            "tip": 0.0,
        },
    )
    bill_id = state.bills[0].bill_id

    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Cake",
            "assigned_to": ["Alice", "Bob"],
            "amount_per_person": {"Alice": 12.0},
        }],
    })

    item = state.bills[0].items[0]
    assert item.cost_allocations == {"Alice": 12.0, "Bob": 6.0}


def test_assign_items_rejects_negative_amount_per_person():
    state, assign_items, _set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"],
        {
            "raw_text": "Cake $18",
            "description": "Dessert",
            "items": [{"name": "Cake", "price": 18.0, "qty": 1}],
            "tax": 0.0,
            "tip": 0.0,
        },
    )
    bill_id = state.bills[0].bill_id

    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Cake",
            "assigned_to": ["Alice", "Bob"],
            "amount_per_person": {"Alice": 25.0, "Bob": -7.0},
        }],
    })

    assert "non-negative" in msg
    assert state.bills[0].items[0].cost_allocations == {}


def test_assign_items_rejects_amount_overshoot_that_only_appears_after_remainder_resolution():
    """Same overshoot-with-remainder class of bug as set_payer's, via
    amount_per_person: one person given more than the item's price, with
    the rest left to split "whatever's left" — which resolves to a
    negative share only after the remainder is distributed."""
    state, assign_items, _set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"],
        {
            "raw_text": "Thing $100",
            "description": "Test",
            "items": [{"name": "Thing", "price": 100.0, "qty": 1}],
            "tax": 0.0,
            "tip": 0.0,
        },
    )
    bill_id = state.bills[0].bill_id

    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Thing",
            "assigned_to": ["Alice", "Bob"],
            "amount_per_person": {"Alice": 1000.0},
        }],
    })

    assert "negative" in msg
    assert state.bills[0].items[0].cost_allocations == {}


def test_assign_items_qty_negative_error_blames_qty_not_converted_dollars():
    """Cosmetic: a negative qty_per_person value must be rejected with an
    error about the qty the caller actually passed, not a dollar amount
    converted from it (which _resolve_contribution_entries' generic
    non-negative-amount check would otherwise report)."""
    state, assign_items, _set_payer, _calc = _legacy_tools(
        ["Alice", "Bob"],
        {
            "raw_text": "Burger $40",
            "description": "Test",
            "items": [{"name": "Burger", "price": 40.0, "qty": 4}],
            "tax": 0.0,
            "tip": 0.0,
        },
    )
    bill_id = state.bills[0].bill_id

    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Burger",
            "assigned_to": ["Alice", "Bob"],
            "qty_per_person": {"Alice": -1},
        }],
    })

    assert "qty_per_person must be non-negative" in msg
    assert "Alice: -1" in msg
    assert state.bills[0].items[0].cost_allocations == {}


def test_assign_items_rejects_more_than_one_allocation_mode():
    state, assign_items, _set_payer, _calc = _legacy_tools(["Alice", "Bob"])
    bill_id = state.bills[0].bill_id

    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Pizza",
            "assigned_to": ["Alice", "Bob"],
            "qty_per_person": {"Alice": 1},
            "amount_per_person": {"Alice": 10.0},
        }],
    })

    assert "only one of" in msg
    assert state.bills[0].items[0].cost_allocations == {}


def test_assign_items_rejects_percentages_that_dont_sum_correctly():
    """A percentage over 100 for a single person can't validate against the
    item price — must error, not silently store a bad allocation."""
    state, assign_items, _set_payer, _calc = _legacy_tools(["Alice", "Bob"])
    bill_id = state.bills[0].bill_id

    msg = assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Pizza",
            "assigned_to": ["Alice"],
            "percentage_per_person": {"Alice": 150},
        }],
    })

    assert "between 0 and 100" in msg
    assert state.bills[0].items[0].cost_allocations == {}


def test_assign_items_default_equal_split_still_works():
    """No allocation dict at all keeps the original behavior exactly."""
    state, assign_items, _set_payer, _calc = _legacy_tools(["Alice", "Bob"])
    bill_id = state.bills[0].bill_id

    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice", "Bob"], "shared": True}],
    })

    item = state.bills[0].items[0]
    assert item.cost_allocations == {}
    assert item.assigned_to == ["Alice", "Bob"]
    assert item.shared is True


def test_mark_items_unassigned_clears_stale_cost_allocations():
    """Regression: compute_balances checks item.cost_allocations BEFORE
    falling back to an equal split across assigned_to, so an item
    previously split by percentage/amount and then marked "split evenly"
    must have that stale map cleared — otherwise mark_items_unassigned
    silently keeps charging the old per-person amounts instead of an
    equal split, with no warning to the caller."""
    state = SessionState()
    state.participants = ["Alice", "Bob", "Carol"]
    (
        add_bill, _set_p, assign_items, set_payer, mark_items_unassigned,
        _rename, _add_item, _rm_item, _rm_bill, _calc,
    ) = _build_tools(
        state, "unused-session-id"
    )
    add_bill.invoke({
        "raw_text": "Thing $30",
        "description": "Test",
        "items": [{"name": "Thing", "price": 30.0, "qty": 1}],
        "tax": 0.0,
        "tip": 0.0,
    })
    bill_id = state.bills[0].bill_id

    # Stale allocation: Alice was on the hook for the whole item.
    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Thing",
            "assigned_to": ["Alice", "Bob", "Carol"],
            "percentage_per_person": {"Alice": 100},
        }],
    })
    # Alice's 100% leaves Bob/Carol a $0 explicit share of this item — a
    # stale allocation that, before mark_items_unassigned clears it, would
    # keep charging them nothing instead of the equal 1/3 share they should
    # get once the item is marked "split evenly."
    assert state.bills[0].items[0].cost_allocations == {"Alice": 30.0, "Bob": 0.0, "Carol": 0.0}

    mark_items_unassigned.invoke({"bill_id": bill_id, "item_names": ["Thing"]})

    item = state.bills[0].items[0]
    assert item.cost_allocations == {}, "stale allocation must be cleared, not left to override the equal split"
    assert item.unassigned is True
    assert item.assigned_to == ["Alice", "Bob", "Carol"]

    set_payer.invoke({"bill_id": bill_id, "payers": [{"name": "Alice"}]})
    balances, warnings = Settlement.compute_balances(state.participants, state.bills)
    assert balances["Alice"] == pytest.approx(20.0)
    assert balances["Bob"] == pytest.approx(-10.0)
    assert balances["Carol"] == pytest.approx(-10.0)


# ---------- calculate_split: live view, not a one-time finalize (task7) ----------

def test_calculate_split_does_not_set_finalized_and_is_repeatable():
    state, assign_items, set_payer, calculate_split = _legacy_tools(["Alice", "Bob"])
    bill_id = state.bills[0].bill_id
    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{"item_name": "Pizza", "assigned_to": ["Alice", "Bob"], "shared": True}],
    })
    set_payer.invoke({"bill_id": bill_id, "payers": [{"name": "Alice"}]})

    report1 = calculate_split.invoke({})
    assert state.finalized is False  # calculate_split must never set this
    report2 = calculate_split.invoke({})
    assert report1 == report2  # pure report, callable repeatedly with no side effects

    # Approving a correction (here: just mutating state directly, as a stand-in
    # for "more bills/corrections get approved in between calls") must be
    # reflected on the very next call — there's no stale "finalized" snapshot.
    state.bills.append(state.bills[0])  # duplicate bill -> balances must change
    report3 = calculate_split.invoke({})
    assert report3 != report1


# ---------- core/settlement.py: item.unassigned defense-in-depth (code review) ----------

def test_compute_balances_unassigned_flag_overrides_stale_cost_allocations():
    """compute_balances must check item.unassigned BEFORE item.cost_allocations,
    as defense-in-depth independent of whatever sets `unassigned` elsewhere —
    not rely solely on the convention that the one call site setting
    unassigned=True (mark_items_unassigned) also happens to clear
    cost_allocations. Construct a ParsedBill directly (bypassing the tool
    layer entirely) with both unassigned=True AND a stale, non-empty
    cost_allocations map, and confirm compute_balances still produces the
    equal split, not the stale per-person amounts."""
    item = LineItem(
        name="Thing", price=30.0, qty=1,
        assigned_to=["Alice", "Bob", "Carol"],
        unassigned=True,
        cost_allocations={"Alice": 30.0},  # stale — must be ignored
    )
    bill = ParsedBill(
        bill_id="b1", raw_text="", description="d",
        items=[item], paid_by={"Alice": 30.0},
    )

    balances, warnings = Settlement.compute_balances(["Alice", "Bob", "Carol"], [bill])

    assert warnings == []
    assert balances["Alice"] == pytest.approx(20.0)
    assert balances["Bob"] == pytest.approx(-10.0)
    assert balances["Carol"] == pytest.approx(-10.0)


def test_calculate_split_end_to_end_across_all_three_allocation_modes():
    """A bill with items split equal/percentage/amount, paid by two people
    on a percentage basis — calculate_split must compute correct balances."""
    state, assign_items, set_payer, calculate_split = _legacy_tools(
        ["Alice", "Bob", "Carol", "Dan"],
        {
            "raw_text": "dinner",
            "description": "Group Dinner",
            "items": [
                {"name": "Nachos", "price": 20.0, "qty": 1},
                {"name": "Beer", "price": 21.0, "qty": 1},
                {"name": "Cake", "price": 18.0, "qty": 1},
            ],
            "tax": 4.0,
            "tip": 7.0,
        },
    )
    bill_id = state.bills[0].bill_id

    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{"item_name": "Nachos", "assigned_to": ["Alice", "Bob", "Carol", "Dan"]}],
    })
    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Beer",
            "assigned_to": ["Alice", "Bob", "Carol", "Dan"],
            "percentage_per_person": {"Alice": 30},
        }],
    })
    assign_items.invoke({
        "bill_id": bill_id,
        "assignments": [{
            "item_name": "Cake",
            "assigned_to": ["Alice", "Bob"],
            "amount_per_person": {"Alice": 12.0},
        }],
    })
    set_payer.invoke({
        "bill_id": bill_id,
        "payers": [{"name": "Alice", "percentage": 60}, {"name": "Bob"}],
    })

    balances, warnings = Settlement.compute_balances(state.participants, state.bills)
    assert warnings == []
    assert balances["Alice"] == pytest.approx(14.3559, abs=0.01)
    assert balances["Bob"] == pytest.approx(9.1356, abs=0.01)
    assert balances["Carol"] == pytest.approx(-11.7458, abs=0.01)
    assert balances["Dan"] == pytest.approx(-11.7458, abs=0.01)

    report = calculate_split.invoke({})
    assert "SETTLEMENT REPORT" in report


# ---------- rename_bill / add_item_to_bill / remove_item_from_bill / remove_bill (task9) ----------

def _approved_bill(session_id, admin_id, proposer_id, bill_kwargs=None):
    """Stage a bill via add_bill and approve it as admin — the shared setup
    for testing edits to an already-approved bill. Returns
    (approved_state, bill_id, approved_row_id)."""
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    db.save_session_state(session_id, state, [])  # persist participants, like server.py every turn

    add_bill, *_rest = _build_tools(state, session_id)
    add_bill.invoke(bill_kwargs or one_item_bill(), config=as_speaker(proposer_id))

    proposal = db.list_pending_proposals(session_id)[0]
    bill_id = proposal["payload"]["bill_id"]
    db.decide_proposal(proposal["id"], admin_id, "approved")
    row_id = db.get_bill_row_id(session_id, bill_id)

    approved_state = SessionState.from_dict(db.load_session_state(session_id))
    return approved_state, bill_id, row_id


def _pending_bill(session_id, proposer_id, bill_kwargs=None):
    """Stage a bill via add_bill but never approve it. Returns
    (state, bill_id, proposal_id)."""
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    db.save_session_state(session_id, state, [])

    add_bill, *_rest = _build_tools(state, session_id)
    add_bill.invoke(bill_kwargs or one_item_bill(), config=as_speaker(proposer_id))

    proposal = db.list_pending_proposals(session_id)[0]
    return state, proposal["payload"]["bill_id"], proposal["id"]


def two_item_bill():
    return {
        "raw_text": "Pizza $30, Soda $5",
        "description": "Pizza night",
        "items": [
            {"name": "Pizza", "price": 30.0, "qty": 1},
            {"name": "Soda", "price": 5.0, "qty": 1},
        ],
        "tax": 0.0,
        "tip": 0.0,
    }


# ---------- rename_bill ----------

def test_rename_bill_on_approved_bill_proposes_correction(fresh_db):
    admin = make_user("admin@example.com", "g-rn-admin")
    alice = make_user("alice@example.com", "g-rn-alice")
    bob = make_user("bob@example.com", "g-rn-bob")
    session_id = "sess-rename-1"
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    approved_state, bill_id, row_id = _approved_bill(session_id, admin["id"], alice["id"])

    _add, _set_p, _assign, _payer, _mark, rename_bill, _add_item, _rm_item, _rm_bill, _calc = _build_tools(
        approved_state, session_id
    )
    msg = rename_bill.invoke(
        {"bill_id": bill_id, "new_description": "Pizza Night Redux"}, config=as_speaker(bob["id"])
    )
    assert "already approved" in msg
    assert "admin approval" in msg

    # Not applied yet — the approved bill must stay untouched before approval.
    still_old = SessionState.from_dict(db.load_session_state(session_id))
    assert still_old.bills[0].description == "Pizza night"

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["supersedes_bill_id"] == row_id
    assert pending[0]["payload"]["description"] == "Pizza Night Redux"
    # Items/payer carried through unchanged in the proposal payload.
    assert pending[0]["payload"]["items"][0]["name"] == "Pizza"
    assert pending[0]["payload"]["items"][0]["price"] == 30.0

    db.decide_proposal(pending[0]["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    assert len(final_state.bills) == 1
    final_bill = final_state.bills[0]
    assert final_bill.bill_id == bill_id
    assert final_bill.description == "Pizza Night Redux"
    assert [i.name for i in final_bill.items] == ["Pizza"]
    assert final_bill.items[0].price == 30.0
    assert final_bill.paid_by == {}


def test_rename_bill_no_op_when_description_unchanged(fresh_db):
    admin = make_user("admin@example.com", "g-rn2-admin")
    alice = make_user("alice@example.com", "g-rn2-alice")
    session_id = "sess-rename-2"
    make_session_with_members(session_id, admin["id"], alice["id"])

    approved_state, bill_id, _row_id = _approved_bill(session_id, admin["id"], alice["id"])

    _add, _set_p, _assign, _payer, _mark, rename_bill, _add_item, _rm_item, _rm_bill, _calc = _build_tools(
        approved_state, session_id
    )
    msg = rename_bill.invoke(
        {"bill_id": bill_id, "new_description": "Pizza night"}, config=as_speaker(alice["id"])
    )
    assert "no change" in msg
    assert db.list_pending_proposals(session_id) == []  # nothing proposed


def test_rename_bill_updates_pending_proposal_in_place(fresh_db):
    admin = make_user("admin@example.com", "g-rn3-admin")
    alice = make_user("alice@example.com", "g-rn3-alice")
    session_id = "sess-rename-3"
    make_session_with_members(session_id, admin["id"], alice["id"])

    state, bill_id, proposal_id = _pending_bill(session_id, alice["id"])
    _add, _set_p, _assign, _payer, _mark, rename_bill, _add_item, _rm_item, _rm_bill, _calc = _build_tools(
        state, session_id
    )
    msg = rename_bill.invoke(
        {"bill_id": bill_id, "new_description": "Renamed Before Approval"}, config=as_speaker(alice["id"])
    )
    assert "still awaiting admin approval" in msg
    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["id"] == proposal_id
    assert pending[0]["payload"]["description"] == "Renamed Before Approval"


def test_rename_bill_legacy_path_mutates_state_directly(fresh_db):
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    add_bill, _set_p, _assign, _payer, _mark, rename_bill, _add_item, _rm_item, _rm_bill, _calc = _build_tools(
        state, "unused-session-id"
    )
    add_bill.invoke(one_item_bill())
    bill_id = state.bills[0].bill_id

    msg = rename_bill.invoke({"bill_id": bill_id, "new_description": "New Name"})
    assert state.bills[0].description == "New Name"
    assert "Renamed" in msg
    assert "admin approval" not in msg


# ---------- add_item_to_bill ----------

def test_add_item_to_bill_on_approved_bill_shows_up_unassigned_after_approval(fresh_db):
    admin = make_user("admin@example.com", "g-ai-admin")
    alice = make_user("alice@example.com", "g-ai-alice")
    session_id = "sess-additem-1"
    make_session_with_members(session_id, admin["id"], alice["id"])

    approved_state, bill_id, row_id = _approved_bill(session_id, admin["id"], alice["id"])

    _add, _set_p, _assign, _payer, _mark, _rename, add_item_to_bill, _rm_item, _rm_bill, _calc = _build_tools(
        approved_state, session_id
    )
    msg = add_item_to_bill.invoke(
        {"bill_id": bill_id, "name": "Garlic Bread", "price": 6.0, "qty": 1},
        config=as_speaker(alice["id"]),
    )
    assert "already approved" in msg

    # Not applied yet.
    still_old = SessionState.from_dict(db.load_session_state(session_id))
    assert [i.name for i in still_old.bills[0].items] == ["Pizza"]

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["supersedes_bill_id"] == row_id
    item_names = [i["name"] for i in pending[0]["payload"]["items"]]
    assert item_names == ["Pizza", "Garlic Bread"]

    db.decide_proposal(pending[0]["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    final_bill = final_state.bills[0]
    assert [i.name for i in final_bill.items] == ["Pizza", "Garlic Bread"]
    new_item = next(i for i in final_bill.items if i.name == "Garlic Bread")
    assert new_item.price == 6.0
    assert new_item.qty == 1
    assert new_item.assigned_to == []
    assert new_item.shared is False
    assert new_item.unassigned is False
    assert new_item.cost_allocations == {}
    # Original item untouched.
    original = next(i for i in final_bill.items if i.name == "Pizza")
    assert original.price == 30.0


def test_add_item_to_bill_legacy_path_mutates_state_directly(fresh_db):
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    add_bill, _set_p, _assign, _payer, _mark, _rename, add_item_to_bill, _rm_item, _rm_bill, _calc = _build_tools(
        state, "unused-session-id"
    )
    add_bill.invoke(one_item_bill())
    bill_id = state.bills[0].bill_id

    msg = add_item_to_bill.invoke({"bill_id": bill_id, "name": "Soda", "price": 5.0, "qty": 2})
    assert "admin approval" not in msg
    assert len(state.bills[0].items) == 2
    new_item = state.bills[0].items[1]
    assert new_item.name == "Soda"
    assert new_item.price == 5.0
    assert new_item.qty == 2
    assert new_item.unassigned is False
    assert new_item.assigned_to == []


# ---------- remove_item_from_bill ----------

def test_remove_item_from_bill_fuzzy_match_on_approved_bill(fresh_db):
    admin = make_user("admin@example.com", "g-ri-admin")
    alice = make_user("alice@example.com", "g-ri-alice")
    session_id = "sess-rmitem-1"
    make_session_with_members(session_id, admin["id"], alice["id"])

    approved_state, bill_id, row_id = _approved_bill(session_id, admin["id"], alice["id"], two_item_bill())

    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, remove_item_from_bill, _rm_bill, _calc = (
        _build_tools(approved_state, session_id)
    )
    # Fuzzy substring match: "sod" -> "Soda"
    msg = remove_item_from_bill.invoke(
        {"bill_id": bill_id, "item_name": "sod"}, config=as_speaker(alice["id"])
    )
    assert "already approved" in msg

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["supersedes_bill_id"] == row_id
    item_names = [i["name"] for i in pending[0]["payload"]["items"]]
    assert item_names == ["Pizza"]  # Soda gone, Pizza untouched

    db.decide_proposal(pending[0]["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    final_bill = final_state.bills[0]
    assert [i.name for i in final_bill.items] == ["Pizza"]
    assert final_bill.items[0].price == 30.0


def test_remove_item_from_bill_not_found_lists_available_items(fresh_db):
    admin = make_user("admin@example.com", "g-ri2-admin")
    alice = make_user("alice@example.com", "g-ri2-alice")
    session_id = "sess-rmitem-2"
    make_session_with_members(session_id, admin["id"], alice["id"])

    approved_state, bill_id, _row_id = _approved_bill(session_id, admin["id"], alice["id"], two_item_bill())

    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, remove_item_from_bill, _rm_bill, _calc = (
        _build_tools(approved_state, session_id)
    )
    msg = remove_item_from_bill.invoke(
        {"bill_id": bill_id, "item_name": "Nachos"}, config=as_speaker(alice["id"])
    )
    assert "not found" in msg
    assert "Pizza" in msg and "Soda" in msg
    assert db.list_pending_proposals(session_id) == []  # no phantom proposal


def test_remove_item_from_bill_legacy_path_mutates_state_directly(fresh_db):
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    add_bill, _set_p, _assign, _payer, _mark, _rename, _add_item, remove_item_from_bill, _rm_bill, _calc = (
        _build_tools(state, "unused-session-id")
    )
    add_bill.invoke(two_item_bill())
    bill_id = state.bills[0].bill_id

    msg = remove_item_from_bill.invoke({"bill_id": bill_id, "item_name": "soda"})
    assert "admin approval" not in msg
    assert [i.name for i in state.bills[0].items] == ["Pizza"]


# ---------- remove_bill ----------

def test_remove_bill_on_approved_bill_requires_approval_before_deletion(fresh_db):
    admin = make_user("admin@example.com", "g-rb-admin")
    alice = make_user("alice@example.com", "g-rb-alice")
    session_id = "sess-rmbill-1"
    make_session_with_members(session_id, admin["id"], alice["id"])

    approved_state, bill_id, row_id = _approved_bill(session_id, admin["id"], alice["id"])

    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, _rm_item, remove_bill, _calc = _build_tools(
        approved_state, session_id
    )
    msg = remove_bill.invoke({"bill_id": bill_id}, config=as_speaker(alice["id"]))
    assert "already approved" in msg
    assert "admin approval" in msg

    # Still present before approval.
    still_there = SessionState.from_dict(db.load_session_state(session_id))
    assert len(still_there.bills) == 1

    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["supersedes_bill_id"] == row_id
    assert pending[0]["payload"]["action"] == "remove_bill"
    assert pending[0]["payload"]["bill_id"] == bill_id

    db.decide_proposal(pending[0]["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    assert final_state.bills == []


def test_remove_bill_approval_deletes_bills_and_bill_items_rows(fresh_db):
    admin = make_user("admin@example.com", "g-rb2-admin")
    alice = make_user("alice@example.com", "g-rb2-alice")
    session_id = "sess-rmbill-2"
    make_session_with_members(session_id, admin["id"], alice["id"])

    approved_state, bill_id, row_id = _approved_bill(session_id, admin["id"], alice["id"])

    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, _rm_item, remove_bill, _calc = _build_tools(
        approved_state, session_id
    )
    remove_bill.invoke({"bill_id": bill_id}, config=as_speaker(alice["id"]))
    pending = db.list_pending_proposals(session_id)
    db.decide_proposal(pending[0]["id"], admin["id"], "approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    try:
        bill_row = conn.execute("SELECT * FROM bills WHERE id = ?", (row_id,)).fetchone()
        item_rows = conn.execute("SELECT * FROM bill_items WHERE bill_id = ?", (row_id,)).fetchall()
    finally:
        conn.close()
    assert bill_row is None
    assert item_rows == []


def test_remove_bill_pending_proposal_is_rejected_directly_no_second_approval(fresh_db):
    """An admin removing a bill that's only a still-pending proposal (never
    approved) discards it directly via decide_proposal(..., "rejected") —
    no correction/removal proposal cycle, since there's nothing real yet."""
    admin = make_user("admin@example.com", "g-rb3-admin")
    alice = make_user("alice@example.com", "g-rb3-alice")
    session_id = "sess-rmbill-3"
    make_session_with_members(session_id, admin["id"], alice["id"])

    state, bill_id, proposal_id = _pending_bill(session_id, alice["id"])
    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, _rm_item, remove_bill, _calc = _build_tools(
        state, session_id
    )
    msg = remove_bill.invoke({"bill_id": bill_id}, config=as_speaker(admin["id"]))
    assert "discarded" in msg.lower() or "rejected" in msg.lower() or "never approved" in msg.lower()

    # Rejected directly, no new proposal created, nothing pending left.
    assert db.list_pending_proposals(session_id) == []
    # Sanity: the proposal row itself is now 'rejected', not re-decidable.
    with pytest.raises(ValueError):
        db.decide_proposal(proposal_id, admin["id"], "approved")

    # Never appeared in bills, ever.
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    assert final_state.bills == []


def test_remove_bill_pending_proposal_requires_admin(fresh_db):
    """Security regression: a non-admin member must not be able to discard
    another member's (or even their own) still-pending proposal via
    remove_bill — that's exactly what db.decide_proposal(..., "rejected")
    does, and every other path to it (server.py's reject endpoint) is
    admin-gated via _require_admin. Before the fix, remove_bill called
    decide_proposal directly with no such check."""
    admin = make_user("admin@example.com", "g-rbperm-admin")
    alice = make_user("alice@example.com", "g-rbperm-alice")
    bob = make_user("bob@example.com", "g-rbperm-bob")
    session_id = "sess-rmbill-perm"
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    state, bill_id, proposal_id = _pending_bill(session_id, alice["id"])
    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, _rm_item, remove_bill, _calc = _build_tools(
        state, session_id
    )
    # Bob (a non-admin member) tries to kill Alice's still-pending proposal.
    msg = remove_bill.invoke({"bill_id": bill_id}, config=as_speaker(bob["id"]))
    assert "admin" in msg.lower()

    # The proposal must still be pending — untouched by the rejected attempt.
    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    assert pending[0]["id"] == proposal_id
    assert pending[0]["status"] == "pending"


def test_remove_bill_dual_existence_rejects_stale_correction_and_proposes_removal(fresh_db):
    """Regression test for a resurrection bug: an already-approved bill with
    a *separate* pending correction proposal already targeting it (e.g. from
    set_payer) is a dual-existence case. remove_bill used to check the
    approved bill first and propose its own independent removal proposal
    while leaving that stale correction proposal pending — if an admin later
    approved the stale correction after the removal had already been
    approved and the bill deleted, decide_proposal's insert-path would fire
    (no existing bill row to find) and silently resurrect the bill.

    remove_bill must instead handle both at once: reject the stale pending
    proposal (admin-gated) AND propose the approved bill's removal, so no
    pending proposal survives that could later resurrect it."""
    admin = make_user("admin@example.com", "g-dual-admin")
    alice = make_user("alice@example.com", "g-dual-alice")
    bob = make_user("bob@example.com", "g-dual-bob")
    session_id = "sess-dual-1"
    make_session_with_members(session_id, admin["id"], alice["id"], bob["id"])

    approved_state, bill_id, row_id = _approved_bill(session_id, admin["id"], alice["id"])

    # Bob proposes a correction (set_payer) on the already-approved bill —
    # this is the stale pending proposal that must not survive removal.
    _add, _set_p, _assign, set_payer, _mark, _rename, _add_item, _rm_item, _rm_bill, _calc = _build_tools(
        approved_state, session_id
    )
    set_payer.invoke({"bill_id": bill_id, "payers": [{"name": "Alice"}]}, config=as_speaker(bob["id"]))
    correction = db.list_pending_proposals(session_id)[0]
    assert correction["payload"]["bill_id"] == bill_id
    assert correction["payload"].get("action") != "remove_bill"

    # Admin removes the bill — both the stale correction and the bill's own
    # removal must be handled in this one call.
    _add2, _set_p2, _assign2, _payer2, _mark2, _rename2, _add_item2, _rm_item2, remove_bill, _calc2 = (
        _build_tools(approved_state, session_id)
    )
    msg = remove_bill.invoke({"bill_id": bill_id}, config=as_speaker(admin["id"]))
    assert "discarded" in msg.lower()
    assert "already approved" in msg.lower()

    # The stale correction is gone; only the new removal proposal remains.
    pending = db.list_pending_proposals(session_id)
    assert len(pending) == 1
    removal_proposal = pending[0]
    assert removal_proposal["id"] != correction["id"]
    assert removal_proposal["payload"].get("action") == "remove_bill"
    assert removal_proposal["payload"]["bill_id"] == bill_id

    # The stale correction can never be approved (and resurrect the bill)
    # later — it's already been decided (rejected), not pending.
    with pytest.raises(ValueError):
        db.decide_proposal(correction["id"], admin["id"], "approved")

    # Approving the removal proposal deletes the bill as expected.
    db.decide_proposal(removal_proposal["id"], admin["id"], "approved")
    final_state = SessionState.from_dict(db.load_session_state(session_id))
    assert final_state.bills == []

    # No pending proposal survives that could resurrect it — the bill stays
    # gone, and there's nothing left to decide.
    assert db.list_pending_proposals(session_id) == []


def test_remove_bill_not_found_error(fresh_db):
    admin = make_user("admin@example.com", "g-rb4-admin")
    alice = make_user("alice@example.com", "g-rb4-alice")
    session_id = "sess-rmbill-4"
    make_session_with_members(session_id, admin["id"], alice["id"])

    state = SessionState()
    state.participants = ["Alice", "Bob"]
    db.save_session_state(session_id, state, [])

    _add, _set_p, _assign, _payer, _mark, _rename, _add_item, _rm_item, remove_bill, _calc = _build_tools(
        state, session_id
    )
    msg = remove_bill.invoke({"bill_id": "bill_does_not_exist"}, config=as_speaker(alice["id"]))
    assert "not found" in msg


def test_remove_bill_legacy_path_mutates_state_directly(fresh_db):
    state = SessionState()
    state.participants = ["Alice", "Bob"]
    add_bill, _set_p, _assign, _payer, _mark, _rename, _add_item, _rm_item, remove_bill, _calc = _build_tools(
        state, "unused-session-id"
    )
    add_bill.invoke(one_item_bill())
    bill_id = state.bills[0].bill_id

    msg = remove_bill.invoke({"bill_id": bill_id})
    assert "admin approval" not in msg
    assert state.bills == []
