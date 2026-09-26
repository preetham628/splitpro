"""Regression tests for server.py's per-session locking around chat turns.

Two review rounds on this code surfaced three real races (lost writes from
an unsynchronized ChatAgent cache-miss, an event-loop freeze from holding a
plain threading.Lock inside an async def, and a session-delete/in-flight-turn
race) — none of which were caught by anything in this suite. These tests
lock in the fixes so a future refactor can't silently reopen one of them.

Endpoint functions are called directly (not through TestClient) so `user`
dicts for different simulated callers can't collide the way FastAPI's
*global*, app-wide `dependency_overrides` dict would under concurrent
requests from multiple threads — that would be a race in the test harness,
not in the code under test. Calling `server.chat(...)`/`server.delete_session(...)`
directly is fine: a FastAPI path function is a plain callable outside the
framework, and `Depends(...)` default values are simply not triggered when a
real value is passed explicitly.
"""

import asyncio
import io
import threading
import time
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

import server
from agents.chat_agent import ChatAgent
from agents.image_analyzer import ImageAnalysisResult, ImageAnalyzer
from core import checkpointer, database as db
from core.session_state import ParsedBill


@pytest.fixture
def fresh_server(tmp_path, monkeypatch):
    """Fresh DB + checkpointer per test, and a clean server-side cache —
    server.sessions / server._session_locks are module-level globals shared
    across the whole test session otherwise."""
    path = str(tmp_path / "test.db")
    db.create_db(path)
    db.init_db(path)
    checkpointer.init_checkpointer(path)

    monkeypatch.setattr(server, "sessions", {})
    monkeypatch.setattr(server, "_session_locks", {})

    return path


def make_user(email, google_id, name=None):
    row = db.upsert_user(google_id=google_id, email=email, name=name or email.split("@")[0], avatar_url="")
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


def make_session_with_members(*users, admin_index=0):
    session_id = str(uuid4())
    db.create_session(session_id, users[admin_index]["id"], name="Test Session")
    for i, u in enumerate(users):
        db.add_session_member(session_id, u["id"], role="admin" if i == admin_index else "member")
    return session_id


def sample_payload(bill_id="bill_1", description="Dinner"):
    return {
        "bill_id": bill_id,
        "description": description,
        "raw_text": "raw text",
        "items": [
            {"name": "Burger", "price": 10.0, "qty": 1, "assigned_to": ["Alice"],
             "shared": False, "unassigned": False, "qty_allocations": {}}
        ],
        "tax": 1.0,
        "tip": 2.0,
        "paid_by": "Alice",
    }


# ---------- Concurrent first-touch race (cache-miss check-then-act) ----------

def test_concurrent_first_touch_does_not_lose_writes(fresh_server, monkeypatch):
    """Two members sending the first-ever messages to a brand-new session at
    nearly the same time must not have one's bill silently discarded by the
    other's save — the bug when _get_agent()'s cache-miss construction ran
    outside the per-session lock."""

    def fake_chat(self, user_message, speaker_name=None, speaker_user_id=None):
        bill_id = self.state.next_bill_id()
        time.sleep(0.05)  # widen the window a real race would need
        self.state.bills.append(
            ParsedBill(bill_id=bill_id, raw_text="", description=f"bill-from-{speaker_name}")
        )
        time.sleep(0.05)
        return f"echo: {user_message}"

    monkeypatch.setattr(ChatAgent, "chat", fake_chat)

    alice = make_user("alice@example.com", "g-alice", "Alice")
    bob = make_user("bob@example.com", "g-bob", "Bob")
    session_id = make_session_with_members(alice, bob)

    results = {}

    def run_turn(user, message):
        response, state = server._run_chat_turn(
            session_id, user, message, user["name"],
            persist_user_message=lambda: db.add_chat_message(
                session_id, "user", message, user_id=user["id"]
            ),
        )
        results[user["name"]] = (response, state)

    t1 = threading.Thread(target=run_turn, args=(alice, "alice's first message"))
    t2 = threading.Thread(target=run_turn, args=(bob, "bob's first message"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert "Alice" in results and "Bob" in results

    persisted = db.load_session_state(session_id)
    descriptions = sorted(b["description"] for b in persisted["bills"])
    assert descriptions == ["bill-from-Alice", "bill-from-Bob"], (
        f"lost write: expected both bills persisted, got {descriptions}"
    )

    # Exactly one ChatAgent should have been constructed for the session —
    # a split-brain cache (two independent agents) would mean this race is
    # back even if the DB happens to end up looking fine.
    assert list(server.sessions.keys()) == [session_id]
    cached_descriptions = sorted(b.description for b in server.sessions[session_id].state.bills)
    assert cached_descriptions == ["bill-from-Alice", "bill-from-Bob"]


# ---------- Session delete vs. in-flight turn ----------

def test_delete_session_waits_for_inflight_turn(fresh_server, monkeypatch):
    """Deleting a session while another member's message is mid-processing
    must not race the turn's save against the delete's FK cascade — it
    should wait for the turn to finish (or the turn should cleanly see the
    session as already gone), never raise an unhandled IntegrityError."""

    turn_started = threading.Event()

    def slow_chat(self, user_message, speaker_name=None, speaker_user_id=None):
        turn_started.set()
        time.sleep(0.2)
        return f"echo: {user_message}"

    monkeypatch.setattr(ChatAgent, "chat", slow_chat)

    admin = make_user("admin@example.com", "g-admin", "Admin")
    member = make_user("member@example.com", "g-member", "Member")
    session_id = make_session_with_members(admin, member)

    turn_error = []

    def run_turn():
        try:
            server.chat(session_id, server.ChatRequest(message="hello"), member)
        except Exception as e:  # noqa: BLE001 - captured for the assertion below
            turn_error.append(e)

    t = threading.Thread(target=run_turn)
    t.start()
    assert turn_started.wait(timeout=2), "turn never started"

    # The turn is now inside its locked section (mid-sleep). Deleting here
    # must not blow up with an unhandled DB error.
    delete_result = server.delete_session(session_id, admin)
    t.join(timeout=5)

    assert delete_result == {"ok": True}
    assert not turn_error, f"in-flight turn raised: {turn_error}"

    # No orphaned lock left behind for a session_id that no longer exists.
    assert session_id not in server._session_locks
    assert session_id not in server.sessions


def test_turn_after_delete_gets_clean_404(fresh_server):
    """A chat turn arriving for an already-deleted session must fail with a
    clean 404 (via _require_member), not an unhandled FK/IntegrityError from
    trying to save against a session row that's gone."""
    admin = make_user("admin2@example.com", "g-admin2", "Admin2")
    session_id = make_session_with_members(admin)

    result = server.delete_session(session_id, admin)
    assert result == {"ok": True}

    with pytest.raises(HTTPException) as exc_info:
        server.chat(session_id, server.ChatRequest(message="hello"), admin)
    assert exc_info.value.status_code == 404


# ---------- get_current_user's real shape vs. chat()/upload_image() ----------
#
# get_current_user() only ever returns {"id": ...} (see core/auth.py) — it
# never carries "name" or "email". make_user()'s dicts (used above) are
# richer than that and would mask a regression here, so these tests build
# the `user` dependency value the same minimal way the real dependency does.

def _real_shaped_user(row: dict) -> dict:
    return {"id": row["id"]}


def test_chat_does_not_crash_on_get_current_user_shaped_dict(fresh_server, monkeypatch):
    """chat() must not do user["name"]/user["email"] directly — the real
    get_current_user() dependency returns only {"id": ...}, so that raises an
    unhandled KeyError (a 500) for every real request. It must instead look
    the full row up via db.get_user_by_id, like auth_me already does."""
    captured = {}

    def fake_chat(self, user_message, speaker_name=None, speaker_user_id=None):
        captured["speaker_name"] = speaker_name
        return f"echo: {user_message}"

    monkeypatch.setattr(ChatAgent, "chat", fake_chat)

    alice_row = make_user("alice@example.com", "g-alice", "Alice")
    session_id = make_session_with_members(alice_row)

    result = server.chat(
        session_id, server.ChatRequest(message="hi"), _real_shaped_user(alice_row)
    )

    assert result.response == "echo: hi"
    assert captured["speaker_name"] == "Alice"


def test_chat_falls_back_to_email_when_name_blank(fresh_server, monkeypatch):
    captured = {}

    def fake_chat(self, user_message, speaker_name=None, speaker_user_id=None):
        captured["speaker_name"] = speaker_name
        return "ok"

    monkeypatch.setattr(ChatAgent, "chat", fake_chat)

    row = db.upsert_user(google_id="g-noname", email="noname@example.com", name="", avatar_url="")
    session_id = make_session_with_members(row)

    server.chat(session_id, server.ChatRequest(message="hi"), _real_shaped_user(row))

    assert captured["speaker_name"] == "noname@example.com"


def test_upload_image_does_not_crash_on_get_current_user_shaped_dict(fresh_server, monkeypatch):
    """Same KeyError hazard as chat(), for upload_image()'s identical
    speaker_name = user["name"] or user["email"] line."""
    captured = {}

    def fake_chat(self, user_message, speaker_name=None, speaker_user_id=None):
        captured["speaker_name"] = speaker_name
        return "ok"

    monkeypatch.setattr(ChatAgent, "chat", fake_chat)
    monkeypatch.setattr(
        ImageAnalyzer,
        "analyze",
        lambda self, image_bytes, media_type="image/jpeg": ImageAnalysisResult(
            is_bill=False, description="a cat", message="not a bill", bill_text=""
        ),
    )

    alice_row = make_user("alice2@example.com", "g-alice2", "Alice")
    session_id = make_session_with_members(alice_row)

    upload = UploadFile(filename="photo.jpg", file=io.BytesIO(b"fake-bytes"))

    result = asyncio.run(
        server.upload_image(session_id, upload, _real_shaped_user(alice_row))
    )

    assert result.response == "ok"
    assert captured["speaker_name"] == "Alice"


# ---------- Proposal approval must invalidate the cached agent ----------

def test_approve_proposal_evicts_cache_so_next_turn_keeps_the_bill(fresh_server, monkeypatch):
    """approve_proposal writes the bill straight to the DB but, before this
    fix, left the in-memory cached agent (built pre-approval) untouched. The
    next chat turn's normal write-through save then deleted+reinserted bills
    from that stale, bill-less cache — silently destroying the just-approved
    bill. Regression test for the full propose -> approve -> verify cycle."""

    def noop_chat(self, user_message, speaker_name=None, speaker_user_id=None):
        return "ok"

    monkeypatch.setattr(ChatAgent, "chat", noop_chat)

    admin = make_user("admin3@example.com", "g-admin3", "Admin3")
    session_id = make_session_with_members(admin)

    # Prime the cache with a pre-approval (bill-less) agent, exactly as a
    # normal GET .../state or chat turn would before any proposal exists.
    server.get_state(session_id, admin)
    assert session_id in server.sessions
    assert server.sessions[session_id].state.bills == []

    pid = db.create_proposal(session_id, admin["id"], sample_payload())
    server.approve_proposal(session_id, pid, admin)

    # The bill must be immediately visible via the served state, not just
    # in the DB underneath a stale cache.
    state = server.get_state(session_id, admin)
    assert [b["description"] for b in state["bills"]] == ["Dinner"]

    # One more normal chat turn must not wipe it back out.
    server.chat(session_id, server.ChatRequest(message="thanks"), admin)

    persisted = db.load_session_state(session_id)
    assert [b["description"] for b in persisted["bills"]] == ["Dinner"]


def test_reject_proposal_evicts_cache(fresh_server, monkeypatch):
    monkeypatch.setattr(ChatAgent, "chat", lambda self, m, speaker_name=None, speaker_user_id=None: "ok")

    admin = make_user("admin4@example.com", "g-admin4", "Admin4")
    session_id = make_session_with_members(admin)

    server.get_state(session_id, admin)
    assert session_id in server.sessions

    pid = db.create_proposal(session_id, admin["id"], sample_payload())
    server.reject_proposal(session_id, pid, admin)

    assert session_id not in server.sessions
