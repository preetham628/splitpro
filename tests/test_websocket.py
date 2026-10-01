"""Tests for task14's /ws/sessions/{id} live-update WebSocket: handshake
auth/membership rejection, and the two broadcast trigger points
(_run_chat_turn's new-message push and approve_proposal/reject_proposal's
state push).

Unlike tests/test_server.py (which calls endpoint functions directly,
bypassing the ASGI stack entirely), these tests need a real running event
loop for the WebSocket connections themselves to live on — server.py's
broadcast helpers (_broadcast_to_session et al.) bridge onto that loop via
asyncio.run_coroutine_threadsafe precisely because REST endpoints run in a
worker thread, not on it. So these use FastAPI's TestClient as a context
manager (which starts the real lifespan, including capturing
server._event_loop) for the WebSocket half, while still calling
server.chat()/approve_proposal()/etc. directly for the REST half, matching
test_server.py's existing convention for everything that isn't the socket
itself.

fresh_server additionally points app_config.auth.db_path at the same
scratch DB the fixture already created, so the TestClient's own lifespan
(db.init_db / checkpointer.init_checkpointer) doesn't re-point core.database
at this machine's real configured DB_PATH.
"""

import threading
import time
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server
from agents.chat_agent import ChatAgent
from core import auth as auth_module, checkpointer, database as db


@pytest.fixture
def fresh_server(tmp_path, monkeypatch):
    """Same scratch-DB-plus-clean-cache fixture tests/test_server.py uses,
    plus pointing app_config.auth.db_path at that same scratch DB (restored
    automatically by monkeypatch) so TestClient's lifespan — which calls
    db.init_db(app_config.auth.db_path) for real — targets it too, instead
    of silently re-pointing core.database at the real configured DB_PATH."""
    path = str(tmp_path / "test.db")
    db.create_db(path)
    db.init_db(path)
    checkpointer.init_checkpointer(path)

    monkeypatch.setattr(server, "sessions", {})
    monkeypatch.setattr(server, "_session_locks", {})
    monkeypatch.setattr(server, "_summarizing_sessions", set())
    monkeypatch.setattr(server, "_session_sockets", {})
    monkeypatch.setattr(server.app_config.auth, "db_path", path)
    # The real app lifespan hasn't run yet for this test — reset to None so
    # a leftover (possibly since-closed) loop from an earlier test's
    # TestClient can't be mistaken for a live one. TestClient(...).__enter__
    # sets this for real a few lines into every test below.
    monkeypatch.setattr(server, "_event_loop", None)

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


def _wait_until(predicate, timeout=2, interval=0.01):
    """Poll `predicate` until it's truthy or `timeout` elapses, returning its
    final value either way (so a caller can still assert on it for a clear
    failure message).

    Needed since task14's round-2 fix for the TOCTOU dangling-socket bug
    moved socket registration behind an extra asyncio.to_thread hop (the
    atomic re-check-and-register against delete_session's lock — see
    server._register_socket_if_alive), so it's no longer guaranteed to have
    completed by the moment TestClient's websocket_connect() context manager
    returns (that only waits for the ASGI "websocket.accept" message, one
    `await` earlier in the route). Same spirit as this suite's
    _receive_with_timeout and the rest of this file's threading.Event().
    wait(timeout=...)/join(timeout=...) pattern for anything that could
    otherwise race or hang.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def _receive_with_timeout(ws, timeout=5):
    """ws.receive_json() blocks with no timeout of its own — if a broadcast
    never arrives (e.g. a regression silently breaks the fan-out), the test
    would otherwise hang instead of failing. Same spirit as this suite's
    existing threading.Event().wait(timeout=...)/join(timeout=...) pattern
    for anything that could deadlock."""
    result = {}

    def target():
        try:
            result["value"] = ws.receive_json()
        except Exception as e:  # noqa: BLE001 - surfaced via pytest.fail/raise below
            result["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"Timed out after {timeout}s waiting for a WebSocket broadcast")
    if "error" in result:
        raise result["error"]
    return result["value"]


# ---------- Handshake rejection ----------
#
# All three rejections below now accept() the connection before closing it
# with 1008, rather than closing pre-accept — see session_websocket's
# docstring for why: verified empirically against a real uvicorn server
# with a real WebSocket client, a pre-accept close() never reaches the
# client as an actual WebSocket close frame at all (the handshake's HTTP
# Upgrade never completes, so the client just sees a bare HTTP error, which
# a real browser's WebSocket API reports as a generic code-1006 abnormal
# closure with no information about why — indistinguishable from a dropped
# connection). TestClient's in-process ASGI transport doesn't reproduce
# that distinction (a pre-accept close still raised WebSocketDisconnect
# with the right code locally, which is what let this gap slip through
# round-1 review's tests), so these now connect successfully first (same
# as a real accepted-then-rejected browser connection would) and only see
# the disconnect on the next receive, exactly like app.js's `ws.onmessage`/
# `ws.onclose` would.

def test_websocket_rejects_missing_cookie(fresh_server):
    alice = make_user("alice-ws@example.com", "g-alice-ws", "Alice")
    session_id = make_session_with_members(alice)

    with TestClient(server.app) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
        assert exc_info.value.code == 1008


def test_websocket_rejects_invalid_cookie(fresh_server):
    alice = make_user("alice-ws-bad@example.com", "g-alice-ws-bad", "Alice")
    session_id = make_session_with_members(alice)

    with TestClient(server.app, cookies={"access_token": "not-a-real-jwt"}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
        assert exc_info.value.code == 1008


def test_websocket_rejects_non_member(fresh_server):
    alice = make_user("alice-ws-nm@example.com", "g-alice-ws-nm", "Alice")
    mallory = make_user("mallory-ws-nm@example.com", "g-mallory-ws-nm", "Mallory")
    session_id = make_session_with_members(alice)  # mallory is not a member

    token = auth_module.create_jwt(mallory["id"])
    with TestClient(server.app, cookies={"access_token": token}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
        assert exc_info.value.code == 1008


def test_websocket_accepts_valid_member(fresh_server):
    """Sanity check for the accept path, independent of any broadcast."""
    alice = make_user("alice-ws-ok@example.com", "g-alice-ws-ok", "Alice")
    session_id = make_session_with_members(alice)

    token = auth_module.create_jwt(alice["id"])
    with TestClient(server.app, cookies={"access_token": token}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}"):
            # Connected and accepted — registered in the in-process registry.
            # (The registered object is the server-side starlette.websockets.
            # WebSocket the ASGI app sees, not the client-side
            # WebSocketTestSession handle this test holds, so this checks
            # the registry's shape rather than identity with `ws`.) Polled
            # rather than asserted immediately — see _wait_until's docstring
            # for why registration isn't guaranteed to have landed yet at
            # this exact point.
            assert _wait_until(lambda: session_id in server._session_sockets)
            assert len(server._session_sockets[session_id]) == 1
    # Disconnected cleanly — receive loop's finally should have unregistered it.
    assert session_id not in server._session_sockets


# ---------- Broadcast: chat turn ----------

def test_chat_turn_broadcasts_new_messages_and_state(fresh_server, monkeypatch):
    """After a full chat turn (_run_chat_turn, as invoked by chat()), every
    open socket for that session should receive the new user + agent
    messages (in the same shape GET /sessions/{id}/messages already returns
    rows in) plus the freshly computed state — without the caller needing
    to poll or reload."""
    monkeypatch.setattr(
        ChatAgent, "chat",
        lambda self, m, speaker_name=None, speaker_user_id=None: f"echo: {m}",
    )

    alice = make_user("alice-ws-chat@example.com", "g-alice-ws-chat", "Alice")
    session_id = make_session_with_members(alice)
    token = auth_module.create_jwt(alice["id"])

    with TestClient(server.app, cookies={"access_token": token}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            def send_chat():
                time.sleep(0.1)
                server.chat(
                    session_id, server.ChatRequest(message="hello there"),
                    background_tasks=BackgroundTasks(), user=alice,
                )

            t = threading.Thread(target=send_chat)
            t.start()
            try:
                payload = _receive_with_timeout(ws)
            finally:
                t.join(timeout=5)

    assert payload["type"] == "messages"
    assert "state" in payload
    roles_and_content = [(m["role"], m["content"]) for m in payload["messages"]]
    assert roles_and_content == [("user", "hello there"), ("agent", "echo: hello there")]
    # Both rows carry a real DB-assigned id, same shape list_chat_messages()
    # returns — the frontend doesn't need to understand a second shape.
    assert all(isinstance(m["id"], int) for m in payload["messages"])


def test_two_sockets_on_same_session_both_receive_the_broadcast(fresh_server, monkeypatch):
    """The core multi-tab/multi-user scenario this task exists for: two
    separate connections to the same session must each get the live
    update, not just whichever happened to be registered first.

    Both connections go through a single TestClient (one lifespan, one
    event loop) with per-call `cookies` overrides for each user — two
    separate `with TestClient(...) as ...:` instances would each run their
    *own* ASGI lifespan and capture their own event loop into the same
    server._event_loop module global, clobbering each other, which has no
    equivalent in production (there's only ever one real server process/
    loop). A single shared client matches that production reality while
    still exercising two independent WebSocket connections for two
    different users, same as two real browser sessions would be."""
    monkeypatch.setattr(
        ChatAgent, "chat",
        lambda self, m, speaker_name=None, speaker_user_id=None: "ok",
    )

    alice = make_user("alice-ws-multi@example.com", "g-alice-ws-multi", "Alice")
    bob = make_user("bob-ws-multi@example.com", "g-bob-ws-multi", "Bob")
    session_id = make_session_with_members(alice, bob)
    alice_token = auth_module.create_jwt(alice["id"])
    bob_token = auth_module.create_jwt(bob["id"])

    with TestClient(server.app) as client:
        client.cookies.set("access_token", alice_token)
        ws_alice_ctx = client.websocket_connect(f"/ws/sessions/{session_id}")
        client.cookies.set("access_token", bob_token)
        ws_bob_ctx = client.websocket_connect(f"/ws/sessions/{session_id}")

        with ws_alice_ctx as ws_alice, ws_bob_ctx as ws_bob:
            # Polled, not asserted immediately — see _wait_until's docstring.
            assert _wait_until(lambda: len(server._session_sockets.get(session_id, [])) == 2)

            def send_chat():
                time.sleep(0.1)
                server.chat(
                    session_id, server.ChatRequest(message="hi all"),
                    background_tasks=BackgroundTasks(), user=alice,
                )

            t = threading.Thread(target=send_chat)
            t.start()
            try:
                payload_alice = _receive_with_timeout(ws_alice)
                payload_bob = _receive_with_timeout(ws_bob)
            finally:
                t.join(timeout=5)

    assert payload_alice["type"] == "messages"
    assert payload_bob["type"] == "messages"


# ---------- Broadcast: proposal decisions ----------

def _sample_payload():
    return {
        "bill_id": "bill_1",
        "description": "Dinner",
        "raw_text": "raw text",
        "items": [
            {"name": "Burger", "price": 10.0, "qty": 1, "assigned_to": ["Alice"],
             "shared": False, "unassigned": False, "cost_allocations": {}}
        ],
        "tax": 1.0,
        "tip": 2.0,
        "paid_by": {"Alice": 13.0},
    }


def test_approve_proposal_broadcasts_updated_state(fresh_server):
    admin = make_user("admin-ws-approve@example.com", "g-admin-ws-approve", "Admin")
    session_id = make_session_with_members(admin)
    token = auth_module.create_jwt(admin["id"])

    pid = db.create_proposal(session_id, admin["id"], _sample_payload())

    with TestClient(server.app, cookies={"access_token": token}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            def do_approve():
                time.sleep(0.1)
                server.approve_proposal(session_id, pid, admin)

            t = threading.Thread(target=do_approve)
            t.start()
            try:
                payload = _receive_with_timeout(ws)
            finally:
                t.join(timeout=5)

    assert payload["type"] == "state"
    assert payload["state"]["pending_proposals_count"] == 0
    assert [b["description"] for b in payload["state"]["bills"]] == ["Dinner"]


def test_reject_proposal_broadcasts_updated_state(fresh_server):
    admin = make_user("admin-ws-reject@example.com", "g-admin-ws-reject", "Admin")
    session_id = make_session_with_members(admin)
    token = auth_module.create_jwt(admin["id"])

    pid = db.create_proposal(session_id, admin["id"], _sample_payload())

    with TestClient(server.app, cookies={"access_token": token}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            def do_reject():
                time.sleep(0.1)
                server.reject_proposal(session_id, pid, admin)

            t = threading.Thread(target=do_reject)
            t.start()
            try:
                payload = _receive_with_timeout(ws)
            finally:
                t.join(timeout=5)

    assert payload["type"] == "state"
    assert payload["state"]["pending_proposals_count"] == 0
    assert payload["state"]["bills"] == []  # rejected, never materialized


# ---------- Cleanup on session deletion ----------

def test_delete_session_closes_and_forgets_sockets(fresh_server):
    admin = make_user("admin-ws-del@example.com", "g-admin-ws-del", "Admin")
    session_id = make_session_with_members(admin)
    token = auth_module.create_jwt(admin["id"])

    with TestClient(server.app, cookies={"access_token": token}) as client:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
            # Polled, not asserted immediately — see _wait_until's docstring.
            # Also makes sure delete_session (started below) can't win the
            # race and run its _drop_session_sockets sweep before this
            # socket has even registered — that'd trivially "pass" today
            # even with the dangling-socket bug reintroduced, since there'd
            # be nothing to clean up either way.
            assert _wait_until(lambda: session_id in server._session_sockets)

            def do_delete():
                time.sleep(0.1)
                server.delete_session(session_id, admin)

            t = threading.Thread(target=do_delete)
            t.start()
            try:
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()
            finally:
                t.join(timeout=5)

    assert session_id not in server._session_sockets


# ---------- Round-2 review fixes ----------

def test_concurrent_delete_does_not_leave_dangling_socket(fresh_server, monkeypatch):
    """Reproduces the exact TOCTOU race round-1 review found: a websocket
    handshake passes its early (unsynchronized, cheap) membership check,
    and delete_session — including its _drop_session_sockets sweep — runs
    to completion *before* this connection reaches registration. Pre-fix,
    the handshake would then go ahead and register a socket for a session
    that's already gone, with delete_session's sweep having already run
    and found nothing to clean up — nothing left would ever close it.

    The early check's answer is forced to True regardless of actual DB
    state, simulating "the real check happened to run and pass in the
    instant just before the concurrent delete" — the whole premise of a
    TOCTOU race. The second call, inside _register_socket_if_alive's
    atomic re-check (round-2's fix), is left to hit the real
    db.is_session_member, which by the time it runs will correctly see
    the session — and, via session_members' ON DELETE CASCADE, this user's
    membership row — as gone.

    Run this against the pre-fix code (session_websocket registering
    unconditionally after accept(), with no re-check) and it fails: the
    socket ends up registered for session_id even though delete_session
    has already completed and swept an empty registry.
    """
    admin = make_user("admin-ws-race@example.com", "g-admin-ws-race", "Admin")
    session_id = make_session_with_members(admin)
    token = auth_module.create_jwt(admin["id"])

    real_is_member = db.is_session_member
    call_count = {"n": 0}
    early_check_started = threading.Event()
    delete_done = threading.Event()

    def patched_is_member(sid, uid):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # This is session_websocket's early, unsynchronized check.
            # Signal the main thread, then block until the concurrent
            # delete_session (below) has fully completed, and report the
            # stale "yes, still a member" answer anyway — exactly what a
            # real check that happened to run a moment earlier would have
            # seen.
            early_check_started.set()
            assert delete_done.wait(timeout=5), "delete_session never completed"
            return True
        # Every later call is the atomic re-check inside
        # _register_socket_if_alive — let it see reality.
        return real_is_member(sid, uid)

    monkeypatch.setattr(db, "is_session_member", patched_is_member)

    outcome = {}

    def connect_attempt():
        try:
            with TestClient(server.app, cookies={"access_token": token}) as client:
                with client.websocket_connect(f"/ws/sessions/{session_id}") as ws:
                    try:
                        ws.receive_text()
                        outcome["result"] = "unexpected-message"
                    except WebSocketDisconnect:
                        outcome["result"] = "disconnected-after-accept"
        except WebSocketDisconnect:
            outcome["result"] = "disconnected-on-connect"
        except Exception as e:  # noqa: BLE001 - surfaced via the assert below
            outcome["result"] = f"error: {e!r}"

    t = threading.Thread(target=connect_attempt)
    t.start()
    try:
        assert early_check_started.wait(timeout=5), "early membership check never ran"
        # Runs to completion while the connecting socket is still blocked
        # on the (forced-stale) early check above -- this is round-1's
        # exact repro: _drop_session_sockets' sweep finds nothing, because
        # nothing has registered yet.
        server.delete_session(session_id, admin)
        delete_done.set()
    finally:
        t.join(timeout=5)

    assert outcome.get("result") in ("disconnected-after-accept", "disconnected-on-connect"), outcome
    # The actual bug: no socket should be left registered for a session
    # that's already deleted.
    assert session_id not in server._session_sockets


def test_slow_membership_check_does_not_stall_other_sockets(fresh_server, monkeypatch):
    """Bug #4 regression test: db.is_session_member's blocking sqlite call
    inside session_websocket must run via asyncio.to_thread, not directly
    on the event loop -- a slow/contended lookup for one handshake must not
    stall every other open (or connecting) socket on this process.

    Forces the membership check for session_a to sleep for a second (a
    stand-in for a slow/contended query), then opens a second, unrelated
    connection to session_b while that's in flight and times it — which can
    only land well under a second if the slow check genuinely isn't running
    on the loop thread. Pre-fix (a direct, unwrapped db.is_session_member
    call), session_b's handshake would queue up behind session_a's sleep on
    the one event loop thread and take just as long.
    """
    alice = make_user("alice-ws-slow@example.com", "g-alice-ws-slow", "Alice")
    session_a = make_session_with_members(alice)
    session_b = make_session_with_members(alice)
    token = auth_module.create_jwt(alice["id"])

    real_is_member = db.is_session_member

    def slow_is_member(sid, uid):
        if sid == session_a:
            time.sleep(1.0)
        return real_is_member(sid, uid)

    monkeypatch.setattr(db, "is_session_member", slow_is_member)

    with TestClient(server.app, cookies={"access_token": token}) as client:
        slow_ctx = client.websocket_connect(f"/ws/sessions/{session_a}")
        slow_opened = threading.Event()

        def open_slow():
            with slow_ctx:
                slow_opened.set()

        t = threading.Thread(target=open_slow)
        t.start()
        # Give the slow handshake a moment to actually enter its
        # asyncio.to_thread(db.is_session_member, ...) sleep before racing
        # the second connection against it.
        time.sleep(0.2)

        start = time.time()
        with client.websocket_connect(f"/ws/sessions/{session_b}"):
            elapsed = time.time() - start

        assert slow_opened.wait(timeout=5), "slow connection for session_a never completed"
        t.join(timeout=5)

    assert elapsed < 0.5, (
        f"second handshake took {elapsed}s while session_a's membership "
        f"check was sleeping -- the event loop was stalled by a blocking "
        f"DB call instead of that call running off-loop via asyncio.to_thread"
    )
