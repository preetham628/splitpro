"""
SplitPro FastAPI Server
Run with: uvicorn server:app --reload
"""

import asyncio
import base64
import secrets
import threading
from typing import Callable, Literal, Optional
from uuid import uuid4

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.chat_agent import ChatAgent, summarize_rounds
from agents.image_analyzer import ImageAnalyzer
from config import ChatAgentConfig, load_config
from core import auth as auth_module
from core.auth import get_current_user
from core import checkpointer
from core import database as db
from core.session_state import SessionState
from core.settlement import Settlement

load_dotenv()

app_config = load_config()

auth_module.configure(
    jwt_secret=app_config.auth.jwt_secret,
    jwt_algorithm=app_config.auth.jwt_algorithm,
    jwt_expire_minutes=app_config.auth.jwt_expire_minutes,
    google_client_id=app_config.auth.google_client_id,
    google_client_secret=app_config.auth.google_client_secret,
    google_redirect_uri=app_config.auth.google_redirect_uri,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _event_loop
    # REST endpoints (chat(), approve_proposal(), ...) are plain `def`s, so
    # Starlette runs them in a worker thread pool rather than on this loop —
    # see _broadcast_to_session's docstring for why that means WebSocket
    # sends from them have to be scheduled back onto this loop explicitly
    # rather than just awaited inline.
    _event_loop = asyncio.get_running_loop()
    db.init_db(app_config.auth.db_path)
    checkpointer.init_checkpointer(app_config.auth.db_path)
    yield


app = FastAPI(title="SplitPro API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# In-memory session cache: session_id -> ChatAgent (write-through with SQLite)
sessions: dict[str, ChatAgent] = {}

# Per-session lock guarding a full chat turn — a session can now have real
# concurrent senders (multiple members messaging at once), and while Task 3's
# speaker-identity fix means they can no longer misattribute proposals to the
# wrong user, concurrent turns could still interleave against the same
# SessionState/checkpoint thread (e.g. two turns reading/writing agent.state
# at once). The lock must wrap the *entire* turn — cache lookup/creation of
# the agent (_get_agent's cache-miss path is itself a check-then-act race),
# agent.chat(), persistence, and the state read for the response — not just
# the agent.chat() call, otherwise two first-touch requests for a brand-new
# session can each build/cache their own ChatAgent+SessionState before either
# takes the lock, and the loser's save silently clobbers the winner's.
_session_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


def _drop_session_lock(session_id: str) -> None:
    """Remove a session's lock entry once the session itself is gone, so
    _session_locks doesn't grow unboundedly over a long-running process.
    Safe even if another thread is currently holding the lock object — it
    still holds its own reference and will release it normally; only a
    future _get_session_lock() call for this (now-deleted) session_id would
    build a fresh Lock, and session_ids are never reused."""
    with _session_locks_guard:
        _session_locks.pop(session_id, None)


# In-process WebSocket registry for live session updates: session_id -> list
# of currently-open WebSocket connections for that session. Same shape and
# lifecycle as `sessions`/`_session_locks` above — populated on connect and
# pruned on disconnect (see session_websocket() below), and swept clean in
# delete_session/end_session alongside the existing lock-cleanup pattern
# there (see _drop_session_sockets).
#
# A single in-process dict is all this needs: unlike WhatsApp/Discord/
# Telegram-scale systems, this app only ever runs as one server process, so
# there's no "which of our many servers holds this connection" problem that
# would call for Redis pub/sub or similar cross-process routing.
_session_sockets: dict[str, list[WebSocket]] = {}
_session_sockets_guard = threading.Lock()

# The asyncio event loop captured at lifespan startup (see lifespan() above).
# None until then — in particular, still None for tests that call
# _run_chat_turn/approve_proposal/etc. directly without ever starting the
# app, same as `sessions`/`_session_locks` being plain empty dicts in that
# context. _broadcast_to_session/_drop_session_sockets treat that as "no
# sockets could possibly be connected yet" and no-op, same effect as a
# session with nothing registered in _session_sockets.
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _register_session_socket(session_id: str, websocket: WebSocket) -> None:
    with _session_sockets_guard:
        _session_sockets.setdefault(session_id, []).append(websocket)


def _unregister_session_socket(session_id: str, websocket: WebSocket) -> None:
    with _session_sockets_guard:
        sockets = _session_sockets.get(session_id)
        if sockets and websocket in sockets:
            sockets.remove(websocket)
            if not sockets:
                _session_sockets.pop(session_id, None)


def _drop_session_sockets(session_id: str) -> None:
    """Close and forget every open socket for a session — called from
    delete_session/end_session, the same spot _drop_session_lock is, so a
    deleted session doesn't leave stale connections (or a stale dict entry)
    behind.

    Closing is scheduled onto the event loop the same way
    _broadcast_to_session's sends are (see its docstring); each socket's own
    receive loop in session_websocket() will also notice the close and call
    _unregister_session_socket itself, but popping the dict entry here too
    means _session_sockets doesn't keep an entry around for a session_id
    that's already gone, even if a socket is slow to notice its own closure.
    """
    with _session_sockets_guard:
        sockets = _session_sockets.pop(session_id, [])
    if _event_loop is None or not sockets:
        return
    for ws in sockets:
        _schedule(_safe_close(ws))


async def _safe_close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except Exception:
        # Already closed/closing — nothing to do.
        pass


def _broadcast_to_session(session_id: str, payload: dict) -> None:
    """Fan out a JSON payload to every open WebSocket registered for
    session_id. Called from _run_chat_turn (new chat message(s) + updated
    state, after a turn completes) and approve_proposal/reject_proposal
    (updated state, after a decision is recorded).

    All of those are plain `def` REST endpoints, which — like every other
    synchronous path function in this file — Starlette runs in a worker
    thread pool rather than on the asyncio event loop that owns the
    WebSocket connections in _session_sockets. asyncio.run_coroutine_
    threadsafe bridges that: it hands the actual send off to the event loop
    and returns immediately without blocking this thread, so a slow or
    half-dead socket can't stall the REST request that triggered the
    broadcast.
    """
    if _event_loop is None:
        return
    with _session_sockets_guard:
        sockets = list(_session_sockets.get(session_id, ()))
    for ws in sockets:
        _schedule(_safe_send(ws, payload))


async def _safe_send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        # Connection is mid-close or otherwise broken — its own receive loop
        # in session_websocket() will detect the disconnect and unregister
        # it; nothing more to do from here.
        pass


def _schedule(coro) -> None:
    """asyncio.run_coroutine_threadsafe, swallowing the one failure mode
    that's actually reachable from a caller's perspective: the captured
    event loop having since closed (e.g. app shutdown, or a test harness
    that tore down its own loop after this module-level reference was set).
    That's just a more emphatic version of "nothing is listening" — the
    same no-op outcome as _event_loop being None in the first place — so it
    gets the same treatment rather than raising out of a REST endpoint that
    has nothing to do with WebSocket plumbing.
    """
    try:
        asyncio.run_coroutine_threadsafe(coro, _event_loop)
    except RuntimeError:
        coro.close()


# Session ids with a context-summarization background task currently
# scheduled or in flight (see _schedule_context_summarization_if_needed /
# _maybe_summarize_context below). Without this, two concurrent turns for
# the same session could each independently see "needs summarization" and
# each schedule their own task, both firing a real (paid) LLM summarization
# call before either reaches the lock — wasted cost, even though the second
# apply would safely no-op via apply_context_summarization's subset-id
# guard.
#
# Guarded by its own dedicated lock, _summarizing_sessions_guard, rather
# than reusing a session's per-session lock (_get_session_lock) — that lock
# is dropped (_drop_session_lock) once a session is deleted, and a pending
# summarization task for that now-deleted session would otherwise silently
# recreate (leak) a lock dict entry for a session_id that will never be
# used again just to clear this trivial O(1) set-membership bookkeeping in
# its finally block. This guard's own lifecycle is fully decoupled from
# that, so it never resurrects a dead session's lock entry.
_summarizing_sessions: set[str] = set()
_summarizing_sessions_guard = threading.Lock()


# ---------- Request / Response models ----------

class SessionRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    state: dict


class RenameRequest(BaseModel):
    name: str


class TitleAutoRequest(BaseModel):
    auto: bool


class AddMemberRequest(BaseModel):
    email: str


class UpdateRoleRequest(BaseModel):
    role: Literal["admin", "member"]


# ---------- Helpers ----------

def _chat_config(req: SessionRequest = None) -> ChatAgentConfig:
    return ChatAgentConfig(
        provider=(req.provider if req else None) or app_config.chat_agent.provider,
        model=(req.model if req else None) or app_config.chat_agent.model,
        temperature=app_config.chat_agent.temperature,
        max_iterations=app_config.chat_agent.max_iterations,
    )


def _serialize_state(state: SessionState, session_id: str) -> dict:
    """Convert SessionState to a JSON-serializable dict for the UI."""
    return {
        "participants": state.participants,
        "pending_proposals_count": db.count_pending_proposals(session_id),
        "bills": [
            {
                "bill_id": bill.bill_id,
                "description": bill.description,
                "subtotal": bill.subtotal(),
                "tax": bill.tax,
                "tip": bill.tip,
                "total": bill.total(),
                "paid_by": bill.paid_by,
                "items": [
                    {
                        "name": item.name,
                        "price": item.price,
                        "assigned_to": item.assigned_to,
                        "shared": item.shared,
                        "cost_allocations": item.cost_allocations,
                    }
                    for item in bill.items
                ],
                "unassigned_items": [i.name for i in bill.unassigned_items()],
            }
            for bill in state.bills
        ],
        "all_bills_ready": state.all_bills_ready(),
        "settlement": _compute_settlement(state),
    }


def _compute_settlement(state: SessionState) -> list:
    """Thin wrapper over the shared balance math (core/settlement.py) — this
    used to duplicate that logic inline and had drifted out of sync with the
    calculate_split tool (missing cost_allocations handling entirely), so the
    state panel and the settlement report could disagree. Not anymore.

    Settlement is a live view, not a one-time "finalize" action — there's no
    precondition to check here. Settlement.compute_balances() already skips
    bills with no payer set (bill.paid_by empty) and tolerates empty
    participants/bills lists, so this naturally returns [] rather than
    crashing when there isn't enough data yet (e.g. no bills, or a bill
    with no payer recorded)."""
    balances, _warnings = Settlement.compute_balances(state.participants, state.bills)
    return Settlement.generate_settlements(balances)


def _get_agent(session_id: str, user: dict) -> ChatAgent:
    """Return agent from cache or restore from DB. Validates ownership.

    Conversation history no longer needs restoring here — LangGraph's
    checkpointer already has it, keyed by thread_id (= session_id), and
    ChatAgent.chat() picks it up transparently on the next invoke(). Only
    the bill-splitting SessionState needs to be reloaded explicitly.
    """
    _require_member(session_id, user)

    if session_id in sessions:
        return sessions[session_id]

    # Cache miss — rebuild the agent; its graph resumes via the checkpointer.
    config = _chat_config()
    agent = ChatAgent(session_id=session_id, config=config)
    agent.set_state(SessionState.from_dict(db.load_session_state(session_id)))

    sessions[session_id] = agent
    return agent


def _save_agent(session_id: str, agent: ChatAgent, name: Optional[str] = None) -> None:
    """Persist bill-splitting state to DB (write-through).

    Conversation messages are already persisted by the checkpointer during
    agent.chat()'s graph.invoke() call — nothing to do for those here.

    Settlement is now a purely derived/computed view (see _serialize_state /
    _compute_settlement above) rather than a snapshot that needs persisting
    once a session is "finalized" — there's no such state anymore — so this
    no longer writes anything to the `settlements` table. The table itself
    is left in the schema (unused) rather than migrating it away here.
    """
    db.save_session_state(session_id, agent.state, settlements=[], name=name)


def _run_chat_turn(
    session_id: str,
    user: dict,
    agent_message: str,
    speaker_name: str,
    persist_user_message: Callable[[], dict],
) -> tuple[str, dict]:
    """Run one full chat turn — agent lookup/creation, the model call,
    persistence, and the response's state snapshot — as a single critical
    section under this session's lock.

    Must cover the *whole* turn, not just agent.chat(): _get_agent()'s
    cache-miss path constructs and caches a new ChatAgent/SessionState, which
    is itself an unsynchronized check-then-act if it can run outside the
    lock — two first-touch requests for the same brand-new session would
    each build an independent state object, get serialized against each
    other for nothing, and have one save silently discard the other's bill.
    Likewise the state snapshot returned to the caller must be read before
    the lock releases, or a concurrent turn could mutate agent.state in the
    gap between this turn's unlock and its own serialization.

    `persist_user_message` is a callback (rather than the raw message here)
    so callers can pass image-specific kwargs (upload_image) without this
    helper needing to know about them; it's called with the lock held, right
    after agent.chat() returns, matching the message ordering the two
    endpoints already had. It now returns the inserted row (same shape
    db.add_chat_message/list_chat_messages use) so it can be broadcast below
    alongside the agent's own reply, in the same shape GET
    /sessions/{id}/messages already returns rows in — the frontend's
    WebSocket handler doesn't need to understand a second message shape.

    Synchronous end-to-end — callers on an async path (upload_image) must
    run this via asyncio.to_thread so the lock (a plain threading.Lock)
    never blocks the event loop.
    """
    with _get_session_lock(session_id):
        agent = _get_agent(session_id, user)

        # Auto-renaming is driven by bills *plus* pending proposals, not
        # just bills: add_bill only creates a pending proposal and doesn't
        # touch agent.state.bills until an admin approves it, so a trigger
        # keyed on state.bills alone would never fire for a session sitting
        # on an unapproved bill. Skipped entirely once the user has taken
        # over naming this session manually (title_auto=False, set by the
        # PATCH .../name endpoint) — title_auto is checked once up front so
        # a manually-named session doesn't even pay for the extra
        # count_pending_proposals() query below.
        title_auto = db.get_title_auto(session_id)
        prev_count = (
            len(agent.state.bills) + db.count_pending_proposals(session_id)
            if title_auto else None
        )

        response = agent.chat(agent_message, speaker_name=speaker_name, speaker_user_id=user["id"])
        user_message_row = persist_user_message()
        agent_message_row = db.add_chat_message(session_id, "agent", response)

        new_name: Optional[str] = None
        if title_auto:
            new_count = len(agent.state.bills) + db.count_pending_proposals(session_id)
            if new_count > prev_count:
                new_name = _session_name_from_agent(agent, session_id)

        _save_agent(session_id, agent, name=new_name)
        state = _serialize_state(agent.state, session_id)

    # Pushed after the lock releases — a slow/half-dead socket send
    # shouldn't hold up the lock any longer than the turn itself needed it
    # for. See _broadcast_to_session's docstring for why this hops onto the
    # event loop rather than sending inline.
    _broadcast_to_session(session_id, {
        "type": "messages",
        "messages": [user_message_row, agent_message_row],
        "state": state,
    })

    return response, state


def _schedule_context_summarization_if_needed(session_id: str, background_tasks: BackgroundTasks) -> None:
    """Called right after a turn's response has already been built (see
    chat()/upload_image() below) — a cheap, synchronous check (a checkpoint
    read via ChatAgent.needs_context_summarization(), no LLM call) for
    whether this session's history has grown past the sliding window or the
    raw-message safety net. Only schedules the (potentially slow, real-LLM-
    call) background task when that's actually true — most turns, most
    sessions, this is a no-op.

    Also closes the race where two concurrent turns for the same session
    could each independently see "needs summarization" and each schedule
    their own task: the needs-check itself runs unlocked (cheap, and
    tolerating staleness is fine for a heuristic trigger), but the
    check-and-set against _summarizing_sessions is done under its own
    dedicated _summarizing_sessions_guard (see that global's comment for
    why this is deliberately NOT the per-session lock), so only the first
    of any such pair actually schedules a task — the loser sees its own
    session_id already marked in-flight and skips, saving a wasted (paid)
    LLM call for something apply_context_summarization would've no-op'd
    anyway.
    """
    agent = sessions.get(session_id)
    if agent is None:
        return
    if not agent.needs_context_summarization():
        return

    with _summarizing_sessions_guard:
        if session_id in _summarizing_sessions:
            return  # already scheduled/in flight -- don't duplicate the LLM call
        _summarizing_sessions.add(session_id)

    background_tasks.add_task(_maybe_summarize_context, session_id)


def _maybe_summarize_context(session_id: str) -> None:
    """Background task body — see _schedule_context_summarization_if_needed
    for the (now conditional) scheduling decision this is only ever
    reached from.

    Scheduled via BackgroundTasks.add_task, which FastAPI only runs *after*
    the triggering request's response has already been sent — so however
    long this takes adds zero latency to that request.

    Two phases, matching ChatAgent.plan_context_summarization /
    summarize_rounds / apply_context_summarization:
      1. The slow LLM call (summarize_rounds) runs with no lock held at
         all, so it never blocks a concurrent chat turn for this session.
      2. Only the final checkpoint-surgery + context_summary persist step
         (apply_context_summarization) takes this session's existing
         per-session lock — the same one _run_chat_turn uses for a whole
         turn — and only for that brief step.

    _summarizing_sessions is cleared in a finally, under its own dedicated
    _summarizing_sessions_guard (NOT _get_session_lock) — deliberately
    decoupled from the per-session lock's lifecycle. A session can be
    deleted (dropping its _get_session_lock entry via _drop_session_lock)
    while a summarization task scheduled for it is still pending; calling
    _get_session_lock(session_id) here in that case would silently
    recreate a lock dict entry for a session_id that will never be reused,
    leaking forever in a long-running process. The dedicated guard has no
    such lifecycle to collide with, so this is always safe to call
    regardless of what happened to the session in the meantime.
    """
    try:
        agent = sessions.get(session_id)
        if agent is None:
            # Session was deleted (or its cache entry otherwise evicted)
            # between the triggering turn finishing and this task running —
            # nothing left to summarize.
            return

        plan = agent.plan_context_summarization()
        if plan is None:
            return
        messages_to_summarize, ids_to_remove = plan

        existing_summary = db.get_context_summary(session_id)
        new_summary = summarize_rounds(agent.config, existing_summary, messages_to_summarize)

        with _get_session_lock(session_id):
            agent.apply_context_summarization(ids_to_remove, new_summary)
    finally:
        with _summarizing_sessions_guard:
            _summarizing_sessions.discard(session_id)


def _session_name_from_agent(agent: ChatAgent, session_id: str) -> Optional[str]:
    """Return a name to auto-rename the session to, or None if there's
    nothing usable yet.

    Prefers the first *approved* bill's description. A bill only lands in
    agent.state.bills once an admin approves the expense proposal that
    created it (see core/database.py's decide_proposal) — before that it
    only exists as a pending proposal, so without this fallback a session
    sitting on an unapproved bill would never get auto-renamed away from
    "New Session" no matter how long the conversation runs. Falls back to
    the first pending proposal with a non-empty description; if none of
    them have one (e.g. a bill proposal still mid-construction), returns
    None rather than renaming to something empty, same as the old
    no-bills-yet behavior.
    """
    if agent.state.bills:
        return agent.state.bills[0].description
    for proposal in db.list_pending_proposals(session_id):
        description = proposal["payload"].get("description")
        if description:
            return description
    return None


def _require_member(session_id: str, user: dict) -> None:
    """Any member (admin or not) may access the session. 404 rather than 403
    on failure — matches the existing behavior of not leaking whether a
    session exists to a caller with no access to it."""
    if not db.is_session_member(session_id, user["id"]):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


def _require_admin(session_id: str, user: dict) -> None:
    _require_member(session_id, user)
    if not db.is_session_admin(session_id, user["id"]):
        raise HTTPException(status_code=403, detail="Admin access required")


def _speaker_name(user: dict) -> str:
    """Full display name for chat attribution. get_current_user() only ever
    returns {"id": ...}, so this re-fetches the row the same way auth_me
    does rather than trusting a "name"/"email" key on `user` that never
    exists. Currently unreachable in practice (no delete_user path exists),
    but raises a clean 401 rather than leaving a latent TypeError on a
    missing row for whenever one does."""
    user_row = db.get_user_by_id(user["id"])
    if not user_row:
        raise HTTPException(status_code=401, detail="User not found")
    return user_row["name"] or user_row["email"]


# ---------- Auth Endpoints ----------

@app.get("/auth/google")
def auth_google():
    """Redirect user to Google's OAuth consent screen."""
    state = secrets.token_urlsafe(16)
    url = auth_module.build_google_auth_url(state=state)
    return RedirectResponse(url)


@app.get("/auth/google/callback")
def auth_google_callback(code: str = None, error: str = None):
    """Handle Google OAuth callback: exchange code → upsert user → set JWT cookie."""
    if error or not code:
        return RedirectResponse("/?auth_error=1")

    try:
        token_data = auth_module.exchange_code_for_token(code)
        user_info = auth_module.get_google_user_info(token_data["access_token"])
    except Exception:
        return RedirectResponse("/?auth_error=1")

    user = db.upsert_user(
        google_id=user_info["id"],
        email=user_info["email"],
        name=user_info.get("name", ""),
        avatar_url=user_info.get("picture", ""),
    )

    token = auth_module.create_jwt(user["id"])
    response = RedirectResponse("/")
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=app_config.auth.jwt_expire_minutes * 60,
    )
    return response


@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    """Return current user info. 401 if not logged in."""
    row = db.get_user_by_id(user["id"])
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "avatar_url": row["avatar_url"],
    }


@app.post("/auth/logout")
def auth_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie("access_token")
    return response


# ---------- Session Management Endpoints ----------

@app.get("/api/sessions")
def list_sessions(user: dict = Depends(get_current_user)):
    """List all sessions for the current user."""
    return db.list_sessions(user["id"])


@app.post("/api/sessions", status_code=201)
def create_session(
    req: SessionRequest = SessionRequest(),
    user: dict = Depends(get_current_user),
):
    """Create a new named session in DB + memory cache."""
    session_id = str(uuid4())
    config = _chat_config(req)
    agent = ChatAgent(session_id=session_id, config=config)
    sessions[session_id] = agent

    db.create_session(session_id, user["id"], name="New Session")
    db.add_session_member(session_id, user["id"], role="admin")
    _save_agent(session_id, agent)

    return {
        "session_id": session_id,
        "name": "New Session",
        "provider": config.provider,
    }


@app.patch("/api/sessions/{session_id}/name")
def rename_session(
    session_id: str,
    req: RenameRequest,
    user: dict = Depends(get_current_user),
):
    """Rename a session. Any member may rename — no admin requirement.

    Always turns off auto-naming (title_auto=False) as a side effect — a
    manual edit should "win" over the next bill/proposal auto-rename, same
    as the normal expectation that a manual edit sticks until the user
    explicitly opts back into auto-naming via PATCH .../title-auto.

    Takes the same per-session lock _run_chat_turn holds for its *entire*
    duration — without it, a rename landing while a turn is mid-flight
    would correctly persist name + title_auto=False to the DB, but that
    turn's own end-of-turn save (already in flight, using the title_auto
    value it read before this rename happened) would run afterward and
    silently overwrite the manual name with an auto-generated one. Taking
    the lock here serializes the two instead: whichever of the turn's save
    or this rename finishes last is what the DB ends up reflecting, and a
    rename that lands *before* a turn even starts is seen by that turn's own
    title_auto read (also taken under the lock), so it skips auto-renaming
    entirely. No change needed to where _run_chat_turn reads title_auto.
    """
    _require_member(session_id, user)
    with _get_session_lock(session_id):
        updated = db.rename_session_by_id(session_id, req.name, title_auto=False)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"name": req.name, "title_auto": False}


@app.patch("/api/sessions/{session_id}/title-auto")
def set_session_title_auto(
    session_id: str,
    req: TitleAutoRequest,
    user: dict = Depends(get_current_user),
):
    """Toggle a session's auto-naming flag — mainly used to turn it back on
    after a manual rename turned it off (see rename_session), though it also
    accepts turning it off directly. Any member may toggle it, same as
    rename_session.

    Takes the per-session lock for the same reason rename_session does — an
    in-flight chat turn's title_auto read and this write must be serialized,
    or a turn already past its title_auto check could save over this toggle.
    """
    _require_member(session_id, user)
    with _get_session_lock(session_id):
        db.set_title_auto(session_id, req.auto)
    return {"title_auto": req.auto}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a session. Admin only.

    Takes the same per-session lock _run_chat_turn holds for the whole
    turn, so this can't interleave with an in-flight chat/image turn for
    this session — without it, deleting mid-turn (cascading away the bills
    a concurrent turn is about to save against) turned that turn's later
    save into an unhandled IntegrityError instead of either finishing
    cleanly first or failing with a clean 404. Dropping the lock entry
    afterward (success or already-gone) means a lock is never left behind
    for a session_id that no longer exists. Any open WebSocket connections
    for this session are closed and forgotten the same way, via
    _drop_session_sockets — see its docstring.
    """
    _require_admin(session_id, user)
    with _get_session_lock(session_id):
        deleted = db.delete_session_by_id(session_id)
        if deleted:
            checkpointer.delete_thread(session_id)
            sessions.pop(session_id, None)
    _drop_session_lock(session_id)
    _drop_session_sockets(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


# ---------- Session Membership Endpoints ----------

@app.get("/api/sessions/{session_id}/members")
def list_members(session_id: str, user: dict = Depends(get_current_user)):
    """List a session's members. Any member may view the roster."""
    _require_member(session_id, user)
    return db.list_session_members(session_id)


@app.post("/api/sessions/{session_id}/members", status_code=201)
def add_member(
    session_id: str,
    req: AddMemberRequest,
    user: dict = Depends(get_current_user),
):
    """Add an existing user (by email) to a session. Admin only.

    Idempotent: adding someone who's already a member is a 200 no-op
    rather than a duplicate-row error — db.add_session_member does the
    existence check and the insert as one atomic statement, so this is
    also race-safe against a concurrent duplicate invite.
    """
    _require_admin(session_id, user)

    target = db.get_user_by_email(req.email)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="User not found — they must sign in at least once before being added",
        )

    inserted = db.add_session_member(session_id, target["id"], role="member")
    if not inserted:
        return JSONResponse(status_code=200, content={"ok": True, "already_member": True})
    return {"ok": True, "already_member": False}


@app.delete("/api/sessions/{session_id}/members/{target_user_id}")
def remove_member(
    session_id: str,
    target_user_id: int,
    user: dict = Depends(get_current_user),
):
    """Remove a member from a session. Admin only."""
    _require_admin(session_id, user)

    try:
        removed = db.remove_session_member(session_id, target_user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not removed:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"ok": True}


@app.patch("/api/sessions/{session_id}/members/{target_user_id}/role")
def update_member_role(
    session_id: str,
    target_user_id: int,
    req: UpdateRoleRequest,
    user: dict = Depends(get_current_user),
):
    """Change a member's role. Admin only."""
    _require_admin(session_id, user)

    try:
        updated = db.update_member_role(session_id, target_user_id, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not updated:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"role": req.role}


# ---------- Expense Proposal Review Endpoints ----------

@app.get("/api/sessions/{session_id}/proposals")
def list_proposals(session_id: str, user: dict = Depends(get_current_user)):
    """List pending expense proposals for a session. Admin only."""
    _require_admin(session_id, user)
    return db.list_pending_proposals(session_id)


def _require_proposal_in_session(session_id: str, proposal_id: int) -> None:
    """decide_proposal() is keyed only by proposal_id, with no session_id
    scoping of its own — without this check, an admin of any session could
    decide a proposal belonging to a different session by id alone."""
    proposal = db.get_proposal(proposal_id)
    if proposal is None or proposal["session_id"] != session_id:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")


def _broadcast_session_state(session_id: str) -> None:
    """Re-serialize this session's state straight from the DB and push it to
    every open socket for it. Used by approve_proposal/reject_proposal,
    which — unlike _run_chat_turn — have no in-memory ChatAgent handy to
    serialize from at the point they need to broadcast (decide_proposal
    writes straight to the DB, and the cache entry has *just* been evicted a
    few lines above each call site, by the same stale-cache fix
    _get_agent's docstring describes). Building a throwaway SessionState via
    SessionState.from_dict(db.load_session_state(...)) — the same call
    _get_agent() makes on its own cache-miss path — is enough for this; it
    avoids constructing a full ChatAgent (and whatever LLM client setup that
    entails) just to read state back out of it.
    """
    state = _serialize_state(SessionState.from_dict(db.load_session_state(session_id)), session_id)
    _broadcast_to_session(session_id, {"type": "state", "state": state})


@app.post("/api/sessions/{session_id}/proposals/{proposal_id}/approve")
def approve_proposal(
    session_id: str,
    proposal_id: int,
    user: dict = Depends(get_current_user),
):
    """Approve a pending expense proposal, materializing it into bills/bill_items. Admin only."""
    _require_admin(session_id, user)
    _require_proposal_in_session(session_id, proposal_id)

    # decide_proposal's bill write and the cache eviction must happen inside
    # the same locked section, exactly like delete_session's DB-delete +
    # cache-pop. Doing the DB write first and only taking the lock afterward
    # (the original version of this fix) leaves a window where a concurrent
    # chat turn can grab the lock, find the still-cached pre-approval agent,
    # and complete its own write-through save before eviction runs —
    # reproducing the exact stale-cache data loss this fix exists to close.
    with _get_session_lock(session_id):
        try:
            result = db.decide_proposal(proposal_id, user["id"], "approved")
        except ValueError as e:
            # _require_proposal_in_session already confirmed the proposal
            # exists in this session, so a ValueError here means
            # decide_proposal's own `WHERE status = 'pending'` guard
            # rejected it — i.e. someone else already decided it between
            # that check and this call. That's a conflict with the
            # proposal's current state, not a missing resource, so 409
            # rather than 404.
            raise HTTPException(status_code=409, detail=str(e))

        sessions.pop(session_id, None)

    _broadcast_session_state(session_id)
    return result


@app.post("/api/sessions/{session_id}/proposals/{proposal_id}/reject")
def reject_proposal(
    session_id: str,
    proposal_id: int,
    user: dict = Depends(get_current_user),
):
    """Reject a pending expense proposal. Admin only."""
    _require_admin(session_id, user)
    _require_proposal_in_session(session_id, proposal_id)

    # See the matching comment in approve_proposal — the DB write and the
    # cache eviction must share one locked section, or a concurrent chat
    # turn can slip in between them and save over the decision with a stale
    # cached agent.
    with _get_session_lock(session_id):
        try:
            result = db.decide_proposal(proposal_id, user["id"], "rejected")
        except ValueError as e:
            # See the matching comment in approve_proposal — already-decided
            # is a conflict (409), not a missing resource (404).
            raise HTTPException(status_code=409, detail=str(e))

        sessions.pop(session_id, None)

    _broadcast_session_state(session_id)
    return result


# ---------- Chat Endpoints ----------

@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(
    session_id: str,
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    speaker_name = _speaker_name(user)
    response, state = _run_chat_turn(
        session_id,
        user,
        req.message,
        speaker_name,
        persist_user_message=lambda: db.add_chat_message(
            session_id, "user", req.message, user_id=user["id"]
        ),
    )
    # Checked (cheap) and scheduled (only if actually needed) after the
    # response above is already built — BackgroundTasks only runs the task
    # once the response has been sent, so a long-running summarization here
    # adds no latency to this request. See _schedule_context_summarization_
    # if_needed's docstring for the needs-check + in-flight guard, and
    # _maybe_summarize_context's for the two-phase lock design.
    _schedule_context_summarization_if_needed(session_id, background_tasks)
    return ChatResponse(response=response, state=state)


@app.get("/sessions/{session_id}/state")
def get_state(session_id: str, user: dict = Depends(get_current_user)):
    # Locked for the same reason _run_chat_turn is: an unlocked read here
    # could race a concurrent chat turn's mutation of agent.state, or (on a
    # cache miss) build a redundant ChatAgent alongside one a concurrent
    # chat turn is already constructing.
    with _get_session_lock(session_id):
        agent = _get_agent(session_id, user)
        return _serialize_state(agent.state, session_id)


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, user: dict = Depends(get_current_user)):
    """Full chat transcript for replay when a session is reopened."""
    _require_member(session_id, user)
    return db.list_chat_messages(session_id)


@app.delete("/sessions/{session_id}")
def end_session(session_id: str, user: dict = Depends(get_current_user)):
    """Duplicate of DELETE /api/sessions/{session_id} (no /api prefix). Admin only."""
    return delete_session(session_id, user)


@app.post("/sessions/{session_id}/image", response_model=ChatResponse)
async def upload_image(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    # Cheap, read-only membership check for early rejection — deliberately
    # not _get_agent() here, since that can construct+cache a brand-new
    # ChatAgent on a cache miss, and doing that outside the per-session lock
    # is exactly the unsynchronized check-then-act _run_chat_turn's docstring
    # warns about. The real agent lookup happens inside _run_chat_turn.
    _require_member(session_id, user)

    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    analyzer = ImageAnalyzer(app_config.image_analyzer)
    # analyzer.analyze() is a synchronous call out to the vision model — the
    # same event-loop-blocking hazard the chat-turn lock had (a slow model
    # response or provider outage would otherwise stall the entire loop for
    # every session, not just this request), so it gets the same to_thread
    # treatment.
    result = await asyncio.to_thread(analyzer.analyze, image_bytes, media_type=content_type)

    if result.is_bill:
        agent_message = (
            f"I've scanned a receipt image. Here are the contents:\n\n{result.bill_text}"
        )
    else:
        agent_message = (
            f"The user uploaded an image. It is NOT a bill or receipt. "
            f"Description: {result.description}. "
            f"Tell the user what you see and ask them to upload a receipt image instead."
        )

    speaker_name = _speaker_name(user)

    # _run_chat_turn is synchronous and holds a plain threading.Lock for its
    # duration — run it off the event loop thread so lock contention on a
    # busy session stalls only this request, not every other request the
    # process is serving.
    response, state = await asyncio.to_thread(
        _run_chat_turn,
        session_id,
        user,
        agent_message,
        speaker_name,
        lambda: db.add_chat_message(
            session_id, "user", "", image_base64=image_b64, image_media_type=content_type,
            user_id=user["id"],
        ),
    )
    # See the matching comment in chat() — checked and scheduled only after
    # the response above is already built, so it adds no latency to this
    # request, and only when actually needed.
    _schedule_context_summarization_if_needed(session_id, background_tasks)
    return ChatResponse(response=response, state=state)


# ---------- WebSocket: live session updates ----------

@app.websocket("/ws/sessions/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """Live push of new chat messages + updated state for a session, so
    every tab with it open stays in sync without polling or a manual
    reload. See _broadcast_to_session (new messages after a chat turn) and
    _broadcast_session_state (state after a proposal decision) for the two
    places that push to this.

    Authenticates during the handshake the same way the REST endpoints do —
    the `access_token` cookie set by the Google OAuth callback — rather than
    inventing a second auth mechanism: a WebSocket's handshake is itself an
    HTTP request, and FastAPI's WebSocket exposes its cookies the same way
    Request does, so this reuses core.auth.verify_jwt directly.

    The connection is rejected (closed, never accepted) if that cookie is
    missing/invalid, or if the caller isn't a member of session_id — the
    same db.is_session_member check _require_member uses, inlined here
    rather than calling _require_member itself, since that raises
    HTTPException, which has no meaning for a handshake that was never
    accepted as an HTTP response in the first place.

    This is a pure push channel: the only thing it ever reads from the
    client is used to detect disconnection, not as a message of any kind —
    a client closing the tab (or the connection otherwise dropping) surfaces
    here as WebSocketDisconnect from receive_text(), which is exactly the
    cue needed to unregister the socket.
    """
    token = websocket.cookies.get("access_token")
    payload = auth_module.verify_jwt(token) if token else None
    if payload is None:
        await websocket.close(code=1008)
        return

    user_id = int(payload["sub"])
    if not db.is_session_member(session_id, user_id):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    _register_session_socket(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _unregister_session_socket(session_id, websocket)


@app.get("/health")
def health():
    return {"status": "ok", "config": {"provider": app_config.chat_agent.provider}}


# Serve the frontend — must be mounted AFTER all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")