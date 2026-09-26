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
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.chat_agent import ChatAgent
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
                    }
                    for item in bill.items
                ],
                "unassigned_items": [i.name for i in bill.unassigned_items()],
            }
            for bill in state.bills
        ],
        "finalized": state.finalized,
        "all_bills_ready": state.all_bills_ready(),
        "settlement": _compute_settlement(state) if state.finalized else [],
    }


def _compute_settlement(state: SessionState) -> list:
    """Thin wrapper over the shared balance math (core/settlement.py) — this
    used to duplicate that logic inline and had drifted out of sync with the
    calculate_split tool (missing qty_allocations handling entirely), so the
    state panel and the finalized report could disagree. Not anymore."""
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
    """
    settlements = _compute_settlement(agent.state) if agent.state.finalized else []
    db.save_session_state(session_id, agent.state, settlements, name=name)


def _run_chat_turn(
    session_id: str,
    user: dict,
    agent_message: str,
    speaker_name: str,
    persist_user_message: Callable[[], None],
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
    endpoints already had.

    Synchronous end-to-end — callers on an async path (upload_image) must
    run this via asyncio.to_thread so the lock (a plain threading.Lock)
    never blocks the event loop.
    """
    with _get_session_lock(session_id):
        agent = _get_agent(session_id, user)
        prev_bill_count = len(agent.state.bills)

        response = agent.chat(agent_message, speaker_name=speaker_name, speaker_user_id=user["id"])
        persist_user_message()
        db.add_chat_message(session_id, "agent", response)

        # Auto-rename session when first bill is added
        new_name: Optional[str] = None
        if len(agent.state.bills) > prev_bill_count:
            new_name = _session_name_from_agent(agent)

        _save_agent(session_id, agent, name=new_name)
        state = _serialize_state(agent.state, session_id)

    return response, state


def _session_name_from_agent(agent: ChatAgent) -> Optional[str]:
    """Return the first bill description if bills exist, else None."""
    if agent.state.bills:
        return agent.state.bills[0].description
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
    """Rename a session. Any member may rename — no admin requirement."""
    _require_member(session_id, user)
    updated = db.rename_session_by_id(session_id, req.name)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"name": req.name}


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
    for a session_id that no longer exists.
    """
    _require_admin(session_id, user)
    with _get_session_lock(session_id):
        deleted = db.delete_session_by_id(session_id)
        if deleted:
            checkpointer.delete_thread(session_id)
            sessions.pop(session_id, None)
    _drop_session_lock(session_id)

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


@app.post("/api/sessions/{session_id}/proposals/{proposal_id}/approve")
def approve_proposal(
    session_id: str,
    proposal_id: int,
    user: dict = Depends(get_current_user),
):
    """Approve a pending expense proposal, materializing it into bills/bill_items. Admin only."""
    _require_admin(session_id, user)
    _require_proposal_in_session(session_id, proposal_id)

    try:
        return db.decide_proposal(proposal_id, user["id"], "approved")
    except ValueError as e:
        # _require_proposal_in_session already confirmed the proposal exists
        # in this session, so a ValueError here means decide_proposal's own
        # `WHERE status = 'pending'` guard rejected it — i.e. someone else
        # already decided it between that check and this call. That's a
        # conflict with the proposal's current state, not a missing
        # resource, so 409 rather than 404.
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/sessions/{session_id}/proposals/{proposal_id}/reject")
def reject_proposal(
    session_id: str,
    proposal_id: int,
    user: dict = Depends(get_current_user),
):
    """Reject a pending expense proposal. Admin only."""
    _require_admin(session_id, user)
    _require_proposal_in_session(session_id, proposal_id)

    try:
        return db.decide_proposal(proposal_id, user["id"], "rejected")
    except ValueError as e:
        # See the matching comment in approve_proposal — already-decided is
        # a conflict (409), not a missing resource (404).
        raise HTTPException(status_code=409, detail=str(e))


# ---------- Chat Endpoints ----------

@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(
    session_id: str,
    req: ChatRequest,
    user: dict = Depends(get_current_user),
):
    speaker_name = user["name"] or user["email"]
    response, state = _run_chat_turn(
        session_id,
        user,
        req.message,
        speaker_name,
        persist_user_message=lambda: db.add_chat_message(
            session_id, "user", req.message, user_id=user["id"]
        ),
    )
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

    speaker_name = user["name"] or user["email"]

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
    return ChatResponse(response=response, state=state)


@app.get("/health")
def health():
    return {"status": "ok", "config": {"provider": app_config.chat_agent.provider}}


# Serve the frontend — must be mounted AFTER all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")