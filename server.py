"""
SplitPro FastAPI Server
Run with: uvicorn server:app --reload
"""

import base64
import secrets
from typing import Literal, Optional
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


def _serialize_state(state: SessionState) -> dict:
    """Convert SessionState to a JSON-serializable dict for the UI."""
    return {
        "participants": state.participants,
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
    """Delete a session. Admin only."""
    _require_admin(session_id, user)
    deleted = db.delete_session_by_id(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    checkpointer.delete_thread(session_id)
    sessions.pop(session_id, None)
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
    rather than a duplicate-row error.
    """
    _require_admin(session_id, user)

    target = db.get_user_by_email(req.email)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="User not found — they must sign in at least once before being added",
        )

    if db.is_session_member(session_id, target["id"]):
        return JSONResponse(status_code=200, content={"ok": True, "already_member": True})

    db.add_session_member(session_id, target["id"], role="member")
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
        raise HTTPException(status_code=404, detail=str(e))


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
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Chat Endpoints ----------

@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(
    session_id: str,
    req: ChatRequest,
    user: dict = Depends(get_current_user),
):
    agent = _get_agent(session_id, user)
    prev_bill_count = len(agent.state.bills)

    response = agent.chat(req.message)
    db.add_chat_message(session_id, "user", req.message)
    db.add_chat_message(session_id, "agent", response)

    # Auto-rename session when first bill is added
    new_name: Optional[str] = None
    if len(agent.state.bills) > prev_bill_count:
        new_name = _session_name_from_agent(agent)

    _save_agent(session_id, agent, name=new_name)
    return ChatResponse(response=response, state=_serialize_state(agent.state))


@app.get("/sessions/{session_id}/state")
def get_state(session_id: str, user: dict = Depends(get_current_user)):
    agent = _get_agent(session_id, user)
    return _serialize_state(agent.state)


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, user: dict = Depends(get_current_user)):
    """Full chat transcript for replay when a session is reopened."""
    _require_member(session_id, user)
    return db.list_chat_messages(session_id)


@app.delete("/sessions/{session_id}")
def end_session(session_id: str, user: dict = Depends(get_current_user)):
    """Duplicate of DELETE /api/sessions/{session_id} (no /api prefix). Admin only."""
    _require_admin(session_id, user)
    db.delete_session_by_id(session_id)
    checkpointer.delete_thread(session_id)
    sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/sessions/{session_id}/image", response_model=ChatResponse)
async def upload_image(
    session_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    agent = _get_agent(session_id, user)

    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    analyzer = ImageAnalyzer(app_config.image_analyzer)
    result = analyzer.analyze(image_bytes, media_type=content_type)

    prev_bill_count = len(agent.state.bills)

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

    response = agent.chat(agent_message)
    db.add_chat_message(session_id, "user", "", image_base64=image_b64, image_media_type=content_type)
    db.add_chat_message(session_id, "agent", response)

    new_name: Optional[str] = None
    if len(agent.state.bills) > prev_bill_count:
        new_name = _session_name_from_agent(agent)

    _save_agent(session_id, agent, name=new_name)
    return ChatResponse(response=response, state=_serialize_state(agent.state))


@app.get("/health")
def health():
    return {"status": "ok", "config": {"provider": app_config.chat_agent.provider}}


# Serve the frontend — must be mounted AFTER all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")