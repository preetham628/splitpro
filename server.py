"""
SplitPro FastAPI Server
Run with: uvicorn server:app --reload
"""

import json
import secrets
from typing import Optional
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
    from collections import defaultdict
    global_balances: dict = defaultdict(float)
    for bill in state.bills:
        if not bill.paid_by:
            continue
        person_subtotal: dict = defaultdict(float)
        for item in bill.items:
            recipients = item.assigned_to if item.assigned_to else state.participants
            share = item.price / len(recipients) if recipients else 0
            for person in recipients:
                person_subtotal[person] += share
        bill_subtotal = sum(person_subtotal.values())
        combined_extra = bill.tax + bill.tip
        if combined_extra > 0 and bill_subtotal > 0:
            for person in list(person_subtotal):
                person_subtotal[person] += (person_subtotal[person] / bill_subtotal) * combined_extra
        bill_total = sum(person_subtotal.values())
        global_balances[bill.paid_by] += bill_total
        for person, amt in person_subtotal.items():
            global_balances[person] -= amt
    return Settlement.generate_settlements(dict(global_balances))


def _get_agent(session_id: str, user: dict) -> ChatAgent:
    """Return agent from cache or restore from DB. Validates ownership."""
    if not db.session_belongs_to_user(session_id, user["id"]):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if session_id in sessions:
        return sessions[session_id]

    # Cache miss — restore from DB
    row = db.load_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    config = _chat_config()
    try:
        saved = {
            "message_history": json.loads(row["message_history"]),
            "session_state": json.loads(row["session_state"]),
        }
        agent = ChatAgent.from_dict(saved, config=config)
    except Exception:
        agent = ChatAgent(config=config)

    sessions[session_id] = agent
    return agent


def _save_agent(session_id: str, agent: ChatAgent, name: Optional[str] = None) -> None:
    """Persist agent state to DB (write-through)."""
    d = agent.to_dict()
    db.save_session(
        session_id,
        message_history_json=json.dumps(d["message_history"]),
        session_state_json=json.dumps(d["session_state"]),
        name=name,
    )


def _session_name_from_agent(agent: ChatAgent) -> Optional[str]:
    """Return the first bill description if bills exist, else None."""
    if agent.state.bills:
        return agent.state.bills[0].description
    return None


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
    agent = ChatAgent(config=config)
    sessions[session_id] = agent

    db.create_session(session_id, user["id"], name="New Session")
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
    """Rename a session."""
    updated = db.rename_session(session_id, user["id"], req.name)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"name": req.name}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a session."""
    deleted = db.delete_session(session_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions.pop(session_id, None)
    return {"ok": True}


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


@app.delete("/sessions/{session_id}")
def end_session(session_id: str, user: dict = Depends(get_current_user)):
    db.delete_session(session_id, user["id"])
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
