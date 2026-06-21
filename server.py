"""
SplitPro FastAPI Server
Run with: uvicorn server:app --reload
"""

import json
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from agents.chat_agent import ChatAgent, derive_display_messages
from agents.image_analyzer import ImageAnalyzer
from config import ChatAgentConfig, load_config
from core.session_state import SessionState
from core.settlement import Settlement

load_dotenv()

app_config = load_config()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database.init_db()
    yield


app = FastAPI(title="SplitPro API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory agent cache: chat_id -> ChatAgent (loaded from DB on cache miss)
agent_cache: dict[str, ChatAgent] = {}


# ---------- Request / Response models ----------

class CreateChatRequest(BaseModel):
    name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class RenameChatRequest(BaseModel):
    name: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    state: dict


# ---------- Helpers ----------

def _build_config(provider: Optional[str] = None, model: Optional[str] = None) -> ChatAgentConfig:
    return ChatAgentConfig(
        provider=provider or app_config.chat_agent.provider,
        model=model or app_config.chat_agent.model,
        temperature=app_config.chat_agent.temperature,
        max_iterations=app_config.chat_agent.max_iterations,
    )


def _serialize_state(state: SessionState) -> dict:
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


def _get_or_load_agent(chat_id: str) -> ChatAgent:
    if chat_id in agent_cache:
        return agent_cache[chat_id]
    row = database.get_chat(chat_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")
    config = _build_config(provider=row["provider"], model=row["model"])
    agent = ChatAgent.from_dict(
        {"state": json.loads(row["state_json"]), "history": json.loads(row["history_json"])},
        config=config,
    )
    agent_cache[chat_id] = agent
    return agent


def _persist_agent(chat_id: str, agent: ChatAgent) -> None:
    d = agent.to_dict()
    database.update_chat(
        chat_id,
        state_json=json.dumps(d["state"]),
        history_json=json.dumps(d["history"]),
    )


# ---------- Chat endpoints ----------

@app.get("/chats")
def list_chats():
    """List all chats ordered by most recently updated."""
    return database.list_chats()


@app.post("/chats", status_code=201)
def create_chat(req: CreateChatRequest = CreateChatRequest()):
    """
    Create a new chat session.
    Sends the opening greeting server-side and persists it immediately.
    """
    chat_id = str(uuid4())
    config = _build_config(provider=req.provider, model=req.model)
    agent = ChatAgent(config=config)

    agent.chat("Hello, I'm ready to help split some bills.")

    d = agent.to_dict()
    name = req.name or "New Chat"
    database.create_chat(
        chat_id=chat_id,
        name=name,
        provider=config.provider,
        model=config.model,
        state_json=json.dumps(d["state"]),
        history_json=json.dumps(d["history"]),
    )
    agent_cache[chat_id] = agent

    row = database.get_chat(chat_id)
    return {
        "id": chat_id,
        "name": name,
        "provider": config.provider,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": derive_display_messages(agent.message_history),
        "state": _serialize_state(agent.state),
    }


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str):
    """Load a chat's full message history and state for UI restoration."""
    agent = _get_or_load_agent(chat_id)
    row = database.get_chat(chat_id)
    return {
        "id": chat_id,
        "name": row["name"],
        "provider": row["provider"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": derive_display_messages(agent.message_history),
        "state": _serialize_state(agent.state),
    }


@app.post("/chats/{chat_id}/chat", response_model=ChatResponse)
def chat(chat_id: str, req: ChatRequest):
    """Send a message to the agent and persist the updated state."""
    agent = _get_or_load_agent(chat_id)
    response = agent.chat(req.message)
    _persist_agent(chat_id, agent)
    return ChatResponse(response=response, state=_serialize_state(agent.state))


@app.post("/chats/{chat_id}/image", response_model=ChatResponse)
async def upload_image(chat_id: str, file: UploadFile = File(...)):
    """
    Upload a bill image. The vision model extracts contents and feeds
    them to the agent automatically.
    """
    agent = _get_or_load_agent(chat_id)

    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    analyzer = ImageAnalyzer(app_config.image_analyzer)
    result = analyzer.analyze(image_bytes, media_type=content_type)

    if result.is_bill:
        agent_message = f"I've scanned a receipt image. Here are the contents:\n\n{result.bill_text}"
    else:
        agent_message = (
            f"The user uploaded an image. It is NOT a bill or receipt. "
            f"Description: {result.description}. "
            f"Tell the user what you see and ask them to upload a receipt image instead."
        )

    response = agent.chat(agent_message)
    _persist_agent(chat_id, agent)
    return ChatResponse(response=response, state=_serialize_state(agent.state))


@app.get("/chats/{chat_id}/state")
def get_state(chat_id: str):
    """Get the current session state (bills, participants, assignments)."""
    agent = _get_or_load_agent(chat_id)
    return _serialize_state(agent.state)


@app.patch("/chats/{chat_id}")
def rename_chat(chat_id: str, req: RenameChatRequest):
    """Rename a chat."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name must not be empty.")
    row = database.get_chat(chat_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found")
    database.rename_chat(chat_id, name)
    return {"id": chat_id, "name": name}


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    """Delete a chat and remove it from the in-memory cache."""
    agent_cache.pop(chat_id, None)
    database.delete_chat(chat_id)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "config": {"provider": app_config.chat_agent.provider}}


# Serve the frontend — must be mounted AFTER all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
