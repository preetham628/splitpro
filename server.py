"""
SplitPro FastAPI Server
Run with: uvicorn server:app --reload
"""

from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.chat_agent import ChatAgent
from config import LLMConfig, load_config
from core.session_state import SessionState

load_dotenv()

app_config = load_config()

app = FastAPI(title="SplitPro API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> ChatAgent
sessions: dict[str, ChatAgent] = {}


# ---------- Request / Response models ----------

class SessionRequest(BaseModel):
    provider: Optional[str] = None   # overrides config; "openai" or "bedrock"
    model: Optional[str] = None      # overrides config model


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    state: dict


# ---------- Helpers ----------

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
    }


def _get_agent(session_id: str) -> ChatAgent:
    agent = sessions.get(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return agent


# ---------- Endpoints ----------

@app.post("/sessions", status_code=201)
def create_session(req: SessionRequest = SessionRequest()):
    """
    Start a new bill-splitting session.
    Provider and model default to values in config/defaults.yaml.
    Pass provider/model in the request body to override per-session.
    """
    llm_config = LLMConfig(
        provider=req.provider or app_config.llm.provider,
        model=req.model or app_config.llm.model,
        temperature=app_config.llm.temperature,
    )
    session_id = str(uuid4())
    sessions[session_id] = ChatAgent(
        llm_config=llm_config,
        agent_config=app_config.agent,
    )
    return {"session_id": session_id, "provider": llm_config.provider}


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, req: ChatRequest):
    """
    Send a message to the agent.
    Returns the agent's response and the current session state.
    """
    agent = _get_agent(session_id)
    response = agent.chat(req.message)
    return ChatResponse(response=response, state=_serialize_state(agent.state))


@app.get("/sessions/{session_id}/state")
def get_state(session_id: str):
    """Get the current session state (bills, participants, assignments)."""
    agent = _get_agent(session_id)
    return _serialize_state(agent.state)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """End a session and free its memory."""
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "config": {"provider": app_config.llm.provider}}
