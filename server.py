"""
SplitPro FastAPI Server
Run with: uvicorn server:app --reload
"""

from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.chat_agent import ChatAgent
from agents.image_analyzer import ImageAnalyzer
from config import ChatAgentConfig, load_config
from core.session_state import SessionState
from core.settlement import Settlement

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
    provider: Optional[str] = None    # overrides config; "openai" or "bedrock"
    model: Optional[str] = None       # overrides config chat model


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
        "settlement": _compute_settlement(state) if state.finalized else [],
    }


def _compute_settlement(state: SessionState) -> list:
    """Recompute simplified settlement transactions for the UI state panel."""
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
    chat_config = ChatAgentConfig(
        provider=req.provider or app_config.chat_agent.provider,
        model=req.model or app_config.chat_agent.model,
        temperature=app_config.chat_agent.temperature,
        max_iterations=app_config.chat_agent.max_iterations,
    )
    session_id = str(uuid4())
    sessions[session_id] = ChatAgent(config=chat_config)
    return {"session_id": session_id, "provider": chat_config.provider}


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


@app.post("/sessions/{session_id}/image", response_model=ChatResponse)
async def upload_image(session_id: str, file: UploadFile = File(...)):
    """
    Upload a bill image for the agent to analyze.

    - If the image is a receipt/bill, the agent extracts items and adds them
      to the session automatically, then asks about participants or assignments.
    - If the image is not a bill, the agent explains what it sees and asks
      the user to upload a receipt instead.

    Accepted formats: JPEG, PNG, WEBP, GIF (max ~5MB recommended).
    """
    agent = _get_agent(session_id)

    # Validate file type
    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    # Analyze the image with the vision model
    analyzer = ImageAnalyzer(app_config.image_analyzer)
    result = analyzer.analyze(image_bytes, media_type=content_type)

    if result.is_bill:
        # Feed the extracted bill text into the chat agent so it calls add_bill naturally
        agent_message = (
            f"I've scanned a receipt image. Here are the contents:\n\n{result.bill_text}"
        )
    else:
        # Not a bill — tell the agent what was seen so it can respond in context
        agent_message = (
            f"The user uploaded an image. It is NOT a bill or receipt. "
            f"Description: {result.description}. "
            f"Tell the user what you see and ask them to upload a receipt image instead."
        )

    response = agent.chat(agent_message)
    return ChatResponse(response=response, state=_serialize_state(agent.state))


@app.get("/health")
def health():
    return {"status": "ok", "config": {"provider": app_config.chat_agent.provider}}


# Serve the frontend — must be mounted AFTER all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
