# SplitPro

Conversational AI-powered bill splitting app. Chat with an AI agent to parse bills, assign items to people, and generate a minimal settlement report. Upload a receipt photo and the vision model extracts the line items automatically.

## Features

- **Conversational interface** — paste bill text or chat naturally; the agent handles parsing and assignment
- **Receipt image upload** — vision model (Google Gemini by default) extracts items from a photo
- **Multi-bill sessions** — add multiple bills in one session; all are settled together
- **Item-level assignment** — assign each item to one or more people, with quantity-based splits (e.g. "Alice had 1 of 4 burgers")
- **Tax & tip distribution** — automatically split proportionally to each person's subtotal
- **Minimal settlements** — output is the smallest number of transactions to settle all debts
- **Multi-provider LLM** — OpenAI, Anthropic, or Google; configurable per tool
- **Web UI + REST API** — browser chat interface served by the same FastAPI server
- **CLI mode** — run the agent entirely in the terminal

---

## Project Structure

```
splitpro/
├── server.py               # FastAPI app — REST API + serves frontend
├── main.py                 # CLI entry point
├── config.py               # Dataclass configs + YAML loader
│
├── config/
│   └── defaults.yaml       # Default provider/model/temperature settings
│
├── agents/
│   ├── chat_agent.py       # Conversational agent with LangChain tool calling
│   ├── image_analyzer.py   # Vision model — detects bills and extracts text
│   ├── llm_factory.py      # Builds LangChain LLMs from config (OpenAI/Anthropic/Google)
│   └── __init__.py
│
├── core/
│   ├── session_state.py    # Data models: LineItem, ParsedBill, SessionState
│   ├── settlement.py       # Balance calculation + settlement minimization
│   └── __init__.py
│
├── frontend/
│   ├── index.html          # Single-page chat UI
│   ├── app.js              # Frontend logic (session management, chat, image upload)
│   └── style.css           # Styles
│
├── Dockerfile
├── pyproject.toml
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

Requires Python 3.9+.

### 2. Configure API keys

Create a `.env` file in the project root with the keys for the providers you want to use:

```env
ANTHROPIC_API_KEY=your_anthropic_key   # default chat provider
GOOGLE_API_KEY=your_google_key         # default image analyzer provider
OPENAI_API_KEY=your_openai_key         # optional, if using OpenAI
```

Only the keys for the providers configured in `config/defaults.yaml` are required.

### 3. (Optional) Edit provider/model settings

`config/defaults.yaml` controls which model is used for each tool:

```yaml
chat_agent:
  provider: anthropic                  # openai | anthropic | google
  model: claude-haiku-4-5-20251001     # omit to use provider default
  temperature: 0.2
  max_iterations: 10

image_analyzer:
  provider: google                     # must be a vision-capable model
  model: gemini-2.5-flash
  temperature: 0.1

server:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "*"
```

---

## Running

### Web app (FastAPI + browser UI)

```bash
uvicorn server:app --reload
```

Open `http://localhost:8000` in your browser.

### CLI

```bash
python main.py                                    # uses config defaults
python main.py --provider openai                  # override provider
python main.py --provider anthropic --model claude-sonnet-4-6
```

CLI commands during a session:
- `summary` — print current session state (bills, assignments, balances)
- `quit` / `exit` — end the session

---

## Docker

```bash
docker build -t splitpro .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  splitpro
```

---

## REST API

All state is scoped to a session. Sessions are held in memory; restarting the server clears them.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create a new session. Returns `session_id`. |
| `POST` | `/sessions/{id}/chat` | Send a chat message. Returns agent response + state. |
| `POST` | `/sessions/{id}/image` | Upload a receipt image (JPEG/PNG/WEBP). Returns agent response + state. |
| `GET`  | `/sessions/{id}/state` | Get current session state (bills, assignments, settlement). |
| `DELETE` | `/sessions/{id}` | Delete a session. |
| `GET`  | `/health` | Health check + active config. |

### Create session

```bash
curl -X POST http://localhost:8000/sessions
# {"session_id": "abc-123", "provider": "anthropic"}
```

Override provider/model per session:

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "model": "gpt-4o"}'
```

### Chat

```bash
curl -X POST http://localhost:8000/sessions/abc-123/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Burger $12, Fries $4, Coke $3. Tax $1.90"}'
```

### Upload image

```bash
curl -X POST http://localhost:8000/sessions/abc-123/image \
  -F "file=@receipt.jpg"
```

---

## How it works

1. **Session created** — a `ChatAgent` and empty `SessionState` are instantiated
2. **User sends a message or image** — the agent receives it and decides which tools to call
3. **Agent tools** (called automatically by the LLM):
   - `add_bill` — parse bill text and store line items
   - `set_participants` — record who is splitting
   - `assign_items` — assign items to specific people (supports qty-based splits)
   - `mark_items_unassigned` — split an item equally among everyone
   - `set_payer` — record who paid each bill
   - `calculate_split` — compute balances and generate the settlement report
4. **Settlement** — balances are computed across all bills; a greedy algorithm minimizes the number of transactions
