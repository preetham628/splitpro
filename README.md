# SplitPro

Conversational AI-powered bill splitting app. Sign in, chat with an AI agent to parse bills,
assign items to people, and generate a minimal settlement report. Upload a receipt photo and a
vision model extracts the line items automatically. Sessions persist across restarts.

For how the system actually works internally (auth flow, the agent's tool-calling loop, the
settlement algorithm, full API reference), see **[docs/index.html](docs/index.html)** — open it
directly in a browser.

---

## Features

- **Conversational interface** — paste bill text or chat naturally; the agent handles parsing and assignment
- **Receipt image upload** — vision model (Google Gemini by default) extracts items from a photo
- **Multi-bill sessions** — add multiple bills in one session; all are settled together
- **Item-level assignment** — assign each item to one or more people, with quantity-based splits (e.g. "Alice had 1 of 4 burgers")
- **Tax & tip distribution** — automatically split proportionally to each person's subtotal
- **Minimal settlements** — output is the smallest number of transactions to settle all debts
- **Multi-provider LLM** — OpenAI, Anthropic, or Google; configurable per tool
- **Google sign-in + persistence** — sessions are tied to your Google account and saved to SQLite, so chat history survives a server restart
- **Web UI + REST API** — browser chat interface served by the same FastAPI server
- **CLI mode** — run the agent entirely in the terminal, no sign-in required

---

## Project Structure

```
splitpro/
├── server.py               # FastAPI app — auth, sessions, chat/image API, serves frontend
├── main.py                 # CLI entry point (no auth, no server)
├── config.py                # Dataclass configs + YAML loader
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
│   ├── auth.py              # Google OAuth + JWT helpers
│   ├── database.py          # SQLite persistence (users, chat_sessions)
│   └── __init__.py
│
├── frontend/
│   ├── index.html          # Single-page chat UI
│   ├── app.js               # Frontend logic (session management, chat, image upload)
│   └── style.css            # Styles
│
├── docs/                    # How-it-works documentation — open docs/index.html in a browser
│   ├── index.html
│   ├── architecture.html
│   ├── api.html
│   └── assets/
│
├── .env.example              # Template for the .env file described below
├── Dockerfile
├── docker-compose.yml        # Local build + run, with volume-backed SQLite persistence
├── requirements.txt
└── .gitignore
```

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+.

### 2. Configure environment variables

```bash
cp .env.example .env
```

Then fill in `.env`. What you need depends on how you're running SplitPro:

| Running | Required |
|---|---|
| `python main.py` (CLI) | An API key for the LLM provider you use — nothing else. |
| `uvicorn server:app` (web app) | An LLM API key **and** `JWT_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`. |

```env
# LLM providers — only the ones matching config/defaults.yaml are required
ANTHROPIC_API_KEY=your_anthropic_key   # default chat provider
GOOGLE_API_KEY=your_google_key         # default image analyzer provider
OPENAI_API_KEY=your_openai_key         # optional, if using OpenAI

# Required to run the web app — see step 3 below
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# Required to run the web app
JWT_SECRET=...
```

> **The web server will not start** without `JWT_SECRET`, `GOOGLE_CLIENT_ID`, and
> `GOOGLE_CLIENT_SECRET` — it validates them at import time and raises immediately if any are
> missing. The CLI doesn't touch `server.py` at all, so it's unaffected.

Generate a JWT secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Set up Google OAuth (web app only)

1. In the [Google Cloud Console](https://console.cloud.google.com/apis/credentials), create an OAuth client ID of type **Web application**.
2. Add an **Authorized redirect URI** that exactly matches `GOOGLE_REDIRECT_URI` above (default: `http://localhost:8000/auth/google/callback`).
3. Copy the generated Client ID and Client Secret into `.env`.

### 4. (Optional) Edit provider/model settings

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
  cors_origins:
    - "*"
```

> `auth.db_path` (the SQLite file location) is deliberately *not* set here — it's controlled by
> the `DB_PATH` env var instead (defaults to `splitpro.db` if unset), so Docker/compose can point
> it at a mounted volume without editing this file. See [Docker](#docker) below.

---

## Running

### Web app (FastAPI + browser UI)

```bash
uvicorn server:app --reload
```

Open `http://localhost:8000` in your browser and sign in with Google.

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

### docker compose (recommended)

Builds the image, wires up your `.env`, and persists the SQLite database in a named volume so
it survives container restarts/rebuilds.

```bash
cp .env.example .env   # fill it in first — see "Configure environment variables" above
docker compose up -d --build
```

Open `http://localhost:8000`. Useful follow-ups:

```bash
docker compose ps        # includes container health (from GET /health)
docker compose logs -f   # tail logs
docker compose down      # stop, keep the splitpro_data volume (DB persists)
docker compose down -v   # stop AND delete the volume (DB is wiped)
```

`docker-compose.yml` pins `DB_PATH=data/splitpro.db` and mounts a named volume at `/app/data`,
so the database survives `docker compose down` / `up` and image rebuilds — only `-v` deletes it.

### Plain `docker run` (manual, no compose)

```bash
docker build -t splitpro .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  -e JWT_SECRET=your_jwt_secret \
  -e GOOGLE_CLIENT_ID=your_client_id \
  -e GOOGLE_CLIENT_SECRET=your_client_secret \
  -e DB_PATH=data/splitpro.db \
  -v splitpro_data:/app/data \
  splitpro
```

Without `-v splitpro_data:/app/data`, the SQLite database lives only inside the container's
writable layer and is lost when the container is removed.

---

## Learn more

- **[docs/index.html](docs/index.html)** — start here; includes a live preview of the frontend
- **[docs/architecture.html](docs/architecture.html)** — auth flow, the agent's tool-calling loop, session persistence, the settlement algorithm
- **[docs/api.html](docs/api.html)** — full REST API reference with examples

The running server also exposes interactive OpenAPI docs at `/docs` and `/redoc`.
