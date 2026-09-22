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
├── scripts/
│   └── init_db.sh           # Provisions the SQLite file from schema.sql (see step 4 below)
│
├── schema.sql                # Canonical DB schema — core/database.py reads this file directly
├── .env.example              # Template for the .env file described below
├── Dockerfile
├── docker-compose.yml        # Local build + run, bind-mounts $DB_PATH into the container
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
| `uvicorn server:app` (web app) | An LLM API key **and** `JWT_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `DB_PATH`. |

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

# Required to run the web app — no default. Local SQLite file path today;
# will point at a cloud database connection string later.
DB_PATH=splitpro.db
```

> **The web server will not start** without `JWT_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`,
> or `DB_PATH` — auth vars are validated at import time, and `DB_PATH` is validated when the server
> starts accepting connections; all four raise immediately if missing. The CLI doesn't touch
> `server.py` at all, so it's unaffected by any of this.

Generate a JWT secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Set up Google OAuth (web app only)

1. In the [Google Cloud Console](https://console.cloud.google.com/apis/credentials), create an OAuth client ID of type **Web application**.
2. Add an **Authorized redirect URI** that exactly matches `GOOGLE_REDIRECT_URI` above (default: `http://localhost:8000/auth/google/callback`).
3. Copy the generated Client ID and Client Secret into `.env`.

### 4. Create the database (first time only, web app only)

The server does **not** auto-create the SQLite file — it requires the path in `DB_PATH` to
already exist and fails fast (`FileNotFoundError`) if it doesn't. This is deliberate: a typo'd
`DB_PATH` should error loudly, not silently stand up a fresh empty database. Create it once:

```bash
./scripts/init_db.sh
```

It reads `DB_PATH` from `.env` automatically (or `DB_PATH=other.db ./scripts/init_db.sh` to
override), runs `schema.sql` through the `sqlite3` CLI, and is safe to re-run — it no-ops if the
file already exists rather than touching it. `schema.sql` is the single source of truth for the
schema; `core/database.py` reads the same file, so there's nothing to keep in sync by hand.

### 5. (Optional) Edit provider/model settings

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
> the `DB_PATH` env var instead, which is **required** (no fallback — the server refuses to start
> without it), so Docker/compose can point it at a mounted volume without editing this file. See
> [Docker](#docker) below.

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

Builds the image, wires up your `.env`, and bind-mounts the exact SQLite file at `DB_PATH` into
the container — the same file your local `./scripts/init_db.sh` / `uvicorn` setup uses, nothing
else is mounted. **`DB_PATH` must be a relative path** (the default, `splitpro.db`, is fine) —
see the comment on `volumes:` in `docker-compose.yml` for why an absolute path breaks the mount.

```bash
cp .env.example .env   # fill it in first — see "Configure environment variables" above
./scripts/init_db.sh   # first time only — creates $DB_PATH on the host, e.g. ./splitpro.db
docker compose up -d --build
```

`init_db.sh` must run **before** `docker compose up` — the bind mount requires the file to
already exist (Docker silently creates an empty *directory* there otherwise, which then breaks
every DB read/write with a confusing error). It's a plain file sitting right in your repo root
afterward (gitignored), so you can also inspect it directly: `sqlite3 splitpro.db ".tables"`.

Open `http://localhost:8000`. Useful follow-ups:

```bash
docker compose ps        # includes container health (from GET /health)
docker compose logs -f   # tail logs
docker compose down      # stop the container — $DB_PATH is a host file, unaffected either way
```

### Plain `docker run` (manual, no compose)

```bash
docker build -t splitpro .
./scripts/init_db.sh   # first time only — same as above

docker run -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/splitpro.db:/app/splitpro.db" \
  splitpro
```

The bind-mount source/target must match `DB_PATH` in `.env` (`splitpro.db` by default) —
`docker-compose.yml` derives this automatically from `${DB_PATH}`; with plain `docker run` you
have to keep the `-v` flag's path in sync with `.env` yourself.

---

## Learn more

- **[docs/index.html](docs/index.html)** — start here; includes a live preview of the frontend
- **[docs/architecture.html](docs/architecture.html)** — auth flow, the agent's tool-calling loop, session persistence, the settlement algorithm
- **[docs/api.html](docs/api.html)** — full REST API reference with examples

The running server also exposes interactive OpenAPI docs at `/docs` and `/redoc`.
