-- SplitPro database schema.
--
-- Canonical source of truth — core/database.py reads this file at runtime
-- rather than embedding its own copy, so there's exactly one place the
-- schema is defined.
--
-- Idempotent: every statement is CREATE TABLE IF NOT EXISTS, so running
-- this against an existing database is always safe and never touches
-- existing data. That makes it safe to run unconditionally before every
-- `docker compose up` / `uvicorn server:app`, not just the first time.
--
-- One exception: chat_sessions used to also have message_history and
-- session_state TEXT columns (JSON blobs). Both are obsolete — conversation
-- history now lives in LangGraph's own checkpoint tables (core/checkpointer.py),
-- and session_state's data now lives in session_participants/bills/bill_items/
-- settlements below. CREATE TABLE IF NOT EXISTS can't retroactively drop
-- columns from an already-existing table, so core/database.py's init_db()
-- runs a small one-time ALTER TABLE migration for those two columns (and to
-- add the new `finalized` column) — see _migrate_chat_sessions() there.
--
-- Same story for chat_messages.user_id and bills.approved_by/approved_at —
-- added after those tables already existed in deployed databases, so they're
-- added via ALTER TABLE ADD COLUMN in _migrate_membership() rather than being
-- retrofitted into the CREATE TABLE statements below.
--
-- Usage (local, requires the sqlite3 CLI — ships with macOS/most Linux):
--   sqlite3 splitpro.db < schema.sql
--
-- Usage (anywhere with Python, no sqlite3 CLI required — e.g. inside the
-- Docker image, which doesn't include the sqlite3 CLI binary):
--   python -m core.database splitpro.db

CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    google_id  TEXT UNIQUE NOT NULL,
    email      TEXT UNIQUE NOT NULL,
    name       TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id               TEXT PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(id),
    name             TEXT NOT NULL DEFAULT 'New Session',
    finalized        INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- UI-facing chat transcript (one row per user/agent turn). Distinct from
-- LangGraph's checkpoint tables, which hold the agent's raw internal
-- memory (system/tool-call plumbing) — not directly renderable in a chat
-- UI. Images are stored inline since each belongs to exactly one message.
CREATE TABLE IF NOT EXISTS chat_messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'agent')),
    content          TEXT NOT NULL DEFAULT '',
    image_base64     TEXT,
    image_media_type TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);

CREATE TABLE IF NOT EXISTS session_participants (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    UNIQUE(session_id, name)
);

CREATE TABLE IF NOT EXISTS bills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    bill_id     TEXT NOT NULL,   -- app-level id used by agent tools, e.g. "bill_1"
    description TEXT NOT NULL DEFAULT '',
    raw_text    TEXT NOT NULL DEFAULT '',
    tax         REAL NOT NULL DEFAULT 0,
    tip         REAL NOT NULL DEFAULT 0,
    paid_by     TEXT,
    UNIQUE(session_id, bill_id)
);

-- assigned_to / qty_allocations stay JSON columns rather than further
-- normalized tables — small, bounded, always read/written as a unit with
-- their item, never queried independently.
CREATE TABLE IF NOT EXISTS bill_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bill_id         INTEGER NOT NULL REFERENCES bills(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    price           REAL NOT NULL,
    qty             INTEGER NOT NULL DEFAULT 1,
    assigned_to     TEXT NOT NULL DEFAULT '[]',
    shared          INTEGER NOT NULL DEFAULT 0,
    unassigned      INTEGER NOT NULL DEFAULT 0,
    qty_allocations TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_bill_items_bill ON bill_items(bill_id);

-- Written only once a session is finalized (calculate_split); cleared and
-- rewritten alongside session_state on every save while finalized stays true.
CREATE TABLE IF NOT EXISTS settlements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    from_person TEXT NOT NULL,
    to_person   TEXT NOT NULL,
    amount      REAL NOT NULL
);

-- Real per-user membership on a session, distinct from chat_sessions.user_id
-- (the original creator). Every pre-existing session is backfilled with its
-- creator as an 'admin' member — see _migrate_membership() in
-- core/database.py.
CREATE TABLE IF NOT EXISTS session_members (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    role       TEXT NOT NULL CHECK (role IN ('admin', 'member')),
    joined_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, user_id)
);

-- AI-drafted expenses staged for admin approval before they count toward
-- bills/bill_items (and therefore balances). `payload` mirrors the shape of
-- a bill (bill_id, description, raw_text, items[], tax, tip, paid_by) as
-- JSON, since it hasn't been materialized into bills/bill_items yet.
CREATE TABLE IF NOT EXISTS expense_proposals (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    proposed_by        INTEGER NOT NULL REFERENCES users(id),
    status             TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    supersedes_bill_id INTEGER REFERENCES bills(id) ON DELETE SET NULL,
    payload            TEXT NOT NULL,
    decided_by         INTEGER REFERENCES users(id),
    decided_at         TIMESTAMP,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_expense_proposals_session ON expense_proposals(session_id, status);
