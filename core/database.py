"""
SQLite persistence layer for SplitPro.

Tables (see schema.sql for the authoritative definitions):
  users                 — Google OAuth users
  chat_sessions         — Named bill-splitting sessions
  chat_messages         — UI-facing chat transcript (role, content, optional image)
  session_participants  — participant names per session
  bills / bill_items    — parsed bills, normalized (replaces the old session_state JSON blob)
  settlements           — computed once a session is finalized
  session_members       — per-user admin/member role on a session
  expense_proposals     — AI-drafted expenses staged for admin approval

Conversation memory for the agent itself (system/tool-call plumbing) lives
separately in LangGraph's own checkpoint tables — see core/checkpointer.py.

The database file itself is never auto-created by the running app — only
create_db() (run explicitly, once, via `python -m core.database <path>`)
creates it. init_db() (called at server startup) requires the file to
already exist and raises otherwise. This is deliberate: a typo'd DB_PATH
should fail loudly instead of silently standing up a fresh, empty database,
and it matches how a future cloud database will need to be provisioned
ahead of time rather than materialized on first connect.
"""

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.session_state import SessionState

_db_path: Optional[str] = None

# schema.sql (repo root) is the single source of truth for the schema — both
# the sqlite3 CLI (`sqlite3 $DB_PATH < schema.sql`) and this module read the
# same file, so there's no separate copy of the CREATE TABLE statements to
# drift out of sync.
_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema.sql")


def _load_schema() -> str:
    with open(_SCHEMA_PATH) as f:
        return f.read()


def init_db(db_path: str) -> None:
    """Point the module at an existing database and ensure its schema is current.

    Call once at server startup. Requires db_path to be set AND the file to
    already exist — does not create either. Use create_db() to provision a
    fresh database first.
    """
    if not db_path:
        raise ValueError(
            "DB_PATH env var must be set (e.g. DB_PATH=splitpro.db for local "
            "SQLite). There is no default — set it explicitly in .env."
        )

    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database file not found at '{db_path}'. The app does not "
            f"auto-create it — provision it once with:\n"
            f"    ./scripts/init_db.sh\n"
            f"or: python -m core.database {db_path}"
        )

    global _db_path
    _db_path = db_path

    with _connect() as conn:
        conn.executescript(_load_schema())
        _migrate_chat_sessions(conn)
        _migrate_membership(conn)


def create_db(db_path: str) -> None:
    """Explicitly provision a new database file with the current schema.

    The one place this module is allowed to create the underlying file.
    Run by hand (or once in CI/deploy tooling) — never called from init_db()
    or anywhere in the request path.
    """
    if not db_path:
        raise ValueError("db_path is required.")

    if os.path.exists(db_path):
        raise FileExistsError(f"'{db_path}' already exists — refusing to overwrite it.")

    dirname = os.path.dirname(db_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_load_schema())
        conn.commit()
    finally:
        conn.close()


def _migrate_chat_sessions(conn: sqlite3.Connection) -> None:
    """One-time column migration for pre-existing chat_sessions tables.

    schema.sql's CREATE TABLE IF NOT EXISTS can't retroactively alter a
    table that already exists — this handles the one shape change needed:
    dropping the old message_history/session_state JSON-blob columns (their
    data now lives in LangGraph's checkpoint tables and the normalized
    bills/session_participants/settlements tables, respectively) and adding
    `finalized`. Safe to call on every startup — checks columns first.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(chat_sessions)")}

    if "session_state" in existing:
        conn.execute("ALTER TABLE chat_sessions DROP COLUMN session_state")
    if "message_history" in existing:
        conn.execute("ALTER TABLE chat_sessions DROP COLUMN message_history")
    if "finalized" not in existing:
        conn.execute("ALTER TABLE chat_sessions ADD COLUMN finalized INTEGER NOT NULL DEFAULT 0")


def _migrate_membership(conn: sqlite3.Connection) -> None:
    """One-time migration for multi-user membership + expense proposals.

    Adds chat_messages.user_id (nullable — null for role='agent' rows) and
    bills.approved_by/approved_at (both nullable, populated once an expense
    proposal referencing that bill is approved) — same ALTER TABLE pattern
    as _migrate_chat_sessions(), since CREATE TABLE IF NOT EXISTS can't add
    columns to a table that already exists.

    Then backfills session_members: every pre-existing chat_sessions row
    that has no session_members row yet gets its original owner
    (chat_sessions.user_id) inserted as an 'admin' member, so existing
    single-owner sessions keep working unchanged under the new membership
    model. Safe to call on every startup — checks columns/rows first.
    """
    chat_messages_cols = {row["name"] for row in conn.execute("PRAGMA table_info(chat_messages)")}
    if "user_id" not in chat_messages_cols:
        conn.execute("ALTER TABLE chat_messages ADD COLUMN user_id INTEGER REFERENCES users(id)")

    bills_cols = {row["name"] for row in conn.execute("PRAGMA table_info(bills)")}
    if "approved_by" not in bills_cols:
        conn.execute("ALTER TABLE bills ADD COLUMN approved_by INTEGER REFERENCES users(id)")
    if "approved_at" not in bills_cols:
        conn.execute("ALTER TABLE bills ADD COLUMN approved_at TIMESTAMP")

    conn.execute("""
        INSERT INTO session_members (session_id, user_id, role)
        SELECT cs.id, cs.user_id, 'admin'
        FROM chat_sessions cs
        WHERE NOT EXISTS (
            SELECT 1 FROM session_members sm WHERE sm.session_id = cs.id
        )
    """)


@contextmanager
def _connect():
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------- Users ----------

def upsert_user(google_id: str, email: str, name: str, avatar_url: str) -> dict:
    """Insert or update a user by google_id. Returns the user row as a dict."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO users (google_id, email, name, avatar_url)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(google_id) DO UPDATE SET
                email      = excluded.email,
                name       = excluded.name,
                avatar_url = excluded.avatar_url
        """, (google_id, email, name, avatar_url))
        row = conn.execute(
            "SELECT * FROM users WHERE google_id = ?", (google_id,)
        ).fetchone()
        return dict(row)


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------- Sessions ----------

def list_sessions(user_id: int) -> list[dict]:
    """Return all sessions for a user, newest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT id, name, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
        """, (user_id,)).fetchall()
        return [dict(r) for r in rows]


def create_session(session_id: str, user_id: int, name: str = "New Session") -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO chat_sessions (id, user_id, name)
            VALUES (?, ?, ?)
        """, (session_id, user_id, name))


def save_session_state(
    session_id: str,
    state: "SessionState",
    settlements: list[dict],
    name: Optional[str] = None,
) -> None:
    """Write-through save of bills/participants/settlements after each turn.

    Delete+reinsert rather than diffing — SessionState is fully mutated in
    place each turn, so this is simpler and correct at this data size, and
    matches the app's existing write-through-every-turn persistence style.
    All in one transaction (single _connect() context) for atomicity.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM session_participants WHERE session_id = ?", (session_id,))
        conn.executemany(
            "INSERT INTO session_participants (session_id, name) VALUES (?, ?)",
            [(session_id, p) for p in state.participants],
        )

        # Deleting bills cascades to bill_items (ON DELETE CASCADE, foreign_keys=ON).
        conn.execute("DELETE FROM bills WHERE session_id = ?", (session_id,))
        for bill in state.bills:
            cursor = conn.execute("""
                INSERT INTO bills (session_id, bill_id, description, raw_text, tax, tip, paid_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, bill.bill_id, bill.description, bill.raw_text,
                  bill.tax, bill.tip, bill.paid_by))
            bill_row_id = cursor.lastrowid
            for item in bill.items:
                conn.execute("""
                    INSERT INTO bill_items
                        (bill_id, name, price, qty, assigned_to, shared, unassigned, qty_allocations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (bill_row_id, item.name, item.price, item.qty,
                      json.dumps(item.assigned_to), int(item.shared), int(item.unassigned),
                      json.dumps(item.qty_allocations)))

        conn.execute("DELETE FROM settlements WHERE session_id = ?", (session_id,))
        conn.executemany(
            "INSERT INTO settlements (session_id, from_person, to_person, amount) VALUES (?, ?, ?, ?)",
            [(session_id, s["from"], s["to"], s["amount"]) for s in settlements],
        )

        if name is not None:
            conn.execute("""
                UPDATE chat_sessions SET finalized = ?, name = ?, updated_at = ? WHERE id = ?
            """, (int(state.finalized), name, now, session_id))
        else:
            conn.execute("""
                UPDATE chat_sessions SET finalized = ?, updated_at = ? WHERE id = ?
            """, (int(state.finalized), now, session_id))


def load_session_state(session_id: str) -> dict:
    """Reconstruct a dict shaped exactly like SessionState.from_dict() expects."""
    with _connect() as conn:
        participants = [
            r["name"] for r in conn.execute(
                "SELECT name FROM session_participants WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]

        session_row = conn.execute(
            "SELECT finalized FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        finalized = bool(session_row["finalized"]) if session_row else False

        bills = []
        for b in conn.execute(
            "SELECT * FROM bills WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall():
            items = [
                {
                    "name": i["name"],
                    "price": i["price"],
                    "qty": i["qty"],
                    "assigned_to": json.loads(i["assigned_to"]),
                    "shared": bool(i["shared"]),
                    "unassigned": bool(i["unassigned"]),
                    "qty_allocations": json.loads(i["qty_allocations"]),
                }
                for i in conn.execute(
                    "SELECT * FROM bill_items WHERE bill_id = ? ORDER BY id", (b["id"],)
                ).fetchall()
            ]
            bills.append({
                "bill_id": b["bill_id"],
                "description": b["description"],
                "raw_text": b["raw_text"],
                "tax": b["tax"],
                "tip": b["tip"],
                "paid_by": b["paid_by"],
                "items": items,
            })

    return {"participants": participants, "bills": bills, "finalized": finalized}


# ---------- Chat transcript (UI-facing) ----------

def add_chat_message(
    session_id: str,
    role: str,
    content: str,
    image_base64: Optional[str] = None,
    image_media_type: Optional[str] = None,
) -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO chat_messages (session_id, role, content, image_base64, image_media_type)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, image_base64, image_media_type))


def list_chat_messages(session_id: str) -> list[dict]:
    """Return the chat transcript for a session, oldest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT role, content, image_base64, image_media_type, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]


def rename_session(session_id: str, user_id: int, name: str) -> bool:
    """Rename a session. Returns True if a row was updated."""
    with _connect() as conn:
        cursor = conn.execute("""
            UPDATE chat_sessions SET name = ?
            WHERE id = ? AND user_id = ?
        """, (name, session_id, user_id))
        return cursor.rowcount > 0


def delete_session(session_id: str, user_id: int) -> bool:
    """Delete a session owned by user_id. Returns True if deleted."""
    with _connect() as conn:
        cursor = conn.execute("""
            DELETE FROM chat_sessions WHERE id = ? AND user_id = ?
        """, (session_id, user_id))
        return cursor.rowcount > 0


def session_belongs_to_user(session_id: str, user_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        return row is not None


def get_user_by_email(email: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None


# ---------- Session membership ----------

def add_session_member(session_id: str, user_id: int, role: str = "member") -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT INTO session_members (session_id, user_id, role)
            VALUES (?, ?, ?)
        """, (session_id, user_id, role))


def list_session_members(session_id: str) -> list[dict]:
    """Return this session's members, joined against users, oldest first."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT u.id AS user_id, u.name, u.email, u.avatar_url, sm.role
            FROM session_members sm
            JOIN users u ON u.id = sm.user_id
            WHERE sm.session_id = ?
            ORDER BY sm.id
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]


def is_session_member(session_id: str, user_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM session_members WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        return row is not None


def is_session_admin(session_id: str, user_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM session_members WHERE session_id = ? AND user_id = ? AND role = 'admin'",
            (session_id, user_id),
        ).fetchone()
        return row is not None


def count_admins(session_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM session_members WHERE session_id = ? AND role = 'admin'",
            (session_id,),
        ).fetchone()
        return row["c"]


def _member_role(session_id: str, user_id: int, role: str) -> bool:
    """Change an existing member's role. Returns False if no row matched."""
    with _connect() as conn:
        cursor = conn.execute("""
            UPDATE session_members SET role = ?
            WHERE session_id = ? AND user_id = ?
        """, (role, session_id, user_id))
        return cursor.rowcount > 0


def remove_session_member(session_id: str, user_id: int) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM session_members WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        return cursor.rowcount > 0


# ---------- Expense proposals ----------

def create_proposal(
    session_id: str,
    proposed_by: int,
    payload: dict,
    supersedes_bill_id: Optional[int] = None,
) -> int:
    """Stage an AI-drafted expense for admin approval. Returns the new proposal id."""
    with _connect() as conn:
        cursor = conn.execute("""
            INSERT INTO expense_proposals (session_id, proposed_by, supersedes_bill_id, payload)
            VALUES (?, ?, ?, ?)
        """, (session_id, proposed_by, supersedes_bill_id, json.dumps(payload)))
        return cursor.lastrowid


def update_proposal_payload(proposal_id: int, payload: dict) -> None:
    """Overwrite a proposal's payload. Only meaningful while status is 'pending' —
    that isn't enforced here; callers are responsible for checking status first.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute("""
            UPDATE expense_proposals SET payload = ?, updated_at = ? WHERE id = ?
        """, (json.dumps(payload), now, proposal_id))


def get_proposal(proposal_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM expense_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result


def list_pending_proposals(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT * FROM expense_proposals WHERE session_id = ? AND status = 'pending'
            ORDER BY id
        """, (session_id,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d["payload"])
            results.append(d)
        return results


def decide_proposal(proposal_id: int, decided_by: int, decision: str) -> dict:
    """Approve or reject a pending proposal. Returns the updated proposal row
    (payload json.loads'd), all inside one transaction.

    Rejection only touches the proposal row itself. Approval additionally
    materializes the payload into bills/bill_items: a fresh bill (plus its
    items) when supersedes_bill_id is None, or an in-place UPDATE of the
    superseded bill row with its bill_items deleted and reinserted from the
    payload — the same delete+reinsert style save_session_state() uses.
    """
    if decision not in ("approved", "rejected"):
        raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM expense_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        conn.execute("""
            UPDATE expense_proposals
            SET status = ?, decided_by = ?, decided_at = ?, updated_at = ?
            WHERE id = ?
        """, (decision, decided_by, now, now, proposal_id))

        if decision == "approved":
            payload = json.loads(row["payload"])
            supersedes_bill_id = row["supersedes_bill_id"]

            if supersedes_bill_id is None:
                cursor = conn.execute("""
                    INSERT INTO bills
                        (session_id, bill_id, description, raw_text, tax, tip, paid_by,
                         approved_by, approved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (row["session_id"], payload["bill_id"], payload.get("description", ""),
                      payload.get("raw_text", ""), payload.get("tax", 0), payload.get("tip", 0),
                      payload.get("paid_by"), decided_by, now))
                bill_row_id = cursor.lastrowid
            else:
                conn.execute("""
                    UPDATE bills
                    SET description = ?, raw_text = ?, tax = ?, tip = ?, paid_by = ?,
                        approved_by = ?, approved_at = ?
                    WHERE id = ?
                """, (payload.get("description", ""), payload.get("raw_text", ""),
                      payload.get("tax", 0), payload.get("tip", 0), payload.get("paid_by"),
                      decided_by, now, supersedes_bill_id))
                conn.execute("DELETE FROM bill_items WHERE bill_id = ?", (supersedes_bill_id,))
                bill_row_id = supersedes_bill_id

            for item in payload.get("items", []):
                conn.execute("""
                    INSERT INTO bill_items
                        (bill_id, name, price, qty, assigned_to, shared, unassigned, qty_allocations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (bill_row_id, item["name"], item["price"], item.get("qty", 1),
                      json.dumps(item.get("assigned_to", [])), int(item.get("shared", False)),
                      int(item.get("unassigned", False)), json.dumps(item.get("qty_allocations", {}))))

        result = dict(conn.execute(
            "SELECT * FROM expense_proposals WHERE id = ?", (proposal_id,)
        ).fetchone())
        result["payload"] = json.loads(result["payload"])
        return result


if __name__ == "__main__":
    # Path is optional — falls back to DB_PATH (from .env, then the
    # environment), the same variable the running app itself requires.
    from dotenv import load_dotenv
    load_dotenv()

    path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DB_PATH", "")
    if not path:
        print(
            "Usage: python -m core.database [db_path]\n"
            "No path given and DB_PATH is not set in .env or the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    create_db(path)
    print(f"Created {path}")
