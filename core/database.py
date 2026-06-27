"""
SQLite persistence layer for SplitPro.

Tables:
  users        — Google OAuth users
  chat_sessions — Named bill-splitting sessions, state stored as JSON blobs
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

_db_path: str = "splitpro.db"


def init_db(db_path: str = "splitpro.db") -> None:
    """Create tables if they don't exist. Call once at server startup."""
    global _db_path
    _db_path = db_path

    with _connect() as conn:
        conn.executescript("""
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
                message_history  TEXT NOT NULL DEFAULT '[]',
                session_state    TEXT NOT NULL DEFAULT '{}',
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
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


def load_session(session_id: str) -> Optional[dict]:
    """Return the raw JSON blobs for message_history and session_state, or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def save_session(
    session_id: str,
    message_history_json: str,
    session_state_json: str,
    name: Optional[str] = None,
) -> None:
    """Write-through save after each chat/image turn."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        if name is not None:
            conn.execute("""
                UPDATE chat_sessions
                SET message_history = ?, session_state = ?, name = ?, updated_at = ?
                WHERE id = ?
            """, (message_history_json, session_state_json, name, now, session_id))
        else:
            conn.execute("""
                UPDATE chat_sessions
                SET message_history = ?, session_state = ?, updated_at = ?
                WHERE id = ?
            """, (message_history_json, session_state_json, now, session_id))


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
