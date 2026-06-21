import sqlite3
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "splitpro.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id           TEXT PRIMARY KEY,
                name         TEXT NOT NULL DEFAULT 'New Chat',
                provider     TEXT NOT NULL DEFAULT 'anthropic',
                model        TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                state_json   TEXT NOT NULL DEFAULT '{"participants":[],"bills":[],"finalized":false}',
                history_json TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_chats() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, provider, model, created_at, updated_at FROM chats ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_chat(chat_id: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return dict(row) if row else None


def create_chat(chat_id: str, name: str, provider: str, model: str,
                state_json: str, history_json: str) -> None:
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO chats (id, name, provider, model, created_at, updated_at, state_json, history_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (chat_id, name, provider, model, now, now, state_json, history_json),
        )
        conn.commit()


def update_chat(chat_id: str, state_json: str, history_json: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            """UPDATE chats
               SET state_json = ?, history_json = ?, updated_at = ?
               WHERE id = ?""",
            (state_json, history_json, _now(), chat_id),
        )
        conn.commit()


def rename_chat(chat_id: str, name: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chats SET name = ?, updated_at = ? WHERE id = ?",
            (name, _now(), chat_id),
        )
        conn.commit()


def delete_chat(chat_id: str) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
