"""
LangGraph checkpoint persistence for ChatAgent's conversation state.

Keyed by thread_id (= our session_id). Lives in the same DB_PATH file as
core/database.py's own tables, via its own long-lived connection — separate
from core/database.py's per-call _connect() pattern, since SqliteSaver
expects to own a persistent connection across the app's lifetime.

SqliteSaver.setup() sets PRAGMA journal_mode=WAL the first time it runs —
we revert that back to the default (DELETE) journal mode immediately after.
WAL mode requires shared-memory (mmap) coordination between readers/writers
that doesn't work reliably across a Docker Desktop bind mount boundary
(confirmed: with WAL on, writes made by the container were invisible to the
sqlite3 CLI run on the host against the same bind-mounted file, even though
the container's own connections saw them fine) — and DB_PATH is deliberately
bind-mounted (not a named volume) specifically so it's inspectable from the
host. At this app's scale (single instance, low write concurrency), the
rollback journal has no meaningful downside.
"""

import sqlite3
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

_saver: Optional[SqliteSaver] = None


def init_checkpointer(db_path: str) -> None:
    """Attach to (and provision, if needed) the checkpoint tables. Call once at server startup."""
    global _saver
    conn = sqlite3.connect(db_path, check_same_thread=False)
    _saver = SqliteSaver(conn)
    _saver.setup()
    conn.execute("PRAGMA journal_mode=DELETE")


def get_checkpointer() -> SqliteSaver:
    if _saver is None:
        raise RuntimeError("Checkpointer not initialized — call init_checkpointer() at startup.")
    return _saver


def delete_thread(session_id: str) -> None:
    """Remove all checkpoint data for a session. chat_sessions row deletion
    cascades to our own tables via FK, but LangGraph's checkpoints/writes
    tables have no such relationship (thread_id is a plain string, not an FK)
    — call this explicitly wherever a session is deleted."""
    get_checkpointer().delete_thread(session_id)
