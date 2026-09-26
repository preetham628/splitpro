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
        _migrate_expense_proposals_fk(conn)


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


def _migrate_expense_proposals_fk(conn: sqlite3.Connection) -> None:
    """Rebuild expense_proposals if its supersedes_bill_id FK predates
    ON DELETE SET NULL.

    CREATE TABLE IF NOT EXISTS is a no-op against an already-existing table,
    so any database provisioned before this fix landed would otherwise keep
    the old FK (no ON DELETE clause) forever — and hit the exact
    IntegrityError the fix was meant to eliminate on the next
    save_session_state() delete+reinsert of bills. SQLite can't ALTER a
    foreign key in place, so this does the standard rename+recreate+copy+
    drop dance. No-ops once the FK is already correct, and no-ops on a
    brand-new DB (schema.sql's own CREATE TABLE already has it right, so
    there's nothing to migrate).
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'expense_proposals'"
    ).fetchone()
    if not exists:
        return

    fks = conn.execute("PRAGMA foreign_key_list(expense_proposals)").fetchall()
    supersedes_fk = next((fk for fk in fks if fk["from"] == "supersedes_bill_id"), None)
    if supersedes_fk is None or supersedes_fk["on_delete"] == "SET NULL":
        return

    conn.executescript("""
        ALTER TABLE expense_proposals RENAME TO expense_proposals_old;

        CREATE TABLE expense_proposals (
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

        INSERT INTO expense_proposals SELECT * FROM expense_proposals_old;
        DROP TABLE expense_proposals_old;

        CREATE INDEX IF NOT EXISTS idx_expense_proposals_session ON expense_proposals(session_id, status);
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
            _insert_bill_items(conn, bill_row_id, [
                {
                    "name": item.name, "price": item.price, "qty": item.qty,
                    "assigned_to": item.assigned_to, "shared": item.shared,
                    "unassigned": item.unassigned, "qty_allocations": item.qty_allocations,
                }
                for item in bill.items
            ])

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


def update_member_role(session_id: str, user_id: int, role: str) -> bool:
    """Change an existing member's role. Returns False if no row matched.

    Refuses (raises ValueError) to demote a session's last remaining admin —
    every session must keep at least one admin.

    The admin-count check and the write are one atomic UPDATE statement
    (the count is a correlated subquery in the WHERE clause), not a
    separate SELECT-then-UPDATE — a check-then-act split would leave a race
    window where two concurrent demotions could both read "2 admins" and
    both proceed, leaving zero. A single UPDATE statement runs to
    completion under SQLite's write lock with no other writer interleaved,
    so this can't happen here.

    The transaction is opened with BEGIN IMMEDIATE (rather than relying on
    the driver's default deferred BEGIN) so the write lock is held from
    before the guarded UPDATE all the way through the diagnostic SELECT
    below — otherwise that second read's consistency with the UPDATE it's
    explaining would depend on unstated lock-escalation timing instead of
    something explicit in the code.
    """
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute("""
            UPDATE session_members
            SET role = ?
            WHERE session_id = ? AND user_id = ?
              AND (
                ? = 'admin'
                OR role != 'admin'
                OR (SELECT COUNT(*) FROM session_members AS sm2
                    WHERE sm2.session_id = session_members.session_id AND sm2.role = 'admin') > 1
              )
        """, (role, session_id, user_id, role))
        if cursor.rowcount > 0:
            return True

        # Either not a member, or the guard blocked it — disambiguate for a
        # clear error. This second read is diagnostic only; it doesn't
        # affect correctness since the atomic UPDATE above already made the
        # real decision. Safe from concurrent modification because BEGIN
        # IMMEDIATE above still holds the write lock at this point.
        current = conn.execute(
            "SELECT role FROM session_members WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if current is not None and current["role"] == "admin" and role != "admin":
            raise ValueError(f"cannot demote the last admin of session {session_id!r}")
        return False


def remove_session_member(session_id: str, user_id: int) -> bool:
    """Remove a member from a session. Returns False if they weren't a member.

    Refuses (raises ValueError) to remove a session's last remaining admin.
    Same atomic-statement approach as update_member_role() — see there for
    why the guard has to be part of the DELETE itself, not a prior SELECT,
    and why the transaction opens with BEGIN IMMEDIATE.
    """
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute("""
            DELETE FROM session_members
            WHERE session_id = ? AND user_id = ?
              AND (
                role != 'admin'
                OR (SELECT COUNT(*) FROM session_members AS sm2
                    WHERE sm2.session_id = session_members.session_id AND sm2.role = 'admin') > 1
              )
        """, (session_id, user_id))
        if cursor.rowcount > 0:
            return True

        current = conn.execute(
            "SELECT role FROM session_members WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if current is not None and current["role"] == "admin":
            raise ValueError(f"cannot remove the last admin of session {session_id!r}")
        return False


# ---------- Expense proposals ----------

def create_proposal(
    session_id: str,
    proposed_by: int,
    payload: dict,
    supersedes_bill_id: Optional[int] = None,
) -> int:
    """Stage an AI-drafted expense for admin approval. Returns the new proposal id.

    Raises ValueError if supersedes_bill_id is given but doesn't belong to
    session_id — the same check decide_proposal() re-runs at approval time,
    since a bill's session_id can't change after this either.
    """
    with _connect() as conn:
        if supersedes_bill_id is not None:
            bill = conn.execute(
                "SELECT session_id FROM bills WHERE id = ?", (supersedes_bill_id,)
            ).fetchone()
            if bill is None or bill["session_id"] != session_id:
                raise ValueError(
                    f"supersedes_bill_id {supersedes_bill_id} does not belong to "
                    f"session {session_id!r}"
                )

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


def _insert_bill_items(conn: sqlite3.Connection, bill_row_id: int, items: list[dict]) -> None:
    """Insert bill_items rows for one bill from plain dicts (keys: name,
    price, qty, assigned_to, shared, unassigned, qty_allocations). Shared by
    save_session_state() (converting LineItem dataclasses to dicts first)
    and decide_proposal()'s approval path (payload items are already dicts),
    so the insert shape only has to change in one place.
    """
    for item in items:
        conn.execute("""
            INSERT INTO bill_items
                (bill_id, name, price, qty, assigned_to, shared, unassigned, qty_allocations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (bill_row_id, item["name"], item["price"], item.get("qty", 1),
              json.dumps(item.get("assigned_to", [])), int(item.get("shared", False)),
              int(item.get("unassigned", False)), json.dumps(item.get("qty_allocations", {}))))


def decide_proposal(proposal_id: int, decided_by: int, decision: str) -> dict:
    """Approve or reject a pending proposal. Returns the updated proposal row
    (payload json.loads'd), all inside one transaction.

    Raises ValueError if the proposal isn't (still) 'pending'. That check is
    folded into the UPDATE itself (`WHERE id = ? AND status = 'pending'`,
    rowcount checked) rather than a prior SELECT — a check-then-act split
    would leave a race window where two concurrent decide_proposal() calls
    on the same id could both read 'pending' and both proceed, the second
    hitting the bills UNIQUE(session_id, bill_id) constraint. A single
    UPDATE statement can't be interleaved with another writer, so only one
    caller ever wins the claim.

    Rejection only touches the proposal row itself. Approval additionally
    materializes the payload into bills/bill_items. Which bill is targeted
    is resolved fresh, by the *natural* key (session_id, bill_id) — not by
    trusting the stored supersedes_bill_id surrogate FK. That FK is
    ON DELETE SET NULL (so it doesn't block save_session_state()'s
    every-turn delete+reinsert of bills), which means it can go stale or
    null between proposal creation and decision if any normal chat turn
    happens on this session in between — the row it pointed at may have
    been deleted and reinserted under a new id, or it may have never
    existed. (session_id, bill_id) is stable across that churn since
    bills.bill_id is the app-level id the agent already keys off, so
    looking a bill up by it always finds the live row if one exists,
    scoped to this proposal's own session — which also means this can never
    write to another session's bill regardless of what supersedes_bill_id
    says.
    """
    if decision not in ("approved", "rejected"):
        raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute("""
            UPDATE expense_proposals
            SET status = ?, decided_by = ?, decided_at = ?, updated_at = ?
            WHERE id = ? AND status = 'pending'
        """, (decision, decided_by, now, now, proposal_id))

        if cursor.rowcount == 0:
            existing = conn.execute(
                "SELECT status FROM expense_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
            if existing is None:
                raise ValueError(f"Proposal {proposal_id} not found")
            raise ValueError(
                f"proposal {proposal_id} was already decided (status={existing['status']!r})"
            )

        row = conn.execute(
            "SELECT * FROM expense_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()

        if decision == "approved":
            payload = json.loads(row["payload"])
            session_id = row["session_id"]

            existing_bill = conn.execute(
                "SELECT id FROM bills WHERE session_id = ? AND bill_id = ?",
                (session_id, payload["bill_id"]),
            ).fetchone()

            if existing_bill is None:
                cursor = conn.execute("""
                    INSERT INTO bills
                        (session_id, bill_id, description, raw_text, tax, tip, paid_by,
                         approved_by, approved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, payload["bill_id"], payload.get("description", ""),
                      payload.get("raw_text", ""), payload.get("tax", 0), payload.get("tip", 0),
                      payload.get("paid_by"), decided_by, now))
                bill_row_id = cursor.lastrowid
            else:
                bill_row_id = existing_bill["id"]
                conn.execute("""
                    UPDATE bills
                    SET description = ?, raw_text = ?, tax = ?, tip = ?, paid_by = ?,
                        approved_by = ?, approved_at = ?
                    WHERE id = ?
                """, (payload.get("description", ""), payload.get("raw_text", ""),
                      payload.get("tax", 0), payload.get("tip", 0), payload.get("paid_by"),
                      decided_by, now, bill_row_id))
                conn.execute("DELETE FROM bill_items WHERE bill_id = ?", (bill_row_id,))

            _insert_bill_items(conn, bill_row_id, payload.get("items", []))

        result = dict(row)
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
