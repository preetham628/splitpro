"""Tests for core/database.py's membership + expense proposal layer.

Each test gets its own fresh scratch SQLite file (tmp_path), provisioned via
create_db()/init_db() exactly like a real deployment would be. Sessions and
users are created through the module's own public functions rather than raw
SQL, so these tests exercise the same paths the app itself uses.
"""

import sqlite3

import pytest

from core import database as db


@pytest.fixture
def fresh_db(tmp_path):
    path = str(tmp_path / "test.db")
    db.create_db(path)
    db.init_db(path)
    return path


def make_user(email, google_id):
    return db.upsert_user(google_id=google_id, email=email, name=email.split("@")[0], avatar_url="")


def make_session(session_id, user_id, name="Test Session"):
    db.create_session(session_id, user_id, name)


def sample_payload(bill_id="bill_1", description="Dinner", item_names=("Burger",)):
    return {
        "bill_id": bill_id,
        "description": description,
        "raw_text": "raw text",
        "items": [
            {"name": name, "price": 10.0, "qty": 1, "assigned_to": ["Alice"],
             "shared": False, "unassigned": False, "qty_allocations": {}}
            for name in item_names
        ],
        "tax": 1.0,
        "tip": 2.0,
        "paid_by": "Alice",
    }


# ---------- Membership ----------

def test_add_and_check_membership(fresh_db):
    admin = make_user("admin@example.com", "g-admin")
    member = make_user("member@example.com", "g-member")
    make_session("sess-1", admin["id"])

    db.add_session_member("sess-1", admin["id"], "admin")
    db.add_session_member("sess-1", member["id"], "member")

    assert db.is_session_member("sess-1", admin["id"])
    assert db.is_session_member("sess-1", member["id"])
    assert not db.is_session_member("sess-1", 999999)

    assert db.is_session_admin("sess-1", admin["id"])
    assert not db.is_session_admin("sess-1", member["id"])


def test_list_session_members(fresh_db):
    admin = make_user("admin2@example.com", "g-admin2")
    member = make_user("member2@example.com", "g-member2")
    make_session("sess-2", admin["id"])
    db.add_session_member("sess-2", admin["id"], "admin")
    db.add_session_member("sess-2", member["id"], "member")

    members = db.list_session_members("sess-2")
    assert {m["user_id"] for m in members} == {admin["id"], member["id"]}
    roles = {m["user_id"]: m["role"] for m in members}
    assert roles[admin["id"]] == "admin"
    assert roles[member["id"]] == "member"
    assert all({"user_id", "name", "email", "avatar_url", "role"} <= m.keys() for m in members)


def test_count_admins_and_update_member_role(fresh_db):
    admin = make_user("a3@example.com", "g-a3")
    member = make_user("m3@example.com", "g-m3")
    make_session("sess-3", admin["id"])
    db.add_session_member("sess-3", admin["id"], "admin")
    db.add_session_member("sess-3", member["id"], "member")

    assert db.count_admins("sess-3") == 1

    assert db.update_member_role("sess-3", member["id"], "admin") is True
    assert db.count_admins("sess-3") == 2
    assert db.is_session_admin("sess-3", member["id"])

    assert db.update_member_role("sess-3", 424242, "admin") is False


def test_update_member_role_blocks_demoting_last_admin(fresh_db):
    admin = make_user("a3b@example.com", "g-a3b")
    member = make_user("m3b@example.com", "g-m3b")
    make_session("sess-3b", admin["id"])
    db.add_session_member("sess-3b", admin["id"], "admin")
    db.add_session_member("sess-3b", member["id"], "member")

    with pytest.raises(ValueError):
        db.update_member_role("sess-3b", admin["id"], "member")
    assert db.is_session_admin("sess-3b", admin["id"])

    # With two admins, demoting one is fine.
    db.update_member_role("sess-3b", member["id"], "admin")
    assert db.update_member_role("sess-3b", admin["id"], "member") is True
    assert db.count_admins("sess-3b") == 1


def test_remove_session_member(fresh_db):
    admin = make_user("a4@example.com", "g-a4")
    member = make_user("m4@example.com", "g-m4")
    make_session("sess-4", admin["id"])
    db.add_session_member("sess-4", admin["id"], "admin")
    db.add_session_member("sess-4", member["id"], "member")

    assert db.remove_session_member("sess-4", member["id"]) is True
    assert not db.is_session_member("sess-4", member["id"])
    assert db.remove_session_member("sess-4", member["id"]) is False


def test_remove_session_member_blocks_removing_last_admin(fresh_db):
    admin = make_user("a4b@example.com", "g-a4b")
    member = make_user("m4b@example.com", "g-m4b")
    make_session("sess-4b", admin["id"])
    db.add_session_member("sess-4b", admin["id"], "admin")
    db.add_session_member("sess-4b", member["id"], "member")

    with pytest.raises(ValueError):
        db.remove_session_member("sess-4b", admin["id"])
    assert db.is_session_member("sess-4b", admin["id"])

    # With two admins, removing one is fine.
    db.update_member_role("sess-4b", member["id"], "admin")
    assert db.remove_session_member("sess-4b", admin["id"]) is True


def test_get_user_by_email(fresh_db):
    user = make_user("lookup@example.com", "g-lookup")
    found = db.get_user_by_email("lookup@example.com")
    assert found is not None
    assert found["id"] == user["id"]
    assert db.get_user_by_email("missing@example.com") is None


# ---------- Expense proposals ----------

def test_create_update_get_list_proposal(fresh_db):
    admin = make_user("a5@example.com", "g-a5")
    member = make_user("m5@example.com", "g-m5")
    make_session("sess-5", admin["id"])
    db.add_session_member("sess-5", admin["id"], "admin")
    db.add_session_member("sess-5", member["id"], "member")

    payload = sample_payload()
    pid = db.create_proposal("sess-5", proposed_by=member["id"], payload=payload)
    assert isinstance(pid, int)

    payload["description"] = "Dinner (edited)"
    db.update_proposal_payload(pid, payload)

    fetched = db.get_proposal(pid)
    assert fetched["status"] == "pending"
    assert fetched["payload"]["description"] == "Dinner (edited)"

    pending = db.list_pending_proposals("sess-5")
    assert [p["id"] for p in pending] == [pid]

    assert db.get_proposal(999999) is None


def test_decide_proposal_approved_creates_new_bill(fresh_db):
    admin = make_user("a6@example.com", "g-a6")
    make_session("sess-6", admin["id"])
    db.add_session_member("sess-6", admin["id"], "admin")

    payload = sample_payload(bill_id="bill_new", item_names=("Burger", "Fries"))
    pid = db.create_proposal("sess-6", proposed_by=admin["id"], payload=payload)

    decided = db.decide_proposal(pid, decided_by=admin["id"], decision="approved")
    assert decided["status"] == "approved"
    assert decided["decided_by"] == admin["id"]

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    try:
        bill = conn.execute(
            "SELECT * FROM bills WHERE session_id = ? AND bill_id = ?", ("sess-6", "bill_new")
        ).fetchone()
        assert bill is not None
        assert bill["approved_by"] == admin["id"]
        assert bill["approved_at"] is not None

        items = conn.execute("SELECT * FROM bill_items WHERE bill_id = ?", (bill["id"],)).fetchall()
        assert {i["name"] for i in items} == {"Burger", "Fries"}
    finally:
        conn.close()


def test_decide_proposal_approved_supersedes_existing_bill(fresh_db):
    admin = make_user("a7@example.com", "g-a7")
    make_session("sess-7", admin["id"])
    db.add_session_member("sess-7", admin["id"], "admin")

    first_payload = sample_payload(bill_id="bill_x", item_names=("Burger",))
    pid1 = db.create_proposal("sess-7", proposed_by=admin["id"], payload=first_payload)
    db.decide_proposal(pid1, decided_by=admin["id"], decision="approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    bill_row = conn.execute(
        "SELECT * FROM bills WHERE session_id = ? AND bill_id = ?", ("sess-7", "bill_x")
    ).fetchone()
    conn.close()
    bill_row_id = bill_row["id"]

    second_payload = sample_payload(
        bill_id="bill_x", description="Replaced", item_names=("Burger", "Fries", "Coke")
    )
    pid2 = db.create_proposal(
        "sess-7", proposed_by=admin["id"], payload=second_payload, supersedes_bill_id=bill_row_id
    )
    db.decide_proposal(pid2, decided_by=admin["id"], decision="approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    try:
        all_bills = conn.execute("SELECT * FROM bills WHERE session_id = ?", ("sess-7",)).fetchall()
        assert len(all_bills) == 1, "supersede must update in place, not insert a duplicate bill"
        assert all_bills[0]["id"] == bill_row_id
        assert all_bills[0]["description"] == "Replaced"

        items = conn.execute("SELECT * FROM bill_items WHERE bill_id = ?", (bill_row_id,)).fetchall()
        assert {i["name"] for i in items} == {"Burger", "Fries", "Coke"}
    finally:
        conn.close()


def test_decide_proposal_rejected_creates_no_bill(fresh_db):
    admin = make_user("a8@example.com", "g-a8")
    make_session("sess-8", admin["id"])
    db.add_session_member("sess-8", admin["id"], "admin")

    payload = sample_payload(bill_id="bill_rej")
    pid = db.create_proposal("sess-8", proposed_by=admin["id"], payload=payload)
    decided = db.decide_proposal(pid, decided_by=admin["id"], decision="rejected")
    assert decided["status"] == "rejected"

    conn = sqlite3.connect(fresh_db)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM bills WHERE session_id = ?", ("sess-8",)
        ).fetchone()[0]
        assert count == 0
    finally:
        conn.close()


def test_decide_proposal_invalid_decision_raises(fresh_db):
    admin = make_user("a9@example.com", "g-a9")
    make_session("sess-9", admin["id"])
    pid = db.create_proposal("sess-9", proposed_by=admin["id"], payload=sample_payload())
    with pytest.raises(ValueError):
        db.decide_proposal(pid, decided_by=admin["id"], decision="maybe")


def test_decide_proposal_missing_raises(fresh_db):
    with pytest.raises(ValueError):
        db.decide_proposal(999999, decided_by=1, decision="approved")


def test_decide_proposal_is_not_repeatable(fresh_db):
    """A second decide_proposal() call on an already-decided proposal must
    raise rather than re-apply (silently, or via an IntegrityError on the
    bills UNIQUE(session_id, bill_id) constraint) — retries/double-clicks
    should fail loudly, not corrupt or duplicate data.
    """
    admin = make_user("a10@example.com", "g-a10")
    make_session("sess-10", admin["id"])
    db.add_session_member("sess-10", admin["id"], "admin")

    pid = db.create_proposal("sess-10", proposed_by=admin["id"], payload=sample_payload())
    db.decide_proposal(pid, decided_by=admin["id"], decision="approved")

    with pytest.raises(ValueError):
        db.decide_proposal(pid, decided_by=admin["id"], decision="approved")
    with pytest.raises(ValueError):
        db.decide_proposal(pid, decided_by=admin["id"], decision="rejected")


def test_create_proposal_rejects_cross_session_supersede(fresh_db):
    admin_a = make_user("a11@example.com", "g-a11")
    admin_b = make_user("b11@example.com", "g-b11")
    make_session("sess-11a", admin_a["id"])
    make_session("sess-11b", admin_b["id"])
    db.add_session_member("sess-11a", admin_a["id"], "admin")
    db.add_session_member("sess-11b", admin_b["id"], "admin")

    pid_b = db.create_proposal("sess-11b", proposed_by=admin_b["id"], payload=sample_payload())
    db.decide_proposal(pid_b, decided_by=admin_b["id"], decision="approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    bill_b = conn.execute("SELECT * FROM bills WHERE session_id = 'sess-11b'").fetchone()
    conn.close()

    with pytest.raises(ValueError):
        db.create_proposal(
            "sess-11a", proposed_by=admin_a["id"],
            payload=sample_payload(description="HIJACKED"),
            supersedes_bill_id=bill_b["id"],
        )


def test_decide_proposal_rejects_cross_session_supersede_even_if_bypassed(fresh_db):
    """Belt-and-suspenders: even if a proposal somehow ends up with a
    supersedes_bill_id from another session (bypassing create_proposal's own
    check), decide_proposal must refuse to act on it rather than overwrite
    another session's bill.
    """
    admin_a = make_user("a12@example.com", "g-a12")
    admin_b = make_user("b12@example.com", "g-b12")
    make_session("sess-12a", admin_a["id"])
    make_session("sess-12b", admin_b["id"])
    db.add_session_member("sess-12a", admin_a["id"], "admin")
    db.add_session_member("sess-12b", admin_b["id"], "admin")

    pid_b = db.create_proposal("sess-12b", proposed_by=admin_b["id"], payload=sample_payload())
    db.decide_proposal(pid_b, decided_by=admin_b["id"], decision="approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    bill_b = conn.execute("SELECT * FROM bills WHERE session_id = 'sess-12b'").fetchone()
    conn.close()

    # Bypass create_proposal's guard by inserting the row directly.
    conn = sqlite3.connect(fresh_db)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.execute(
        "INSERT INTO expense_proposals (session_id, proposed_by, supersedes_bill_id, payload) "
        "VALUES (?, ?, ?, ?)",
        ("sess-12a", admin_a["id"], bill_b["id"], '{"bill_id": "x", "items": []}'),
    )
    pid_a = cursor.lastrowid
    conn.commit()
    conn.close()

    with pytest.raises(ValueError):
        db.decide_proposal(pid_a, decided_by=admin_a["id"], decision="approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    bill_b_after = conn.execute("SELECT * FROM bills WHERE id = ?", (bill_b["id"],)).fetchone()
    conn.close()
    assert bill_b_after["session_id"] == "sess-12b"
    assert bill_b_after["description"] == sample_payload()["description"]


def test_save_session_state_after_supersede_does_not_raise(fresh_db):
    """Regression test: expense_proposals.supersedes_bill_id must not block
    the normal delete-and-reinsert save_session_state() does on bills every
    turn. Previously this raised sqlite3.IntegrityError on the very next
    save for any session with a decided correction-proposal, because the FK
    had no ON DELETE clause.
    """
    from core.session_state import SessionState

    admin = make_user("a13@example.com", "g-a13")
    make_session("sess-13", admin["id"])
    db.add_session_member("sess-13", admin["id"], "admin")

    pid1 = db.create_proposal("sess-13", proposed_by=admin["id"], payload=sample_payload())
    db.decide_proposal(pid1, decided_by=admin["id"], decision="approved")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    bill = conn.execute("SELECT * FROM bills WHERE session_id = 'sess-13'").fetchone()
    conn.close()

    pid2 = db.create_proposal(
        "sess-13", proposed_by=admin["id"],
        payload=sample_payload(description="v2"), supersedes_bill_id=bill["id"],
    )
    db.decide_proposal(pid2, decided_by=admin["id"], decision="approved")

    state = SessionState(participants=[], bills=[], finalized=False)
    db.save_session_state("sess-13", state, settlements=[])  # must not raise

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    try:
        assert conn.execute("SELECT COUNT(*) FROM bills").fetchone()[0] == 0
        proposal_row = conn.execute(
            "SELECT supersedes_bill_id FROM expense_proposals WHERE id = ?", (pid2,)
        ).fetchone()
        assert proposal_row["supersedes_bill_id"] is None, (
            "FK should SET NULL rather than block the bill delete"
        )
    finally:
        conn.close()


# ---------- Legacy backfill migration ----------

def test_legacy_sessions_backfilled_as_admin(tmp_path):
    """Simulate a pre-existing DB (old schema shape: no session_members /
    expense_proposals tables, no user_id/approved_by/approved_at columns)
    and confirm init_db() backfills exactly one admin session_members row
    per existing session, and is idempotent on re-run.
    """
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            google_id  TEXT UNIQUE NOT NULL,
            email      TEXT UNIQUE NOT NULL,
            name       TEXT,
            avatar_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chat_sessions (
            id               TEXT PRIMARY KEY,
            user_id          INTEGER NOT NULL REFERENCES users(id),
            name             TEXT NOT NULL DEFAULT 'New Session',
            finalized        INTEGER NOT NULL DEFAULT 0,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chat_messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role             TEXT NOT NULL CHECK (role IN ('user', 'agent')),
            content          TEXT NOT NULL DEFAULT '',
            image_base64     TEXT,
            image_media_type TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE bills (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            bill_id     TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            raw_text    TEXT NOT NULL DEFAULT '',
            tax         REAL NOT NULL DEFAULT 0,
            tip         REAL NOT NULL DEFAULT 0,
            paid_by     TEXT,
            UNIQUE(session_id, bill_id)
        );
        INSERT INTO users (id, google_id, email, name, avatar_url) VALUES (1, 'g1', 'a@example.com', 'A', '');
        INSERT INTO users (id, google_id, email, name, avatar_url) VALUES (2, 'g2', 'b@example.com', 'B', '');
        INSERT INTO chat_sessions (id, user_id, name) VALUES ('legacy-1', 1, 'Legacy 1');
        INSERT INTO chat_sessions (id, user_id, name) VALUES ('legacy-2', 2, 'Legacy 2');
    """)
    conn.commit()
    conn.close()

    db.init_db(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT session_id, user_id, role FROM session_members ORDER BY session_id"
        ).fetchall()
        by_session = {r["session_id"]: r for r in rows}
        assert set(by_session) == {"legacy-1", "legacy-2"}
        assert by_session["legacy-1"]["user_id"] == 1
        assert by_session["legacy-1"]["role"] == "admin"
        assert by_session["legacy-2"]["user_id"] == 2
        assert by_session["legacy-2"]["role"] == "admin"

        cm_cols = {r["name"] for r in conn.execute("PRAGMA table_info(chat_messages)")}
        bills_cols = {r["name"] for r in conn.execute("PRAGMA table_info(bills)")}
        assert "user_id" in cm_cols
        assert {"approved_by", "approved_at"} <= bills_cols
    finally:
        conn.close()

    # Idempotent re-run: no duplicate backfilled rows, no errors.
    db.init_db(path)
    conn = sqlite3.connect(path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM session_members").fetchone()[0]
        assert count == 2
    finally:
        conn.close()
