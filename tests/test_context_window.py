"""Tests for Task 10's sliding-window context management: round-based
windowing, the raw-message safety net, and the two-phase background
summarization flow (slow unlocked LLM call + fast locked checkpoint
surgery / persist).

Two fixtures mirror the two existing suites this straddles:
  - fresh_db (tests/test_chat_agent.py's pattern): a scratch SQLite file for
    core/database.py's own tables (chat_sessions.context_summary lives
    here).
  - fresh_server (tests/test_server.py's pattern): on top of fresh_db, also
    resets server.py's module-level session cache/locks so tests don't leak
    into each other.

The checkpointer itself is always ":memory:" — conversation history lives
there, completely independent of core/database.py's file, same as
production (core/checkpointer.py).
"""

import threading
import time
import uuid

import pytest
from fastapi import BackgroundTasks
from langchain_core.messages import AIMessage, HumanMessage

import server
from agents.chat_agent import (
    ChatAgent,
    MESSAGE_SAFETY_NET,
    ROUND_WINDOW,
    ROUNDS_TO_SUMMARIZE,
    _context_summary_section,
    _exceeds_summarization_thresholds,
    _round_start_indices,
    summarize_rounds,
)
from config import ChatAgentConfig
from core import checkpointer, database as db


@pytest.fixture
def fresh_db(tmp_path):
    path = str(tmp_path / "test.db")
    db.create_db(path)
    db.init_db(path)
    return path


@pytest.fixture
def fresh_server(fresh_db, monkeypatch):
    checkpointer.init_checkpointer(":memory:")
    monkeypatch.setattr(server, "sessions", {})
    monkeypatch.setattr(server, "_session_locks", {})
    monkeypatch.setattr(server, "_summarizing_sessions", set())
    return fresh_db


def make_user(email, google_id, name=None):
    row = db.upsert_user(google_id=google_id, email=email, name=name or email.split("@")[0], avatar_url="")
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


def make_session_with_members(*users, admin_index=0):
    session_id = str(uuid.uuid4())
    db.create_session(session_id, users[admin_index]["id"], name="Test Session")
    for i, u in enumerate(users):
        db.add_session_member(session_id, u["id"], role="admin" if i == admin_index else "member")
    return session_id


class FakeLLM:
    """Minimal fake chat model covering both call sites create_llm() feeds
    into: ChatAgent's tool-bound agent node (bind_tools().invoke(...), one
    plain AIMessage ends each round immediately — no tool calls needed for
    these tests) and summarize_rounds()'s direct .invoke() call (detected by
    the summarization prompt's distinctive opening line)."""

    def __init__(self, summary_text="a summary of the earlier rounds"):
        self.summary_text = summary_text
        self.summarize_prompts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        last = messages[-1]
        content = last.content if isinstance(last.content, str) else str(last.content)
        if content.startswith("You maintain a running summary"):
            self.summarize_prompts.append(content)
            return AIMessage(content=self.summary_text)
        return AIMessage(content="ok")  # ends the round immediately, no tool calls


def make_agent(session_id, monkeypatch, fake_llm=None):
    fake_llm = fake_llm or FakeLLM()
    monkeypatch.setattr("agents.chat_agent.create_llm", lambda config: fake_llm)
    agent = ChatAgent(session_id=session_id, config=ChatAgentConfig())
    return agent, fake_llm


# ---------- Pure helpers ----------

def test_round_start_indices_counts_human_messages():
    messages = [
        HumanMessage(content="h1"), AIMessage(content="a1"),
        HumanMessage(content="h2"), AIMessage(content="a2"), AIMessage(content="a2b"),
    ]
    assert _round_start_indices(messages) == [0, 2]


def test_exceeds_thresholds_on_round_count():
    messages = []
    for i in range(ROUND_WINDOW):
        messages += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
    assert not _exceeds_summarization_thresholds(messages)  # exactly at the window, not over

    messages += [HumanMessage(content="one more"), AIMessage(content="a")]
    assert _exceeds_summarization_thresholds(messages)  # round_count now > ROUND_WINDOW


def test_exceeds_thresholds_on_raw_message_safety_net():
    # Two rounds (well under ROUND_WINDOW) but the raw message count alone
    # crosses MESSAGE_SAFETY_NET -- simulates one round with a huge number
    # of tool calls (e.g. a giant pasted bill).
    messages = [HumanMessage(content="h0")]
    messages += [AIMessage(content=f"tool-call-{i}") for i in range(MESSAGE_SAFETY_NET)]
    messages += [HumanMessage(content="h1"), AIMessage(content="a1")]
    assert len(_round_start_indices(messages)) == 2
    assert _exceeds_summarization_thresholds(messages)


# ---------- End-to-end: round window triggers + checkpoint actually shrinks ----------

def test_summarization_fires_after_more_than_8_rounds_and_shrinks_checkpoint(fresh_db, monkeypatch):
    checkpointer.init_checkpointer(":memory:")
    admin = make_user("admin@example.com", "g-ctx-admin")
    session_id = make_session_with_members(admin)

    agent, fake_llm = make_agent(session_id, monkeypatch)

    # Drive 10 full rounds through the real ChatAgent.chat() -> graph.invoke()
    # path (no tool calls, so each round is exactly HumanMessage + AIMessage).
    for i in range(10):
        agent.chat(f"message {i}", speaker_name="Admin", speaker_user_id=admin["id"])

    messages_before = agent.checkpoint_messages()
    assert len(_round_start_indices(messages_before)) == 10
    assert agent.needs_context_summarization() is True

    plan = agent.plan_context_summarization()
    assert plan is not None
    messages_to_summarize, ids_to_remove = plan
    # Oldest ROUNDS_TO_SUMMARIZE rounds, capped to leave at least one round verbatim.
    assert len(ids_to_remove) == ROUNDS_TO_SUMMARIZE * 2  # HumanMessage + AIMessage per round
    assert set(ids_to_remove).issubset({m.id for m in messages_before})

    existing_summary = db.get_context_summary(session_id)
    assert existing_summary == ""  # nothing summarized yet

    new_summary = summarize_rounds(agent.config, existing_summary, messages_to_summarize)
    assert new_summary == fake_llm.summary_text
    assert len(fake_llm.summarize_prompts) == 1

    applied = agent.apply_context_summarization(ids_to_remove, new_summary)
    assert applied is True

    # context_summary actually persisted to the DB.
    assert db.get_context_summary(session_id) == fake_llm.summary_text

    # The checkpoint's message count actually dropped -- not just "nothing
    # crashed." And the specific summarized-away ids are genuinely gone.
    messages_after = agent.checkpoint_messages()
    assert len(messages_after) == len(messages_before) - len(ids_to_remove)
    after_ids = {m.id for m in messages_after}
    assert after_ids.isdisjoint(set(ids_to_remove))

    # The 4 newest rounds remain, verbatim and in order.
    assert len(_round_start_indices(messages_after)) == 10 - ROUNDS_TO_SUMMARIZE

    # The agent keeps working cleanly after the checkpoint surgery.
    reply = agent.chat("one more after summarization", speaker_name="Admin", speaker_user_id=admin["id"])
    assert reply == "ok"
    assert len(_round_start_indices(agent.checkpoint_messages())) == 10 - ROUNDS_TO_SUMMARIZE + 1


def test_apply_context_summarization_is_noop_when_ids_already_gone(fresh_db, monkeypatch):
    """Defensive race guard: if the snapshotted ids are no longer all present
    in the checkpoint (e.g. something else already trimmed them), apply
    must no-op rather than raise or double-remove."""
    checkpointer.init_checkpointer(":memory:")
    admin = make_user("admin@example.com", "g-ctx-race")
    session_id = make_session_with_members(admin)
    agent, _fake_llm = make_agent(session_id, monkeypatch)

    for i in range(10):
        agent.chat(f"message {i}", speaker_name="Admin", speaker_user_id=admin["id"])

    plan = agent.plan_context_summarization()
    messages_to_summarize, ids_to_remove = plan

    # Simulate the race: remove them "out from under" the pending apply call.
    assert agent.apply_context_summarization(ids_to_remove, "first summary") is True
    assert db.get_context_summary(session_id) == "first summary"

    # A second, stale attempt with the same (now-gone) ids must no-op, not
    # raise, and must not clobber the summary that was already persisted.
    applied_again = agent.apply_context_summarization(ids_to_remove, "stale summary")
    assert applied_again is False
    assert db.get_context_summary(session_id) == "first summary"


def test_apply_context_summarization_noop_with_no_ids():
    agent = object.__new__(ChatAgent)  # avoid building a real graph for this trivial check
    assert ChatAgent.apply_context_summarization(agent, [], "anything") is False


# ---------- Safety net triggers under the round window ----------

def test_safety_net_triggers_with_round_count_under_window(fresh_db, monkeypatch):
    """A single round stuffed with many tool-call/tool-response messages
    must trip the raw-message safety net even though round_count is well
    under ROUND_WINDOW."""
    checkpointer.init_checkpointer(":memory:")
    admin = make_user("admin@example.com", "g-ctx-safety")
    session_id = make_session_with_members(admin)
    agent, _fake_llm = make_agent(session_id, monkeypatch)

    # Two ordinary rounds first.
    agent.chat("hi", speaker_name="Admin", speaker_user_id=admin["id"])
    agent.chat("hi again", speaker_name="Admin", speaker_user_id=admin["id"])
    assert agent.needs_context_summarization() is False

    # Now simulate one giant round (e.g. a huge pasted bill triggering dozens
    # of tool calls) by injecting many extra messages directly into the
    # checkpoint for the graph's real "agent" node, round-aligned (starts
    # with a HumanMessage).
    config = {"configurable": {"thread_id": session_id}}
    huge_round = [HumanMessage(content="huge bill paste")]
    huge_round += [AIMessage(content=f"processing item {i}") for i in range(MESSAGE_SAFETY_NET)]
    agent._graph.update_state(config, {"messages": huge_round}, as_node="agent")

    messages = agent.checkpoint_messages()
    round_count = len(_round_start_indices(messages))
    assert round_count == 3  # still well under ROUND_WINDOW (8)
    assert len(messages) > MESSAGE_SAFETY_NET

    assert agent.needs_context_summarization() is True
    plan = agent.plan_context_summarization()
    assert plan is not None
    _messages_to_summarize, ids_to_remove = plan
    assert ids_to_remove, "safety net fired but nothing was planned for removal"

    # Capped to leave at least one round verbatim (the huge round itself
    # must survive this trim, or the window would have destroyed the only
    # context immediately after triggering on it), and something must
    # actually have been trimmed.
    remaining_messages = [m for m in messages if m.id not in ids_to_remove]
    remaining_round_count = len(_round_start_indices(remaining_messages))
    assert 1 <= remaining_round_count < round_count
    assert len(remaining_messages) < len(messages)

    # Applying it for real actually drops the checkpoint by exactly the
    # removed-round count, confirming this isn't just a planning no-op. Note
    # this one pass does NOT necessarily drop back under MESSAGE_SAFETY_NET
    # itself here -- the only round left standing is the oversized one that
    # tripped the safety net in the first place, and plan_context_
    # summarization deliberately never folds away the last remaining round
    # (see its docstring). A later turn adding ordinary rounds on top would
    # eventually let a further pass absorb it the same way the round-window
    # path does.
    new_summary = summarize_rounds(agent.config, db.get_context_summary(session_id), _messages_to_summarize)
    assert agent.apply_context_summarization(ids_to_remove, new_summary) is True
    messages_after = agent.checkpoint_messages()
    assert len(messages_after) == len(messages) - len(ids_to_remove)
    assert db.get_context_summary(session_id) == new_summary


# ---------- System prompt injection ----------

def test_context_summary_section_empty_when_no_summary(fresh_db):
    session_id = "sess-no-summary"
    db.create_session(session_id, db.upsert_user("g1", "a@x.com", "A", "")["id"])
    assert _context_summary_section(session_id) == ""


def test_context_summary_injected_into_system_prompt(fresh_db, monkeypatch):
    checkpointer.init_checkpointer(":memory:")
    admin = make_user("admin@example.com", "g-ctx-prompt")
    session_id = make_session_with_members(admin)
    db.set_context_summary(session_id, "Alice and Bob were chatting about dinner plans.")

    captured_system_prompts = []

    class RecordingLLM(FakeLLM):
        def invoke(self, messages):
            system_msg = messages[0]
            captured_system_prompts.append(system_msg.content)
            return super().invoke(messages)

    agent, _fake_llm = make_agent(session_id, monkeypatch, fake_llm=RecordingLLM())
    agent.chat("hello", speaker_name="Admin", speaker_user_id=admin["id"])

    assert captured_system_prompts, "agent node never invoked the LLM"
    assert "EARLIER CONVERSATION (summarized):" in captured_system_prompts[0]
    assert "Alice and Bob were chatting about dinner plans." in captured_system_prompts[0]


def test_context_summary_not_injected_for_cli_legacy_path(fresh_db, monkeypatch):
    """No speaker_user_id (the CLI's shape) must never touch core.database at
    all for the context summary -- mirrors how pending-proposals injection
    is already gated the same way."""
    checkpointer.init_checkpointer(":memory:")
    session_id = "sess-cli-" + str(uuid.uuid4())
    # Deliberately no chat_sessions row created for this session_id -- a
    # legacy/CLI call must never query core.database at all, so this would
    # raise if it did.
    agent, _fake_llm = make_agent(session_id, monkeypatch)
    reply = agent.chat("hello")  # no speaker_name/speaker_user_id, like main.py
    assert reply == "ok"


# ---------- Background task orchestration + locking (server.py) ----------

def test_chat_schedules_background_task_without_invoking_it_synchronously(fresh_server, monkeypatch):
    """When summarization IS needed, background_tasks.add_task must only
    register the check for FastAPI to run after the response is sent --
    confirms chat() never awaits/calls it inline, which is what actually
    guarantees zero added latency on the triggering request."""
    # Avoid a real (unmocked) create_llm() call here -- the first-ever call
    # pays a one-time real-provider-SDK import cost unrelated to anything
    # under test here, which would make the timing assertion below flaky.
    monkeypatch.setattr("agents.chat_agent.create_llm", lambda config: FakeLLM())
    monkeypatch.setattr(ChatAgent, "chat", lambda self, m, speaker_name=None, speaker_user_id=None: "ok")
    # ChatAgent.chat is fully mocked out above (bypassing the real graph), so
    # the checkpoint never actually grows -- force the needs-check itself so
    # the scheduling path under test is actually exercised.
    monkeypatch.setattr(ChatAgent, "needs_context_summarization", lambda self: True)

    summarize_called = threading.Event()

    def spy_maybe_summarize(session_id):
        summarize_called.set()
        time.sleep(1)

    monkeypatch.setattr(server, "_maybe_summarize_context", spy_maybe_summarize)

    admin = make_user("admin@example.com", "g-bgtask-admin")
    session_id = make_session_with_members(admin)

    bg = BackgroundTasks()
    start = time.time()
    result = server.chat(session_id, server.ChatRequest(message="hi"), background_tasks=bg, user=admin)
    elapsed = time.time() - start

    assert result.response == "ok"
    assert elapsed < 0.5, f"chat() took {elapsed}s -- summarization must never run inline"
    assert not summarize_called.is_set(), "background task body ran synchronously instead of being scheduled"
    assert len(bg.tasks) == 1


def test_chat_does_not_schedule_background_task_when_not_needed(fresh_server, monkeypatch):
    """The common case -- most turns, most sessions -- must not schedule a
    background task at all. A deviation from this was flagged in review:
    the original implementation scheduled unconditionally on every single
    turn and only checked inside the task, meaning even a session's very
    first message paid for an extra background thread + checkpoint read
    for nothing."""
    monkeypatch.setattr("agents.chat_agent.create_llm", lambda config: FakeLLM())
    monkeypatch.setattr(ChatAgent, "chat", lambda self, m, speaker_name=None, speaker_user_id=None: "ok")
    # ChatAgent.chat is mocked out, so the checkpoint never grows -- this is
    # also naturally true for a session's first real message either way.
    assert ChatAgent.needs_context_summarization  # sanity: not already removed/renamed

    admin = make_user("admin@example.com", "g-noneed-admin")
    session_id = make_session_with_members(admin)

    bg = BackgroundTasks()
    result = server.chat(session_id, server.ChatRequest(message="hi"), background_tasks=bg, user=admin)

    assert result.response == "ok"
    assert len(bg.tasks) == 0, "a background task was scheduled even though summarization wasn't needed"
    assert session_id not in server._summarizing_sessions


def test_concurrent_turns_only_schedule_one_summarization(fresh_server, monkeypatch):
    """Two REAL concurrent threads that would each independently see "needs
    summarization" must not each schedule their own background task / fire
    their own (paid) LLM summarization call -- the in-flight guard
    (_summarizing_sessions, guarded by its own dedicated lock) must let
    only one of them through.

    Uses a threading.Barrier to force genuine interleaving: both threads'
    needs_context_summarization() calls release together, so both threads
    actually race into _schedule_context_summarization_if_needed's
    check-and-set at essentially the same instant, rather than one
    trivially finishing before the other starts (which would pass even with
    the locking removed entirely) -- same real-thread spirit as this file's
    other two concurrency tests above.
    """
    monkeypatch.setattr("agents.chat_agent.create_llm", lambda config: FakeLLM())
    monkeypatch.setattr(ChatAgent, "chat", lambda self, m, speaker_name=None, speaker_user_id=None: "ok")

    release_together = threading.Barrier(2, timeout=5)

    def needs_summarization_after_barrier(self):
        release_together.wait()
        return True

    monkeypatch.setattr(ChatAgent, "needs_context_summarization", needs_summarization_after_barrier)

    admin = make_user("admin@example.com", "g-dedupe-admin")
    session_id = make_session_with_members(admin)

    # Prime the cache with one agent shared by both "concurrent" turns, same
    # as a real session would have (server.sessions caches one ChatAgent per
    # session_id, reused across requests/users).
    server._get_agent(session_id, admin)

    bg1, bg2 = BackgroundTasks(), BackgroundTasks()
    errors = []

    def run(background_tasks):
        try:
            server._schedule_context_summarization_if_needed(session_id, background_tasks)
        except Exception as e:  # noqa: BLE001 - captured for the assertion below
            errors.append(e)

    t1 = threading.Thread(target=run, args=(bg1,))
    t2 = threading.Thread(target=run, args=(bg2,))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive() and not t2.is_alive(), "a thread never returned -- deadlock?"
    assert not errors, f"a thread raised: {errors}"

    scheduled_counts = sorted([len(bg1.tasks), len(bg2.tasks)])
    assert scheduled_counts == [0, 1], (
        f"expected exactly one of the two concurrent callers to schedule a task, "
        f"got bg1={len(bg1.tasks)} bg2={len(bg2.tasks)}"
    )
    assert session_id in server._summarizing_sessions


def test_background_summarization_unlocked_phase_does_not_block_concurrent_chat_turn(fresh_server, monkeypatch):
    """The slow LLM call (phase 1) must run with no lock held -- a
    concurrent chat turn for the same session must complete immediately
    while phase 1 is in flight, not wait for it."""
    admin = make_user("admin@example.com", "g-concur-admin")
    session_id = make_session_with_members(admin)

    agent, _fake_llm = make_agent(session_id, monkeypatch)
    server.sessions[session_id] = agent

    monkeypatch.setattr(agent, "plan_context_summarization", lambda: ([], ["fake-id"]))

    applied = threading.Event()

    def fake_apply(ids, summary):
        applied.set()
        return True

    monkeypatch.setattr(agent, "apply_context_summarization", fake_apply)

    phase1_started = threading.Event()
    release_phase1 = threading.Event()

    def slow_summarize_rounds(config, existing_summary, messages):
        phase1_started.set()
        assert release_phase1.wait(timeout=5), "test deadlocked waiting to be released"
        return "a summary"

    monkeypatch.setattr(server, "summarize_rounds", slow_summarize_rounds)

    bg_thread = threading.Thread(target=server._maybe_summarize_context, args=(session_id,))
    bg_thread.start()
    assert phase1_started.wait(timeout=2), "phase 1 never started"

    monkeypatch.setattr(ChatAgent, "chat", lambda self, m, speaker_name=None, speaker_user_id=None: "ok")

    start = time.time()
    response, _state = server._run_chat_turn(
        session_id, admin, "hello", admin["name"],
        persist_user_message=lambda: db.add_chat_message(session_id, "user", "hello", user_id=admin["id"]),
    )
    elapsed = time.time() - start

    assert response == "ok"
    assert elapsed < 1.0, (
        f"concurrent chat turn took {elapsed}s -- it must not be blocked by "
        f"in-flight summarization's unlocked (slow LLM call) phase"
    )

    release_phase1.set()
    bg_thread.join(timeout=5)
    assert applied.is_set()


def test_background_summarization_locked_phase_blocks_concurrent_chat_turn_briefly(fresh_server, monkeypatch):
    """The final persist step (phase 2) DOES take the per-session lock, same
    as any other mutation of this session -- a concurrent chat turn must
    wait for it (confirming the lock is real, not a no-op), but only for
    that brief step."""
    admin = make_user("admin@example.com", "g-concur2-admin")
    session_id = make_session_with_members(admin)

    agent, _fake_llm = make_agent(session_id, monkeypatch)
    server.sessions[session_id] = agent

    monkeypatch.setattr(agent, "plan_context_summarization", lambda: ([], ["fake-id"]))
    monkeypatch.setattr(server, "summarize_rounds", lambda config, existing, messages: "a summary")

    phase2_started = threading.Event()
    release_phase2 = threading.Event()

    def blocking_apply(ids, summary):
        phase2_started.set()
        assert release_phase2.wait(timeout=5), "test deadlocked waiting to be released"
        return True

    monkeypatch.setattr(agent, "apply_context_summarization", blocking_apply)

    bg_thread = threading.Thread(target=server._maybe_summarize_context, args=(session_id,))
    bg_thread.start()
    assert phase2_started.wait(timeout=2), "phase 2 (locked) never started"

    monkeypatch.setattr(ChatAgent, "chat", lambda self, m, speaker_name=None, speaker_user_id=None: "ok")

    chat_done = threading.Event()

    def run_chat():
        server._run_chat_turn(
            session_id, admin, "hi", admin["name"],
            persist_user_message=lambda: db.add_chat_message(session_id, "user", "hi", user_id=admin["id"]),
        )
        chat_done.set()

    chat_thread = threading.Thread(target=run_chat)
    chat_thread.start()

    # The lock is held by the background task's phase 2 -- the chat turn
    # must still be waiting a short while later.
    assert not chat_done.wait(timeout=0.3), "chat turn completed despite the persist-phase lock being held"

    release_phase2.set()
    assert chat_done.wait(timeout=5), "chat turn never completed after the lock was released"
    bg_thread.join(timeout=5)


def test_maybe_summarize_context_noop_when_session_not_cached(fresh_server):
    """A session evicted (or never cached) between the triggering turn
    finishing and the background task running must be a clean no-op, not a
    crash."""
    server._maybe_summarize_context("some-session-id-not-in-cache")  # must not raise


def test_maybe_summarize_context_does_not_leak_lock_entry_for_deleted_session(fresh_server):
    """Regression test: _maybe_summarize_context's finally block must clear
    _summarizing_sessions via its own dedicated guard, NOT _get_session_lock
    -- a session can be deleted (its _session_locks entry dropped via
    _drop_session_lock) while a previously-scheduled summarization task for
    it is still pending. If that finally block called _get_session_lock(
    session_id) in that case, it would silently recreate a _session_locks
    entry for a session_id that will never be used again, leaking forever
    in a long-running process. The agent-is-None early return below is
    exactly the path a deleted session's pending task hits."""
    session_id = "sess-never-cached-" + str(uuid.uuid4())
    assert session_id not in server.sessions
    assert session_id not in server._session_locks

    server._maybe_summarize_context(session_id)

    assert session_id not in server._session_locks, (
        "finally block leaked a _session_locks entry for a deleted/never-cached session"
    )


# ---------- main.py's CLI path is completely unaffected ----------

def test_cli_path_never_touches_core_database(monkeypatch):
    """Faithfully mirrors main.py: core.database.init_db() is never called
    at all (main.py doesn't even import core.database), and ChatAgent.chat()
    is invoked with no speaker_name/speaker_user_id, exactly like the CLI's
    calls. If _agent_node's new context-summary injection (or anything else
    this task added) ever touched core.database on that path, this would
    raise -- db._db_path is still None, so even the cheapest read raises a
    TypeError (sqlite3.connect(None)) rather than silently succeeding.
    main.py also never imports fastapi.BackgroundTasks or server.py, so
    there is no summarization scheduling machinery reachable from the CLI
    at all -- this test only needs to cover the one thing that IS reachable
    from the CLI: ChatAgent.chat() itself.

    core.database._db_path is forced to None here (rather than asserted)
    since other tests in this suite call db.init_db()/db.create_db() and
    that module-global state is process-wide, not reset between tests --
    monkeypatch restores whatever it was afterward either way."""
    monkeypatch.setattr(db, "_db_path", None)
    monkeypatch.setattr("agents.chat_agent.create_llm", lambda config: FakeLLM())
    checkpointer.init_checkpointer(":memory:")

    agent = ChatAgent(session_id=str(uuid.uuid4()), config=ChatAgentConfig())
    # No speaker_name/speaker_user_id passed -- exactly main.py's call shape.
    reply = agent.chat("Hello, I'm ready to help split some bills.")
    assert reply == "ok"
    reply2 = agent.chat("we had pizza")
    assert reply2 == "ok"
