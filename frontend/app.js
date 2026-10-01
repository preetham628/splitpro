// ── Config ──────────────────────────────────────────────────────────────────
const API = '';

// ── State ────────────────────────────────────────────────────────────────────
let sessionId  = null;
let isLoading  = false;
let currentUser = null;
let members     = new Map();  // user_id -> {user_id, name, email, avatar_url, role}
let isCurrentUserAdmin = false;
let activeTab   = 'session';
// Last list fetched by loadSessionList() — kept around so the title bar
// (which needs just the *current* session's name/title_auto) doesn't need
// its own fetch; it's refreshed every time the sidebar is.
let sessionsCache = [];
// Bumped only by createNewSession/loadSession, checked only by
// createNewSession — this is "did the active session change" and is a
// distinct concern from loadProposals()'s own reentrancy guard below. They
// must not share a counter: an unrelated loadProposals() call (opening the
// Approvals tab, deciding a proposal) would otherwise falsely tell an
// in-flight createNewSession() it had been superseded when nobody actually
// navigated away.
let navSeq      = 0;
// Private to loadProposals() — "did a newer call to *this* function
// supersede an older one" (same-session reentrancy: reopening the tab,
// back-to-back decideProposal() calls). Combined with a sessionId-identity
// check (the cross-session case) rather than folded into navSeq above.
let proposalsLoadSeq = 0;
// Private to decideProposal() — guards only its final renderState() call
// against an older decideProposal() call's response arriving after a newer
// one's (e.g. approve one proposal, then immediately reject another). Bumped
// right before each call's own state fetch is issued (not at call-start),
// since issuance order — not call-start order — tracks actual data
// freshness: the gap between call-start and state-fetch-issuance (waiting on
// the decide POST and loadProposals()) has variable length per call, so a
// call that started first can still issue its state fetch last. Each call's
// own loadProposals()/state-fetch still runs to completion regardless of
// this counter — only the render is skipped when stale — so an older call
// isn't starved of its refresh just because a newer one has since started.
// Kept separate from proposalsLoadSeq/navSeq for the same reason those two
// are kept separate from each other: each counter is scoped to exactly the
// calls that can supersede one another, and no others.
let decideProposalSeq = 0;

// ── Live updates (WebSocket) ────────────────────────────────────────────────
// The currently-open live-update socket, and which session it belongs to.
// Kept as a pair (rather than inferring "whose socket is this" from
// `sessionId` alone) because closing the *old* socket when switching
// sessions needs to happen even after `sessionId` has already been
// reassigned to the new one.
let socket          = null;
let socketSessionId = null;
let socketReconnectTimer    = null;
let socketReconnectAttempts = 0;
// Session ids with a sendToAgent()/sendImageToAgent() REST call currently in
// flight *for this tab*. The server broadcasts a turn's new messages to
// every open socket on the session, including the one belonging to the tab
// that sent it — but that tab already renders the user's message
// (optimistically, in handleSend()) and the agent's reply (from the POST
// response itself) via the existing REST flow, so appending them *again*
// from the socket would duplicate every bubble this tab sends. Scoped by
// session (a Set, not a single boolean) rather than gated solely by the
// global `isLoading` flag, since loadSession() can switch the active
// session while a turn for the *previous* one is still in flight (isLoading
// doesn't block navigation) — a global flag would then also suppress the
// new session's unrelated incoming messages until that stale turn resolved.
let awaitingOwnTurnSessions = new Set();

// ── DOM refs ─────────────────────────────────────────────────────────────────
const messagesEl    = document.getElementById('messages');
const inputEl       = document.getElementById('message-input');
const sendBtn       = document.getElementById('send-btn');
const fileInput     = document.getElementById('file-input');
const providerBadge = document.getElementById('provider-badge');
const stateContent  = document.getElementById('state-content');
const loginOverlay  = document.getElementById('login-overlay');
const appEl         = document.getElementById('app');
const sessionList   = document.getElementById('session-list');
const newSessionBtn = document.getElementById('new-session-btn');
const chatTitleName  = document.getElementById('chat-title-name');
const autoNameToggle = document.getElementById('auto-name-toggle');
const sessionSidebarEl = document.querySelector('.session-sidebar');
const statePanelEl     = document.getElementById('state-panel');
const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
const toggleStateBtn   = document.getElementById('toggle-state-btn');
const panelBackdrop    = document.getElementById('panel-backdrop');
const userAvatar    = document.getElementById('user-avatar');
const userName      = document.getElementById('user-name');
const logoutBtn     = document.getElementById('logout-btn');

const stateTabButtons  = document.querySelectorAll('.state-tab');
const tabPanels = {
  session:   document.getElementById('tab-panel-session'),
  members:   document.getElementById('tab-panel-members'),
  approvals: document.getElementById('tab-panel-approvals'),
};
const approvalsTabBtn   = document.getElementById('approvals-tab');
const approvalsTabBadge = document.getElementById('approvals-tab-badge');
const membersContent    = document.getElementById('members-content');
const inviteEmailInput  = document.getElementById('invite-email-input');
const inviteBtn         = document.getElementById('invite-btn');
const approvalsContent  = document.getElementById('approvals-content');

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const res = await fetch(`${API}/auth/me`, { credentials: 'include' });
    if (!res.ok) {
      loginOverlay.classList.remove('hidden');
      return;
    }
    currentUser = await res.json();
  } catch {
    loginOverlay.classList.remove('hidden');
    return;
  }

  // Show app
  loginOverlay.classList.add('hidden');
  applyResponsivePanelDefaults();
  appEl.style.display = 'flex';

  // Populate user info in header
  userAvatar.src = currentUser.avatar_url || '';
  userAvatar.alt = currentUser.name || '';
  const displayName = currentUser.name || currentUser.email;
  userName.textContent = displayName;
  // Note: unlike the item-name spans (set via innerHTML template strings,
  // where title="${escapeHtml(...)}" is required so quotes/angle-brackets
  // in the name don't break out of the attribute), this is a direct DOM
  // property assignment — the string is used verbatim, so escaping it
  // would corrupt the tooltip for names containing &, <, >, ' or ".
  userName.title = displayName;

  await loadSessionList();

  // Restore last used session or create a new one
  const lastId = localStorage.getItem('lastSessionId');
  if (lastId) {
    // Verify it still exists in the list
    const res = await fetch(`${API}/api/sessions`, { credentials: 'include' });
    const sessions = await res.json();
    const found = sessions.find(s => s.id === lastId);
    if (found) {
      await loadSession(lastId);
      return;
    }
  }
  await createNewSession();
}

// ── Auth ──────────────────────────────────────────────────────────────────────
logoutBtn.addEventListener('click', async () => {
  await fetch(`${API}/auth/logout`, { method: 'POST', credentials: 'include' });
  location.reload();
});

// ── Session management ────────────────────────────────────────────────────────
async function loadSessionList() {
  const res = await fetch(`${API}/api/sessions`, { credentials: 'include' });
  const sessionsData = await res.json();
  sessionsCache = sessionsData;
  renderSessionList(sessionsData);
  updateChatTitleBar();
}

// Reflects the current session's name + auto-naming toggle in the title bar
// above the chat panel. Reads from sessionsCache rather than issuing its own
// fetch — it's refreshed everywhere the sidebar already is (loadSessionList
// is the single source for both), including right after a manual rename
// (startRename) or an auto-rename (sendToAgent/sendImageToAgent).
function updateChatTitleBar() {
  // Don't clobber an in-progress inline edit (startRename() flips this to
  // 'true' while the user is actively typing a new name) — a sidebar
  // refresh landing mid-edit would otherwise wipe out whatever they've
  // typed so far out from under them.
  if (chatTitleName.contentEditable === 'true') return;

  const current = sessionsCache.find(s => s.id === sessionId);
  if (!current) {
    chatTitleName.textContent = '';
    autoNameToggle.checked = false;
    return;
  }
  chatTitleName.textContent = current.name;
  chatTitleName.title = current.name;
  autoNameToggle.checked = !!current.title_auto;
}

function renderSessionList(sessions) {
  // Same bug class as updateChatTitleBar()'s guard, just on the sidebar's
  // own rename path: this function unconditionally wipes and rebuilds every
  // <li>, including whichever one's nameSpan startRename() just flipped to
  // contentEditable — a refresh landing mid-edit (e.g. from loadSessionList()
  // racing a dblclick-to-rename) would otherwise destroy that span and
  // recreate it fresh with the old name, silently discarding whatever the
  // user has typed so far. Bail out of the whole rebuild while any sidebar
  // item is being edited; the next call (e.g. right after that rename's own
  // finish() triggers loadSessionList()) picks up the latest data once the
  // edit is no longer in progress.
  if (sessionList.querySelector('.session-item-name[contenteditable="true"]')) return;

  sessionList.innerHTML = '';
  sessions.forEach(s => {
    const li = document.createElement('li');
    li.className = 'session-item' + (s.id === sessionId ? ' active' : '');
    li.dataset.id = s.id;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'session-item-name';
    nameSpan.textContent = s.name;
    nameSpan.title = s.name;

    // Double-click to rename
    nameSpan.addEventListener('dblclick', () => startRename(s.id, nameSpan));

    const delBtn = document.createElement('button');
    delBtn.className = 'session-delete-btn';
    delBtn.textContent = '✕';
    delBtn.title = 'Delete session';
    delBtn.addEventListener('click', e => {
      e.stopPropagation();
      deleteSession(s.id);
    });

    li.appendChild(nameSpan);
    li.appendChild(delBtn);
    li.addEventListener('click', () => loadSession(s.id));
    sessionList.appendChild(li);
  });
}

async function createNewSession() {
  const mySeq = ++navSeq;
  const res = await fetch(`${API}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({}),
  });
  const data = await res.json();
  // The user may have switched to (or created) a different session via
  // loadSession/createNewSession while the POST above was in flight — in
  // that case this call has nothing left to do; making its session active
  // now would clobber whatever the user has since navigated to.
  if (mySeq !== navSeq) return;

  sessionId = data.session_id;
  providerBadge.textContent = data.provider || '–';
  localStorage.setItem('lastSessionId', sessionId);
  connectSessionSocket(sessionId);

  await loadSessionList();
  if (mySeq !== navSeq) return;

  messagesEl.innerHTML = '';
  stateContent.innerHTML = '<p class="muted">No data yet.</p>';
  membersContent.innerHTML = '<p class="muted">Loading…</p>';
  switchTab('session');

  setActiveSession(sessionId);
  await loadMembers(sessionId);
  if (mySeq !== navSeq) return;
  await sendToAgent("Hello, I'm ready to help split some bills.");
}

async function loadSession(id) {
  navSeq++;  // supersede any in-flight createNewSession/loadSession call
  sessionId = id;
  localStorage.setItem('lastSessionId', id);
  setActiveSession(id);
  // Opened alongside (not awaited against) the REST catch-up fetches below —
  // this only needs to cover "stay live while already looking at it"; the
  // fetches below still do the "catch up on what happened while this tab
  // was closed/elsewhere" job they always have.
  connectSessionSocket(id);

  messagesEl.innerHTML = '';
  stateContent.innerHTML = '<p class="muted">Loading…</p>';
  membersContent.innerHTML = '<p class="muted">Loading…</p>';
  approvalsContent.innerHTML = '<p class="muted">No pending proposals.</p>';
  switchTab('session');
  updateChatTitleBar();

  // Member list is loaded first (and awaited) so both message attribution
  // (sender names) and the admin-only approval/member controls have the
  // current user's role and the id→name map ready before anything below
  // tries to use it — re-fetched on every session load, not cached
  // indefinitely, in case the user's role changed since they last opened it.
  await loadMembers(id);
  // The user may have switched to a different session while any of the
  // awaits above/below were in flight — `sessionId` (global) would then
  // point at that newer session while `id` (this call's own session) is
  // stale. Applying a stale response here would overwrite the now-current
  // session's messages/state/members with the abandoned one's, so every
  // checkpoint below bails out instead once that's detected.
  if (id !== sessionId) return;

  try {
    const res = await fetch(`${API}/sessions/${id}/messages`, { credentials: 'include' });
    if (res.ok) {
      const messages = await res.json();
      if (id !== sessionId) return;
      messages.forEach(m => {
        const imageDataUrl = m.image_base64
          ? `data:${m.image_media_type};base64,${m.image_base64}`
          : null;
        appendBubble(m.role, m.content, imageDataUrl, m.user_id);
      });
    }
  } catch {
    // Transcript failed to load — chat panel just starts empty for this session.
  }

  if (id !== sessionId) return;

  try {
    const res = await fetch(`${API}/sessions/${id}/state`, { credentials: 'include' });
    if (res.ok) {
      const state = await res.json();
      if (id !== sessionId) return;
      renderState(state);
    }
  } catch {
    if (id === sessionId) stateContent.innerHTML = '<p class="muted">No data yet.</p>';
  }
}

// ── Live updates (WebSocket) ────────────────────────────────────────────────
// Opens a fresh live-update connection for `id`, first tearing down whatever
// connection (and pending reconnect) belonged to the previously active
// session — a tab only ever needs one of these open at a time, for whichever
// session it's currently looking at.
function connectSessionSocket(id) {
  disconnectSessionSocket();
  socketSessionId = id;
  socketReconnectAttempts = 0;
  openSessionSocket(id);
}

function openSessionSocket(id) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws/sessions/${id}`);

  ws.onmessage = ev => {
    // Guards the same staleness the REST catch-up fetches in loadSession()
    // guard against — the user may have switched away from (or reloaded
    // past) the session this particular socket was opened for.
    if (id !== socketSessionId || id !== sessionId) return;
    let payload;
    try {
      payload = JSON.parse(ev.data);
    } catch {
      return;
    }
    handleSocketPayload(id, payload);
  };

  // Both a clean close (e.g. the server restarting) and a transport error
  // surface here — WebSocket always fires 'close' after 'error', so a single
  // reconnect path off onclose covers both without double-scheduling.
  ws.onclose = () => scheduleSocketReconnect(id);

  socket = ws;
}

function disconnectSessionSocket() {
  if (socketReconnectTimer) {
    clearTimeout(socketReconnectTimer);
    socketReconnectTimer = null;
  }
  socketSessionId = null;
  if (socket) {
    const s = socket;
    socket = null;
    // Detach first — this is an intentional close (switching sessions, or
    // superseded by a newer connectSessionSocket call), not a drop that
    // should trigger scheduleSocketReconnect.
    s.onmessage = null;
    s.onclose = null;
    s.close();
  }
}

function scheduleSocketReconnect(id) {
  // Only reconnect for the session the user is still actually looking at —
  // a close belonging to a session they've since navigated away from (or
  // one disconnectSessionSocket already detached) should just be dropped.
  if (id !== socketSessionId || id !== sessionId) return;
  socketReconnectAttempts++;
  // Capped exponential backoff (1s, 2s, 4s, ... up to 15s) rather than
  // hammering the server through an outage, but never gives up outright —
  // per task14's acceptance bar, silently going stale forever isn't an
  // option, so this keeps trying indefinitely at a bounded rate. A page
  // reload (which re-runs init()/loadSession() from scratch) remains the
  // immediate fallback if this is somehow still reconnecting.
  const delay = Math.min(1000 * 2 ** (socketReconnectAttempts - 1), 15000);
  socketReconnectTimer = setTimeout(() => {
    if (id === socketSessionId && id === sessionId) openSessionSocket(id);
  }, delay);
}

// Applies one pushed update to the chat log / state panel in place — no
// full page reload, no re-running loadSession()'s fetch sequence.
function handleSocketPayload(id, payload) {
  if (payload.state) renderState(payload.state);

  if (payload.messages && !awaitingOwnTurnSessions.has(id)) {
    payload.messages.forEach(m => {
      const imageDataUrl = m.image_base64
        ? `data:${m.image_media_type};base64,${m.image_base64}`
        : null;
      appendBubble(m.role, m.content, imageDataUrl, m.user_id);
    });
  }
}

// ── Members / roles ──────────────────────────────────────────────────────────
async function loadMembers(id) {
  let list = null;
  try {
    const res = await fetch(`${API}/api/sessions/${id}/members`, { credentials: 'include' });
    if (res.ok) list = await res.json();
  } catch {
    // list stays null — treated as a failed fetch below.
  }

  // Bail out without touching global state if the user has since switched
  // to a different session — an in-flight fetch for a session that's no
  // longer current must not clobber the now-current session's member list
  // or admin flag (see the matching guard in loadSession()).
  if (id !== sessionId) return;

  if (list) {
    members = new Map(list.map(m => [m.user_id, m]));
    const me = members.get(currentUser.id);
    isCurrentUserAdmin = !!(me && me.role === 'admin');
  } else {
    members = new Map();
    isCurrentUserAdmin = false;
  }
  renderMembers();
  updateAdminVisibility();
}

function updateAdminVisibility() {
  approvalsTabBtn.hidden = !isCurrentUserAdmin;
  if (!isCurrentUserAdmin && activeTab === 'approvals') switchTab('session');
}

function senderName(userId) {
  if (userId == null) return 'Someone';
  const m = members.get(userId);
  return m ? (m.name || m.email) : 'Unknown';
}

function renderMembers() {
  membersContent.innerHTML = '';
  if (members.size === 0) {
    membersContent.innerHTML = '<p class="muted">No members.</p>';
    return;
  }

  const list = document.createElement('div');
  list.className = 'member-list';

  members.forEach(m => {
    const row = document.createElement('div');
    row.className = 'member-row';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    if (m.avatar_url) {
      const img = document.createElement('img');
      img.src = m.avatar_url;
      img.alt = '';
      avatar.appendChild(img);
    } else {
      avatar.textContent = (m.name || m.email || '?').charAt(0).toUpperCase();
    }

    const info = document.createElement('div');
    info.className = 'member-info';
    const nameLine = document.createElement('div');
    nameLine.className = 'member-name';
    const label = m.name || m.email;
    nameLine.textContent = m.user_id === currentUser.id ? `${label} (you)` : label;
    const roleLine = document.createElement('div');
    roleLine.className = 'member-role';
    roleLine.textContent = m.role;
    info.appendChild(nameLine);
    info.appendChild(roleLine);

    row.appendChild(avatar);
    row.appendChild(info);

    if (isCurrentUserAdmin && m.user_id !== currentUser.id) {
      const actions = document.createElement('div');
      actions.className = 'member-actions';

      const toggleBtn = document.createElement('button');
      toggleBtn.className = 'member-action-btn';
      toggleBtn.textContent = m.role === 'admin' ? 'Demote' : 'Promote';
      toggleBtn.addEventListener('click', () =>
        changeMemberRole(m.user_id, m.role === 'admin' ? 'member' : 'admin'));

      const removeBtn = document.createElement('button');
      removeBtn.className = 'member-action-btn danger';
      removeBtn.textContent = 'Remove';
      removeBtn.addEventListener('click', () => removeMember(m.user_id));

      actions.appendChild(toggleBtn);
      actions.appendChild(removeBtn);
      row.appendChild(actions);
    }

    list.appendChild(row);
  });

  membersContent.appendChild(list);
}

async function changeMemberRole(userId, role) {
  try {
    const res = await fetch(`${API}/api/sessions/${sessionId}/members/${userId}/role`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ role }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || 'Failed to update role.');
      return;
    }
    await loadMembers(sessionId);
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

async function removeMember(userId) {
  if (!confirm('Remove this member from the session?')) return;
  try {
    const res = await fetch(`${API}/api/sessions/${sessionId}/members/${userId}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (!res.ok) {
      // Surface the backend's own message (e.g. "last admin") rather than
      // pre-validating that rule client-side — the backend is the source
      // of truth for it.
      const err = await res.json().catch(() => ({}));
      alert(err.detail || 'Failed to remove member.');
      return;
    }
    await loadMembers(sessionId);
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

inviteBtn.addEventListener('click', async () => {
  const email = inviteEmailInput.value.trim();
  if (!email || !sessionId) return;
  inviteBtn.disabled = true;
  try {
    const res = await fetch(`${API}/api/sessions/${sessionId}/members`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ email }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      alert(data.detail || 'Failed to invite member.');
      return;
    }
    inviteEmailInput.value = '';
    await loadMembers(sessionId);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    inviteBtn.disabled = false;
  }
});

inviteEmailInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); inviteBtn.click(); }
});

// ── Approval view ────────────────────────────────────────────────────────────
async function loadProposals() {
  // Two independent checks for two independent kinds of staleness:
  // requestSessionId catches a session switch (loadSession changed which
  // session is active — doesn't necessarily call loadProposals() itself,
  // e.g. it always resets to the Session tab, not Approvals); mySeq catches
  // a newer loadProposals() call for the *same* session superseding an
  // older, slower one (reopening the tab, back-to-back decideProposal()
  // calls). Neither alone covers both cases, and this deliberately doesn't
  // touch navSeq — that counter is for createNewSession/loadSession's own,
  // different, purpose.
  const requestSessionId = sessionId;
  const mySeq = ++proposalsLoadSeq;
  const isStale = () => requestSessionId !== sessionId || mySeq !== proposalsLoadSeq;
  approvalsContent.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const res = await fetch(`${API}/api/sessions/${requestSessionId}/proposals`, { credentials: 'include' });
    if (isStale()) return;
    if (!res.ok) {
      approvalsContent.innerHTML = '<p class="muted">Failed to load proposals.</p>';
      return;
    }
    const proposals = await res.json();
    if (isStale()) return;
    renderProposals(proposals);
  } catch {
    if (!isStale()) {
      approvalsContent.innerHTML = '<p class="muted">Failed to load proposals.</p>';
    }
  }
}

function renderProposals(proposals) {
  if (!proposals.length) {
    approvalsContent.innerHTML = '<p class="muted">No pending proposals.</p>';
    return;
  }

  approvalsContent.innerHTML = '';
  proposals.forEach(p => {
    const payload = p.payload || {};

    // A remove_bill proposal (see agents/chat_agent.py's remove_bill tool)
    // has no items/tax/tip/paid_by — it's just {action, bill_id}. Render it
    // as its own simple card instead of falling through to the normal
    // items/total rendering below, which would otherwise show a misleading
    // "Untitled expense · Payer unknown · $0.00" card with no items.
    if (payload.action === 'remove_bill') {
      const card = document.createElement('div');
      card.className = 'bill-card proposal-card';
      card.innerHTML = `<div class="bill-title">Remove bill ${escapeHtml(payload.bill_id || '')}</div>
        <div class="bill-meta">This will delete the bill and all its items.</div>`;

      const actions = document.createElement('div');
      actions.className = 'proposal-actions';

      const approveBtn = document.createElement('button');
      approveBtn.className = 'proposal-btn approve';
      approveBtn.textContent = 'Approve';
      approveBtn.addEventListener('click', () => decideProposal(p.id, 'approve'));

      const rejectBtn = document.createElement('button');
      rejectBtn.className = 'proposal-btn reject';
      rejectBtn.textContent = 'Reject';
      rejectBtn.addEventListener('click', () => decideProposal(p.id, 'reject'));

      actions.appendChild(approveBtn);
      actions.appendChild(rejectBtn);
      card.appendChild(actions);

      approvalsContent.appendChild(card);
      return;
    }

    const items = payload.items || [];
    const tax = payload.tax || 0;
    const tip = payload.tip || 0;
    // item.price is already the line total for all of its qty units (see
    // LineItem.unit_price / Bill.subtotal in core/session_state.py) — do
    // not re-multiply by qty here, that double-counts.
    const total = items.reduce((sum, i) => sum + i.price, 0) + tax + tip;
    const payer = formatPaidBy(payload.paid_by);

    const card = document.createElement('div');
    card.className = 'bill-card proposal-card';

    // p.is_correction (server-computed, freshly checked against live bills
    // rather than trusted from the stale supersedes_bill_id FK — see
    // core/database.py's _is_correction) marks a proposal that will
    // overwrite an existing bill rather than create a new one. Falls back
    // to false — and the badge just doesn't render — on any backend build
    // that doesn't send the field yet.
    const correctionBadge = p.is_correction
      ? '<span class="proposal-badge">Correction</span>' : '';

    let html = `<div class="bill-title">${escapeHtml(payload.description || 'Untitled expense')}${correctionBadge}</div>
      <div class="bill-meta">${escapeHtml(payer)} · $${total.toFixed(2)}</div>`;

    items.forEach(item => {
      const isUnassigned = item.unassigned || !item.assigned_to || item.assigned_to.length === 0;
      const assignText = isUnassigned
        ? 'unassigned'
        : item.assigned_to.join(', ') + (item.shared ? ' (shared)' : '');
      const splitText = formatItemSplit(item);
      html += `<div class="item-block">
        <div class="item-row">
          <span class="item-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
          <span class="item-price">$${item.price.toFixed(2)}</span>
          <span class="item-assign ${isUnassigned ? 'unassigned' : ''}">${escapeHtml(assignText)}</span>
        </div>${splitText ? `<div class="item-split">${escapeHtml(splitText)}</div>` : ''}
      </div>`;
    });

    if (tax > 0 || tip > 0) {
      html += `<div class="item-block"><div class="item-row">
        <span class="item-name" style="color:var(--muted)">Tax + Tip</span>
        <span class="item-price">$${(tax + tip).toFixed(2)}</span>
        <span class="item-assign">proportional</span>
      </div></div>`;
    }

    card.innerHTML = html;

    const actions = document.createElement('div');
    actions.className = 'proposal-actions';

    const approveBtn = document.createElement('button');
    approveBtn.className = 'proposal-btn approve';
    approveBtn.textContent = 'Approve';
    approveBtn.addEventListener('click', () => decideProposal(p.id, 'approve'));

    const rejectBtn = document.createElement('button');
    rejectBtn.className = 'proposal-btn reject';
    rejectBtn.textContent = 'Reject';
    rejectBtn.addEventListener('click', () => decideProposal(p.id, 'reject'));

    actions.appendChild(approveBtn);
    actions.appendChild(rejectBtn);
    card.appendChild(actions);

    approvalsContent.appendChild(card);
  });
}

async function decideProposal(proposalId, decision) {
  // The user can switch sessions while this is in flight (approve/reject
  // isn't gated the way isLoading gates chat sends) — re-check before
  // refreshing the approvals/state panels so a slow decision on a session
  // the user has since left can't overwrite what's now on screen.
  //
  // mySeq covers the same-session case: two decideProposal() calls in quick
  // succession (e.g. approve one proposal, then immediately reject another)
  // whose state re-fetches can resolve out of order. It's captured right
  // before the state fetch is issued — not at the top of this function —
  // because the gap between call-start and state-fetch-issuance (waiting on
  // the decide POST and loadProposals()) has variable length per call, so
  // issuance order (which tracks actual data freshness) can differ from
  // call-start order. It only gates the final renderState() call, not
  // loadProposals()/the state fetch themselves — an older call must still be
  // allowed to run its own refresh to completion (loadProposals() already
  // guards its own render internally), otherwise a newer call starting
  // before the older call's POST even resolves would make the older call
  // skip refreshing entirely, and if the newer call then failed before
  // reaching its own refresh, nobody would refresh the UI at all despite the
  // decision having committed server-side.
  const requestSessionId = sessionId;
  try {
    const res = await fetch(
      `${API}/api/sessions/${requestSessionId}/proposals/${proposalId}/${decision}`,
      { method: 'POST', credentials: 'include' },
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || `Failed to ${decision} proposal.`);
      return;
    }
    if (requestSessionId !== sessionId) return;
    // A decided proposal changes bills/settlement/pending_proposals_count —
    // refresh both the proposal list and the state panel.
    await loadProposals();
    const mySeq = ++decideProposalSeq;
    const isStale = () => requestSessionId !== sessionId || mySeq !== decideProposalSeq;
    const stateRes = await fetch(`${API}/sessions/${requestSessionId}/state`, { credentials: 'include' });
    if (stateRes.ok) {
      const state = await stateRes.json();
      if (!isStale()) renderState(state);
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(name) {
  if (name === 'approvals' && !isCurrentUserAdmin) return;
  activeTab = name;
  stateTabButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === name));
  Object.entries(tabPanels).forEach(([key, el]) => el.classList.toggle('hidden', key !== name));
  if (name === 'approvals') loadProposals();
}

stateTabButtons.forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// ── Collapsible panel toggles ────────────────────────────────────────────────
// Standard drawer pattern: on a wide screen the toggle buttons just let the
// user reclaim screen space from a panel they don't need; below the
// responsive breakpoints (900px for the session sidebar, 640px for the
// state panel) each panel starts collapsed, and these same buttons are the
// *only* way to bring it back — there is no other dead-end `display: none`
// left in style.css. Once a panel has been toggled by hand during this page
// load, resizing the window no longer overrides that explicit choice.
//
// Below its breakpoint, a reopened panel overlays the chat panel (fixed
// position, via the media queries in style.css) instead of sharing flex
// space with it — pushing the layout instead would squeeze the chat column
// down to near nothing on a phone-width screen. The toggle buttons live in
// the top header (not the chat title bar, and not the panels themselves)
// precisely so they're never covered by that overlay and stay reachable
// the whole time. The backdrop below is the overlay's scrim.
const sidebarOverlayMQ = window.matchMedia('(max-width: 900px)');
const statePanelOverlayMQ = window.matchMedia('(max-width: 640px)');
let sidebarManuallySet = false;
let statePanelManuallySet = false;

function setSidebarCollapsed(collapsed) {
  sessionSidebarEl.classList.toggle('collapsed', collapsed);
  toggleSidebarBtn.classList.toggle('active', !collapsed);
  updatePanelBackdrop();
}

function setStatePanelCollapsed(collapsed) {
  statePanelEl.classList.toggle('collapsed', collapsed);
  toggleStateBtn.classList.toggle('active', !collapsed);
  updatePanelBackdrop();
}

// Shows the scrim whenever a panel is open *and* currently in overlay mode
// (i.e. narrow enough that style.css positions it fixed over the chat
// panel) — never while a panel is merely sharing flex space on a wide
// screen, where there's nothing behind it that needs dimming.
function updatePanelBackdrop() {
  const sidebarOverlayOpen = sidebarOverlayMQ.matches && !sessionSidebarEl.classList.contains('collapsed');
  const stateOverlayOpen   = statePanelOverlayMQ.matches && !statePanelEl.classList.contains('collapsed');
  panelBackdrop.hidden = !(sidebarOverlayOpen || stateOverlayOpen);
}

// Closes the *other* panel first when it's currently open as an overlay —
// both panels can coexist fine side-by-side on a wide screen, but two
// fixed-position overlays at a narrow width physically overlap each other
// (see style.css's media queries), so only one overlay may be open at a
// time. No-ops (and leaves manuallySet alone) when the other panel isn't
// actually in overlay mode or isn't open, so this never fights the normal
// wide-screen "both panels visible" case.
function closeStatePanelOverlayIfOpen() {
  if (statePanelOverlayMQ.matches && !statePanelEl.classList.contains('collapsed')) {
    statePanelManuallySet = true;
    setStatePanelCollapsed(true);
  }
}

function closeSidebarOverlayIfOpen() {
  if (sidebarOverlayMQ.matches && !sessionSidebarEl.classList.contains('collapsed')) {
    sidebarManuallySet = true;
    setSidebarCollapsed(true);
  }
}

toggleSidebarBtn.addEventListener('click', () => {
  sidebarManuallySet = true;
  const willOpen = sessionSidebarEl.classList.contains('collapsed');
  if (willOpen && sidebarOverlayMQ.matches) closeStatePanelOverlayIfOpen();
  setSidebarCollapsed(!willOpen);
});

toggleStateBtn.addEventListener('click', () => {
  statePanelManuallySet = true;
  const willOpen = statePanelEl.classList.contains('collapsed');
  if (willOpen && statePanelOverlayMQ.matches) closeSidebarOverlayIfOpen();
  setStatePanelCollapsed(!willOpen);
});

// Tapping the scrim closes whichever overlay panel(s) are currently open —
// same as tapping outside a mobile drawer in Discord/Slack/WhatsApp Web.
panelBackdrop.addEventListener('click', () => {
  closeSidebarOverlayIfOpen();
  closeStatePanelOverlayIfOpen();
});

// Applies the default collapsed/expanded state for each panel based on the
// current viewport width — but only for whichever panel the user hasn't
// already overridden by hand via the toggle buttons above. Always
// recomputes the backdrop afterwards, unconditionally: setSidebarCollapsed/
// setStatePanelCollapsed only run (and so only update the backdrop) inside
// their respective `if (!...ManuallySet)` branch, so once both panels have
// been toggled by hand at least once, neither branch would otherwise fire
// on a later resize — leaving a stale backdrop visible (e.g. open both as
// overlays at a narrow width, then resize back to desktop: without this,
// the backdrop would stay up and block every click across the whole app).
function applyResponsivePanelDefaults() {
  if (!sidebarManuallySet) {
    setSidebarCollapsed(sidebarOverlayMQ.matches);
  }
  if (!statePanelManuallySet) {
    setStatePanelCollapsed(statePanelOverlayMQ.matches);
  }
  updatePanelBackdrop();
}

window.addEventListener('resize', applyResponsivePanelDefaults);

function setActiveSession(id) {
  document.querySelectorAll('.session-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });
}

async function deleteSession(id) {
  if (!confirm('Delete this session?')) return;
  await fetch(`${API}/api/sessions/${id}`, {
    method: 'DELETE', credentials: 'include',
  });
  if (id === sessionId) {
    sessionId = null;
    localStorage.removeItem('lastSessionId');
    messagesEl.innerHTML = '';
    stateContent.innerHTML = '<p class="muted">No data yet.</p>';
    // The server already drops its side of any sockets on this session (see
    // server.py's delete_session), but this tab's own socket would
    // otherwise sit around pointed at a session that no longer exists until
    // createNewSession()/loadSession() replaces it below.
    disconnectSessionSocket();
  }
  await loadSessionList();
  if (!sessionId) await createNewSession();
}

function startRename(id, nameSpan) {
  nameSpan.contentEditable = 'true';
  nameSpan.focus();
  // Select all text
  const range = document.createRange();
  range.selectNodeContents(nameSpan);
  window.getSelection().removeAllRanges();
  window.getSelection().addRange(range);

  const finish = async () => {
    nameSpan.contentEditable = 'false';
    const newName = nameSpan.textContent.trim() || 'New Session';
    nameSpan.textContent = newName;
    await fetch(`${API}/api/sessions/${id}/name`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ name: newName }),
    });
    // A manual rename also turns off auto-naming server-side (title_auto ->
    // false) — refresh so both the sidebar and the title bar's toggle pick
    // that up, regardless of which of the two UI spots started this edit.
    await loadSessionList();
  };

  nameSpan.addEventListener('blur', finish, { once: true });
  nameSpan.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); nameSpan.blur(); }
    if (e.key === 'Escape') { nameSpan.blur(); }
  }, { once: true });
}

newSessionBtn.addEventListener('click', createNewSession);

// Title bar: double-click to rename, same wiring as the sidebar's own
// session-name span — reuses startRename() rather than duplicating its
// rename/PATCH logic.
chatTitleName.addEventListener('dblclick', () => {
  if (!sessionId) return;
  startRename(sessionId, chatTitleName);
});

autoNameToggle.addEventListener('change', async () => {
  if (!sessionId) return;
  const auto = autoNameToggle.checked;
  try {
    await fetch(`${API}/api/sessions/${sessionId}/title-auto`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ auto }),
    });
  } finally {
    await loadSessionList();
  }
});

// ── Send helpers ──────────────────────────────────────────────────────────────
async function sendToAgent(text) {
  if (!sessionId || isLoading) return;
  // loadSession() isn't gated by isLoading, so the user can switch sessions
  // while this turn is in flight — capture which session this response
  // belongs to and re-check before applying it to (what may now be a
  // different session's) DOM.
  const requestSessionId = sessionId;
  setLoading(true);
  // The server broadcasts this turn's messages to every open socket on
  // requestSessionId, including this tab's own — but this tab is about to
  // render the agent's reply itself from the POST response below (the
  // user's own message is already in the log, appended optimistically by
  // handleSend()), so the socket's echo of this same turn must be ignored
  // rather than appended a second time. See awaitingOwnTurnSessions' own
  // comment for why this is scoped per-session rather than a single flag.
  awaitingOwnTurnSessions.add(requestSessionId);
  const typing = appendTyping();
  try {
    const res  = await fetch(`${API}/sessions/${requestSessionId}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body:    JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.remove();
    if (requestSessionId !== sessionId) return;
    appendBubble('agent', data.response);
    renderState(data.state);
    // Refresh sidebar in case session was auto-renamed
    await loadSessionList();
    setActiveSession(sessionId);
  } catch (err) {
    typing.remove();
    if (requestSessionId === sessionId) appendBubble('agent', `⚠️ Error: ${err.message}`);
  } finally {
    setLoading(false);
    awaitingOwnTurnSessions.delete(requestSessionId);
  }
}

async function sendImageToAgent(file) {
  if (!sessionId || isLoading) return;
  // See the matching comment in sendToAgent() — loadSession() can run
  // concurrently with this (it isn't gated by isLoading), so every step
  // below re-checks it's still acting on the session this upload started in.
  const requestSessionId = sessionId;
  setLoading(true);
  // See the matching comment in sendToAgent() — suppresses this tab's own
  // socket echo of the turn this call is about to produce.
  awaitingOwnTurnSessions.add(requestSessionId);

  const reader = new FileReader();
  reader.onload = e => {
    if (requestSessionId === sessionId) appendBubble('user', '', e.target.result, currentUser.id);
  };
  reader.readAsDataURL(file);

  const typing = appendTyping();
  try {
    const form = new FormData();
    form.append('file', file);
    const res  = await fetch(`${API}/sessions/${requestSessionId}/image`, {
      method: 'POST',
      credentials: 'include',
      body:   form,
    });
    const data = await res.json();
    typing.remove();
    if (requestSessionId !== sessionId) return;
    appendBubble('agent', data.response);
    renderState(data.state);
    await loadSessionList();
    setActiveSession(sessionId);
  } catch (err) {
    typing.remove();
    if (requestSessionId === sessionId) appendBubble('agent', `⚠️ Error: ${err.message}`);
  } finally {
    setLoading(false);
    awaitingOwnTurnSessions.delete(requestSessionId);
  }
}

// ── UI builders ───────────────────────────────────────────────────────────────
function appendBubble(role, text, imageDataUrl = null, userId = null) {
  const isOwn = role === 'user' && !!currentUser && userId === currentUser.id;

  const row = document.createElement('div');
  row.className = `bubble-row ${role}` + (isOwn ? ' own' : '');

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  if (role === 'agent') {
    avatar.textContent = '🤖';
  } else if (isOwn) {
    avatar.textContent = '🙂';
  } else {
    const m = members.get(userId);
    if (m && m.avatar_url) {
      const img = document.createElement('img');
      img.src = m.avatar_url;
      img.alt = '';
      avatar.appendChild(img);
    } else {
      avatar.textContent = ((m && (m.name || m.email)) || '?').charAt(0).toUpperCase();
    }
  }

  const col = document.createElement('div');
  col.className = 'bubble-col';

  // Own messages are self-evident; label everyone else's (other members and
  // the agent) so the transcript reads as a group chat, not a 1:1.
  if (!isOwn) {
    const label = document.createElement('div');
    label.className = 'sender-label';
    label.textContent = role === 'agent' ? 'SplitPro' : senderName(userId);
    col.appendChild(label);
  }

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (imageDataUrl) {
    const img = document.createElement('img');
    img.className = 'preview';
    img.src = imageDataUrl;
    bubble.appendChild(img);
  }

  if (text) {
    const content = document.createElement('div');
    content.innerHTML = role === 'agent'
      ? marked.parse(text)
      : escapeHtml(text);
    bubble.appendChild(content);
  }

  col.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(col);
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function appendTyping() {
  const row = document.createElement('div');
  row.className = 'bubble-row agent typing';
  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function setLoading(on) {
  isLoading = on;
  sendBtn.disabled = on;
  inputEl.disabled = on;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
            .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// paid_by is always a dict (person -> dollar amount, possibly empty), never
// a plain string — returns plain (unescaped) text; callers are responsible
// for passing the result through escapeHtml(), same as the old plain-string
// `payer` variable this replaces.
function formatPaidBy(paidBy) {
  const entries = Object.entries(paidBy || {});
  if (entries.length === 0) return 'Payer unknown';
  if (entries.length === 1) return `Paid by ${entries[0][0]}`;
  return 'Paid by ' + entries.map(([name, amount]) => `${name} ($${amount.toFixed(2)})`).join(', ');
}

// Mirrors core/settlement.py's compute_balances cost_allocations handling
// (explicit per-person dollar amounts, remainder split equally among the
// rest of assigned_to) so the displayed per-item breakdown matches what the
// settlement actually charges each person. Returns null when the item has
// no cost_allocations to show (plain equal split, nothing to break down).
function computeItemShares(item) {
  // compute_balances checks `unassigned` before `cost_allocations` — an
  // explicit "split this evenly" marker always wins over a stale allocation
  // map left behind by whatever previously split this item (see
  // mark_items_unassigned, which sets unassigned=True precisely to force
  // this). Mirror that ordering here, or a stale map renders a wrong
  // breakdown right next to the "unassigned" badge.
  if (item.unassigned) return null;

  const assigned = item.assigned_to || [];
  const allocations = item.cost_allocations || {};
  if (!assigned.length || !Object.keys(allocations).length) return null;

  const shares = {};
  let allocatedAmount = 0;
  for (const [person, amount] of Object.entries(allocations)) {
    shares[person] = amount;
    allocatedAmount += amount;
  }

  const remaining = item.price - allocatedAmount;
  const remainderPeople = assigned.filter(p => !(p in allocations));
  if (remaining > 1e-9 && remainderPeople.length) {
    const share = remaining / remainderPeople.length;
    remainderPeople.forEach(p => { shares[p] = (shares[p] || 0) + share; });
  }

  // Defensive fallback for a narrow edge case this per-item function can't
  // fully replicate: compute_balances' remainder distribution falls back to
  // *all session participants* (not just this item's assigned_to) when
  // cost_allocations already names everyone in assigned_to but sums
  // slightly short of item.price under N-way validation tolerance — context
  // this function doesn't have without threading the full participant list
  // through. Rather than do that, just suppress the breakdown if the shares
  // computed here don't actually add back up to item.price within a sane
  // (per-person cent-rounding) tolerance, rather than risk showing numbers
  // that silently disagree with the real settlement.
  const sum = Object.values(shares).reduce((a, b) => a + b, 0);
  const sumTolerance = 0.01 * Math.max(assigned.length, 1);
  if (Math.abs(sum - item.price) > sumTolerance) return null;

  return shares;
}

// Returns plain (unescaped) "Name: $X.XX, Name: $Y.YY" text for an item's
// per-person breakdown, or null when there's nothing worth showing — either
// no cost_allocations at all, or the resulting shares are indistinguishable
// from a plain equal split (price / number of assigned people), which the
// item-assign text already conveys without a redundant number.
function formatItemSplit(item) {
  const shares = computeItemShares(item);
  if (!shares) return null;

  const assigned = item.assigned_to || [];
  const n = assigned.length;

  // Per-person tolerance scaled by group size, same idea as
  // validate_contribution_map (core/session_state.py): splitting a total
  // evenly to the cent is frequently impossible (e.g. $100 / 3 =
  // {33.33, 33.33, 33.34}, ~$0.0067 off the exact 33.333... share), and a
  // flat threshold tight enough to catch a real uneven split is too tight
  // for that unavoidable rounding remainder once there are 3+ people.
  const equalShare = item.price / n;
  const tolerance = 0.005 * Math.max(n, 1);
  const isEqual = assigned.every(p => Math.abs((shares[p] || 0) - equalShare) < tolerance);
  if (isEqual) return null;

  return assigned.map(p => `${p}: $${(shares[p] || 0).toFixed(2)}`).join(', ');
}

// ── State panel renderer ──────────────────────────────────────────────────────
function renderState(state) {
  if (!state) return;

  const pendingCount = state.pending_proposals_count || 0;
  approvalsTabBadge.textContent = String(pendingCount);
  approvalsTabBadge.hidden = pendingCount === 0;

  let html = '';

  if (pendingCount > 0) {
    const plural = pendingCount === 1 ? '' : 's';
    html += isCurrentUserAdmin
      ? `<div class="pending-alert">⏳ <strong>${pendingCount}</strong> pending proposal${plural} — <button class="link-btn" id="review-proposals-btn">Review</button></div>`
      : `<div class="pending-alert readonly">⏳ <strong>${pendingCount}</strong> pending proposal${plural} awaiting admin review</div>`;
  }

  html += '<div class="section-label">Participants</div>';
  if (state.participants && state.participants.length) {
    html += '<div class="pills">';
    state.participants.forEach(p => {
      html += `<span class="pill">${escapeHtml(p)}</span>`;
    });
    html += '</div>';
  } else {
    html += '<p class="muted">Not set yet</p>';
  }

  if (state.bills && state.bills.length) {
    html += '<div class="section-label" style="margin-top:16px">Bills</div>';
    state.bills.forEach(bill => {
      const payer = formatPaidBy(bill.paid_by);
      html += `<div class="bill-card">
        <div class="bill-title">${escapeHtml(bill.description)}</div>
        <div class="bill-meta">${escapeHtml(payer)} · $${bill.total.toFixed(2)}</div>`;

      bill.items.forEach(item => {
        const isUnassigned = item.unassigned || !item.assigned_to || item.assigned_to.length === 0;
        const assignText   = isUnassigned
          ? 'unassigned'
          : item.assigned_to.join(', ') + (item.shared ? ' (shared)' : '');
        const splitText = formatItemSplit(item);
        html += `<div class="item-block">
          <div class="item-row">
            <span class="item-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
            <span class="item-price">$${item.price.toFixed(2)}</span>
            <span class="item-assign ${isUnassigned ? 'unassigned' : ''}">${escapeHtml(assignText)}</span>
          </div>${splitText ? `<div class="item-split">${escapeHtml(splitText)}</div>` : ''}
        </div>`;
      });

      if (bill.tax > 0 || bill.tip > 0) {
        html += `<div class="item-block"><div class="item-row">
          <span class="item-name" style="color:var(--muted)">Tax + Tip</span>
          <span class="item-price">$${(bill.tax + bill.tip).toFixed(2)}</span>
          <span class="item-assign">proportional</span>
        </div></div>`;
      }

      html += '</div>';
    });
  }

  // Settlement is a live view, recomputed from current bills/payers on every
  // state fetch (see server.py's _serialize_state) — there's no "finalize"
  // step anymore, so this renders as soon as there's anything to settle.
  if (state.settlement && state.settlement.length) {
    html += `<div class="settlement-card">
      <h3>✅ Settlement</h3>
      <p class="settlement-subtitle">Updates live as bills and payers change</p>`;
    state.settlement.forEach(txn => {
      html += `<div class="txn-row">
        ${escapeHtml(txn.from)} → ${escapeHtml(txn.to)}: <strong>$${txn.amount.toFixed(2)}</strong>
      </div>`;
    });
    html += '</div>';
  }

  stateContent.innerHTML = html || '<p class="muted">No data yet.</p>';

  const reviewBtn = document.getElementById('review-proposals-btn');
  if (reviewBtn) reviewBtn.addEventListener('click', () => switchTab('approvals'));
}

// ── Event listeners ───────────────────────────────────────────────────────────
sendBtn.addEventListener('click', handleSend);

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    sendImageToAgent(file);
    fileInput.value = '';
  }
});

function handleSend() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;
  appendBubble('user', text, null, currentUser.id);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendToAgent(text);
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
