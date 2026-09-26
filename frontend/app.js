// ── Config ──────────────────────────────────────────────────────────────────
const API = '';

// ── State ────────────────────────────────────────────────────────────────────
let sessionId  = null;
let isLoading  = false;
let currentUser = null;
let members     = new Map();  // user_id -> {user_id, name, email, avatar_url, role}
let isCurrentUserAdmin = false;
let activeTab   = 'session';

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
  appEl.style.display = 'flex';

  // Populate user info in header
  userAvatar.src = currentUser.avatar_url || '';
  userAvatar.alt = currentUser.name || '';
  userName.textContent = currentUser.name || currentUser.email;

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
  const sessions = await res.json();
  renderSessionList(sessions);
}

function renderSessionList(sessions) {
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
  const res = await fetch(`${API}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({}),
  });
  const data = await res.json();

  sessionId = data.session_id;
  providerBadge.textContent = data.provider || '–';
  localStorage.setItem('lastSessionId', sessionId);

  await loadSessionList();
  messagesEl.innerHTML = '';
  stateContent.innerHTML = '<p class="muted">No data yet.</p>';
  membersContent.innerHTML = '<p class="muted">Loading…</p>';
  switchTab('session');

  setActiveSession(sessionId);
  await loadMembers(sessionId);
  await sendToAgent("Hello, I'm ready to help split some bills.");
}

async function loadSession(id) {
  sessionId = id;
  localStorage.setItem('lastSessionId', id);
  setActiveSession(id);

  messagesEl.innerHTML = '';
  stateContent.innerHTML = '<p class="muted">Loading…</p>';
  membersContent.innerHTML = '<p class="muted">Loading…</p>';
  approvalsContent.innerHTML = '<p class="muted">No pending proposals.</p>';
  switchTab('session');

  // Member list is loaded first (and awaited) so both message attribution
  // (sender names) and the admin-only approval/member controls have the
  // current user's role and the id→name map ready before anything below
  // tries to use it — re-fetched on every session load, not cached
  // indefinitely, in case the user's role changed since they last opened it.
  await loadMembers(id);

  try {
    const res = await fetch(`${API}/sessions/${id}/messages`, { credentials: 'include' });
    if (res.ok) {
      const messages = await res.json();
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

  try {
    const res = await fetch(`${API}/sessions/${id}/state`, { credentials: 'include' });
    if (res.ok) {
      const state = await res.json();
      renderState(state);
    }
  } catch {
    stateContent.innerHTML = '<p class="muted">No data yet.</p>';
  }
}

// ── Members / roles ──────────────────────────────────────────────────────────
async function loadMembers(id) {
  try {
    const res = await fetch(`${API}/api/sessions/${id}/members`, { credentials: 'include' });
    if (res.ok) {
      const list = await res.json();
      members = new Map(list.map(m => [m.user_id, m]));
      const me = members.get(currentUser.id);
      isCurrentUserAdmin = !!(me && me.role === 'admin');
    } else {
      members = new Map();
      isCurrentUserAdmin = false;
    }
  } catch {
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
  approvalsContent.innerHTML = '<p class="muted">Loading…</p>';
  try {
    const res = await fetch(`${API}/api/sessions/${sessionId}/proposals`, { credentials: 'include' });
    if (!res.ok) {
      approvalsContent.innerHTML = '<p class="muted">Failed to load proposals.</p>';
      return;
    }
    renderProposals(await res.json());
  } catch {
    approvalsContent.innerHTML = '<p class="muted">Failed to load proposals.</p>';
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
    const items = payload.items || [];
    const tax = payload.tax || 0;
    const tip = payload.tip || 0;
    const total = items.reduce((sum, i) => sum + i.price * (i.qty || 1), 0) + tax + tip;
    const payer = payload.paid_by ? `Paid by ${payload.paid_by}` : 'Payer unknown';

    const card = document.createElement('div');
    card.className = 'bill-card proposal-card';

    let html = `<div class="bill-title">${escapeHtml(payload.description || 'Untitled expense')}</div>
      <div class="bill-meta">${escapeHtml(payer)} · $${total.toFixed(2)}</div>`;

    items.forEach(item => {
      const isUnassigned = !item.assigned_to || item.assigned_to.length === 0;
      const assignText = isUnassigned
        ? 'unassigned'
        : item.assigned_to.join(', ') + (item.shared ? ' (shared)' : '');
      html += `<div class="item-row">
        <span class="item-name">${escapeHtml(item.name)}</span>
        <span class="item-price">$${item.price.toFixed(2)}</span>
        <span class="item-assign ${isUnassigned ? 'unassigned' : ''}">${escapeHtml(assignText)}</span>
      </div>`;
    });

    if (tax > 0 || tip > 0) {
      html += `<div class="item-row">
        <span class="item-name" style="color:var(--muted)">Tax + Tip</span>
        <span class="item-price">$${(tax + tip).toFixed(2)}</span>
        <span class="item-assign">proportional</span>
      </div>`;
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
  try {
    const res = await fetch(
      `${API}/api/sessions/${sessionId}/proposals/${proposalId}/${decision}`,
      { method: 'POST', credentials: 'include' },
    );
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || `Failed to ${decision} proposal.`);
      return;
    }
    // A decided proposal changes bills/settlement/pending_proposals_count —
    // refresh both the proposal list and the state panel.
    await loadProposals();
    const stateRes = await fetch(`${API}/sessions/${sessionId}/state`, { credentials: 'include' });
    if (stateRes.ok) renderState(await stateRes.json());
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
  };

  nameSpan.addEventListener('blur', finish, { once: true });
  nameSpan.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); nameSpan.blur(); }
    if (e.key === 'Escape') { nameSpan.blur(); }
  }, { once: true });
}

newSessionBtn.addEventListener('click', createNewSession);

// ── Send helpers ──────────────────────────────────────────────────────────────
async function sendToAgent(text) {
  if (!sessionId || isLoading) return;
  setLoading(true);
  const typing = appendTyping();
  try {
    const res  = await fetch(`${API}/sessions/${sessionId}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body:    JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.remove();
    appendBubble('agent', data.response);
    renderState(data.state);
    // Refresh sidebar in case session was auto-renamed
    await loadSessionList();
    setActiveSession(sessionId);
  } catch (err) {
    typing.remove();
    appendBubble('agent', `⚠️ Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

async function sendImageToAgent(file) {
  if (!sessionId || isLoading) return;
  setLoading(true);

  const reader = new FileReader();
  reader.onload = e => appendBubble('user', '', e.target.result, currentUser.id);
  reader.readAsDataURL(file);

  const typing = appendTyping();
  try {
    const form = new FormData();
    form.append('file', file);
    const res  = await fetch(`${API}/sessions/${sessionId}/image`, {
      method: 'POST',
      credentials: 'include',
      body:   form,
    });
    const data = await res.json();
    typing.remove();
    appendBubble('agent', data.response);
    renderState(data.state);
    await loadSessionList();
    setActiveSession(sessionId);
  } catch (err) {
    typing.remove();
    appendBubble('agent', `⚠️ Error: ${err.message}`);
  } finally {
    setLoading(false);
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
      const payer = bill.paid_by ? `Paid by ${bill.paid_by}` : 'Payer unknown';
      html += `<div class="bill-card">
        <div class="bill-title">${escapeHtml(bill.description)}</div>
        <div class="bill-meta">${payer} · $${bill.total.toFixed(2)}</div>`;

      bill.items.forEach(item => {
        const isUnassigned = !item.assigned_to || item.assigned_to.length === 0;
        const assignText   = isUnassigned
          ? 'unassigned'
          : item.assigned_to.join(', ') + (item.shared ? ' (shared)' : '');
        html += `<div class="item-row">
          <span class="item-name">${escapeHtml(item.name)}</span>
          <span class="item-price">$${item.price.toFixed(2)}</span>
          <span class="item-assign ${isUnassigned ? 'unassigned' : ''}">${escapeHtml(assignText)}</span>
        </div>`;
      });

      if (bill.tax > 0 || bill.tip > 0) {
        html += `<div class="item-row">
          <span class="item-name" style="color:var(--muted)">Tax + Tip</span>
          <span class="item-price">$${(bill.tax + bill.tip).toFixed(2)}</span>
          <span class="item-assign">proportional</span>
        </div>`;
      }

      html += '</div>';
    });
  }

  if (state.finalized && state.settlement && state.settlement.length) {
    html += `<div class="settlement-card">
      <h3>✅ Settlement</h3>`;
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
