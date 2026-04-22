// ── Config ──────────────────────────────────────────────────────────────────
const API = '';

// ── State ────────────────────────────────────────────────────────────────────
let sessionId  = null;
let isLoading  = false;
let currentUser = null;

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

  setActiveSession(sessionId);
  await sendToAgent("Hello, I'm ready to help split some bills.");
}

async function loadSession(id) {
  sessionId = id;
  localStorage.setItem('lastSessionId', id);
  setActiveSession(id);

  // Clear chat panel (visual history isn't restored — AI context is in DB)
  messagesEl.innerHTML = '';
  stateContent.innerHTML = '<p class="muted">Loading…</p>';

  try {
    const res = await fetch(`${API}/sessions/${id}/state`, { credentials: 'include' });
    if (res.ok) {
      const state = await res.json();
      renderState(state);
    }
  } catch {
    stateContent.innerHTML = '<p class="muted">No data yet.</p>';
  }

  appendBubble('agent', 'Session loaded. How can I help you continue?');
}

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
  reader.onload = e => appendBubble('user', '', e.target.result);
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
function appendBubble(role, text, imageDataUrl = null) {
  const row = document.createElement('div');
  row.className = `bubble-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'agent' ? '🤖' : '🙂';

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

  row.appendChild(avatar);
  row.appendChild(bubble);
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

  let html = '';

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
  appendBubble('user', text);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendToAgent(text);
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
