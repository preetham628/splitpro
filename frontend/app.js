// ── Config ───────────────────────────────────────────────────────────────────
const API = '';

// ── State ────────────────────────────────────────────────────────────────────
let currentChatId = null;
let isLoading     = false;
let chatList      = [];   // [{id, name, provider, created_at, updated_at}]

// ── DOM refs ─────────────────────────────────────────────────────────────────
const messagesEl    = document.getElementById('messages');
const inputEl       = document.getElementById('message-input');
const sendBtn       = document.getElementById('send-btn');
const fileInput     = document.getElementById('file-input');
const providerBadge = document.getElementById('provider-badge');
const stateContent  = document.getElementById('state-content');
const chatListEl    = document.getElementById('chat-list');
const newChatBtn    = document.getElementById('new-chat-btn');

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  try {
    await loadChatList();
    if (chatList.length === 0) {
      await createNewChat();
    } else {
      await switchToChat(chatList[0].id);
    }
  } catch (err) {
    messagesEl.innerHTML = `<p style="color:red;padding:20px">
      Could not connect to the API. Make sure the server is running on port 8000.<br>
      <code>${err.message}</code></p>`;
  }
}

// ── Chat list ─────────────────────────────────────────────────────────────────
async function loadChatList() {
  const res = await fetch(`${API}/chats`);
  chatList = await res.json();
  renderChatList();
}

function renderChatList() {
  if (chatList.length === 0) {
    chatListEl.innerHTML = '<p style="font-size:12px;color:var(--muted);padding:10px 10px">No chats yet</p>';
    return;
  }
  chatListEl.innerHTML = chatList.map(c => `
    <div class="chat-item ${c.id === currentChatId ? 'active' : ''}" data-id="${escapeAttr(c.id)}">
      <span class="chat-item-name">${escapeHtml(c.name)}</span>
      <button class="chat-delete-btn" data-id="${escapeAttr(c.id)}" title="Delete">×</button>
    </div>
  `).join('');

  chatListEl.querySelectorAll('.chat-item').forEach(el => {
    el.addEventListener('click', e => {
      if (e.target.classList.contains('chat-delete-btn')) return;
      switchToChat(el.dataset.id);
    });
  });
  chatListEl.querySelectorAll('.chat-delete-btn').forEach(btn => {
    btn.addEventListener('click', () => deleteChat(btn.dataset.id));
  });
}

// ── Create / switch / delete ───────────────────────────────────────────────────
async function createNewChat() {
  setLoading(true);
  try {
    const res = await fetch(`${API}/chats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    currentChatId = data.id;
    providerBadge.textContent = data.provider || 'openai';

    chatList.unshift({
      id: data.id, name: data.name, provider: data.provider,
      created_at: data.created_at, updated_at: data.updated_at,
    });
    renderChatList();

    messagesEl.innerHTML = '';
    (data.messages || []).forEach(msg => {
      try { appendBubble(msg.role, msg.content); }
      catch (e) { console.warn('Failed to render message:', msg, e); }
    });
    renderState(data.state);
  } finally {
    setLoading(false);
  }
}

async function switchToChat(id) {
  if (id === currentChatId || isLoading) return;
  setLoading(true);
  try {
    const res = await fetch(`${API}/chats/${id}`);
    if (!res.ok) { await loadChatList(); return; }
    const data = await res.json();

    currentChatId = data.id;
    providerBadge.textContent = data.provider || 'openai';

    messagesEl.innerHTML = '';
    (data.messages || []).forEach(msg => {
      try { appendBubble(msg.role, msg.content); }
      catch (e) { console.warn('Failed to render message:', msg, e); }
    });
    renderState(data.state);
    renderChatList();
    scrollToBottom();
  } finally {
    setLoading(false);
  }
}

async function deleteChat(id) {
  await fetch(`${API}/chats/${id}`, { method: 'DELETE' });
  chatList = chatList.filter(c => c.id !== id);

  if (currentChatId === id) {
    currentChatId = null;
    messagesEl.innerHTML = '';
    stateContent.innerHTML = '<p class="muted">No data yet.</p>';
    if (chatList.length > 0) {
      await switchToChat(chatList[0].id);
    } else {
      await createNewChat();
    }
  } else {
    renderChatList();
  }
}

// ── Send helpers ──────────────────────────────────────────────────────────────
async function sendToAgent(text) {
  if (!currentChatId || isLoading) return;
  setLoading(true);
  const typing = appendTyping();
  try {
    const res = await fetch(`${API}/chats/${currentChatId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.remove();
    appendBubble('agent', data.response);
    renderState(data.state);
    bumpChatToTop(currentChatId);
  } catch (err) {
    typing.remove();
    appendBubble('agent', `⚠️ Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

async function sendImageToAgent(file) {
  if (!currentChatId || isLoading) return;
  setLoading(true);

  const reader = new FileReader();
  reader.onload = e => appendBubble('user', '', e.target.result);
  reader.readAsDataURL(file);

  const typing = appendTyping();
  try {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API}/chats/${currentChatId}/image`, {
      method: 'POST',
      body: form,
    });
    const data = await res.json();
    typing.remove();
    appendBubble('agent', data.response);
    renderState(data.state);
    bumpChatToTop(currentChatId);
  } catch (err) {
    typing.remove();
    appendBubble('agent', `⚠️ Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// Move the active chat to the top of the list after a new message
function bumpChatToTop(id) {
  const idx = chatList.findIndex(c => c.id === id);
  if (idx > 0) {
    const [chat] = chatList.splice(idx, 1);
    chat.updated_at = new Date().toISOString();
    chatList.unshift(chat);
    renderChatList();
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
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function escapeAttr(str) {
  return str.replace(/"/g, '&quot;');
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
newChatBtn.addEventListener('click', createNewChat);

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
