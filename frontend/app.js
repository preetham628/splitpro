// ── Config ──────────────────────────────────────────────────────────────────
// Use relative URLs since the frontend is served from the same FastAPI origin.
const API = '';

// ── State ────────────────────────────────────────────────────────────────────
let sessionId = null;
let isLoading = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const messagesEl    = document.getElementById('messages');
const inputEl       = document.getElementById('message-input');
const sendBtn       = document.getElementById('send-btn');
const fileInput     = document.getElementById('file-input');
const providerBadge = document.getElementById('provider-badge');
const stateContent  = document.getElementById('state-content');

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  try {
    const res  = await fetch(`${API}/sessions`, { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const data = await res.json();
    sessionId = data.session_id;
    providerBadge.textContent = data.provider || 'openai';

    // Clear the "Connecting…" placeholder
    messagesEl.innerHTML = '';

    // Ask the agent to open the conversation
    await sendToAgent("Hello, I'm ready to help split some bills.");
  } catch (err) {
    messagesEl.innerHTML = `<p style="color:red;padding:20px">
      Could not connect to the API. Make sure the server is running on port 8000.<br>
      <code>${err.message}</code></p>`;
  }
}

// ── Send helpers ──────────────────────────────────────────────────────────────
async function sendToAgent(text) {
  if (!sessionId || isLoading) return;
  setLoading(true);
  const typing = appendTyping();
  try {
    const res  = await fetch(`${API}/sessions/${sessionId}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.remove();
    appendBubble('agent', data.response);
    renderState(data.state);
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

  // Show user bubble with image preview
  const reader = new FileReader();
  reader.onload = e => {
    appendBubble('user', '', e.target.result);
  };
  reader.readAsDataURL(file);

  const typing = appendTyping();
  try {
    const form = new FormData();
    form.append('file', file);
    const res  = await fetch(`${API}/sessions/${sessionId}/image`, {
      method: 'POST',
      body:   form,
    });
    const data = await res.json();
    typing.remove();
    appendBubble('agent', data.response);
    renderState(data.state);
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
    // Render markdown for agent messages, plain text for user
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

  // Participants
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

  // Bills
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

  // Settlement (when finalized)
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

// Auto-resize textarea
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    sendImageToAgent(file);
    fileInput.value = '';   // reset so same file can be re-selected
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
