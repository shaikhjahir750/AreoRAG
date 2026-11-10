/**
 * ==========================================================================
 * AERORAG // MISSION INTEL — INTERACTION CONTROLLER & API CLIENT
 * Flight Operations Workstation: Navigation, Telemetry & Dual-Pane Inspector
 * ==========================================================================
 */

// Global API Endpoint Base:
// Use same-origin relative URLs whenever served from FastAPI (HTTP/HTTPS on any port, e.g. 8000, 10000, or Render).
// Only fall back to local backend ports if opened directly via file:// or a static dev server (e.g. VS Code Live Server 5500).
const isStaticPreview =
  window.location.protocol === 'file:' ||
  ['5500', '5501', '5502'].includes(window.location.port);

let API_BASE = isStaticPreview ? 'http://127.0.0.1:10000' : '';

async function resolvePreviewApiBase() {
  if (!isStaticPreview) return;
  for (const candidate of ['http://127.0.0.1:10000', 'http://127.0.0.1:8000']) {
    try {
      const res = await fetch(`${candidate}/healthz`);
      if (res.ok) {
        API_BASE = candidate;
        return;
      }
    } catch (_) {
      // Try next candidate port
    }
  }
}

// Operational State Store
const appState = {
  currentView: 'home',
  retrievalMode: 'dense',
  missionScope: 'all',
  chatHistory: [],
  sessionTokens: 0,
  quotaInterval: null,
  clockInterval: null
};

// ============================================================================
// 1. Lifecycle Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  initNavigation();
  initGlobalShortcuts();
  initSystemClock();
  await resolvePreviewApiBase();
  fetchLiveQuota();

  // Poll live quota every 15 seconds
  appState.quotaInterval = setInterval(fetchLiveQuota, 15000);
});

// Live UTC Mission Clock Ticker
function initSystemClock() {
  const clockEl = document.getElementById('ops-clock-utc');
  function updateClock() {
    if (clockEl) {
      const now = new Date();
      const utcStr = now.toUTCString().split(' ')[4];
      clockEl.textContent = `SYS_TIME: UTC ${utcStr}`;
    }
  }
  updateClock();
  appState.clockInterval = setInterval(updateClock, 1000);
}

// Global Keyboard Shortcuts (Ctrl+K, Esc, Ctrl+Enter)
function initGlobalShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl+K or Cmd+K: Focus AeroRAG Search Console
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      switchView('rag');
      setTimeout(() => {
        const input = document.getElementById('rag-query-input');
        if (input) {
          input.focus();
          input.select();
        }
      }, 50);
    }

    // Escape: Close expanded chunks or clear active focus
    if (e.key === 'Escape') {
      const cards = document.querySelectorAll('.chunk-card.expanded');
      cards.forEach((c) => c.classList.remove('expanded'));
    }
  });

  // Ctrl+Enter on textarea executes RAG query
  const ragInput = document.getElementById('rag-query-input');
  if (ragInput) {
    ragInput.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        submitRAGQuery();
      }
    });
  }
}

// ============================================================================
// 2. Multi-View Navigation Controller
// ============================================================================
function initNavigation() {
  const hash = window.location.hash.replace('#', '');
  if (['home', 'rag', 'aerobot', 'about'].includes(hash)) {
    switchView(hash, false);
  } else {
    switchView('home', false);
  }

  window.addEventListener('hashchange', () => {
    const newHash = window.location.hash.replace('#', '');
    if (['home', 'rag', 'aerobot', 'about'].includes(newHash)) {
      switchView(newHash, false);
    }
  });
}

function switchView(viewName, updateHash = true) {
  appState.currentView = viewName;
  if (updateHash) {
    window.location.hash = viewName;
  }

  // Update tab buttons
  document.querySelectorAll('.nav-btn').forEach((btn) => {
    if (btn.dataset.view === viewName) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Switch view containers
  document.querySelectorAll('.view-section').forEach((sec) => {
    sec.classList.remove('active');
  });

  const activeSection = document.getElementById(`view-${viewName}`);
  if (activeSection) {
    activeSection.classList.add('active');
  }

  // Auto-focus input on desktop navigation (avoid unexpected virtual keyboard popup on mobile)
  if (window.innerWidth > 768) {
    if (viewName === 'rag') {
      setTimeout(() => document.getElementById('rag-query-input')?.focus(), 120);
    } else if (viewName === 'aerobot') {
      setTimeout(() => document.getElementById('aerobot-input')?.focus(), 120);
    }
  }
  if (viewName === 'about') {
    fetchLiveQuota();
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// 3. Live Gemini Free-Tier Quota Polling & Telemetry
// ============================================================================
async function fetchLiveQuota() {
  try {
    const res = await fetch(`${API_BASE}/api/quota`);
    if (!res.ok) return;
    const data = await res.json();
    renderQuotaDisplay(data);
  } catch (err) {
    console.warn('Quota telemetry poll warning:', err);
  }
}

function renderQuotaDisplay(data) {
  // Navbar Pill
  const navVal = document.getElementById('nav-quota-remaining');
  if (navVal) {
    navVal.textContent = Number(data.daily_remaining).toLocaleString();
  }

  // About View Dashboard Meters
  const reqRemaining = document.getElementById('dash-requests-remaining');
  const dailyProg = document.getElementById('dash-daily-progress');
  const dailyUsed = document.getElementById('dash-daily-used');
  const dailyPct = document.getElementById('dash-daily-pct');

  if (reqRemaining) reqRemaining.textContent = Number(data.daily_remaining).toLocaleString();
  if (dailyUsed) dailyUsed.textContent = `Used Today: ${data.daily_used}`;
  if (dailyPct) dailyPct.textContent = `${data.daily_percent_remaining}% Available`;
  if (dailyProg) dailyProg.style.width = `${data.daily_percent_remaining}%`;

  // RPM metrics
  const rpmUsed = document.getElementById('dash-rpm-used');
  const rpmProg = document.getElementById('dash-rpm-progress');
  const rpmRemaining = document.getElementById('dash-rpm-remaining');
  const rpmStatus = document.getElementById('dash-rpm-status');

  if (rpmUsed) rpmUsed.textContent = data.minute_used;
  if (rpmRemaining) rpmRemaining.textContent = `Remaining this min: ${data.minute_remaining}`;
  if (rpmProg) {
    const rpmPct = Math.min(100, (data.minute_used / data.minute_limit) * 100);
    rpmProg.style.width = `${rpmPct}%`;
  }
  if (rpmStatus) {
    rpmStatus.textContent = `● ${data.status}`;
    rpmStatus.style.color = (data.minute_used >= 12) ? 'var(--amber-warning)' : 'var(--emerald-active)';
  }

  // Midnight UTC Reset countdown
  const resetCountdown = document.getElementById('dash-reset-countdown');
  if (resetCountdown && data.reset_countdown) {
    resetCountdown.textContent = data.reset_countdown;
  }
}

// ============================================================================
// 4. AeroRAG Terminal Search & Dual-Pane Transparency Inspector
// ============================================================================
function selectMissionScope(scope) {
  appState.missionScope = scope;
  document.querySelectorAll('.scope-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.scope === scope);
  });
}

function updateModeSelection(mode) {
  appState.retrievalMode = mode;
  document.getElementById('mode-dense-lbl')?.classList.toggle('active', mode === 'dense');
  document.getElementById('mode-hybrid-lbl')?.classList.toggle('active', mode === 'hybrid');
}

function setQuery(text) {
  const input = document.getElementById('rag-query-input');
  if (input) {
    input.value = text;
    if (window.innerWidth > 768) {
      input.focus();
    }
  }
}

function loadSamplePrompt(text) {
  switchView('rag');
  setQuery(text);
  setTimeout(submitRAGQuery, 150);
}

async function submitRAGQuery() {
  const input = document.getElementById('rag-query-input');
  let query = input ? input.value.trim() : '';
  if (!query) return;

  // Append mission scope if user picked a targeted filter and not already specified
  if (appState.missionScope !== 'all' && !query.toLowerCase().includes(appState.missionScope.toLowerCase())) {
    query = `[${appState.missionScope}] ${query}`;
  }

  const btnSubmit = document.getElementById('btn-submit-rag');
  const loadingEl = document.getElementById('rag-loading');
  const resultsEl = document.getElementById('rag-results');
  const stepText = document.getElementById('loading-step-text');

  // UI state: loading
  if (btnSubmit) btnSubmit.disabled = true;
  if (loadingEl) loadingEl.classList.remove('hidden');
  if (resultsEl) resultsEl.classList.add('hidden');

  // Terminal telemetry progression
  const steps = [
    'DECOMPOSING MULTI-MISSION QUERY & TARGETING METADATA...',
    'EXECUTING CHROMADB VECTOR RETRIEVAL (k=15)...',
    'RUNNING FLASHRANK MINILM-L-12 CROSS-ENCODER (k=8)...',
    'SYNTHESIZING GROUNDED INVESTIGATION DEBRIEF...'
  ];
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    if (stepText) stepText.textContent = steps[stepIdx];
  }, 650);

  try {
    const res = await fetch(`${API_BASE}/api/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        mode: appState.retrievalMode
      })
    });

    clearInterval(stepInterval);

    const rawText = await res.text();
    let data;
    try {
      data = JSON.parse(rawText);
    } catch (parseErr) {
      throw new Error(`Server returned non-JSON response (status ${res.status}): ${rawText || '[Empty Body]'}`);
    }

    if (!res.ok) {
      throw new Error(data.detail || data.message || `Server error (status ${res.status})`);
    }

    renderRAGResults(data);
    if (data.quota) renderQuotaDisplay(data.quota);
  } catch (err) {
    clearInterval(stepInterval);
    const msg = (err.message === 'Failed to fetch')
      ? 'Cannot reach the Python FastAPI backend (app.py). index.html cannot process queries on its own—please run `python app.py` and open http://127.0.0.1:10000.'
      : err.message;
    alert(`AERORAG EXECUTION NOTICE: ${msg}`);
  } finally {
    if (btnSubmit) btnSubmit.disabled = false;
    if (loadingEl) loadingEl.classList.add('hidden');
  }
}

function renderRAGResults(data) {
  const resultsEl = document.getElementById('rag-results');
  if (!resultsEl) return;
  resultsEl.classList.remove('hidden');

  // Latency telemetry in cards and top bar
  const latencyBadge = document.getElementById('rag-latency-badge');
  if (latencyBadge) {
    latencyBadge.textContent = `LATENCY: ${data.latency_ms} ms // MODE: ${data.mode.toUpperCase()}`;
  }

  const navLatency = document.getElementById('nav-latency-badge');
  if (navLatency) {
    navLatency.textContent = `p95: ${(data.latency_ms / 1000).toFixed(2)}s`;
  }

  // 1. Render Subquery Decomposition Plan
  const planContainer = document.getElementById('rag-plan-container');
  if (planContainer) {
    planContainer.innerHTML = '';
    data.plan.forEach((sq) => {
      const row = document.createElement('div');
      row.className = 'subquery-row';
      row.innerHTML = `
        <span class="subquery-mission">[${sq.mission.toUpperCase()}]</span>
        <span class="subquery-text">${escapeHtml(sq.sub_question)}</span>
      `;
      planContainer.appendChild(row);
    });
  }

  // 2. Render Answer with Provenance Highlighting
  const answerContent = document.getElementById('rag-answer-content');
  if (answerContent) {
    answerContent.innerHTML = formatInvestigatorDebrief(data.answer);
  }

  // 3. Refusal Alert Banner
  const refusalAlert = document.getElementById('rag-refusal-alert');
  if (refusalAlert) {
    if (data.is_refusal) {
      refusalAlert.classList.remove('hidden');
    } else {
      refusalAlert.classList.add('hidden');
    }
  }

  // 4. Render Retrieved Context Chunks Deck
  const chunkBadge = document.getElementById('chunk-count-badge');
  if (chunkBadge) chunkBadge.textContent = data.contexts.length;

  const chunksContainer = document.getElementById('rag-chunks-container');
  if (chunksContainer) {
    chunksContainer.innerHTML = '';
    data.contexts.forEach((c) => {
      const missionClass = `tag-${c.mission.toLowerCase().replace(/[^a-z0-9]/g, '')}`;
      const card = document.createElement('div');
      card.className = 'chunk-card';
      card.innerHTML = `
        <div class="chunk-header" onclick="toggleChunk(this)">
          <div class="chunk-title-group">
            <span class="chunk-badge">#${String(c.chunk_id).padStart(2, '0')}</span>
            <span class="mission-tag ${missionClass}">${c.mission.toUpperCase()}</span>
            <span class="chunk-source">${escapeHtml(c.source_file)}</span>
          </div>
          <span class="chunk-toggle-icon">▼</span>
        </div>
        <div class="chunk-body">${escapeHtml(c.content)}</div>
      `;
      chunksContainer.appendChild(card);
    });
  }

  // Smoothly scroll down to inspector
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function toggleChunk(headerEl) {
  const card = headerEl.closest('.chunk-card');
  if (card) {
    card.classList.toggle('expanded');
  }
}

function toggleAllChunks() {
  const cards = document.querySelectorAll('.chunk-card');
  const anyExpanded = Array.from(cards).some((c) => c.classList.contains('expanded'));
  cards.forEach((c) => {
    if (anyExpanded) {
      c.classList.remove('expanded');
    } else {
      c.classList.add('expanded');
    }
  });
}

function copyAnswerText() {
  const answerContent = document.getElementById('rag-answer-content');
  if (!answerContent) return;
  navigator.clipboard.writeText(answerContent.innerText).then(() => {
    alert('INVESTIGATION DEBRIEF COPIED TO CLIPBOARD');
  });
}

// ============================================================================
// 5. AeroBot Conversational Flight Recorder Debrief
// ============================================================================
function handleChatKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendAeroBotMessage();
  }
}

function sendQuickChatMessage(promptText) {
  const input = document.getElementById('aerobot-input');
  if (input) {
    input.value = promptText;
    sendAeroBotMessage();
  }
}

async function sendAeroBotMessage() {
  const input = document.getElementById('aerobot-input');
  const text = input ? input.value.trim() : '';
  if (!text) return;

  input.value = '';
  const stream = document.getElementById('chat-messages-stream');

  // Append User Turn
  appendChatMessage('user', text);

  // Append Monospaced Telemetry Loading State
  const typingEl = document.createElement('div');
  typingEl.className = 'chat-entry assistant-turn';
  typingEl.id = 'bot-typing-placeholder';
  typingEl.innerHTML = `
    <div class="chat-meta-bar">
      <span>AEROBOT // COMPUTING</span>
      <span>...</span>
    </div>
    <div class="chat-bubble-card mono" style="color: var(--text-muted); font-size: 0.8rem;">
      &gt; SYNTHESIZING REASONING DEBRIEF...
    </div>
  `;
  stream.appendChild(typingEl);
  stream.scrollTop = stream.scrollHeight;

  // Track conversation turn
  appState.chatHistory.push({ role: 'user', content: text });

  try {
    const res = await fetch(`${API_BASE}/api/aerobot/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: text,
        chat_history: appState.chatHistory.slice(-6)
      })
    });

    const placeholder = document.getElementById('bot-typing-placeholder');
    if (placeholder) placeholder.remove();

    const rawText = await res.text();
    let data;
    try {
      data = JSON.parse(rawText);
    } catch (parseErr) {
      throw new Error(`AeroBot returned non-JSON response (status ${res.status}): ${rawText || '[Empty Body]'}`);
    }

    if (!res.ok) {
      throw new Error(data.detail || data.message || `AeroBot error (status ${res.status})`);
    }

    appState.chatHistory.push({ role: 'assistant', content: data.response });

    // Update Session Telemetry
    if (data.tokens && data.tokens.total_tokens) {
      appState.sessionTokens += data.tokens.total_tokens;
      const totalTokensEl = document.getElementById('chat-total-tokens');
      if (totalTokensEl) totalTokensEl.textContent = appState.sessionTokens.toLocaleString();
    }

    const turnsEl = document.getElementById('chat-turns-count');
    if (turnsEl) {
      turnsEl.textContent = `${Math.min(6, Math.floor(appState.chatHistory.length / 2))} / 6 max`;
    }

    // Append Assistant Message
    appendChatMessage('assistant', data.response, data.tokens, data.latency_ms, data.is_restricted_topic, text);

    // Refresh live quota after chat turn
    fetchLiveQuota();

  } catch (err) {
    const placeholder = document.getElementById('bot-typing-placeholder');
    if (placeholder) placeholder.remove();
    const msg = (err.message === 'Failed to fetch')
      ? 'Connection notice: Cannot reach the FastAPI backend (`app.py`). Make sure `python app.py` is running and access the UI through the backend URL (e.g. `http://127.0.0.1:10000`).'
      : `Connection notice: Unable to contact reasoning engine (${err.message}). Please verify your Gemini API key credentials.`;
    appendChatMessage('assistant', msg);
  }
}

function appendChatMessage(role, text, tokens = null, latencyMs = 0, isRestricted = false, originalQuery = '') {
  const stream = document.getElementById('chat-messages-stream');
  if (!stream) return;

  const entryEl = document.createElement('div');
  entryEl.className = `chat-entry ${role === 'user' ? 'user-turn' : 'assistant-turn'}`;

  // Telemetry Meta Bar
  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];
  const senderLabel = (role === 'user') ? 'OPERATOR' : 'AEROBOT // DEBRIEF';

  let telemetrySnippet = '';
  if (tokens && tokens.total_tokens) {
    telemetrySnippet = ` • ${tokens.total_tokens} TOK • ${latencyMs}ms`;
  }

  // Boundary Routing Banner
  let routingBannerHtml = '';
  if (isRestricted) {
    routingBannerHtml = `
      <div class="scope-routing-banner">
        <span class="scope-banner-text">
          <strong>NOTICE: Out of Corpus Scope.</strong> Verbatim flight telemetry and board investigation records are indexed in the AeroRAG tool.
        </span>
        <button class="scope-banner-btn" onclick="loadSamplePrompt('${escapeQuotes(originalQuery)}')">
          INSPECT IN AERORAG &rarr;
        </button>
      </div>
    `;
  }

  entryEl.innerHTML = `
    <div class="chat-meta-bar">
      <span>${senderLabel}</span>
      <span>${timeStr}${telemetrySnippet}</span>
    </div>
    <div class="chat-bubble-card">
      <div>${formatInvestigatorDebrief(text)}</div>
      ${routingBannerHtml}
    </div>
  `;

  stream.appendChild(entryEl);
  stream.scrollTop = stream.scrollHeight;
}

function clearAeroBotChat() {
  appState.chatHistory = [];
  appState.sessionTokens = 0;
  
  const turnsEl = document.getElementById('chat-turns-count');
  if (turnsEl) turnsEl.textContent = '0 / 6 max';
  
  const totalTokensEl = document.getElementById('chat-total-tokens');
  if (totalTokensEl) totalTokensEl.textContent = '0';

  const stream = document.getElementById('chat-messages-stream');
  if (stream) {
    stream.innerHTML = `
      <div class="chat-entry assistant-turn">
        <div class="chat-meta-bar">
          <span>AEROBOT // FLIGHT ASSISTANT</span>
          <span>SYS_RESET: OK</span>
        </div>
        <div class="chat-bubble-card">
          <p><strong>AeroBot online.</strong> Conversation history has been reset. All previous turns have been cleared from context memory.</p>
          <p style="margin-top: 0.4rem; font-size: 0.8rem; color: var(--text-muted);">
            <em>Ready for new orbital mechanics, rocketry, or spaceflight engineering inquiries.</em>
          </p>
        </div>
      </div>
    `;
  }
}

// ============================================================================
// 6. Text Formatting & Citation Utilities
// ============================================================================
function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeQuotes(str) {
  if (!str) return '';
  return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function formatInvestigatorDebrief(text) {
  if (!text) return '';
  let formatted = escapeHtml(text);

  // Bold markup
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Italic markup
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  
  // Inline code / Document citations: [Doc: Cortright-1970 §4.2] or [MSC-00171]
  formatted = formatted.replace(/`([^`]+)`/g, '<code class="mono">$1</code>');
  formatted = formatted.replace(/(\[(?:Doc:|Cortright|MSC-|Rogers)[^\]]+\])/gi, '<span class="provenance-citation">$1</span>');

  // Paragraph breaks
  formatted = formatted.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  return formatted;
}
