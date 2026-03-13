/**
 * static/js/app.js
 * ─────────────────
 * FashionSense — state machine to match the backend session flow.
 *
 * States: IDLE → CAPTURING → ANALYSING → RESULTS → (CAPTURING | RESULTS)
 */

// ── Config ──────────────────────────────────────────────────────────────────
const WS_URL = `${location.origin.replace(/^http/, 'ws')}/ws/camera`;

const CAT_COLORS = {
  short_sleeve_top:      '#ff6464', long_sleeve_top:       '#ffa03c',
  short_sleeve_outwear:  '#50c878', long_sleeve_outwear:   '#28b4c8',
  vest:                  '#b450ff', sling:                 '#ff50b4',
  shorts:                '#64b4ff', trousers:              '#3c50c8',
  skirt:                 '#dcb428', short_sleeve_dress:    '#ff7850',
  long_sleeve_dress:     '#8c3cc8', vest_dress:            '#c86496',
  sling_dress:           '#64c8b4',
};

const CAT_EMOJI = {
  short_sleeve_top:'👕', long_sleeve_top:'👔', short_sleeve_outwear:'🧥',
  long_sleeve_outwear:'🧥', vest:'🦺', sling:'👗', shorts:'🩳',
  trousers:'👖', skirt:'👗', short_sleeve_dress:'👗',
  long_sleeve_dress:'👗', vest_dress:'👗', sling_dress:'👗',
};

// ── State ────────────────────────────────────────────────────────────────────
let ws    = null;
let phase = 'idle';   // idle | capturing | analysing | results

// ── DOM ───────────────────────────────────────────────────────────────────────
const feed        = document.getElementById('feed');
const idleOverlay = document.getElementById('idle-overlay');
const cameraWrap  = document.getElementById('camera-wrap');
const dot         = document.getElementById('dot');
const statusText  = document.getElementById('status-text');
const step1       = document.getElementById('step-1');
const step2       = document.getElementById('step-2');
const step3       = document.getElementById('step-3');
const phaseLabel  = document.getElementById('phase-label');
const btnStart    = document.getElementById('btn-start');
const actionBtns  = document.getElementById('action-btns');
const tagsEl      = document.getElementById('tags');
const recsScroll  = document.getElementById('recs-scroll');

// ── Public — button handlers ─────────────────────────────────────────────────

function startScan() {
  if (ws) { ws.close(); ws = null; }

  setStatus('connecting', 'Connecting…');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('live', 'Connected');
    setPhase('capturing');
    feed.style.display = 'block';
    idleOverlay.classList.add('hidden');
    cameraWrap.classList.add('active');
    btnStart.disabled = true;
    actionBtns.classList.remove('visible');
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);

    if (msg.type === 'frame') {
      feed.src = 'data:image/jpeg;base64,' + msg.frame;
      if (msg.phase === 'capturing') {
        setPhase('capturing');
        setStatus('scanning', `Scanning… ${msg.countdown}s`);
        updateSteps('capturing');
      } else if (msg.phase === 'analysing') {
        setPhase('analysing');
        setStatus('scanning', 'Analysing…');
        updateSteps('analysing');
      }

    } else if (msg.type === 'results') {
      setPhase('results');
      setStatus('live', 'Done');
      updateSteps('results');
      renderTags(msg.detections || []);
      renderRecs(msg.recommendations || []);
      actionBtns.classList.add('visible');
      btnStart.disabled = true;   // hide start — use retry instead
      phaseLabel.innerHTML = 'Outfit scanned — choose an action below';

    } else if (msg.type === 'error') {
      setStatus('error', 'Error');
      console.error('Server error:', msg.message);
      resetToIdle();
    }
  };

  ws.onerror = () => {
    setStatus('error', 'Connection error');
    resetToIdle();
  };

  ws.onclose = () => {
    if (phase !== 'results') resetToIdle();
  };
}

function sendCommand(cmd) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    // WebSocket closed — reconnect then immediately start capture
    startScan();
    return;
  }

  ws.send(JSON.stringify({ cmd }));

  if (cmd === 'retry') {
    actionBtns.classList.remove('visible');
    tagsEl.innerHTML = '<span class="no-tag">Scanning…</span>';
    recsScroll.innerHTML = '<div class="recs-placeholder">Scanning your outfit…</div>';
    setPhase('capturing');
    updateSteps('capturing');
    setStatus('scanning', 'Starting scan…');

  } else if (cmd === 'more_recs') {
    recsScroll.innerHTML = '<div class="recs-placeholder">Finding new suggestions…</div>';
    setStatus('live', 'Loading suggestions…');
  }
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderTags(detections) {
  if (!detections.length) {
    tagsEl.innerHTML = '<span class="no-tag">No clothing detected</span>';
    return;
  }
  // Deduplicate by class_name, keep highest confidence
  const seen = {};
  detections.forEach(d => {
    if (!seen[d.class_name] || seen[d.class_name].confidence < d.confidence)
      seen[d.class_name] = d;
  });
  tagsEl.innerHTML = Object.values(seen).map(d => {
    const color = CAT_COLORS[d.class_name] || '#888';
    const label = d.class_name.replace(/_/g, ' ');
    const conf  = Math.round(d.confidence * 100);
    return `
      <div class="tag">
        <div class="tag-dot" style="background:${color}"></div>
        <span>${label}</span>
        <span class="tag-conf">${conf}%</span>
      </div>`;
  }).join('');
}

function renderRecs(recs) {
  if (!recs.length) {
    recsScroll.innerHTML = '<div class="recs-placeholder">No recommendations found — try a retry scan</div>';
    return;
  }
  recsScroll.innerHTML = recs.map(r => {
    const emoji = CAT_EMOJI[r.category] || '👔';
    const cat   = (r.category || '').replace(/_/g, ' ');
    return `
      <div class="rec-card">
        <div class="rec-thumb">${emoji}</div>
        <div class="rec-info">
          <div class="rec-name">${esc(r.name)}</div>
          <div class="rec-cat">${cat}</div>
          <div class="rec-why">${esc(r.reason)}</div>
          <div class="rec-price">${esc(r.price)}</div>
        </div>
      </div>`;
  }).join('');
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function setPhase(p) {
  phase = p;
}

function updateSteps(p) {
  // Step 1 = capturing, Step 2 = analysing, Step 3 = results
  step1.className = 'phase-step';
  step2.className = 'phase-step';
  step3.className = 'phase-step';

  if (p === 'capturing') {
    step1.classList.add('active');
    phaseLabel.innerHTML = '<span>Scanning</span> your outfit…';
  } else if (p === 'analysing') {
    step1.classList.add('done');
    step2.classList.add('active');
    phaseLabel.innerHTML = '<span>Analysing</span> detections…';
  } else if (p === 'results') {
    step1.classList.add('done');
    step2.classList.add('done');
    step3.classList.add('done');
  }
}

function setStatus(state, text) {
  dot.className = 'dot';
  if (state === 'live')       dot.classList.add('live');
  else if (state === 'scanning') dot.classList.add('scanning');
  else if (state === 'error') dot.classList.add('error');
  statusText.textContent = text;
}

function resetToIdle() {
  phase = 'idle';
  feed.style.display = 'none';
  idleOverlay.classList.remove('hidden');
  cameraWrap.classList.remove('active');
  btnStart.disabled = false;
  actionBtns.classList.remove('visible');
  step1.className = 'phase-step';
  step2.className = 'phase-step';
  step3.className = 'phase-step';
  phaseLabel.innerHTML = 'Press start to begin';
  setStatus('idle', 'Ready');
}

function esc(str) {
  return String(str || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}