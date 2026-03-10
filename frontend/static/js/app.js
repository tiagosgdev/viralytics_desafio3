/**
 * static/js/app.js
 * ─────────────────
 * FashionSense dashboard — WebSocket camera stream + upload mode
 */

// ── Config ─────────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;
const WS_URL   = `${API_BASE.replace(/^http/, 'ws')}/ws/camera`;

// Category → colour (matches Python CATEGORY_COLORS, as CSS hex)
const CAT_COLORS = {
  short_sleeve_top:       '#ff6464',
  long_sleeve_top:        '#ffa03c',
  short_sleeve_outwear:   '#50c878',
  long_sleeve_outwear:    '#28b4c8',
  vest:                   '#b450ff',
  sling:                  '#ff50b4',
  shorts:                 '#64b4ff',
  trousers:               '#3c50c8',
  skirt:                  '#dcb428',
  short_sleeve_dress:     '#ff7850',
  long_sleeve_dress:      '#8c3cc8',
  vest_dress:             '#c86496',
  sling_dress:            '#64c8b4',
};

// Clothing emoji for rec cards
const CAT_EMOJI = {
  short_sleeve_top: '👕', long_sleeve_top: '👔', short_sleeve_outwear: '🧥',
  long_sleeve_outwear: '🧥', vest: '🦺', sling: '👗', shorts: '🩳',
  trousers: '👖', skirt: '👗', short_sleeve_dress: '👗',
  long_sleeve_dress: '👗', vest_dress: '👗', sling_dress: '👗',
};

// ── State ───────────────────────────────────────────────────────────────────
let ws        = null;
let mode      = 'live';   // 'live' | 'upload'
let confValue = 0.40;
let lastDetRender = 0;
let lastRecsRender = 0;
const RENDER_INTERVAL = 500; // ms — throttle sidebar updates

// ── DOM refs ─────────────────────────────────────────────────────────────────
const feed          = document.getElementById('camera-feed');
const overlay       = document.getElementById('camera-overlay');
const scanLine      = document.getElementById('scan-line');
const statusDot     = document.getElementById('status-dot');
const statusText    = document.getElementById('status-text');
const fpsCounter    = document.getElementById('fps-counter');
const detList       = document.getElementById('detections-list');
const recsList      = document.getElementById('recs-list');
const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const infBar        = document.getElementById('inf-bar');

// ── Mode switching ───────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('tab-live').classList.toggle('active', m === 'live');
  document.getElementById('tab-upload').classList.toggle('active', m === 'upload');
  document.getElementById('live-controls').style.display   = m === 'live'   ? 'block' : 'none';
  document.getElementById('upload-controls').style.display = m === 'upload' ? 'block' : 'none';
  if (m === 'upload') stopCamera();
}

// ── Camera stream ─────────────────────────────────────────────────────────────
function startCamera() {
  if (ws) ws.close();

  setStatus('connecting', 'Connecting…');
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('active', 'Live');
    scanLine.classList.add('active');
    btnStart.disabled = true;
    btnStop.disabled  = false;
    overlay.classList.add('hidden');
    feed.style.display = 'block';
  };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    renderFrame(data.frame);
    updateFPS(data.fps, data.inference_ms);

    // Throttle sidebar updates to avoid flickering
    const now = Date.now();
    if (now - lastDetRender > RENDER_INTERVAL) {
      renderDetections(data.detections || []);
      lastDetRender = now;
    }
    if (now - lastRecsRender > RENDER_INTERVAL) {
      renderRecommendations(data.recommendations || []);
      lastRecsRender = now;
    }
  };

  ws.onerror = () => setStatus('error', 'Error');

  ws.onclose = () => {
    setStatus('idle', 'Disconnected');
    scanLine.classList.remove('active');
    btnStart.disabled = false;
    btnStop.disabled  = true;
    feed.style.display = 'none';
    overlay.classList.remove('hidden');
  };
}

function stopCamera() {
  if (ws) { ws.close(); ws = null; }
}

// ── Upload mode ───────────────────────────────────────────────────────────────
async function handleUpload(evt) {
  const file = evt.target.files[0];
  if (!file) return;

  setStatus('connecting', 'Processing…');
  overlay.classList.add('hidden');

  // Preview instantly
  const reader = new FileReader();
  reader.onload = (e) => {
    feed.src = e.target.result;
    feed.style.display = 'block';
  };
  reader.readAsDataURL(file);

  // Call API
  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch(`${API_BASE}/api/detect/image`, { method: 'POST', body: form });
    const data = await resp.json();

    // Show annotated frame
    if (data.annotated_frame) {
      feed.src = 'data:image/jpeg;base64,' + data.annotated_frame;
    }

    renderDetections(data.detections || []);
    renderRecommendations(data.recommendations || []);
    updateFPS(null, data.inference_ms);
    setStatus('active', 'Done');
  } catch (err) {
    setStatus('error', 'API Error');
    console.error(err);
  }
}

// Drag-and-drop
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; });
dropZone.addEventListener('dragleave', ()=> { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  const file = e.dataTransfer.files[0];
  if (file) {
    const input = document.getElementById('upload-input');
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    handleUpload({ target: input });
  }
});

// ── Render helpers ────────────────────────────────────────────────────────────
function renderFrame(b64) {
  feed.src = 'data:image/jpeg;base64,' + b64;
}

function renderDetections(detections) {
  if (!detections.length) {
    detList.innerHTML = '<span class="no-detection">No clothing detected</span>';
    return;
  }

  // Deduplicate by class_name
  const seen = {};
  detections.forEach(d => {
    if (!seen[d.class_name] || seen[d.class_name].confidence < d.confidence) {
      seen[d.class_name] = d;
    }
  });

  detList.innerHTML = Object.values(seen).map(d => {
    const color = CAT_COLORS[d.class_name] || '#888';
    const label = d.class_name.replace(/_/g, ' ');
    const conf  = Math.round(d.confidence * 100);
    return `
      <div class="detection-tag">
        <div class="detection-dot" style="background:${color}"></div>
        <span>${label}</span>
        <span class="detection-conf">${conf}%</span>
      </div>`;
  }).join('');
}

function renderRecommendations(recs) {
  if (!recs.length) {
    recsList.innerHTML = '<div class="empty-recs">Point the camera at clothing to get recommendations</div>';
    return;
  }

  recsList.innerHTML = recs.map(r => {
    const emoji = CAT_EMOJI[r.category] || '👔';
    const cat   = (r.category || '').replace(/_/g, ' ');
    return `
      <div class="rec-card">
        <div class="rec-thumb">${emoji}</div>
        <div class="rec-info">
          <div class="rec-name">${escHtml(r.name)}</div>
          <div class="rec-cat">${cat}</div>
          <div class="rec-reason">${escHtml(r.reason)}</div>
          <div class="rec-price">${escHtml(r.price)}</div>
        </div>
      </div>`;
  }).join('');
}

function updateFPS(fps, ms) {
  if (fps != null) {
    fpsCounter.textContent = `${fps.toFixed(1)} FPS`;
  } else if (ms != null) {
    fpsCounter.textContent = `${ms.toFixed(0)} ms`;
  }

  // Inference bar (capped at 200ms = 100%)
  if (ms != null) {
    const pct = Math.min(100, (ms / 200) * 100);
    infBar.style.width = pct + '%';
    infBar.style.background = ms < 80 ? 'var(--accent3)' : ms < 150 ? 'var(--accent)' : 'var(--accent2)';
  }
}

function setStatus(state, text) {
  statusDot.className  = 'status-dot';
  if (state === 'active')     statusDot.classList.add('active');
  else if (state === 'error') statusDot.classList.add('error');
  statusText.textContent = text;
}

function updateConf(val) {
  confValue = val / 100;
  document.getElementById('conf-label').textContent = val + '%';
  // In a real app you'd send this to the backend
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
