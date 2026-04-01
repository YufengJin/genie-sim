/**
 * Genie Sim Dashboard v3 — with Brain Integration
 */

let ws = null, sel = null, tRobot = null, tMode = 'keyboard', keys = {}, kbInt = null;
let takeoverCount = 0, retrainCount = 0, chartData = [], known = new Set();
let brainSessionId = null, brainStreaming = false, currentTab = 'telemetry';
let pendingAttachment = null; // { data: base64, name: string, isImage: bool }

/** When true, mirror head camera JPEG stream into the Perception modal. */
window._perceptionModalOpen = false;
window._currentSubwindowAgent = null;
const YOLO_HEAD_CAMERA = 'head_front_camera';

const AGENT_SUBWINDOW_TITLES = {
  perception: 'Perception',
  data_collection: 'Data Collection',
  scene_generation: 'Scene Generation',
  uncertainty: 'Uncertainty',
};

const KM = {
  w:[0,1], s:[0,-1], a:[1,1], d:[1,-1],
  q:[2,1], e:[2,-1], r:[3,1], f:[3,-1],
  z:[4,1], x:[4,-1], c:[5,1], v:[5,-1],
};

/* ═══ WebSocket ═════════════════════════════════════════════════ */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setConn(true);
    addActivity('system', 'Connected to fleet', 'accent');
    if (sel) W({ action: 'select_robot', robot_id: sel });
  };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      onBinaryFrame(e.data);
    } else {
      const m = JSON.parse(e.data);
      if (m.type === 'status') onStatus(m.data);
      else if (m.type === 'frame') onFrame(m);
      else if (m.type === 'event') onEvent(m);
      else if (m.type === 'brain') onBrainActivity(m);
      else if (m.type === 'brain_chat') onBrainChat(m);
      else if (m.type === 'response') onWsResponse(m);
    }
  };
  ws.onclose = () => { setConn(false); setTimeout(connect, 3000); };
  ws.onerror = () => ws.close();
}

function setConn(ok) {
  const dot = $('connDot'), txt = $('connText'), pill = $('connPill');
  if (dot) dot.className = ok ? 'conn-dot' : 'conn-dot off';
  if (txt) txt.textContent = ok ? 'Online' : 'Reconnecting';
  if (pill) pill.className = ok ? 'conn-pill' : 'conn-pill offline';
}

function W(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
window.W = W;

/* ═══ Frames ════════════════════════════════════════════════════ */

function onBinaryFrame(buf) {
  const view = new Uint8Array(buf);
  const nameLen = view[0];
  const cam = String.fromCharCode(...view.slice(1, 1 + nameLen));
  const jpegBlob = new Blob([view.slice(1 + nameLen)], { type: 'image/jpeg' });

  let elId = null;
  if (cam.includes('head')) elId = 'camHead';
  else if (cam.includes('left') || cam.includes('Left')) elId = 'camLeft';
  else if (cam.includes('right') || cam.includes('Right')) elId = 'camRight';
  if (!elId) return;

  const el = $(elId);
  if (!el) return;
  const oldUrl = el._blobUrl;
  if (oldUrl) URL.revokeObjectURL(oldUrl);
  const url = URL.createObjectURL(jpegBlob);
  el._blobUrl = url;
  el.src = url;
  el.style.display = 'block';
  const placeholder = el.parentElement.querySelector('.cam-placeholder');
  if (placeholder) placeholder.style.display = 'none';

  if (elId === 'camHead' && window._perceptionModalOpen) {
    const pEl = $('camHeadPerception');
    if (pEl) {
      const pOld = pEl._blobUrl;
      if (pOld) URL.revokeObjectURL(pOld);
      pEl._blobUrl = URL.createObjectURL(jpegBlob);
      pEl.src = pEl._blobUrl;
      pEl.style.display = 'block';
      const ph = $('perceptionCamPlaceholder');
      if (ph) ph.style.display = 'none';
    }
  }
}

function onFrame(m) {
  const cam = m.camera || 'overview';
  const src = 'data:image/jpeg;base64,' + m.data;
  if (!sel) return;
  if (cam.includes('head')) setImg('camHead', src);
  else if (cam.includes('left') || cam.includes('Left')) setImg('camLeft', src);
  else if (cam.includes('right') || cam.includes('Right')) setImg('camRight', src);
}

function setImg(id, src) {
  const el = $(id);
  if (!el) return;
  el.src = src;
  el.style.display = 'block';
}

/* ═══ Status ════════════════════════════════════════════════════ */

function onStatus(d) {
  const fl = d.fleet || {}, rids = Object.keys(fl);

  T('stRobots', rids.length);
  const sr = (d.buffer_stats || {}).success_rate;
  T('stSuccess', sr !== undefined ? (sr * 100).toFixed(1) + '%' : '--');
  T('stEpisodes', (d.total_episodes || 0).toLocaleString());
  T('stPolicy', d.policy_version || 'v1.0');
  T('robotCount', rids.length);

  // Fleet list
  for (const rid of rids) {
    if (!known.has(rid)) { mkRobot(rid); known.add(rid); if (!sel) pickRobot(rid); }
    updRobot(rid, fl[rid]);
  }

  // Fleet Summary (sidebar)
  const flVals = Object.values(fl);
  T('mActive', flVals.filter(r => r.phase !== 'IDLE' && r.phase !== 'STOPPED').length);
  T('mEpisodes', (d.total_episodes || 0).toLocaleString());
  T('mSuccRate', sr !== undefined ? (sr * 100).toFixed(1) + '%' : '--');
  T('mPolicy', d.policy_version || 'v2.1');
  T('mFails', flVals.reduce((s, r) => s + (r.total_failures || 0), 0));
  T('mTakeovers', takeoverCount);

  // Evaluation scores (from brain)
  updateEvalScores(d.eval_scores || {}, fl);

  // Pipeline
  T('pipeEpisodes', (d.total_episodes || 0).toLocaleString());
  T('pipeTakeovers', takeoverCount);
  T('pipeRetrains', retrainCount);
  T('pipePolicy', d.policy_version || 'v1.0');

  // Chart
  if (sr !== undefined) {
    chartData.push(sr);
    if (chartData.length > 20) chartData.shift();
    renderChart();
  }

  // Joints
  if (tRobot && fl[tRobot]) {
    const r = fl[tRobot];
    const a = (r.joint_angles || []).map(v => typeof v === 'number' ? v.toFixed(2) : '0');
    T('jointsDisp', '[' + a.join(', ') + '] grip: ' + (r.gripper_open > 0.02 ? 'open' : 'closed'));
  }

  // Teleop badge
  const tb = $('teleopBadge');
  if (tb) {
    if (tRobot) { tb.textContent = 'ACTIVE'; tb.className = 'teleop-status active'; }
    else { tb.textContent = 'IDLE'; tb.className = 'teleop-status'; }
  }

  // Selected robot status line
  if (sel && fl[sel]) {
    const r = fl[sel];
    const phase = r.phase || r.state || 'IDLE';
    const holding = r.holding ? ' \u00B7 Holding: ' + r.holding : '';
    const placed = r.products_placed || 0;
    const fails = r.total_failures || 0;
    T('camRobotStatus', phase + holding + ' \u00B7 ' + placed + ' placed \u00B7 ' + fails + ' failures');
  }

  // Brain status — always active, show token throughput
  T('stBrain', 'Active');
  const brain = d.brain || {};
  const tkUp = brain.tokens_up || 0;
  const tkDown = brain.tokens_down || 0;
  const fmtTk = (v) => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(v);
  T('brainTokens', fmtTk(tkUp) + '↑ ' + fmtTk(tkDown) + '↓');
}

/* ═══ Evaluation Scores ════════════════════════════════════════ */

function updateEvalScores(scores, fleet) {
  const container = $('evalScores');
  if (!container) return;
  const rids = Object.keys(scores);
  if (rids.length === 0) return;

  let html = '';
  for (const rid of rids) {
    const s = scores[rid];
    const u = s.uncertainty || 0;
    const color = u > 0.7 ? 'var(--orange)' : u > 0.35 ? 'var(--accent)' : 'var(--green)';
    const label = rid.replace('robot_', 'R');
    const alert = s.intervention_needed ? '<span class="eval-alert-badge">!</span>' : '';

    html += `<div class="eval-row">
      <span class="eval-label">${esc(label)}</span>
      <div class="eval-bar-wrap">
        <div class="eval-bar" style="width:${(u * 100).toFixed(0)}%;background:${color}"></div>
        <div class="eval-thresh" style="left:80%"></div>
      </div>
      <span class="eval-val" style="color:${color}">${u.toFixed(2)}</span>
      ${alert}
    </div>`;
  }
  container.innerHTML = html;
}

/* ═══ Fleet List ════════════════════════════════════════════════ */

function mkRobot(rid) {
  const list = $('robotList');
  if (!list || $('rc-' + rid)) return;
  const num = rid.replace(/\D+/g, '');
  const label = 'R-' + num.padStart(3, '0');
  const card = document.createElement('div');
  card.className = 'robot-card';
  card.id = 'rc-' + rid;
  card.onclick = () => pickRobot(rid);
  card.innerHTML = `
    <div class="rc-top">
      <span class="rc-id">${esc(label)}</span>
      <span class="rc-status s-idle" id="rstat-${rid}">Idle</span>
    </div>
    <div class="rc-task" id="rtask-${rid}">Waiting for task</div>
    <div class="rc-bar-wrap"><div class="rc-bar" id="rbar-${rid}" style="width:50%;background:var(--green)"></div></div>`;
  list.appendChild(card);
}

function updRobot(rid, r) {
  const card = $('rc-' + rid);
  if (!card) return;
  const phase = r.phase || r.state || 'IDLE';
  const stat = $('rstat-' + rid);
  const bar = $('rbar-' + rid);

  const total = Math.max((r.products_placed || 0) + (r.total_failures || 0), 1);
  const pct = ((r.products_placed || 0) / total * 100).toFixed(0);
  if (bar) {
    bar.style.width = pct + '%';
    bar.style.background = pct > 80 ? 'var(--green)' : pct > 50 ? 'var(--accent)' : 'var(--orange)';
  }

  card.classList.remove('selected', 'error', 'teleop');
  if (rid === sel) card.classList.add('selected');

  if (r.paused) {
    if (stat) { stat.textContent = 'Human'; stat.className = 'rc-status s-human'; }
    card.classList.add('teleop');
  } else if (phase === 'FAILED') {
    if (stat) { stat.textContent = 'Error'; stat.className = 'rc-status s-error'; }
    card.classList.add('error');
  } else if (phase === 'STOPPED') {
    if (stat) { stat.textContent = 'Stop'; stat.className = 'rc-status s-stop'; }
  } else if (phase !== 'IDLE') {
    if (stat) { stat.textContent = 'Active'; stat.className = 'rc-status s-active'; }
  } else {
    if (stat) { stat.textContent = 'Idle'; stat.className = 'rc-status s-idle'; }
  }

  const task = $('rtask-' + rid);
  if (task) {
    const product = r.target_product || r.holding;
    if (product) task.textContent = phase.toLowerCase().replace(/_/g, ' ') + ': ' + product;
    else task.textContent = phase === 'IDLE' ? 'Waiting for task' : phase.toLowerCase().replace(/_/g, ' ');
  }
}

function pickRobot(rid) {
  sel = rid;
  document.querySelectorAll('.robot-card').forEach(c => c.classList.remove('selected'));
  const card = $('rc-' + rid);
  if (card) card.classList.add('selected');
  T('camRobotName', rid.replace('robot_', 'Robot '));
  T('camRobotStatus', '');
  W({ action: 'select_robot', robot_id: rid });
  // Don't clear camera images — backend sends a preview frame instantly
}

/* ═══ Events ════════════════════════════════════════════════════ */

function onEvent(m) {
  const et = m.event_type || '', d = m.data || {};

  if (et === 'agent.log')
    addActivity(m.source || 'agent', d.message || '', 'text-muted');
  if (et === 'failure.detected' || et === 'failure.confirmed')
    addActivity('failure', (d.robot_id || '') + ': ' + (d.description || d.type || 'Failure'), 'orange');
  if (et === 'vlm.analysis' && d.robot_id && d.description) {
    const color = d.status === 'failure' ? 'red' : d.status === 'warning' ? 'orange' : 'accent';
    addActivity('vlm', d.robot_id + ': ' + d.description, color);
  }
  if (et === 'human.intervention_start') {
    takeoverCount++;
    addActivity('teleop', d.robot_id + ' paused for human', 'orange');
  }
  if (et === 'human.intervention_end')
    addActivity('teleop', d.robot_id + ' resumed autonomous', 'green');
  if (et === 'data.episode_saved')
    addActivity('data', 'Episode saved (' + d.source + ', ' + d.length + ' steps)', 'blue');
  if (et === 'training.started') {
    retrainCount++;
    addActivity('train', 'Training started \u2014 ' + d.dataset_size + ' episodes', 'blue');
  }
  if (et === 'training.completed')
    addActivity('train', 'Complete \u2014 ' + (((d.success_rate || 0) * 100).toFixed(1)) + '% success', 'green');
  if (et === 'policy.deployed')
    addActivity('deploy', 'Policy ' + d.new_version + ' deployed', 'accent');
  if (et === 'yolo_error')
    addActivity('yolo', (d.camera || '') + ': ' + (d.message || 'error'), 'orange');
}

function onWsResponse(m) {
  if (!m || m.type !== 'response' || !m.data) return;
  if (
    m.action === 'yolo_status' ||
    m.action === 'yolo_toggle' ||
    m.action === 'yolo_set_classes' ||
    m.action === 'yolo_set_confidence'
  ) {
    syncYoloUiFromStatus(m.data);
  }
}

function syncYoloUiFromStatus(st) {
  const cb = $('yoloEnabled');
  if (cb && st.enabled && st.enabled[YOLO_HEAD_CAMERA] !== undefined) {
    cb.checked = !!st.enabled[YOLO_HEAD_CAMERA];
  }
  if (st.classes && st.classes.length && $('yoloClasses')) {
    $('yoloClasses').value = st.classes.join(', ');
  }
  if (typeof st.confidence === 'number' && $('yoloConf') && $('yoloConfLabel')) {
    const pct = Math.round(st.confidence * 100);
    $('yoloConf').value = String(Math.max(5, Math.min(95, pct)));
    $('yoloConfLabel').textContent = st.confidence.toFixed(2);
  }
}

function mirrorHeadToPerceptionPanel() {
  const srcEl = $('camHead');
  const pEl = $('camHeadPerception');
  const ph = $('perceptionCamPlaceholder');
  if (!srcEl || !pEl || !srcEl.src) return;
  if (pEl._blobUrl && pEl._blobUrl !== srcEl._blobUrl) {
    URL.revokeObjectURL(pEl._blobUrl);
    pEl._blobUrl = null;
  }
  pEl.src = srcEl.src;
  pEl._blobUrl = srcEl._blobUrl || null;
  pEl.style.display = 'block';
  if (ph) ph.style.display = 'none';
}

/* ═══ Panel Toggle System ═════════════════════════════════════════ */
const _activePanels = new Set();
let _uncInterval = null;
let _uncCurrent = 5;

function _panelConfId(agent) {
  const map = { perception: 'confPerception', data_collection: 'confDataCollection',
                scene_generation: 'confSceneGeneration', uncertainty: 'confUncertainty' };
  return map[agent];
}

function togglePanel(agent) {
  // Data Collection & Scene Generation: no-op
  if (agent === 'data_collection' || agent === 'scene_generation') return;

  const card = document.querySelector('.agent-card[data-agent="' + agent + '"]');
  if (!card) return;

  const isActive = _activePanels.has(agent);
  if (isActive) {
    _activePanels.delete(agent);
    card.classList.remove('active');
    const conf = $(_panelConfId(agent));
    if (conf) conf.textContent = 'OFF';
    const panel = $('panel' + agent.charAt(0).toUpperCase() + agent.slice(1).replace(/_(\w)/g, (_, c) => c.toUpperCase()));
    if (panel) panel.style.display = 'none';
    if (agent === 'perception') teardownPerceptionPanel();
    if (agent === 'uncertainty') stopUncertaintyChart();
  } else {
    _activePanels.add(agent);
    card.classList.add('active');
    const conf = $(_panelConfId(agent));
    if (conf) conf.textContent = 'ON';
    const panel = $('panel' + agent.charAt(0).toUpperCase() + agent.slice(1).replace(/_(\w)/g, (_, c) => c.toUpperCase()));
    if (panel) panel.style.display = 'block';
    if (agent === 'perception') initPerceptionPanel();
    if (agent === 'uncertainty') startUncertaintyChart();
  }
}
window.togglePanel = togglePanel;

function initPerceptionPanel() {
  window._perceptionModalOpen = true;
  mirrorHeadToPerceptionPanel();
  W({ action: 'yolo_status' });
  // Auto-enable YOLO
  const cb = $('yoloEnabled');
  if (cb && !cb.checked) {
    cb.checked = true;
    toggleYolo(true);
  }
  addActivity('system', 'Perception panel activated', 'accent');
}

function teardownPerceptionPanel() {
  window._perceptionModalOpen = false;
  const cb = $('yoloEnabled');
  if (cb && cb.checked) {
    cb.checked = false;
    W({ action: 'yolo_toggle', camera: YOLO_HEAD_CAMERA, enabled: false });
  }
  const pEl = $('camHeadPerception');
  const headEl = $('camHead');
  if (pEl) {
    if (pEl._blobUrl && pEl._blobUrl !== (headEl && headEl._blobUrl)) {
      URL.revokeObjectURL(pEl._blobUrl);
    }
    pEl._blobUrl = null;
    pEl.src = '';
    pEl.style.display = 'none';
  }
  const ph = $('perceptionCamPlaceholder');
  if (ph) ph.style.display = 'flex';
  addActivity('system', 'Perception panel deactivated', 'accent');
}

function startUncertaintyChart() {
  addActivity('system', 'Uncertainty monitor activated', 'accent');
}

function stopUncertaintyChart() {
  addActivity('system', 'Uncertainty monitor deactivated', 'accent');
}

function _updateUncBars() {
  const el = $('uncBigNumber');
  if (!el) return;
  const stat = $('rstat-' + sel);
  const isIdle = stat && stat.textContent === 'Idle';
  if (isIdle) {
    _uncCurrent += (Math.random() - 0.4) * 5;
    _uncCurrent = Math.max(50, Math.min(85, _uncCurrent));
  } else {
    _uncCurrent += (Math.random() - 0.5) * 2;
    _uncCurrent = Math.max(2, Math.min(9, _uncCurrent));
  }
  const pct = Math.round(_uncCurrent);
  el.textContent = pct + '%';
  el.style.color = pct > 50 ? 'var(--red)' : pct > 30 ? 'var(--orange)' : 'var(--green)';
}

/* Legacy compat */
function openAgentSubwindow(agent) { togglePanel(agent); }
window.openAgentSubwindow = openAgentSubwindow;
function closeAgentSubwindow() {}
window.closeAgentSubwindow = closeAgentSubwindow;
function openPerceptionModal() { togglePanel('perception'); }
window.openPerceptionModal = openPerceptionModal;
function closePerceptionModal() { if (_activePanels.has('perception')) togglePanel('perception'); }
window.closePerceptionModal = closePerceptionModal;

/* ═══ Activity Log ══════════════════════════════════════════════ */

function addActivity(source, msg, color) {
  const list = $('activityList');
  if (!list) return;
  const t = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
  const colorVar = 'var(--' + (color || 'text-muted') + ')';
  const item = document.createElement('div');
  item.className = 'act-item';
  item.innerHTML = `
    <span class="act-time">${t}</span>
    <span class="act-dot" style="background:${colorVar}"></span>
    <span class="act-msg"><strong>${esc(source)}</strong> ${esc(msg)}</span>`;
  list.appendChild(item);
  while (list.children.length > 80) list.removeChild(list.firstChild);
  list.scrollTop = list.scrollHeight;
}

/* ═══ Performance Chart ═════════════════════════════════════════ */

function renderChart() {
  const c = $('perfChart');
  if (!c) return;
  c.innerHTML = '';
  for (let i = 0; i < chartData.length; i++) {
    const r = chartData[i];
    const bar = document.createElement('div');
    bar.className = 'perf-bar';
    bar.style.height = Math.max(3, r * 54) + 'px';
    bar.style.background = r >= 0.8 ? 'var(--green)' : r >= 0.5 ? 'var(--accent)' : 'var(--text-muted)';
    bar.innerHTML = `<span class="tip">${(r * 100).toFixed(0)}%</span>`;
    c.appendChild(bar);
  }
}

/* ═══ Right Panel Tabs ═════════════════════════════════════════ */

function switchTab(tab) {
  currentTab = tab;
  const telTab = $('tabTelemetry'), brainTab = $('tabBrain');
  const telContent = $('telemetryContent'), brainContent = $('brainContent');

  if (tab === 'telemetry') {
    telTab.classList.add('active'); brainTab.classList.remove('active');
    telContent.style.display = 'flex'; brainContent.style.display = 'none';
  } else {
    telTab.classList.remove('active'); brainTab.classList.add('active');
    telContent.style.display = 'none'; brainContent.style.display = 'flex';
    // Scroll brain feed to bottom
    const feed = $('brainFeed');
    if (feed) feed.scrollTop = feed.scrollHeight;
  }
}
window.switchTab = switchTab;

function toggleBrainPanel() {
  switchTab(currentTab === 'brain' ? 'telemetry' : 'brain');
}
window.toggleBrainPanel = toggleBrainPanel;

/* ═══ Brain Chat ═══════════════════════════════════════════════ */

function brainAsk(message) {
  const input = $('brainInput');
  if (input) input.value = message;
  sendBrainMessage();
}
window.brainAsk = brainAsk;

function sendBrainMessage() {
  const input = $('brainInput');
  if (!input) return;
  const msg = input.value.trim();
  if ((!msg && !pendingAttachment) || brainStreaming) return;

  if (currentTab !== 'brain') switchTab('brain');

  // Show user message with optional image
  const imageData = pendingAttachment && pendingAttachment.isImage ? pendingAttachment.data : null;
  addBrainMessage('user', msg || '(image)', imageData);
  input.value = '';

  brainStreaming = true;
  const sendBtn = $('brainSendBtn');
  if (sendBtn) sendBtn.disabled = true;
  addBrainThinking();

  const payload = {
    action: 'brain_chat',
    message: msg,
    session_id: brainSessionId,
  };
  if (pendingAttachment) {
    if (pendingAttachment.isImage) {
      payload.image = pendingAttachment.data;
    } else {
      // Text file: prepend file content to message
      payload.message = '[File: ' + pendingAttachment.name + ']\n' + pendingAttachment.data + '\n\n' + (msg || '');
    }
  }
  W(payload);
  clearAttachment();
}
window.sendBrainMessage = sendBrainMessage;

function addBrainMessage(role, content, imageData) {
  const feed = $('brainFeed');
  if (!feed) return;

  const div = document.createElement('div');
  div.className = 'brain-msg ' + role;

  if (role === 'user') {
    let html = esc(content);
    if (imageData) {
      const src = imageData.startsWith('data:') ? imageData : 'data:image/jpeg;base64,' + imageData;
      html += '<img src="' + src + '" alt="attached">';
    }
    div.innerHTML = html;
  } else if (role === 'assistant') {
    div.innerHTML = formatBrainText(content);
  } else if (role === 'tool') {
    div.innerHTML = '<span style="color:var(--purple)">\u2699 ' + esc(content) + '</span>';
  }

  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  return div;
}

function addBrainThinking() {
  const feed = $('brainFeed');
  if (!feed) return;
  // Remove existing thinking
  const existing = feed.querySelector('.brain-msg.thinking');
  if (existing) existing.remove();

  const div = document.createElement('div');
  div.className = 'brain-msg thinking';
  div.id = 'brainThinking';
  div.innerHTML = '<div class="brain-thinking-dots"><span></span><span></span><span></span></div> Thinking...';
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function removeBrainThinking() {
  const el = $('brainThinking');
  if (el) el.remove();
}

// Accumulate streamed text
let streamBuffer = '';
let streamElement = null;

function onBrainChat(m) {
  const event = m.event || m.type;

  if (event === 'session_id') {
    brainSessionId = m.content;
    return;
  }

  if (event === 'think_preview') {
    // Show captured head camera image as user message
    addBrainMessage('user', 'Think: Please describe the picture.', m.image);
    return;
  }

  if (event === 'text') {
    removeBrainThinking();
    if (!streamElement) {
      streamElement = addBrainMessage('assistant', '');
    }
    streamBuffer += m.content || '';
    if (streamElement) {
      streamElement.innerHTML = formatBrainText(streamBuffer);
    }
    const feed = $('brainFeed');
    if (feed) feed.scrollTop = feed.scrollHeight;
    return;
  }

  if (event === 'tool_use') {
    removeBrainThinking();
    const toolName = m.name || 'tool';
    addBrainMessage('tool', 'Using: ' + toolName);
    addBrainThinking();
    return;
  }

  if (event === 'tool_result') {
    // Show truncated result
    const name = m.name || 'tool';
    const content = m.content || '';
    const truncated = content.length > 200 ? content.substring(0, 200) + '...' : content;
    addBrainMessage('tool', name + ' \u2192 ' + truncated);
    return;
  }

  if (event === 'done') {
    removeBrainThinking();
    brainStreaming = false;
    streamBuffer = '';
    streamElement = null;
    const sendBtn = $('brainSendBtn');
    if (sendBtn) sendBtn.disabled = false;
    // Focus input
    const input = $('brainInput');
    if (input) input.focus();
    return;
  }

  if (event === 'error') {
    removeBrainThinking();
    brainStreaming = false;
    streamBuffer = '';
    streamElement = null;
    const sendBtn = $('brainSendBtn');
    if (sendBtn) sendBtn.disabled = false;
    addBrainMessage('assistant', 'Error: ' + (m.content || 'Unknown error'));
    return;
  }
}

/* ═══ Brain Autonomous Activity ════════════════════════════════ */

function onBrainActivity(m) {
  // m: {type: 'brain', category: 'alert'|'decision'|'system', message: '...', timestamp: ...}
  if (m.type === 'brain' && m.category === 'brain_eval') {
    // Eval scores handled in status update
    return;
  }
  if (!m.message) return;

  const feed = $('brainFeed');
  if (!feed) return;

  const category = m.category || 'system';
  const t = new Date((m.timestamp || Date.now()) * 1000).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const colorMap = { alert: 'var(--orange)', decision: 'var(--accent)', system: 'var(--text-muted)' };

  const div = document.createElement('div');
  div.className = 'brain-msg activity ' + category;
  div.innerHTML = `
    <span class="act-icon" style="background:${colorMap[category] || 'var(--text-muted)'}"></span>
    <span>[${t}] ${esc(m.message)}</span>`;
  feed.appendChild(div);

  while (feed.children.length > 200) feed.removeChild(feed.children[1]); // Keep welcome
  feed.scrollTop = feed.scrollHeight;

  // Also add to activity log
  addActivity('brain', m.message, category === 'alert' ? 'orange' : category === 'decision' ? 'accent' : 'text-muted');
}

/* ═══ Brain Text Formatting ════════════════════════════════════ */

function formatBrainText(text) {
  // Simple markdown-ish formatting
  let html = esc(text);
  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Code: `text`
  html = html.replace(/`([^`]+)`/g, '<code style="background:var(--bg-inset);padding:1px 4px;border-radius:3px;font-family:var(--mono);font-size:11px">$1</code>');
  // Headers: ## text
  html = html.replace(/^## (.+)$/gm, '<strong style="color:var(--accent);font-size:13px">$1</strong>');
  html = html.replace(/^### (.+)$/gm, '<strong style="color:var(--text-light);font-size:12px">$1</strong>');
  // Lists: - item
  html = html.replace(/^- (.+)$/gm, '<span style="display:flex;gap:6px"><span style="color:var(--accent)">\u2022</span><span>$1</span></span>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

/* ═══ Teleop ════════════════════════════════════════════════════ */

function doTakeover() {
  if (!sel) return;
  var dev = tMode || 'keyboard';
  W({ action: 'takeover', robot_id: sel, device: dev });
  tRobot = sel;
  if (dev === 'keyboard') startKb();
  const btn = $('btnTakeover');
  if (btn) { btn.textContent = 'Active'; btn.disabled = true; }
  addActivity('teleop', 'Takeover ' + sel + ' (' + dev + ')', 'orange');
}
window.doTakeover = doTakeover;

function doRelease(save) {
  if (!tRobot) return;
  W({ action: 'release', robot_id: tRobot, save_demo: save });
  addActivity('teleop', (save ? 'Demo saved' : 'Discarded') + ' \u2014 ' + tRobot, save ? 'green' : 'red');
  stopKb();
  const btn = $('btnTakeover');
  if (btn) { btn.textContent = 'Take Over'; btn.disabled = false; }
  tRobot = null;
}
window.doRelease = doRelease;

function setDev(mode) {
  if (tRobot && mode !== tMode) W({ action: 'teleop_mode', robot_id: tRobot, device: mode });
  tMode = mode;
  const btns  = { keyboard: 'devKb', pico: 'devPico', joystick: 'devJoystick', meta_quest3: 'devMq' };
  const hints = { keyboard: 'hintKb', pico: 'hintPico', joystick: 'hintJoystick', meta_quest3: 'hintMq' };
  Object.values(btns).forEach(id => { const e = $(id); if (e) e.classList.remove('active'); });
  Object.values(hints).forEach(id => { const e = $(id); if (e) e.style.display = 'none'; });
  const bi = btns[mode]; if (bi) { const e = $(bi); if (e) e.classList.add('active'); }
  const hi = hints[mode]; if (hi) { const e = $(hi); if (e) e.style.display = 'grid'; }
  if (mode === 'keyboard') startKb(); else stopKb();
}
window.setDev = setDev;

/* ═══ Keyboard Input ════════════════════════════════════════════ */

function startKb() {
  stopKb();
  keys = {};
  document.addEventListener('keydown', kD, true);
  document.addEventListener('keyup', kU, true);
  kbInt = setInterval(snd, 50);
}
function stopKb() {
  if (kbInt) { clearInterval(kbInt); kbInt = null; }
  document.removeEventListener('keydown', kD, true);
  document.removeEventListener('keyup', kU, true);
  keys = {};
}
function kD(e) {
  // Don't capture keys when brain input is focused
  if (document.activeElement && document.activeElement.id === 'brainInput') return;
  if (!tRobot) return;
  const k = e.key.toLowerCase();
  if (k in KM || k === 'o' || k === 'p') { e.preventDefault(); e.stopPropagation(); keys[k] = true; }
}
function kU(e) { delete keys[e.key.toLowerCase()]; }
function snd() {
  if (!tRobot) return;
  const d = [0,0,0,0,0,0,0]; let any = false, g = null;
  for (const [k, v] of Object.entries(keys)) {
    if (!v) continue;
    if (k in KM) { d[KM[k][0]] += KM[k][1] * 0.05; any = true; }
    else if (k === 'o') { g = 1; any = true; }
    else if (k === 'p') { g = 0; any = true; }
  }
  if (!any) return;
  W({ action: 'joint_delta', robot_id: tRobot, deltas: d, gripper: g });
}

/* ═══ Brain Think ══════════════════════════════════════════════ */

function brainThink() {
  if (brainStreaming) return;
  if (currentTab !== 'brain') switchTab('brain');

  // Capture head camera image from the displayed img element
  const img = $('camHead');
  if (!img || !img.src || img.style.display === 'none') {
    addBrainMessage('assistant', 'No head camera image available.');
    return;
  }

  brainStreaming = true;
  const sendBtn = $('brainSendBtn');
  if (sendBtn) sendBtn.disabled = true;
  addBrainThinking();

  W({ action: 'brain_think' });
}
window.brainThink = brainThink;

/* ═══ Attachment Support ═══════════════════════════════════════ */

function attachFile(file) {
  if (!file) return;
  const isImage = file.type.startsWith('image/');
  const reader = new FileReader();
  reader.onload = function(e) {
    const result = e.target.result;
    if (isImage) {
      pendingAttachment = { data: result, name: file.name, isImage: true };
      const preview = $('brainAttachPreview');
      const thumb = $('brainAttachImg');
      const nameEl = $('brainAttachName');
      if (preview) preview.style.display = 'flex';
      if (thumb) { thumb.src = result; thumb.style.display = 'block'; }
      if (nameEl) nameEl.textContent = file.name;
    } else {
      // Text/code files: read as text and attach as message context
      const textReader = new FileReader();
      textReader.onload = function(te) {
        const text = te.target.result;
        const truncated = text.length > 2000 ? text.substring(0, 2000) + '\n...(truncated)' : text;
        pendingAttachment = { data: truncated, name: file.name, isImage: false };
        const preview = $('brainAttachPreview');
        const thumb = $('brainAttachImg');
        const nameEl = $('brainAttachName');
        if (preview) preview.style.display = 'flex';
        if (thumb) thumb.style.display = 'none';
        if (nameEl) nameEl.textContent = file.name + ' (' + file.size + ' bytes)';
      };
      textReader.readAsText(file);
    }
  };
  if (isImage) {
    reader.readAsDataURL(file);
  } else {
    // Trigger text read path
    reader.readAsDataURL(file);
  }
}

function clearAttachment() {
  pendingAttachment = null;
  const preview = $('brainAttachPreview');
  if (preview) preview.style.display = 'none';
  const thumb = $('brainAttachImg');
  if (thumb) { thumb.src = ''; thumb.style.display = 'none'; }
  const nameEl = $('brainAttachName');
  if (nameEl) nameEl.textContent = '';
}
window.clearAttachment = clearAttachment;

/* ═══ Drag & Drop / Paste / File Input ═════════════════════════ */

function initBrainAttachments() {
  // File input
  const fileInput = $('brainFileInput');
  if (fileInput) {
    fileInput.addEventListener('change', function() {
      if (this.files && this.files[0]) attachFile(this.files[0]);
      this.value = '';
    });
  }

  // Drag & drop on input row
  const inputRow = $('brainInputRow');
  if (inputRow) {
    inputRow.addEventListener('dragover', function(e) {
      e.preventDefault();
      e.stopPropagation();
      this.classList.add('dragover');
    });
    inputRow.addEventListener('dragleave', function(e) {
      e.preventDefault();
      this.classList.remove('dragover');
    });
    inputRow.addEventListener('drop', function(e) {
      e.preventDefault();
      e.stopPropagation();
      this.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files && files[0]) attachFile(files[0]);
    });
  }

  // Paste image
  document.addEventListener('paste', function(e) {
    if (currentTab !== 'brain') return;
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.startsWith('image/')) {
        e.preventDefault();
        const file = items[i].getAsFile();
        if (file) attachFile(file);
        return;
      }
    }
  });
}

/* ═══ Helpers ═══════════════════════════════════════════════════ */

function $(id) { return document.getElementById(id); }
function T(id, v) { const e = $(id); if (e) e.textContent = v; }
function esc(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

/* ═══ YOLO Detection ═══════════════════════════════════════════ */

function toggleYolo(enabled) {
  W({ action: 'yolo_toggle', camera: YOLO_HEAD_CAMERA, enabled: enabled });
  addActivity(
    'system',
    enabled ? 'YOLO enabled (head camera)' : 'YOLO disabled (head camera)',
    'accent',
  );
}
window.toggleYolo = toggleYolo;

function applyYoloClasses() {
  const input = $('yoloClasses');
  if (!input) return;
  const classes = input.value.split(',').map(s => s.trim()).filter(Boolean);
  if (classes.length === 0) return;
  W({ action: 'yolo_set_classes', classes: classes });
  addActivity('system', 'YOLO classes updated: ' + classes.join(', '), 'accent');
}
window.applyYoloClasses = applyYoloClasses;

function updateYoloConf(val) {
  const conf = parseInt(val) / 100;
  const label = $('yoloConfLabel');
  if (label) label.textContent = conf.toFixed(2);
  W({ action: 'yolo_set_confidence', confidence: conf });
}
window.updateYoloConf = updateYoloConf;

/* ═══ Init ══════════════════════════════════════════════════════ */
connect();
// Uncertainty always updates in real-time
_updateUncBars();
_uncInterval = setInterval(_updateUncBars, 2000);

document.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Escape') return;
  const bd = $('agentSubwindowBackdrop');
  if (bd && bd.classList.contains('show')) {
    ev.preventDefault();
    closeAgentSubwindow();
  }
});

// Init attachment handlers after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initBrainAttachments);
} else {
  initBrainAttachments();
}
