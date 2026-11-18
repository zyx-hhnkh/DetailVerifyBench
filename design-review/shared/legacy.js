
// ─── Column definitions ───────────────────────────────────────────────────────
// Real: 8 dimension types (no Scene/Camera)
const REAL_COLS = [
  {key:'p',     label:'P<sub>tok</sub>',  group:'Overall', sep:true,  src:'overall'},
  {key:'r',     label:'R<sub>tok</sub>',  group:'Overall',            src:'overall'},
  {key:'f1',    label:'F1<sub>tok</sub>', group:'Overall',            src:'overall'},
  {key:'num',   label:'Num',   group:'Dimension R<sub>tok</sub>', sep:true, src:'dim'},
  {key:'clr',   label:'Clr',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'cat',   label:'Cat',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'shp',   label:'Shp',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'mat',   label:'Mat',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'spat',  label:'Spat',  group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'ocr',   label:'OCR',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'other', label:'Other', group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'chart',  label:'Chart',  group:'Domain F1', sep:true, src:'domain'},
  {key:'movie',  label:'Movie',  group:'Domain F1',           src:'domain'},
  {key:'nature', label:'Nature', group:'Domain F1',           src:'domain'},
  {key:'poster', label:'Poster', group:'Domain F1',           src:'domain'},
  {key:'gui',    label:'GUI',    group:'Domain F1',           src:'domain'},
];

// Synthetic: 10 dimension types (+ Scene, Camera)
const SYNTH_COLS = [
  {key:'p',     label:'P<sub>tok</sub>',  group:'Overall', sep:true,  src:'overall'},
  {key:'r',     label:'R<sub>tok</sub>',  group:'Overall',            src:'overall'},
  {key:'f1',    label:'F1<sub>tok</sub>', group:'Overall',            src:'overall'},
  {key:'num',   label:'Num',   group:'Dimension R<sub>tok</sub>', sep:true, src:'dim'},
  {key:'clr',   label:'Clr',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'cat',   label:'Cat',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'shp',   label:'Shp',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'mat',   label:'Mat',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'spat',  label:'Spat',  group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'ocr',   label:'OCR',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'scene', label:'Scene', group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'cam',   label:'Cam',   group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'other', label:'Other', group:'Dimension R<sub>tok</sub>',           src:'dim'},
  {key:'chart',  label:'Chart',  group:'Domain F1', sep:true, src:'domain'},
  {key:'movie',  label:'Movie',  group:'Domain F1',           src:'domain'},
  {key:'nature', label:'Nature', group:'Domain F1',           src:'domain'},
  {key:'poster', label:'Poster', group:'Domain F1',           src:'domain'},
  {key:'gui',    label:'GUI',    group:'Domain F1',           src:'domain'},
];

// ─── State ───────────────────────────────────────────────────────────────────
let lbModels   = [];   // raw data from leaderboard.json
let currentTab    = 'real';
let currentFilter = 'all';
let sortColIdx    = null;
let sortDir       = 'desc';

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmt(v) {
  if (v === null || v === undefined) return '<span class="dash-val">—</span>';
  return v.toFixed(2);
}

function getVal(model, col) {
  const src = currentTab === 'real' ? model.real : model.synthetic;
  if (col.src === 'overall') return src[col.key];
  if (col.src === 'dim')     return src.dim[col.key] ?? null;
  if (col.src === 'domain')  return src.domain[col.key] ?? null;
  return null;
}

function getBestValues(rows, cols) {
  const best = {};
  cols.forEach((col, i) => {
    let max = -Infinity;
    rows.forEach(m => {
      const v = getVal(m, col);
      if (v !== null && v !== undefined && v > max) max = v;
    });
    best[i] = max;
  });
  return best;
}

// ─── Render ──────────────────────────────────────────────────────────────────
function render() {
  if (!lbModels.length) return;
  const cols = currentTab === 'real' ? REAL_COLS : SYNTH_COLS;

  // Filter
  let rows = lbModels;
  if (currentFilter === 'open')   rows = lbModels.filter(m => m.type === 'open');
  if (currentFilter === 'closed') rows = lbModels.filter(m => m.type === 'closed');

  // Sort
  if (sortColIdx !== null) {
    const col = cols[sortColIdx];
    rows = [...rows].sort((a, b) => {
      const va = getVal(a, col) ?? -Infinity;
      const vb = getVal(b, col) ?? -Infinity;
      return sortDir === 'desc' ? vb - va : va - vb;
    });
  } else {
    rows = [...rows].sort((a, b) => {
      const src = currentTab === 'real';
      const fa = (src ? a.real : a.synthetic).f1 ?? -Infinity;
      const fb = (src ? b.real : b.synthetic).f1 ?? -Infinity;
      return fb - fa;
    });
  }

  const best = getBestValues(rows, cols);

  // ── Build thead ──
  const groups = [];
  let curGroup = null, curCount = 0, curSep = false;
  cols.forEach(col => {
    if (col.group !== curGroup) {
      if (curGroup !== null) groups.push({label: curGroup, count: curCount, sep: curSep});
      curGroup = col.group; curCount = 1; curSep = !!col.sep;
    } else { curCount++; }
  });
  if (curGroup !== null) groups.push({label: curGroup, count: curCount, sep: curSep});

  let row1 = `<th class="sticky-1" rowspan="2">Rank</th>`;
  row1 += `<th class="sticky-2" rowspan="2">Model</th>`;
  row1 += `<th class="sticky-3" rowspan="2">Type</th>`;
  groups.forEach(g => {
    row1 += `<th colspan="${g.count}"${g.sep ? ' class="col-sep"' : ''}>${g.label}</th>`;
  });

  let row2 = '';
  cols.forEach((col, i) => {
    const isSorted = sortColIdx === i;
    const cls = [
      isSorted ? (sortDir === 'desc' ? 'sort-desc' : 'sort-asc') : '',
      col.sep ? 'col-sep' : ''
    ].filter(Boolean).join(' ');
    row2 += `<th${cls ? ` class="${cls}"` : ''} onclick="sortBy(${i})">${col.label}</th>`;
  });

  document.getElementById('lb-thead').innerHTML = `<tr>${row1}</tr><tr>${row2}</tr>`;

  // ── Build tbody ──
  let tbody = '';
  rows.forEach((model, ri) => {
    const rank = ri + 1;
    const rankClass = rank <= 3 ? `rank-${rank}` : '';
    const medal = String(rank).padStart(2, '0');
    const badgeClass = model.type === 'open' ? 'badge-open' : 'badge-closed';
    const badgeLabel = model.type === 'open' ? 'Open' : 'Closed';

    let cells = `<td class="sticky-1 rank-medal ${rankClass}">${medal}</td>`;
    cells += `<td class="sticky-2 align-left">${model.model}</td>`;
    cells += `<td class="sticky-3"><span class="badge ${badgeClass}">${badgeLabel}</span></td>`;

    cols.forEach((col, i) => {
      const v = getVal(model, col);
      const isBest = v !== null && v !== undefined && v === best[i] && v > 0;
      const sepClass = col.sep ? 'col-sep-body' : '';
      const bestClass = isBest ? 'best' : '';
      const cls = [sepClass, bestClass].filter(Boolean).join(' ');
      cells += `<td${cls ? ` class="${cls}"` : ''}>${fmt(v)}</td>`;
    });

    tbody += `<tr>${cells}</tr>`;
  });

  document.getElementById('lb-tbody').innerHTML = tbody;
}

// ─── Controls ────────────────────────────────────────────────────────────────
function switchTab(tab) {
  currentTab = tab;
  sortColIdx = null;
  document.querySelectorAll('.lb-tab').forEach((el, i) => {
    el.classList.toggle('active', (i === 0 && tab === 'real') || (i === 1 && tab === 'synthetic'));
  });
  render();
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(el => {
    el.classList.toggle('active', el.textContent.trim().toLowerCase().startsWith(
      f === 'all' ? 'all' : f === 'open' ? 'open' : 'closed'
    ));
  });
  render();
}

function sortBy(colIdx) {
  sortDir = sortColIdx === colIdx ? (sortDir === 'desc' ? 'asc' : 'desc') : 'desc';
  sortColIdx = colIdx;
  render();
}

// ─── Injection Demo ──────────────────────────────────────────────────────────
let injData = [];
let injCurrent = null;
let injRound = 0;

function renderInjAllMarked(text) {
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/&lt;\/?HALLUCINATION&gt;/gi, m =>
      m.toLowerCase().includes('/') ? '__HALEND__' : '__HALSTART__'
    )
    .replace(/__HALSTART__([\s\S]*?)__HALEND__/g, (_, inner) =>
      '<mark class="hal">' + inner + '</mark>'
    );
}

function renderInjDetected(text, matchedSpans, unmatchedSpans) {
  // First HTML-escape, then convert tags to markers
  let html = text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/&lt;\/?HALLUCINATION&gt;/gi, m =>
      m.toLowerCase().includes('/') ? '__HALEND__' : '__HALSTART__'
    );

  // Build lookup for matched spans (detected = yellow)
  const matchedSet = new Set(matchedSpans.map(s => s.trim()));

  html = html.replace(/__HALSTART__([\s\S]*?)__HALEND__/g, (_, inner) => {
    const raw = inner.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>').trim();
    const cls = matchedSet.has(raw) ? 'hal-detected' : 'hal';
    return '<mark class="' + cls + '">' + inner + '</mark>';
  });

  return html;
}

function renderInjViewer() {
  const viewer = document.getElementById('inj-viewer');
  if (!injCurrent) { viewer.innerHTML = ''; return; }

  const maxRound = injCurrent.iterations.length;
  const round = Math.min(injRound, maxRound - 1);
  const it = injCurrent.iterations[round];

  // Update round button states
  document.querySelectorAll('.inj-round-tabs .pipe-tab').forEach((b, i) => {
    b.classList.toggle('active', i === round);
    b.disabled = i >= maxRound;
    b.style.opacity = i >= maxRound ? '0.4' : '1';
  });

  // Left: all injected hallucinations marked red (before detection)
  const allMarked = renderInjAllMarked(it.description_tag_nodetect);
  // Right: after detection — matched=yellow, unmatched=red
  const afterDetect = renderInjDetected(it.description_tag_nodetect, it.matched_spans, it.unmatched_spans);

  const s = it.stats;
  const survived = s.num_injected - s.num_detected;
  const statsHtml =
    '<div class="inj-stats-inline">'
    + '<span class="inj-stat"><strong>' + s.num_injected + '</strong> injected</span>'
    + '<span class="inj-stat inj-stat-detected"><strong>' + s.num_detected + '</strong> detected (' + Math.round(s.detection_rate * 100) + '%)</span>'
    + '<span class="inj-stat inj-stat-survived"><strong>' + survived + '</strong> survived</span>'
    + '</div>';

  const legendHtml =
    '<div class="inj-legend-inline">'
    + '<span class="inj-legend-item"><mark class="hal">text</mark>: Injected hallucination</span>'
    + '<span class="inj-legend-item"><mark class="hal-detected">text</mark>: Hallucination detected by the Detector</span>'
    + '<span class="inj-legend-note">Note: Detected hallucinations are excluded from the final synthetic dataset.</span>'
    + '</div>';

  viewer.innerHTML =
    '<div class="inj-panel inj-panel-img">'
    + '<img src="' + injCurrent.image + '" alt="' + escHtml(injCurrent.filename) + '" loading="lazy">'
    + '</div>'
    + '<div class="inj-right">'
    + '<div class="inj-texts">'
    +   '<div class="inj-panel inj-panel-text">'
    +   '<div class="inj-panel-title">Injected Description (Round ' + (round+1) + ')</div>'
    +   '<div class="inj-panel-body">' + allMarked + '</div>'
    +   '</div>'
    +   '<div class="inj-panel inj-panel-text">'
    +   '<div class="inj-panel-title">Detection Result (Round ' + (round+1) + ')</div>'
    +   '<div class="inj-panel-body">' + afterDetect + '</div>'
    +   '</div>'
    + '</div>'
    + statsHtml
    + legendHtml
    + '</div>';

}

function setInjRound(r) {
  injRound = r;
  renderInjViewer();
}

function setInjExample(filename) {
  injCurrent = injData.find(e => e.filename === filename) || null;
  injRound = 0;
  renderInjViewer();
}

function initInjDemo(data) {
  // Remove the first Poster example (no data available)
  let posterCount = 0;
  data = data.filter(e => {
    if (e.category === 'Poster') { posterCount++; return posterCount > 1; }
    return true;
  });

  injData = data;
  const sel = document.getElementById('inj-example-sel');
  // Count per-category for fallback labels
  const catCount = {};
  sel.innerHTML = data.map(e => {
    catCount[e.category] = (catCount[e.category] || 0) + 1;
    const label = INJ_NAMES[e.filename] || (e.category + ' - Example ' + catCount[e.category]);
    return '<option value="' + e.filename + '">' + label + '</option>';
  }).join('');
  injCurrent = data[0] || null;
  injRound = 0;
  renderInjViewer();
}

// ─── Detection Visualization ─────────────────────────────────────────────────
const DET_MODELS = [
  'GPT-5.2','GPT-5.4','Gemini3-Pro','Gemini-3.1-Pro','Claude-Opus-4.6',
  'Step3-VL-10B','Qwen3-VL-8B-Think','VCD+Qwen3-VL-8B',
  'GLM-4.6V-Flash','Qwen3.5-9B','Qwen3.5-35B-A3B','Qwen3.5-397B-A17B',
  'KIMI-K2.5','Seed2.0-pro','Mimo-v2-pro'
];
const DET_COLORS = [
  '#2563eb','#7c3aed','#0ea5e9','#ec4899','#f97316',
  '#22c55e','#ef4444','#6366f1','#14b8a6','#e11d48',
  '#84cc16','#a855f7','#06b6d4','#d946ef','#f59e0b'
];

let detData = [];
let detMode = 'real';
let detCurrent = null;
let detChecked = new Set(['GPT-5.4']);
let detDropdownOpen = false;

function initDetVis(data) {
  detData = data;
  // Populate example dropdown (same style as injection demo)
  const sel = document.getElementById('det-example-sel');
  const catCount = {};
  sel.innerHTML = data.map(e => {
    catCount[e.category] = (catCount[e.category] || 0) + 1;
    const label = e.category + ' - Example ' + catCount[e.category];
    return '<option value="' + e.filename + '">' + label + '</option>';
  }).join('');
  detCurrent = data[0] || null;

  // Build model dropdown menu
  const menu = document.getElementById('det-model-menu');
  menu.innerHTML = DET_MODELS.map((m, i) => {
    const checked = detChecked.has(m);
    return '<div class="det-model-option' + (checked ? ' selected' : '') + '" data-model="' + m + '" onclick="toggleDetModel(\'' + m + '\')">'
      + '<span class="det-model-swatch" style="background:' + DET_COLORS[i] + '"></span>'
      + '<span>' + escHtml(m) + '</span>'
      + '<span class="det-model-check">' + (checked ? '✓' : '') + '</span>'
      + '</div>';
  }).join('');

  renderDetViewer();
}

function toggleDetDropdown() {
  detDropdownOpen = !detDropdownOpen;
  document.getElementById('det-model-menu').classList.toggle('open', detDropdownOpen);
}

// Close dropdown when clicking outside
document.addEventListener('click', function(e) {
  if (!e.target.closest('#det-model-dropdown')) {
    detDropdownOpen = false;
    const menu = document.getElementById('det-model-menu');
    if (menu) menu.classList.remove('open');
  }
});

function setDetMode(mode) {
  detMode = mode;
  document.getElementById('det-btn-real').classList.toggle('active', mode === 'real');
  document.getElementById('det-btn-synth').classList.toggle('active', mode === 'synthetic');
  renderDetViewer();
}

function setDetExample(filename) {
  detCurrent = detData.find(e => e.filename === filename) || null;
  renderDetViewer();
}

function toggleDetModel(model) {
  if (detChecked.has(model)) detChecked.delete(model);
  else detChecked.add(model);
  // Update dropdown item UI
  document.querySelectorAll('.det-model-option').forEach(el => {
    const m = el.dataset.model;
    const on = detChecked.has(m);
    el.classList.toggle('selected', on);
    el.querySelector('.det-model-check').textContent = on ? '✓' : '';
  });
  document.getElementById('det-model-count').textContent = detChecked.size;
  renderDetViewer();
}

function renderDetViewer() {
  const viewer = document.getElementById('det-viewer');
  if (!detCurrent) { viewer.innerHTML = ''; return; }

  const src = detCurrent[detMode];
  if (!src || !src.gt_tagged) {
    viewer.innerHTML = '<div style="padding:24px;color:#94a3b8;">No data available for this mode.</div>';
    return;
  }

  // Build annotated text
  const html = buildDetText(src);

  // Build legend (below description, larger font)
  const legendItems = ['<span class="det-legend-item"><span class="det-legend-swatch" style="background:#fee2e2;border:1px solid #f87171"></span> GT</span>'];
  for (const m of detChecked) {
    const idx = DET_MODELS.indexOf(m);
    const c = DET_COLORS[idx] || '#888';
    legendItems.push('<span class="det-legend-item"><span class="det-legend-swatch" style="background:' + c + '"></span> ' + escHtml(m) + '</span>');
  }
  const legendHtml = '<div class="det-legend">' + legendItems.join('') + '</div>';

  viewer.innerHTML =
    '<div class="det-panel-img">'
    + '<img src="' + detCurrent.image + '" alt="' + escHtml(detCurrent.filename) + '" loading="lazy">'
    + '</div>'
    + '<div class="det-panel-text">'
    + '<div class="det-text-body">' + html + '</div>'
    + legendHtml
    + '</div>';
}

function buildDetText(src) {
  // Parse GT tagged text to extract plain text and GT span positions
  const tagged = src.gt_tagged;
  const gtSpans = [];  // {start, end} in plain text
  let plain = '';
  const tagRe = /<\/?HALLUCINATION>/gi;
  let lastIdx = 0, inHal = false, halStart = 0;
  let match;
  // Reset lastIndex
  tagRe.lastIndex = 0;
  while ((match = tagRe.exec(tagged)) !== null) {
    // Add text before this tag to plain
    plain += tagged.substring(lastIdx, match.index);
    if (match[0].toLowerCase().includes('/')) {
      // Closing tag
      if (inHal) {
        gtSpans.push({start: halStart, end: plain.length});
        inHal = false;
      }
    } else {
      // Opening tag
      halStart = plain.length;
      inHal = true;
    }
    lastIdx = match.index + match[0].length;
  }
  plain += tagged.substring(lastIdx);

  // Collect all annotations
  const annotations = [];

  // GT annotations from parsed positions
  for (const gs of gtSpans) {
    annotations.push({start: gs.start, end: gs.end, type: 'gt', label: 'GT'});
  }

  // Model spans — find each in plain text (deduplicated per model)
  const checkedModels = [...detChecked];
  for (const m of checkedModels) {
    const spans = src.models[m] || [];
    const ci = DET_MODELS.indexOf(m);
    const color = DET_COLORS[ci] || '#888';
    // Deduplicate spans (case-insensitive) and track used positions
    const seen = new Set();
    const usedPositions = new Set(); // "start,end" strings
    for (const span of spans) {
      if (!span) continue;
      const trimmed = span.trim();
      if (!trimmed) continue;
      const key = trimmed.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      const lower = plain.toLowerCase();
      let searchFrom = 0;
      while (true) {
        const idx = lower.indexOf(key, searchFrom);
        if (idx < 0) break;
        const posKey = idx + ',' + (idx + trimmed.length);
        if (!usedPositions.has(posKey)) {
          usedPositions.add(posKey);
          annotations.push({start: idx, end: idx + trimmed.length, type: 'model', label: m, color: color});
        }
        searchFrom = idx + 1;
      }
    }
  }

  // Sort by start position
  annotations.sort((a, b) => a.start - b.start || a.end - b.end);

  // Build HTML: collect all boundary points
  const points = new Set([0, plain.length]);
  for (const a of annotations) { points.add(a.start); points.add(a.end); }
  const sorted = [...points].sort((a, b) => a - b);

  let html = '';
  for (let i = 0; i < sorted.length - 1; i++) {
    const s = sorted[i], e = sorted[i + 1];
    const chunk = escHtml(plain.substring(s, e));

    const gtHere = annotations.some(a => a.type === 'gt' && a.start <= s && a.end >= e);
    const modelsHere = annotations.filter(a => a.type === 'model' && a.start <= s && a.end >= e);

    let out = chunk;
    if (gtHere) {
      out = '<span class="det-gt">' + out + '</span>';
    }
    for (const m of modelsHere) {
      const isStart = m.start === s;
      const isEnd = m.end === e;
      const pre = isStart ? '<span class="det-bracket" style="color:' + m.color + '">[</span>' : '';
      const post = isEnd ? '<span class="det-bracket" style="color:' + m.color + '">]</span>' : '';
      out = pre + out + post;
    }
    html += out;
  }

  return html;
}

// ─── Pipeline step slider ─────────────────────────────────────────────────────
const pipeImages = ['data/images/Pipeline_step1.png','data/images/Pipeline_step2.png','data/images/Pipeline_step3.png'];
function setPipeStep(i) {
  document.getElementById('pipe-img').src = pipeImages[i];
  document.querySelectorAll('#pipeline .pipe-tab').forEach((b, j) => b.classList.toggle('active', j === i));
}

// ─── Download ────────────────────────────────────────────────────────────────
function downloadTable() {
  const el = document.getElementById('lb-container');
  html2canvas(el, { scale: 2, useCORS: true, backgroundColor: '#ffffff' }).then(canvas => {
    const link = document.createElement('a');
    link.download = `DetailVerifyBench_leaderboard_${currentTab}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  });
}

// ─── Example display names (edit here to customize dropdown labels) ──────────
// Map filename → display name; falls back to 'Example N' if not set
const EX_NAMES = {
  // e.g. 'movie_001.jpg': 'Aerial City Scene',
};

// Injection demo display names; falls back to 'Category - Example N'
const INJ_NAMES = {
  // e.g. 'poster_002.jpg': 'Movie Poster - Battle Scene',
};

// ─── Examples ────────────────────────────────────────────────────────────────
let exData      = [];
let exDomain    = 'Movie';
let exMode      = 'natural';
let exCurrent   = null;
let exShowGT    = false;
let exMethod    = 'adversarial';
let exHighlight = true;   // default: highlights ON

const domainColors = {Chart:'#0ea5e9',Movie:'#8b5cf6',Nature:'#22c55e',Poster:'#f97316',UI:'#ec4899'};

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function getExSrc() {
  if (exMode === 'natural') return exCurrent.natural;
  return exCurrent.injected[exMethod];
}

function stripHalTags(tag) {
  return tag.replace(/<[/]?HALLUCINATION>/gi,'');
}

function renderHalHTML(tag, hallucination_labels) {
  const labelMap = {};
  for (const h of (hallucination_labels || [])) labelMap[h.text.trim()] = h.labels || (h.label ? [h.label] : []);
  return tag
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/&lt;\/?HALLUCINATION&gt;/gi, m =>
      m.toLowerCase().includes('/') ? '__HALEND__' : '__HALSTART__'
    )
    .replace(/__HALSTART__([\s\S]*?)__HALEND__/g, (_, inner) => {
      const raw = inner.replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>');
      const labels = labelMap[raw.trim()] || [];
      return '<mark class="hal" data-labels="' + labels.join('; ') + '">' + inner + '</mark>';
    });
}

// Word-level LCS diff: render GT with words not in original highlighted green
function wordDiffGT(stripped, gt) {
  const norm = w => w.toLowerCase().replace(/[^a-z0-9']/g,'');
  const a = stripped.trim().split(/\s+/);
  const b = gt.trim().split(/\s+/);
  const m = a.length, n = b.length;
  const dp = Array.from({length:m+1}, () => new Array(n+1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = norm(a[i-1]) === norm(b[j-1])
        ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
  const ops = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && norm(a[i-1]) === norm(b[j-1])) { ops.unshift('same'); i--; j--; }
    else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) { ops.unshift('add'); j--; }
    else { ops.unshift('del'); i--; }
  }
  let bi = 0, ai = 0;
  const parts = [];
  for (const op of ops) {
    if (op === 'same')     { parts.push(escHtml(b[bi++])); ai++; }
    else if (op === 'add') { parts.push('<mark class="gt-cor">' + escHtml(b[bi++]) + '</mark>'); }
    else                   { ai++; }
  }
  return parts.join(' ');
}

function buildCaption(src) {
  if (!exHighlight) {
    const plain = exShowGT
      ? (src.GT_description || stripHalTags(src.description_tag))
      : stripHalTags(src.description_tag);
    return escHtml(plain);
  }
  if (exShowGT) {
    if (exMode === 'natural' && exCurrent.category === 'Nature') {
      return escHtml(src.GT_description || stripHalTags(src.description_tag));
    }
    return wordDiffGT(
      stripHalTags(src.description_tag),
      src.GT_description || stripHalTags(src.description_tag)
    );
  }
  return renderHalHTML(src.description_tag, src.hallucination_labels);
}

function methodSliderHTML() {
  if (exMode !== 'synthetic') return '';
  const methods = [['adversarial','Adversarial (Ours)'],['tldr','TLDR'],['halloc','Halloc']];
  return '<div class="ex-method-slider">'
    + methods.map(([k,l]) =>
        '<button class="' + (exMethod===k?'active':'') + '" data-method="' + k
        + '" onclick="setExMethod(\'' + k + '\')">' + l + '</button>'
      ).join('')
    + '</div>';
}

function renderViewer() {
  const viewer = document.getElementById('ex-viewer');
  if (!exCurrent) { viewer.innerHTML = ''; return; }
  const src = getExSrc();
  const dc = domainColors[exCurrent.category] || '#6b7280';
  const domainLabel = exCurrent.category === 'UI' ? 'GUI' : exCurrent.category;
  viewer.innerHTML =
    '<div class="ex-viewer-card">'
    + '<div class="ex-viewer-img">'
    +   '<img src="' + exCurrent.image + '" alt="' + escHtml(exCurrent.filename) + '" loading="lazy">'
    + '</div>'
    + '<div class="ex-viewer-body">'
    +   '<div class="ex-viewer-header">'
    +     '<span class="ex-domain-badge" style="background:' + dc + '20;color:' + dc + '">' + domainLabel + '</span>'
    +     '<span class="ex-filename">' + document.getElementById('ex-image-sel').selectedOptions[0].text + '</span>'
    +     methodSliderHTML()
    +   '</div>'
    +   '<button id="btn-show-gt" class="ex-gt-btn' + (exShowGT?' active':'') + '" onclick="toggleGT()">Show GT</button>'
    +   '<div class="ex-caption" id="ex-caption">' + buildCaption(src) + '</div>'
    + '</div>'
    + '</div>';
}

// Partial update — avoids reloading the image
function refreshCaption() {
  const src = getExSrc();
  document.getElementById('ex-caption').innerHTML = buildCaption(src);
  // Show GT button: only active state changes, text stays "Show GT"
  document.getElementById('btn-show-gt')?.classList.toggle('active', exShowGT);
  // Method slider active state
  document.querySelectorAll('.ex-method-slider button').forEach(b =>
    b.classList.toggle('active', b.dataset.method === exMethod)
  );
}

// Unhighlight button: active = currently UN-highlighted (exHighlight=false)
function syncUnhighlightBtn() {
  const btn = document.getElementById('btn-highlight');
  if (btn) btn.classList.toggle('active', !exHighlight);
}

function toggleHighlight() {
  exHighlight = !exHighlight;
  syncUnhighlightBtn();
  refreshCaption();
}

function toggleGT() {
  exShowGT = !exShowGT;
  refreshCaption();
}

function setExMethod(method) {
  exMethod = method;
  refreshCaption();
}

function setExDomain(domain) {
  exDomain = domain;
  populateImageSelect(domain);
}

function setExImage(filename) {
  const pool = exData.filter(e => e.category === exDomain);
  exCurrent = pool.find(e => e.filename === filename) || null;
  exShowGT = false;
  renderViewer();
}

function setExMode(mode) {
  exMode = mode;
  document.getElementById('btn-natural').classList.toggle('active', mode === 'natural');
  document.getElementById('btn-synthetic').classList.toggle('active', mode === 'synthetic');
  exShowGT = false; exMethod = 'adversarial';
  renderViewer();
}

function populateImageSelect(domain) {
  const filtered = exData.filter(e => e.category === domain);
  const sel = document.getElementById('ex-image-sel');
  sel.innerHTML = filtered.map((e, i) => {
    const label = EX_NAMES[e.filename] || ('Example ' + (i + 1));
    return '<option value="' + e.filename + '">' + label + '</option>';
  }).join('');
  exCurrent = filtered[0] || null;
  exShowGT = false; exMethod = 'adversarial';
  renderViewer();
}

exData = LEGACY_DATA.examples;
populateImageSelect('Movie');
initInjDemo(LEGACY_DATA.injection);
initDetVis(LEGACY_DATA.detection);

// ─── Floating tooltip for .hal marks ──────────────────────────────────────────
const halTip = document.createElement('div');
halTip.className = 'hal-tooltip';
halTip.style.display = 'none';
document.body.appendChild(halTip);

document.addEventListener('mouseover', e => {
  const mark = e.target.closest('mark.hal');
  if (!mark) return;
  const labels = mark.getAttribute('data-labels');
  if (!labels) return;
  halTip.textContent = labels;
  halTip.style.opacity = '0';
  halTip.style.display = 'block';
  const r = mark.getBoundingClientRect();
  const tipW = halTip.offsetWidth, tipH = halTip.offsetHeight;
  let left = r.left + r.width / 2 - tipW / 2;
  left = Math.max(4, Math.min(left, window.innerWidth - tipW - 4));
  let top = r.top - tipH - 5;
  if (top < 4) top = r.bottom + 5;
  halTip.style.left = left + 'px';
  halTip.style.top = top + 'px';
  halTip.style.opacity = '1';
});
document.addEventListener('mouseout', e => {
  if (e.target.closest('mark.hal')) {
    halTip.style.opacity = '0';
    halTip.style.display = 'none';
  }
});

// Leaderboard is initialized with current paper data in enhancements.js.
