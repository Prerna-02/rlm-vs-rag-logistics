const LABEL = { rag_bm25: "RAG · BM25", rag_hybrid: "RAG · hybrid", direct_llm: "Direct LLM",
  rlm_depth0: "RLM · depth 0", rlm_depth1: "RLM · depth 1", rlm_depth2: "RLM · depth 2" };
// Categorical slots 1-6 of the validated palette (light / dark steps), fixed per method; marker shape is the second cue.
const COLORS = { light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
                 dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"] };
const MARKERS = ["circle", "rect", "triangle", "rectRot", "rectRounded", "triangle"];
const ROTATION = [0, 0, 0, 0, 0, 180];  // depth 2: downward triangle
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (v, d = 2) => v == null ? "—" : v.toFixed(d);
const money = v => v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(3)}`;
const secs = v => v < 60 ? `${Math.round(v)} s` : `${(v / 60).toFixed(1)} min`;
let DATA, sizeChart, costChart, tier = "T2";

/* ---------- theme ---------- */
const isDark = () => document.documentElement.dataset.theme
  ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
const palette = () => COLORS[isDark() ? "dark" : "light"];
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch {}
$("#theme").addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch {}
  drawCharts();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawCharts);

/* ---------- reveal on scroll + count-up ---------- */
const io = new IntersectionObserver(entries => entries.forEach(e => {
  if (!e.isIntersecting) return;
  e.target.classList.add("in");
  const b = e.target.querySelector("[data-count]");
  if (b) countUp(b);
  io.unobserve(e.target);
}), { threshold: 0.15 });
document.querySelectorAll(".reveal").forEach(el => io.observe(el));
function countUp(el) {
  const end = +el.dataset.count, start = performance.now(), dur = 1200;
  const tick = now => {
    const p = Math.min(1, (now - start) / dur);
    el.textContent = Math.round(end * (1 - Math.pow(1 - p, 3))).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ---------- leaderboard ---------- */
function renderBoard(key = "hard") {
  const rows = [...DATA.leaderboard].sort((a, b) =>
    key === "method" ? DATA.methods.indexOf(a.method) - DATA.methods.indexOf(b.method)
      : key.startsWith("cost") || key.startsWith("seconds") ? a[key] - b[key] : (b[key] ?? -1) - (a[key] ?? -1));
  const heat = v => v == null ? "<span>—</span>"
    : `<span style="background:rgba(${css("--heat")},${(0.08 + v * 0.5).toFixed(2)})">${fmt(v)}</span>`;
  $("#board tbody").innerHTML = rows.map(r => `<tr>
    <td><i class="swatch" style="background:${palette()[DATA.methods.indexOf(r.method)]}"></i>${LABEL[r.method]}</td>
    <td class="heat">${heat(r.T1)}</td><td class="heat">${heat(r.T2)}</td><td class="heat">${heat(r.T3)}</td>
    <td>${money(r.cost_median)}</td><td>${secs(r.seconds_median)}</td></tr>`).join("");
  document.querySelectorAll("#board th").forEach(th => th.classList.toggle("sorted", th.dataset.k === key));
}
document.querySelectorAll("#board th").forEach(th => th.addEventListener("click", () => renderBoard(th.dataset.k)));

/* ---------- charts ---------- */
function chartDefaults() {
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = css("--text-2");
  Chart.defaults.borderColor = css("--grid");
}
const tooltip = { backgroundColor: css("--surface-solid"), titleColor: css("--text"), bodyColor: css("--text-2"),
  borderColor: css("--grid"), borderWidth: 1, padding: 10, boxPadding: 4, usePointStyle: true };

function drawSize() {
  const datasets = DATA.methods.map((m, i) => ({
    label: LABEL[m], method: m,
    data: DATA.sizes.map(s => DATA.by_size[m][tier][s]?.score ?? null),
    n: DATA.sizes.map(s => DATA.by_size[m][tier][s]?.n ?? 0),
    borderColor: palette()[i], backgroundColor: palette()[i], pointStyle: MARKERS[i], rotation: ROTATION[i],
    pointRadius: 6, pointHoverRadius: 9, borderWidth: 2, tension: 0, spanGaps: false,
    pointBorderColor: css("--surface-solid"), pointBorderWidth: 2,
  })).filter(d => d.data.some(v => v != null));
  sizeChart?.destroy();
  sizeChart = new Chart($("#sizeChart"), {
    type: "line",
    data: { labels: DATA.sizes.map(s => `${s.toUpperCase()} · ${DATA.contexts[s].records.toLocaleString()} orders`), datasets },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { y: { min: 0, max: 1, ticks: { stepSize: 0.25, callback: v => v.toFixed(2) }, title: { display: true, text: "mean score" } },
                x: { grid: { display: false } } },
      plugins: {
        legend: { position: "top", labels: { usePointStyle: true, boxWidth: 8, padding: 16 } },
        tooltip: { ...tooltip, callbacks: { label: c => ` ${c.dataset.label}: ${fmt(c.raw)}  (${c.dataset.n[c.dataIndex]} questions)` } },
      },
    },
  });
  const head = `<tr><th>Method</th>${DATA.sizes.map(s => `<th>${s}</th>`).join("")}</tr>`;
  $("#sizeTable").innerHTML = `<table>${head}${DATA.methods.map(m => `<tr><td>${LABEL[m]}</td>${
    DATA.sizes.map(s => `<td>${fmt(DATA.by_size[m][tier][s]?.score)}</td>`).join("")}</tr>`).join("")}</table>`;
}

const labelPlugin = {  // direct labels next to each scatter point
  id: "pointLabels",
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    ctx.save();
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillStyle = css("--text-2");
    const pts = chart.data.datasets.map((d, i) => ({ label: d.label, p: chart.getDatasetMeta(i).data[0] })).filter(o => o.p);
    pts.forEach(({ label, p }) => {
      // when two points nearly touch, the lower one's label goes below it, the upper one's above
      const below = pts.some(o => o.p !== p && Math.abs(o.p.x - p.x) < 90 && Math.abs(o.p.y - p.y) < 18 && o.p.y < p.y);
      ctx.fillText(label, p.x + 12, below ? p.y + 20 : p.y - 10);
    });
    ctx.restore();
  },
};

function drawCost() {
  costChart?.destroy();
  costChart = new Chart($("#costChart"), {
    type: "scatter",
    data: { datasets: DATA.leaderboard.map((r, i) => ({
      label: LABEL[r.method], data: [{ x: r.cost_median, y: r.hard }], p90: r.cost_p90,
      backgroundColor: palette()[i], borderColor: css("--surface-solid"), borderWidth: 2,
      pointStyle: MARKERS[i], rotation: ROTATION[i], pointRadius: 9, pointHoverRadius: 12,
    })) },
    options: {
      maintainAspectRatio: false,
      layout: { padding: { right: 70, top: 10 } },
      scales: {
        x: { type: "logarithmic", min: 0.0003, max: 0.2, title: { display: true, text: "median cost per question (USD, log scale)" },
             ticks: { autoSkip: false, maxRotation: 0, callback: v => `$${v}` },
             afterBuildTicks: axis => { axis.ticks = [0.001, 0.01, 0.1].map(value => ({ value })); } },
        y: { min: 0, max: 1, title: { display: true, text: "mean score on T2 + T3" } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { ...tooltip, callbacks: { label: c =>
          ` ${c.dataset.label}: score ${fmt(c.raw.y)}, median ${money(c.raw.x)}, 90th pct ${money(c.dataset.p90)}` } },
      },
    },
    plugins: [labelPlugin],
  });
}

function drawCharts() {
  if (!DATA) return;
  chartDefaults();
  Object.assign(tooltip, { backgroundColor: css("--surface-solid"), titleColor: css("--text"), bodyColor: css("--text-2"), borderColor: css("--grid") });
  drawSize();
  drawCost();
  renderBoard($("#board th.sorted")?.dataset.k ?? "hard");
}
document.querySelectorAll("#tierSeg button").forEach(b => b.addEventListener("click", () => {
  tier = b.dataset.tier;
  document.querySelectorAll("#tierSeg button").forEach(x => x.setAttribute("aria-selected", x === b));
  drawSize();
}));

/* ---------- trajectory viewer ---------- */
const rlmRuns = () => DATA.runs.filter(r => r.method.startsWith("rlm"));
function fillPickers() {
  const methods = [...new Set(rlmRuns().map(r => r.method))].sort();
  $("#pMethod").innerHTML = methods.map(m => `<option value="${m}">${LABEL[m]}</option>`).join("");
  $("#pMethod").value = "rlm_depth1";
  fillSizes();
}
function fillSizes() {
  const m = $("#pMethod").value;
  const sizes = DATA.sizes.filter(s => rlmRuns().some(r => r.method === m && r.size === s));
  $("#pSize").innerHTML = sizes.map(s => `<option>${s}</option>`).join("");
  fillQuestions();
}
function fillQuestions() {
  const m = $("#pMethod").value, s = $("#pSize").value;
  const runs = rlmRuns().filter(r => r.method === m && r.size === s).sort((a, b) => a.id.localeCompare(b.id));
  $("#pQuestion").innerHTML = runs.map(r =>
    `<option value="${r.id}">${r.tier} · ${esc(DATA.questions[r.id].question.slice(0, 70))}…</option>`).join("");
  showTrace();
}
async function showTrace() {
  const m = $("#pMethod").value, id = $("#pQuestion").value;
  const run = DATA.runs.find(r => r.method === m && r.id === id);
  const q = DATA.questions[id];
  if (!run) return;
  const ok = run.score >= 0.999 ? "ok" : run.score > 0 ? "" : "no";
  $("#traceHead").innerHTML = `<p class="q">${esc(q.question)}</p>
    <p><span class="pill">correct: ${esc(q.truth)}</span><span class="pill ${ok}">answered: ${esc(run.answer.slice(0, 120))}</span></p>
    <p><span class="pill">score ${fmt(run.score)}</span><span class="pill">${run.steps} steps</span>
       <span class="pill">${run.sub_calls} helper calls</span><span class="pill">${money(run.cost)}</span><span class="pill">${secs(run.seconds)}</span></p>`;
  $("#trace").innerHTML = "<li data-n='…'><div class='step'><p class='say'>Loading…</p></div></li>";
  try {
    const steps = await fetch(`data/traces/${m}/${id}.json`).then(r => r.json());
    $("#trace").innerHTML = steps.map((s, i) => `<li data-n="${i + 1}" style="animation-delay:${i * 60}ms">
      <div class="step"><h4>Step ${i + 1} · Claude wrote</h4>${renderResponse(s.response)}
      ${s.outputs.map(o => `<div class="out"><h4>Python printed</h4><pre>${esc(o || "(nothing)")}</pre></div>`).join("")}
      ${s.sub_calls.length ? `<div class="subcalls">${s.sub_calls.map((c, j) =>
        `<span title="${esc(c.reply)}">helper call ${j + 1} · ${c.prompt_chars.toLocaleString()} chars</span>`).join("")}</div>` : ""}
      </div></li>`).join("");
  } catch {
    $("#trace").innerHTML = "<li data-n='!'><div class='step'><p class='say'>Could not load this trajectory.</p></div></li>";
  }
}
function renderResponse(text) {  // prose stays prose, ```repl blocks become code
  return text.split(/```repl\s*\n([\s\S]*?)```/).map((part, i) =>
    i % 2 ? `<pre>${esc(part.trim())}</pre>` : part.trim() ? `<p class="say">${esc(part.trim())}</p>` : "").join("");
}
$("#pMethod").addEventListener("change", fillSizes);
$("#pSize").addEventListener("change", fillQuestions);
$("#pQuestion").addEventListener("change", showTrace);

/* ---------- boot ---------- */
fetch("data/results.json").then(r => r.json()).then(d => {
  DATA = d;
  drawCharts();
  fillPickers();
});
