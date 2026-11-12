/* SafeMine Monitor - front-end logic (vanilla JS + Chart.js) */
const $ = (s) => document.querySelector(s);
const api = async (url, opt) => {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
const pct = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const lvl = (l) => (l ? `<span class="lvl ${l}">${l}</span>` : "–");
const CAT_COLORS = { NO_HELMET: "#f85149", NO_VEST: "#f0883e", RESTRICTED_ZONE: "#bc8cff", MACHINE_PROXIMITY: "#ff6b9a" };
const charts = {};
function chart(id, cfg) {
  if (!window.Chart) return;
  if (charts[id]) charts[id].destroy();
  Chart.defaults.color = "#8aa0b5";
  Chart.defaults.borderColor = "#26384b";
  charts[id] = new Chart(document.getElementById(id), cfg);
}

/* ---------------------------------------------------------------- navigation */
const loaders = {};
document.querySelectorAll("nav a").forEach((a) =>
  a.addEventListener("click", () => show(a.dataset.page))
);
function show(page) {
  document.querySelectorAll("nav a").forEach((x) => x.classList.toggle("active", x.dataset.page === page));
  document.querySelectorAll(".page").forEach((x) => x.classList.toggle("active", x.id === "page-" + page));
  if (loaders[page]) loaders[page]().catch((e) => console.error(e));
}

/* ---------------------------------------------------------------- model info */
async function loadModel() {
  try {
    const m = await api("/api/model");
    if (!m.loaded) { $("#modelBox").innerHTML = `<b style="color:#f85149">Model not loaded</b><br>${esc(m.error)}`; return; }
    const caps = Object.entries(m.capabilities).map(([k, v]) => `${v ? "✔" : "✖"} ${k}`).join("<br>");
    $("#modelBox").innerHTML = `<b>${esc(m.name)}</b> ${m.is_custom ? "(custom PPE)" : "(COCO fallback)"}<br>${caps}`;
  } catch (e) { $("#modelBox").textContent = "Model: API offline"; }
}

/* ---------------------------------------------------------------- dashboard */
loaders.dashboard = async () => {
  const d = await api("/api/dashboard");
  $("#kJobs").textContent = d.totals.jobs_done;
  $("#kEvents").textContent = d.totals.events;
  $("#kHigh").textContent = d.totals.high_risk;
  $("#kWorkers").textContent = d.totals.workers_observed;
  $("#kUBR").textContent = pct(d.totals.unsafe_behaviour_rate);
  $("#kHelmet").textContent = pct(d.totals.helmet_compliance_rate);
  const cats = Object.keys(d.by_category);
  chart("chCat", { type: "bar", data: { labels: cats.map((c) => d.categories[c] || c),
    datasets: [{ data: cats.map((c) => d.by_category[c]), backgroundColor: cats.map((c) => CAT_COLORS[c] || "#4c9aff") }] },
    options: { indexAxis: "y", plugins: { legend: { display: false } } } });
  chart("chLevel", { type: "doughnut", data: { labels: Object.keys(d.by_level),
    datasets: [{ data: Object.values(d.by_level), backgroundColor: ["#3fb950", "#f0883e", "#f85149"] }] } });
  $("#recent").innerHTML = d.recent_events.length ? d.recent_events.map((e) => `
    <div class="ev">${e.snapshot ? `<img src="${e.snapshot}">` : `<img>`}
      <div><b>${esc(e.label)}</b><br>${esc(e.camera)} · t=${e.start_s}s ${lvl(e.risk_level)}</div></div>`).join("")
    : '<p class="hint">No detections yet. Go to “Upload & Detect”.</p>';
  $("#histTbl").innerHTML = "<tr><th>Time</th><th>Camera</th><th>File</th><th>Events</th><th>Risk</th></tr>" +
    d.history.map((h) => `<tr class="click" data-id="${h.id}"><td>${esc(h.created_at)}</td><td>${esc(h.camera)}</td>
      <td>${esc(h.filename)}</td><td>${h.events ?? h.status}</td><td>${lvl(h.overall_risk)}</td></tr>`).join("");
  document.querySelectorAll("#histTbl tr.click").forEach((r) => r.addEventListener("click", () => openJob(r.dataset.id)));
};

/* ---------------------------------------------------------------- upload + zone */
let zone = [];
let zoneImg = null;
const cv = $("#zoneCanvas"), ctx = cv.getContext("2d");
function drawZone() {
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (zoneImg) ctx.drawImage(zoneImg, 0, 0, cv.width, cv.height);
  else { ctx.fillStyle = "#8aa0b5"; ctx.font = "14px sans-serif"; ctx.fillText("Select a video/image to preview the first frame", 150, 180); }
  if (zone.length) {
    ctx.strokeStyle = "#f85149"; ctx.fillStyle = "rgba(248,81,73,.25)"; ctx.lineWidth = 2; ctx.beginPath();
    zone.forEach(([x, y], i) => (i ? ctx.lineTo(x * cv.width, y * cv.height) : ctx.moveTo(x * cv.width, y * cv.height)));
    if (zone.length > 2) ctx.closePath();
    ctx.fill(); ctx.stroke();
    zone.forEach(([x, y]) => { ctx.fillStyle = "#f85149"; ctx.fillRect(x * cv.width - 3, y * cv.height - 3, 6, 6); });
  }
  $("#zoneInfo").textContent = zone.length ? `${zone.length} points` : "no zone";
}
cv.addEventListener("click", (e) => {
  const r = cv.getBoundingClientRect();
  zone.push([+((e.clientX - r.left) / r.width).toFixed(4), +((e.clientY - r.top) / r.height).toFixed(4)]);
  drawZone();
});
$("#clearZone").onclick = () => { zone = []; drawZone(); };
function setPreview(src, isVideo) {
  if (!isVideo) { const im = new Image(); im.onload = () => { zoneImg = im; fitCanvas(im.width, im.height); }; im.src = src; return; }
  const v = $("#hiddenVideo"); v.src = src;
  v.onloadeddata = () => { v.currentTime = 0.1; };
  v.onseeked = () => { const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0); const im = new Image(); im.onload = () => { zoneImg = im; fitCanvas(v.videoWidth, v.videoHeight); }; im.src = c.toDataURL(); };
}
function fitCanvas(w, h) { cv.width = 640; cv.height = Math.round(640 * h / w); drawZone(); }
$("#fileInput").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return; $("#sampleSelect").value = "";
  setPreview(URL.createObjectURL(f), f.type.startsWith("video"));
};
$("#sampleSelect").onchange = (e) => { if (e.target.value) { $("#fileInput").value = ""; setPreview(`/api/samples/${encodeURIComponent(e.target.value)}/frame`, false); } };
loaders.upload = async () => {
  const s = await api("/api/samples");
  $("#sampleSelect").innerHTML = '<option value="">(none)</option>' + s.samples.map((n) => `<option>${esc(n)}</option>`).join("");
  drawZone();
};
$("#submitBtn").onclick = async () => {
  const f = $("#fileInput").files[0], sample = $("#sampleSelect").value;
  if (!f && !sample) { $("#uploadMsg").textContent = "Choose a file or a sample first."; return; }
  const fd = new FormData();
  fd.append("camera", $("#cameraInput").value); fd.append("zone", JSON.stringify(zone.length > 2 ? zone : []));
  $("#uploadMsg").textContent = "Uploading…"; $("#submitBtn").disabled = true;
  try {
    let r;
    if (f) { fd.append("file", f); r = await api("/api/upload", { method: "POST", body: fd }); }
    else { fd.append("name", sample); r = await api("/api/samples/process", { method: "POST", body: fd }); }
    $("#uploadMsg").textContent = `Job ${r.job_id} queued.`;
    openJob(r.job_id);
  } catch (e) { $("#uploadMsg").textContent = "Error: " + e.message; }
  $("#submitBtn").disabled = false;
};

/* ---------------------------------------------------------------- monitoring */
let pollTimer = null, currentJob = null;
function openJob(id) { currentJob = id; show("monitor"); startMonitor(id); }
async function startMonitor(id) {
  clearInterval(pollTimer);
  $("#eventTbl").innerHTML = ""; $("#workerPanel").innerHTML = ""; $("#replayBtn").disabled = true;
  const j = await api(`/api/jobs/${id}`);
  $("#monSub").textContent = `${j.filename} · ${j.camera} · job ${id}`;
  if (j.status === "done") return finishMonitor(j);
  let streaming = false;
  const startStream = () => { streaming = true; $("#streamImg").src = `/api/jobs/${id}/stream?ts=${Date.now()}`; };
  if (j.status === "processing") startStream();
  $("#monBadge").textContent = "ANALYSING (REPLAY MODE)"; $("#monBadge").className = "badge run";
  pollTimer = setInterval(async () => {
    const jj = await api(`/api/jobs/${id}`);
    if (jj.status === "processing" && !streaming) startStream();
    $("#progBar").style.width = (jj.progress || 0) + "%";
    $("#monStatus").textContent = `${jj.status} · ${(jj.progress || 0).toFixed(0)}% · t=${jj.live.t ?? 0}s · events ${jj.live.events ?? 0}`;
    renderWorkers(jj.live.workers);
    if (jj.status === "done" || jj.status === "error") { clearInterval(pollTimer); finishMonitor(jj); }
  }, 1000);
}
function renderWorkers(ws) {
  const yn = (v) => `<span class="${v === "Yes" ? "yes" : v === "No" ? "no" : "unk"}">${v === "Yes" ? "✔ Yes" : v === "No" ? "✖ No" : "? Unknown"}</span>`;
  const b = (v, good) => `<span class="${v ? (good ? "yes" : "no") : "yes"}">${v ? "⚠ Yes" : "No"}</span>`;
  $("#workerPanel").innerHTML = (ws || []).map((w) => `<div class="w"><b>Worker ID: W${w.track_id}</b>
    <div class="row">Helmet ${yn(w.helmet)}</div><div class="row">High-vis vest ${yn(w.vest)}</div>
    <div class="row">In restricted zone ${b(w.in_zone)}</div><div class="row">Near machinery ${b(w.near_machine)}</div></div>`).join("")
    || '<p class="hint">No worker in the current frame.</p>';
}
async function finishMonitor(j) {
  $("#progBar").style.width = "100%";
  $("#monBadge").textContent = j.status === "error" ? "ERROR" : "COMPLETED"; $("#monBadge").className = "badge " + (j.status === "error" ? "run" : "done");
  if (j.status === "error") { $("#monStatus").textContent = j.error; return; }
  const s = j.summary || {};
  $("#monStatus").textContent = `${s.events_total} events · ${s.unique_workers} workers · overall risk ${s.risk?.overall_level} · ` +
    (s.processing_fps ? `${s.processing_fps} fps processing` : "");
  if (j.media_type === "video") {
    $("#replayBtn").disabled = false;
    $("#streamImg").src = `/api/jobs/${j.id}/replay?ts=${Date.now()}`;
  } else { $("#streamImg").src = j.output_path + "?ts=" + Date.now(); renderWorkers(s.workers); }
  $("#downloadBtn").href = j.output_path || "#";
  const ev = await api(`/api/jobs/${j.id}/events`);
  $("#eventTbl").innerHTML = "<tr><th>Snapshot</th><th>Worker</th><th>Category</th><th>Time (s)</th><th>Conf.</th><th>Exposure</th><th>L×S</th><th>Risk</th><th>Recommendation</th></tr>" +
    ev.map((e) => `<tr><td>${e.snapshot ? `<a href="${e.snapshot}" target="_blank"><img class="snap" src="${e.snapshot}"></a>` : ""}</td>
      <td>W${e.track_id}</td><td>${esc(e.label)}</td><td>${e.start_s} – ${e.end_s}<br><span class="hint">${e.duration_s}s</span></td>
      <td>${e.max_conf}</td><td>${pct(e.exposure_ratio)}</td><td>${e.likelihood}×${e.severity}=${e.risk_score}</td>
      <td>${lvl(e.risk_level)}</td><td class="hint">${esc(e.recommendation)}</td></tr>`).join("");
}
$("#replayBtn").onclick = () => { if (currentJob) $("#streamImg").src = `/api/jobs/${currentJob}/replay?ts=${Date.now()}`; };

/* ---------------------------------------------------------------- records */
loaders.records = async () => {
  const js = await api("/api/jobs");
  $("#jobsTbl").innerHTML = "<tr><th>Created</th><th>File</th><th>Type</th><th>Camera</th><th>Status</th><th>Events</th><th>UBR</th><th>Risk</th><th></th></tr>" +
    js.map((j) => `<tr class="click" data-id="${j.id}"><td>${esc(j.created_at)}</td><td>${esc(j.filename)}</td><td>${j.media_type}</td>
      <td>${esc(j.camera)}</td><td>${j.status}</td><td>${j.summary?.events_total ?? "–"}</td><td>${pct(j.summary?.unsafe_behaviour_rate)}</td>
      <td>${lvl(j.summary?.risk?.overall_level)}</td><td><button class="btn del" data-id="${j.id}">🗑</button></td></tr>`).join("");
  document.querySelectorAll("#jobsTbl tr.click").forEach((r) => r.addEventListener("click", (e) => {
    if (e.target.classList.contains("del")) return; openJob(r.dataset.id); }));
  document.querySelectorAll("#jobsTbl .del").forEach((b) => b.addEventListener("click", async () => {
    if (confirmDelete(b)) { await api(`/api/jobs/${b.dataset.id}`, { method: "DELETE" }); loaders.records(); } }));
};
function confirmDelete(b) { if (b.dataset.armed) return true; b.dataset.armed = 1; b.textContent = "confirm?"; return false; }

/* ---------------------------------------------------------------- risk */
loaders.risk = async () => {
  const r = await api("/api/risk");
  const band = (s) => (s >= 15 ? "High" : s >= 5 ? "Medium" : "Low");
  let h = '<div class="matrix"><div></div>' + [1, 2, 3, 4, 5].map((s) => `<div><b>S${s}</b></div>`).join("");
  for (let L = 5; L >= 1; L--) {
    h += `<div><b>L${L}</b></div>`;
    for (let S = 1; S <= 5; S++) { const n = r.cells[`${L}-${S}`] || 0; h += `<div class="cell ${band(L * S)}">${L * S}${n ? `<br><b>${n} ev</b>` : ""}</div>`; }
  }
  $("#matrix").innerHTML = h + "</div>";
  $("#sevTbl").innerHTML = '<table class="tbl"><tr><th>Category</th><th>S</th><th>Basis</th></tr>' +
    Object.entries(r.matrix.severity).map(([c, v]) => `<tr><td>${c}</td><td>${v.S}</td><td class="hint">${esc(v.basis)}</td></tr>`).join("") +
    "</table><h3 style='margin-top:12px'>Likelihood scale</h3><table class='tbl'>" +
    Object.entries(r.matrix.likelihood_scale).map(([k, v]) => `<tr><td>L${k}</td><td class="hint">${esc(v)}</td></tr>`).join("") + "</table>";
  const bc = r.summary.by_category || {};
  $("#recTbl").innerHTML = "<tr><th>Category</th><th>Events</th><th>Max score</th><th>Level</th><th>Recommended control</th></tr>" +
    (Object.entries(bc).map(([c, v]) => `<tr><td>${c}</td><td>${v.count}</td><td>${v.max_score}</td><td>${lvl(v.level)}</td><td class="hint">${esc(v.recommendation)}</td></tr>`).join("")
     || '<tr><td colspan="5" class="hint">No events yet.</td></tr>');
};

/* ---------------------------------------------------------------- analytics */
loaders.analytics = async () => {
  const a = await api("/api/analytics");
  const syn = a.dataset?.synthetic;
  $("#dataBanner").className = "banner " + (syn ? "syn" : "real");
  $("#dataBanner").textContent = a.dataset?.source
    ? (syn ? `⚠ Using SYNTHETIC demo data (${a.dataset.source}) — replace with real research data before reporting results.`
           : `Research data: ${a.dataset.source}`) : "No research dataset found.";
  const inc = a.incidents || {};
  if (inc.pareto_category) {
    const p = inc.pareto_category;
    chart("chPareto", { data: { labels: p.map((x) => x.label), datasets: [
      { type: "bar", label: "Count", data: p.map((x) => x.count), backgroundColor: "#4c9aff", yAxisID: "y" },
      { type: "line", label: "Cumulative %", data: p.map((x) => x.cum_pct), borderColor: "#f5a524", yAxisID: "y1" }] },
      options: { scales: { y1: { position: "right", max: 100, min: 0 } } } });
    chart("chTrend", { type: "line", data: { labels: inc.monthly_trend.labels, datasets: [{ label: "Records", data: inc.monthly_trend.values, borderColor: "#3fb950" }] } });
    chart("chSev", { type: "bar", data: { labels: Object.keys(inc.severity_distribution).map((k) => "S" + k),
      datasets: [{ label: "Records", data: Object.values(inc.severity_distribution), backgroundColor: "#f0883e" }] } });
    $("#briTbl").innerHTML = "<tr><th>Category</th><th>n</th><th>Share</th><th>Mean S</th><th>Days lost</th><th>Fatal</th><th>BRI</th></tr>" +
      inc.category_indicators.map((r) => `<tr><td>${esc(r.category)}</td><td>${r.n}</td><td>${pct(r.share)}</td><td>${r.mean_severity}</td><td>${r.days_lost}</td><td>${r.fatal}</td><td><b>${r.bri}</b></td></tr>`).join("");
    const xt = inc.experience_crosstab, cs = xt.chi_square;
    $("#chiBox").innerHTML = `<p class="hint">χ² = ${cs.chi2}, dof = ${cs.dof}, p = ${cs.p_value === null ? "–" : cs.p_value.toExponential(3)}, Cramér's V = ${cs.cramers_v}</p>`;
    $("#xtTbl").innerHTML = `<tr><th></th>${xt.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` +
      xt.index.map((r, i) => `<tr><td>${esc(r)}</td>${xt.values[i].map((v) => `<td>${v}</td>`).join("")}</tr>`).join("");
  }
  const d = a.detection || {};
  $("#cvInd").innerHTML = [["Media analysed", d.jobs_analysed], ["Video minutes", d.video_minutes], ["Person-frames", d.person_frames],
    ["Unsafe Behaviour Rate", pct(d.unsafe_behaviour_rate)], ["Helmet compliance", pct(d.helmet_compliance_rate)],
    ["Vest compliance", pct(d.vest_compliance_rate)], ["Events", d.events_total], ["Events / minute", d.events_per_minute ?? "–"]]
    .map(([k, v]) => `<tr><td>${k}</td><td><b>${v ?? "–"}</b></td></tr>`).join("");
  $("#dq").textContent = JSON.stringify(a.data_quality, null, 2);
};
$("#csvBtn").onclick = async () => {
  const f = $("#csvInput").files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  await api("/api/analytics/upload", { method: "POST", body: fd }); loaders.analytics();
};

/* ---------------------------------------------------------------- EGT */
let egtDefaults = null;
loaders.egt = async () => {
  if (!egtDefaults) {
    egtDefaults = await api("/api/egt/defaults");
    $("#scenarioSel").innerHTML = Object.entries(egtDefaults.scenarios).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("");
    $("#initGrid").innerHTML = ["x0", "y0", "z0"].map((k) => `<label>${k}<input type="number" step="0.05" min="0.01" max="0.99" id="i_${k}" value="0.5"></label>`).join("");
    $("#paramGrid").innerHTML = Object.entries(egtDefaults.params).map(([k, v]) =>
      `<label title="${esc(egtDefaults.param_doc[k])}">${k}<input type="number" step="any" id="p_${k}" value="${v}"></label>`).join("");
    $("#sensParam").innerHTML = Object.keys(egtDefaults.params).map((k) => `<option ${k === "Fw" ? "selected" : ""}>${k}</option>`).join("");
    const c = egtDefaults.calibration_source;
    $("#calibInfo").textContent = `CV data: UBR=${c.unsafe_behaviour_rate ?? "n/a"}, recall=${c.recall ?? "n/a (run validation)"}`;
  }
  runEgt();
};
$("#scenarioSel").onchange = () => {
  const sc = egtDefaults.scenarios[$("#scenarioSel").value].params;
  Object.entries(egtDefaults.params).forEach(([k, v]) => ($(`#p_${k}`).value = sc[k] ?? v));
};
$("#calibBtn").onclick = () => {
  const c = egtDefaults.calibration; $("#i_x0").value = c.x0; $("#p_phi").value = c.phi; $("#p_r").value = c.r;
};
$("#runEgt").onclick = () => runEgt();
async function runEgt() {
  const params = {}; Object.keys(egtDefaults.params).forEach((k) => (params[k] = +$(`#p_${k}`).value));
  const body = { params, initial: { x0: +$("#i_x0").value, y0: +$("#i_y0").value, z0: +$("#i_z0").value }, t_max: 50,
    sensitivity: { param: $("#sensParam").value, range: [+$("#sensLo").value, +$("#sensHi").value, 13] } };
  const r = await api("/api/egt/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  chart("chEgt", { type: "line", data: { labels: r.t, datasets: [
    { label: "x: workers safe", data: r.x, borderColor: "#3fb950", pointRadius: 0 },
    { label: "y: supervisors strict", data: r.y, borderColor: "#4c9aff", pointRadius: 0 },
    { label: "z: management invests", data: r.z, borderColor: "#f5a524", pointRadius: 0 }] },
    options: { scales: { y: { min: 0, max: 1 }, x: { ticks: { maxTicksLimit: 11 } } } } });
  const f = r.final, ta = r.time_average;
  $("#egtFinal").textContent = `Final state (t=50): x=${f.x.toFixed(3)}, y=${f.y.toFixed(3)}, z=${f.z.toFixed(3)} · time-average (2nd half): x=${ta.x.toFixed(3)}, y=${ta.y.toFixed(3)}, z=${ta.z.toFixed(3)} · detection prob. with investment: strict ${r.detection_probabilities.strict_invest}, lax ${r.detection_probabilities.lax_invest}`;
  const s = r.sensitivity.results;
  chart("chSens", { type: "line", data: { labels: s.map((v) => v.value.toFixed(1)), datasets: [
    { label: "x", data: s.map((v) => v.x), borderColor: "#3fb950" }, { label: "y", data: s.map((v) => v.y), borderColor: "#4c9aff" },
    { label: "z", data: s.map((v) => v.z), borderColor: "#f5a524" }] },
    options: { scales: { y: { min: 0, max: 1 }, x: { title: { display: true, text: r.sensitivity.param } } } } });
  $("#eqTbl").innerHTML = "<tr><th>Point</th><th>Meaning</th><th>Eigenvalues</th><th>Stability</th></tr>" +
    r.equilibria.map((e) => `<tr><td>${e.point}</td><td class="hint">${e.meaning}</td><td>${e.eigenvalues.join(", ")}</td><td>${e.stability}</td></tr>`).join("");
  $("#payTbl").innerHTML = "<tr><th>Worker</th><th>Supervisor</th><th>Management</th><th>U_W</th><th>U_S</th><th>U_M</th></tr>" +
    r.payoff_table.map((p) => `<tr><td>${p.worker}</td><td>${p.supervisor}</td><td>${p.management}</td><td>${p.U_W}</td><td>${p.U_S}</td><td>${p.U_M}</td></tr>`).join("");
}

/* ---------------------------------------------------------------- validation */
loaders.validation = async () => {
  const js = await api("/api/jobs");
  $("#valJob").innerHTML = js.filter((j) => j.status === "done").map((j) => `<option value="${j.id}">${esc(j.filename)} (${j.id})</option>`).join("");
  renderValidation(await api("/api/validation"));
};
function renderValidation(v) {
  const ev = v.event_level;
  $("#valTbl").innerHTML = ev ? "<tr><th>Category</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr>" +
    Object.entries({ OVERALL: ev.overall, ...ev.per_category }).map(([c, m]) =>
      `<tr><td>${c}</td><td>${m.TP}</td><td>${m.FP}</td><td>${m.FN}</td><td>${m.precision}</td><td>${m.recall}</td><td><b>${m.f1}</b></td></tr>`).join("")
    : '<tr><td class="hint">No evaluation yet.</td></tr>';
  const mv = v.model_level;
  $("#mvTbl").innerHTML = mv ? Object.entries(mv.overall || mv).filter(([k, x]) => typeof x !== "object").map(([k, x]) => `<tr><td>${k}</td><td>${x}</td></tr>`).join("") +
    (mv.per_class ? "<tr><th>Class</th><th>AP50</th></tr>" + Object.entries(mv.per_class).map(([k, x]) => `<tr><td>${k}</td><td>${x}</td></tr>`).join("") : "")
    : '<tr><td class="hint">Not available.</td></tr>';
}
$("#valBtn").onclick = async () => {
  const f = $("#gtInput").files[0], id = $("#valJob").value; if (!f || !id) return;
  const fd = new FormData(); fd.append("file", f);
  await api(`/api/validation/${id}`, { method: "POST", body: fd });
  renderValidation(await api("/api/validation"));
};

/* ---------------------------------------------------------------- init */
loadModel();
loaders.dashboard().catch(console.error);
setInterval(() => { if ($("#page-dashboard").classList.contains("active")) loaders.dashboard().catch(() => {}); }, 8000);
