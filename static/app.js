"use strict";
// Curamate front end. The browser keeps the interview state and sends it with every request.
//
// Screens: 1 intake -> 2 confirm matches -> 3 interview -> 4 report.
// API calls to the Flask server (app.py), all JSON over HTTP on 127.0.0.1:
//   GET  /api/health  status pills          GET  /api/meta    travel regions
//   POST /api/match   free text -> symptoms POST /api/step    next question / detail form / done
//   POST /api/report  final diagnosis + cited report

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : Math.round(p * 100) + "%");
const sevTag = (d) => `<span class="sev sev-${d.severity}" title="Severity ${d.severity}/5 (1 = most severe)">${esc(d.severity_label)}</span>`;
const STEPS = ["intake", "confirm", "interview", "report"];

// Interview state: everything known about the patient so far. Sent with every /api/step and /api/report call.
const S = { age: null, sex: null, travel: "unknown", answers: {}, details: {}, asked: [], detailsDone: [], log: [], labels: {} };
let matches = [];

// The one helper for every server call: GET when there is no body, POST with a JSON body otherwise.
async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// The request body for /api/step and /api/report: the full interview state.
const payload = (extra = {}) => ({
  age: S.age, sex: S.sex, travel: S.travel, answers: S.answers, details: S.details,
  asked: S.asked, details_done: S.detailsDone, ...extra,
});

function show(step) {
  STEPS.forEach((s) => $("#step-" + s).classList.toggle("hidden", s !== step));
  const idx = STEPS.indexOf(step);
  document.querySelectorAll("#stepper li").forEach((li, i) => {
    li.classList.toggle("active", i === idx);
    li.classList.toggle("done", i < idx);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setBusy(busy) {
  document.querySelectorAll("#step-interview button").forEach((b) => (b.disabled = busy));
  document.body.style.cursor = busy ? "progress" : "";
}

// ---------- start-up ----------
// API: GET /api/health -> green/red status pills (model, knowledge base, Ollama, embeddings, LLM).
async function loadStatus() {
  const names = { model: "Model", kb: "Knowledge base", ollama: "Ollama", embed: "Embeddings", llm: "LLM" };
  try {
    const h = await api("/api/health");
    $("#status").innerHTML = Object.entries(names).filter(([k]) => h[k])
      .map(([k, n]) => `<span class="pill ${h[k].ok ? "ok" : "bad"}" title="${esc(h[k].detail)}">${n}${k === "llm" && h[k].ok ? ": " + esc(h[k].detail) : ""}</span>`)
      .join("");
  } catch {
    $("#status").innerHTML = `<span class="pill bad">Server not reachable</span>`;
  }
}

// API: GET /api/meta -> travel regions for the intake dropdown.
async function loadMeta() {
  const m = await api("/api/meta");
  $("#travel").insertAdjacentHTML("beforeend", m.regions.map((r) => `<option value="${r.value}">Travelled: ${esc(r.label)}</option>`).join(""));
}

// ---------- 1. intake -> symptom matching (RAG use 1) ----------
// API: POST /api/match {text} -> matched symptom codes per phrase; then show the confirm screen.
$("#intake-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  S.age = parseInt($("#age").value, 10);
  S.sex = $("#sex").value;
  S.travel = $("#travel").value;
  const btn = $("#match-btn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Matching…`;
  try {
    const r = await api("/api/match", { text: $("#symptoms").value });
    matches = r.results.map((m) => ({ ...m, choice: m.selected || "", answer: m.negated ? "no" : "yes", removed: false }));
    $("#match-info").textContent = `Each phrase was embedded with Nomic and compared by cosine similarity (HNSW index in Chroma) ` +
      `against the DDXPlus symptom list. Top ${r.top_k} matches shown; the best is pre-selected when similarity ≥ ${r.threshold}. (${r.ms} ms)`;
    renderMatches();
    show("confirm");
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Match symptoms";
  }
});

function renderMatches() {
  const rows = matches.map((m, i) => (m.removed ? "" : `
    <div class="match ${m.selected ? "" : "unmatched"}">
      <div>
        <div class="phrase">You wrote: “${esc(m.phrase)}”${m.selected ? "" : " - no confident match: pick one or ignore"}</div>
        <select class="m-choice" data-i="${i}">
          <option value="">- ignore this phrase -</option>
          ${m.options.map((o) => `<option value="${o.code}" ${o.code === m.choice ? "selected" : ""}>${esc(o.label)}${o.history ? " (history)" : ""} · similarity ${o.score.toFixed(2)}</option>`).join("")}
        </select>
      </div>
      <div class="seg">
        <button class="m-ans ${m.answer === "yes" ? "on-yes" : ""}" data-i="${i}" data-a="yes">Has it</button>
        <button class="m-ans ${m.answer === "no" ? "on-no" : ""}" data-i="${i}" data-a="no">Denies</button>
      </div>
      <button class="x m-del" data-i="${i}" title="Remove">✕</button>
    </div>`)).join("");
  $("#matches").innerHTML = rows || `<p class="muted">No phrases found - go back and describe the symptoms.</p>`;
}

$("#matches").addEventListener("change", (e) => {
  if (e.target.classList.contains("m-choice")) matches[e.target.dataset.i].choice = e.target.value;
});
$("#matches").addEventListener("click", (e) => {
  const t = e.target.closest("button");
  if (!t) return;
  const m = matches[t.dataset.i];
  if (t.classList.contains("m-ans")) m.answer = t.dataset.a;
  if (t.classList.contains("m-del")) m.removed = true;
  renderMatches();
});
$("#back-intake").onclick = () => show("intake");

// "Start interview": copy the doctor-confirmed matches into the state, then run the first interview round.
$("#start-btn").onclick = () => {
  Object.assign(S, { answers: {}, details: {}, asked: [], detailsDone: [], log: [], labels: {} });
  for (const m of matches) {
    if (m.removed || !m.choice) continue;
    S.answers[m.choice] = m.answer;
    S.labels[m.choice] = (m.options.find((o) => o.code === m.choice) || {}).label || m.choice;
    S.log.push({ kind: "intake", code: m.choice });
  }
  if (!Object.values(S.answers).includes("yes")) {
    alert("Confirm at least one symptom the patient HAS before starting.");
    return;
  }
  show("interview");
  next();
};

// ---------- 2. interview loop (LightGBM + information gain) ----------
// API: POST /api/step with the full state. The server answers with status:
//   "question" -> show the next question, "details" -> show the detail form, "done" -> load the report.
async function next(extra) {
  setBusy(true);
  try {
    const r = await api("/api/step", payload(extra));
    renderDiff(r.differential, r);
    renderHistory();
    $("#q-counter").textContent = `${r.n_asked} of max ${r.max_questions} questions`;
    $("#stop-reason").textContent = r.reason ? "Stopping rule: " + r.reason : "";
    if (r.status === "question") renderQuestion(r);
    else if (r.status === "details") renderDetails(r.details);
    else loadReport(r.reason);
  } catch (err) {
    $("#question-area").innerHTML = `<div class="banner bad">${esc(err.message)}</div>`;
  } finally {
    setBusy(false);
  }
}

function renderQuestion(r) {
  const q = r.question;
  $("#question-area").innerHTML = `
    <div class="muted small">Most informative next question - chosen from ${r.n_candidates} candidates in ${r.what_if_ms} ms</div>
    <div class="question">${esc(q.text)}</div>
    <div class="answers">
      <button class="yes" data-a="yes">Yes</button>
      <button class="no" data-a="no">No</button>
      <button data-a="unknown">Don't know</button>
    </div>
    <details class="why"><summary>Why this question?</summary>
      <p>For every candidate question Curamate predicted the differential twice - once as if the answer were yes, once as if no -
         and picked the question expected to reduce uncertainty the most (information gain).</p>
      <div class="why-grid">
        <div><b>Expected gain</b><br>${q.gain.toFixed(2)} bits</div>
        <div><b>Chance of "yes"</b><br>${pct(q.p_yes)}</div>
        <div><b>If yes →</b><br>${esc(q.if_yes.display)} ${pct(q.if_yes.prob)}</div>
        <div><b>If no →</b><br>${esc(q.if_no.display)} ${pct(q.if_no.prob)}</div>
      </div>
      ${r.alternatives.length ? `<div class="small muted">Next best: ${r.alternatives.map((a) => `${esc(a.label)} (${a.gain.toFixed(2)})`).join(" · ")}</div>` : ""}
    </details>`;
  // Yes / No / Don't know -> save the answer in the state and ask the server for the next round.
  $("#question-area").querySelectorAll(".answers button").forEach((b) => (b.onclick = () => {
    S.answers[q.code] = b.dataset.a;
    S.asked.push(q.code);
    S.labels[q.code] = q.label;
    S.log.push({ kind: "answer", code: q.code });
    next();
  }));
}

function renderDetails(form) {
  const fields = form.fields.map((f) => {
    const head = `<div class="field-head"><span>${esc(f.label)}</span><label><input type="checkbox" class="unk" checked> Unknown</label></div>`;
    if (f.type === "scale") {
      return `<div class="field" data-code="${f.code}" data-type="scale">${head}
        <div class="scale-row"><span>0 ${esc(f.hints[0])}</span><input type="range" min="0" max="10" value="0">
        <span>10 ${esc(f.hints[1])}</span><span class="scale-val">-</span></div></div>`;
    }
    if (f.type === "choice") {
      return `<div class="field" data-code="${f.code}" data-type="choice">${head}<div class="opts">
        ${f.options.map((o) => `<label><input type="radio" name="r-${f.code}" value="${o.value}"> ${esc(o.label)}</label>`).join("")}</div></div>`;
    }
    const filter = f.options.length > 20 ? `<input type="search" class="filter" placeholder="Filter ${f.options.length} options…">` : "";
    return `<div class="field" data-code="${f.code}" data-type="multi">${head}${filter}<div class="opts">
      <label><input type="checkbox" value="__none__"> <i>None of these</i></label>
      ${f.options.map((o) => `<label data-l="${esc(o.label.toLowerCase())}"><input type="checkbox" value="${o.value}"> ${esc(o.label)}</label>`).join("")}
      </div></div>`;
  }).join("");
  const area = $("#question-area");
  area.innerHTML = `
    <div class="muted small">The patient has: ${esc(S.labels[form.root] || form.root)}. Fill in what is known - anything left as Unknown is handled by the model.</div>
    <h3 style="margin-top:10px">${esc(form.title)}</h3>${fields}
    <div class="actions"><button class="ghost" id="skip-details">Skip (all unknown)</button><button class="primary" id="save-details">Save details</button></div>`;
  area.querySelectorAll(".field").forEach((el) => {
    const unk = el.querySelector(".unk");
    const touched = (e) => {
      if (e.target === unk || e.target.classList.contains("filter")) return;
      unk.checked = false;
      if (el.dataset.type === "scale") el.querySelector(".scale-val").textContent = el.querySelector("input[type=range]").value;
    };
    el.addEventListener("input", touched);
    el.addEventListener("change", touched);
    const filter = el.querySelector(".filter");
    if (filter) filter.addEventListener("input", () => {
      const q = filter.value.toLowerCase();
      el.querySelectorAll(".opts label[data-l]").forEach((l) => (l.style.display = l.dataset.l.includes(q) ? "" : "none"));
    });
  });
  $("#save-details").onclick = () => submitDetails(form, false);
  $("#skip-details").onclick = () => submitDetails(form, true);
}

// Saves the pain / rash / swelling detail form into S.details (blank fields = "unknown"), then continues.
function submitDetails(form, skip) {
  const values = {};
  document.querySelectorAll("#question-area .field").forEach((el) => {
    const code = el.dataset.code;
    if (skip || el.querySelector(".unk").checked) return void (values[code] = "unknown");
    if (el.dataset.type === "scale") values[code] = parseInt(el.querySelector("input[type=range]").value, 10);
    else if (el.dataset.type === "choice") {
      const c = el.querySelector("input[type=radio]:checked");
      values[code] = c ? c.value : "unknown";
    } else values[code] = [...el.querySelectorAll(".opts input:checked")].map((i) => i.value).filter((v) => v !== "__none__");
  });
  S.details[form.root] = values;
  S.detailsDone.push(form.root);
  S.log.push({ kind: "details", code: form.root });
  next();
}

$("#undo-btn").onclick = () => {
  const last = S.log[S.log.length - 1];
  if (!last || last.kind === "intake") return;
  S.log.pop();
  if (last.kind === "answer") {
    delete S.answers[last.code];
    S.asked = S.asked.filter((c) => c !== last.code);
  } else {
    delete S.details[last.code];
    S.detailsDone = S.detailsDone.filter((c) => c !== last.code);
  }
  next();
};
$("#finish-btn").onclick = () => next({ finish: true });

function dxRows(list, n) {
  return list.slice(0, n).map((d) => `
    <div class="dx"><div class="dx-top"><span class="dx-name">${esc(d.display)}</span><span>${sevTag(d)} <b>${pct(d.prob)}</b></span></div>
    <div class="bar"><div style="width:${Math.max(1, d.prob * 100)}%"></div></div></div>`).join("");
}

function renderDiff(list, r) {
  $("#uncertainty").textContent = r ? `uncertainty ${r.uncertainty} bits ≈ ${r.suspects} suspects` : "";
  $("#diff").innerHTML = dxRows(list, 6);
}

function renderHistory() {
  const items = Object.entries(S.answers).map(([c, a]) => `<li><span>${esc(S.labels[c] || c)}${S.asked.includes(c) ? "" : ' <span class="muted small">(intake)</span>'}</span>
    <span class="tag ${a}">${a === "unknown" ? "don't know" : a}</span></li>`);
  if (S.travel !== "unknown") {
    const t = S.travel === "no" ? "no" : $("#travel").selectedOptions[0].textContent.replace("Travelled: ", "");
    items.unshift(`<li><span>Travel abroad</span><span class="tag ${S.travel === "no" ? "no" : "yes"}">${esc(t)}</span></li>`);
  }
  $("#history").innerHTML = items.join("") || `<li class="muted">None yet.</li>`;
}

// ---------- 3. report (RAG use 2 + local LLM) ----------
// API: POST /api/report with the final state -> diagnosis, differential, sources and the LLM's cited report.
async function loadReport(reason) {
  show("report");
  $("#report-area").innerHTML = `<p><span class="spinner"></span>Retrieving clinical knowledge from Chroma and writing the summary with the local LLM… usually 10-30 seconds.</p>`;
  try {
    renderReport(await api("/api/report", payload()), reason);
  } catch (err) {
    $("#report-area").innerHTML = `<div class="banner bad">${esc(err.message)}</div>`;
  }
}

const FLAG_TEXT = {
  no_citation: "No citation for this sentence",
  unknown_source: "Cites a source that was not retrieved",
  weak_support: "The cited source does not clearly contain this claim",
  contradicts_patient: "States a finding the patient DENIED",
};

// Turns the LLM's markdown into HTML: highlights sentences the citation checker flagged, makes [S#] clickable.
function renderMarkdown(text, checks) {
  const flagged = checks.filter((c) => c.status !== "ok");
  let html = "", inList = false;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const bullet = /^[-*•]\s+/.test(line);
    if (inList && !bullet) { html += "</ul>"; inList = false; }
    if (/^#{1,6}\s/.test(line)) { html += `<h4>${esc(line.replace(/^#+\s*/, ""))}</h4>`; continue; }
    let body = esc(line.replace(/^[-*•]\s+/, ""));
    for (const f of flagged) {
      const s = esc(f.sentence);
      if (s && body.includes(s)) body = body.replace(s, `<span class="flag-${f.status}" title="${FLAG_TEXT[f.status]}">${s}</span>`);
    }
    body = body.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\[(S\d+|P)\]/g, '<span class="cite" data-src="$1">$1</span>');
    if (bullet) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${body}</li>`; }
    else html += `<p>${body}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

// Report page: diagnosis + severity, differential, key findings (LightGBM), cited summary (LLM), sources (RAG).
function renderReport(r, reason) {
  const d = r.diagnosis;
  const flagged = r.citations.filter((c) => c.status !== "ok").length;
  const urgent = d.severity === 1 ? "Emergency - act immediately according to local protocol."
    : d.severity === 2 ? "Urgent clinical assessment recommended." : "";
  $("#report-area").innerHTML = `
    <div class="dx-hero">
      <div><div class="muted small">Most likely diagnosis · ICD-10 ${esc(d.icd10)}</div><h2>${esc(d.display)}</h2>
        ${sevTag(d)} <span class="muted small">severity ${d.severity}/5</span></div>
      <div style="text-align:right"><div class="big">${pct(d.prob)}</div>
        <div class="muted small">LightGBM probability · ${r.questions_asked} follow-up questions</div></div>
    </div>
    ${urgent ? `<div class="banner bad"><b>${esc(d.severity_label)}:</b> ${urgent}</div>` : ""}
    <div class="banner ${d.prob >= 0.9 ? "" : "warn"}">${esc(r.confidence_note)}${reason ? `<br><span class="small muted">${esc(reason)}</span>` : ""}</div>
    <div class="grid2">
      <div><h3>Differential</h3>${dxRows(r.differential, 5)}</div>
      <div>
        <h3>What drove the prediction</h3>
        <div class="chips">${r.findings.map((f) => `<span class="chip ${f.effect > 0 ? "up" : "down"}" title="LightGBM contribution ${f.effect}">
          ${f.effect > 0 ? "▲" : "▼"} ${esc(f.label)}${f.answer ? ": " + esc(f.answer) : ""}</span>`).join("") || '<span class="muted small">-</span>'}</div>
        <h3>Typical of ${esc(d.display)}, not yet asked</h3>
        <div class="chips">${r.not_yet_asked.map((n) => `<span class="chip" title="${Math.round(n.freq * 100)}% of training patients with this disease">${esc(n.label)}</span>`).join("") || '<span class="muted small">None</span>'}</div>
      </div>
    </div>
    <h3>Clinical summary</h3>
    <div class="report-text">${renderMarkdown(r.report.text, r.citations)}</div>
    <div class="meta">${r.report.mode === "llm" ? `Written by ${esc(r.report.model)} running locally in ${r.report.seconds}s.` : esc(r.report.note)}
      Citation check: ${r.citations.length - flagged}/${r.citations.length} sentences supported${flagged ? ` - ${flagged} flagged (highlighted)` : ""}.
      Diagnosis and percentages come from the classifier, not the LLM.</div>
    <h3 style="margin-top:18px">Sources retrieved (RAG)</h3>
    <p class="hint">Query: “${esc(r.retrieval.query)}” - embedded with Nomic, filtered to the diagnosis in Chroma, ranked by cosine similarity (top ${r.retrieval.top_k}) in ${r.retrieval.ms} ms.</p>
    <div class="sources">${r.sources.map((s) => `
      <details id="src-${s.id}"><summary>[${s.id}] ${esc(s.display)} - ${esc(s.section_title)} <span class="muted small">similarity ${s.score.toFixed(2)}</span></summary>
        <p>${esc(s.body)}</p>
        <p class="muted small">${s.origin === "dataset" ? "Generated from the DDXPlus dataset." : "Synthesized for Curamate from: " + esc(s.references) + " - not clinically reviewed."}</p>
      </details>`).join("")}</div>`;
  $("#report-area").querySelectorAll(".cite").forEach((c) => (c.onclick = () => {
    const el = document.getElementById("src-" + c.dataset.src);
    if (!el) return;
    el.open = true;
    el.classList.add("flash");
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => el.classList.remove("flash"), 1500);
  }));
}

$("#back-interview").onclick = () => { show("interview"); next(); };
$("#new-btn").onclick = () => {
  Object.assign(S, { age: null, sex: null, travel: "unknown", answers: {}, details: {}, asked: [], detailsDone: [], log: [], labels: {} });
  $("#intake-form").reset();
  $("#diff").innerHTML = `<p class="muted">Appears once the interview starts.</p>`;
  $("#uncertainty").textContent = "";
  renderHistory();
  show("intake");
};

loadStatus();
loadMeta().catch(() => {});
