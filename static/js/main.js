(function () {
  "use strict";

  // ── SVG constants ──────────────────────────────────────────────────────────
  const BOT_SVG = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none">
    <rect x="2" y="7" width="20" height="13" rx="3" fill="#fff" opacity=".9"/>
    <circle cx="8.5" cy="12" r="2" fill="#2563eb"/>
    <circle cx="15.5" cy="12" r="2" fill="#2563eb"/>
    <path d="M9 17 Q12 19.5 15 17" stroke="#2563eb" stroke-width="1.3" fill="none" stroke-linecap="round"/>
    <rect x="10" y="2" width="4" height="5" rx="1" fill="#fff" opacity=".7"/>
    <line x1="12" y1="2" x2="12" y2="7" stroke="#60a5fa" stroke-width="1.5"/>
  </svg>`;

  const SEND_SVG = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2" fill="currentColor" stroke="none"/>
  </svg>`;

  const SPIN_SVG = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
    <path d="M12 2a10 10 0 1 0 10 10" />
  </svg>`;

  // ── State ──────────────────────────────────────────────────────────────────
  let currentConversationId = null;

  // ── DOM refs ───────────────────────────────────────────────────────────────
  const messagesScroll = document.getElementById("messagesScroll");
  const messagesInner  = document.getElementById("messagesInner");
  const corpusList     = document.getElementById("corpusList");
  const convList       = document.getElementById("convList");
  const corpusSelect   = document.getElementById("corpusSelect");
  const refSelect      = document.getElementById("refSelect");
  const msgInput       = document.getElementById("msgInput");
  const sendBtn        = document.getElementById("sendBtn");
  const uploadBtn      = document.getElementById("uploadBtn");
  const fileInput      = document.getElementById("fileInput");
  const newConvBtn     = document.getElementById("newConvBtn");

  // ── Utilities ──────────────────────────────────────────────────────────────
  function esc(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmt(v, decimals = 0) {
    return Number(v ?? 0).toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function scrollBottom() {
    messagesScroll.scrollTop = messagesScroll.scrollHeight;
  }

  function timestamp() {
    return new Date().toLocaleTimeString("en-US", {
      hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true,
    });
  }

  // ── Message bubbles ────────────────────────────────────────────────────────
  function addBubble(text, role = "assistant") {
    const ts  = timestamp();
    const row = document.createElement("div");
    row.className = "msg-row " + role;
    if (role === "assistant") {
      row.innerHTML = `
        <div class="msg-avatar bot-av">${BOT_SVG}</div>
        <div class="msg-body">
          <div class="msg-bubble">${esc(text)}</div>
          <span class="msg-time">${ts}</span>
        </div>`;
    } else {
      row.innerHTML = `
        <div class="msg-body">
          <div class="msg-bubble">${esc(text)}</div>
          <span class="msg-time">${ts}</span>
        </div>
        <div class="msg-avatar user-av">U</div>`;
    }
    messagesInner.appendChild(row);
    scrollBottom();
  }

  // ── Analysis result cards ──────────────────────────────────────────────────
  const RESULT_TITLES = {
    frequency:          "Frequency Analysis",
    kwic:               "Keywords in Context (KWIC)",
    ngram_collocation:  "N-gram & Collocation Analysis",
    keyword_comparison: "Keyword Comparison",
  };

  function addResultCard(payload) {
    if (!payload.safe) {
      addErrorCard(payload.error || "This analysis could not be displayed safely.");
      return;
    }
    const result = payload.result || {};
    const type   = result.analysis_type;

    if (payload.conversational || type === "conversational") {
      addBubble(payload.reply || result.reply || "", "assistant");
      return;
    }

    const card  = document.createElement("article");
    card.className = "result-card";
    const title = RESULT_TITLES[type] || "Analysis Result";
    card.innerHTML = `
      <div class="result-card-head">
        <div>
          <h3 class="result-card-title">${esc(title)}</h3>
          <p class="result-card-sub">Derived from corpus evidence · validated</p>
        </div>
        <span class="result-badge">Ready</span>
      </div>
      <div class="result-card-body">${renderResult(result)}</div>`;
    messagesInner.appendChild(card);
    scrollBottom();
  }

  function addErrorCard(message) {
    const card = document.createElement("article");
    card.className = "result-card result-card--error";
    card.innerHTML = `
      <div class="result-card-head">
        <h3 class="result-card-title">Analysis Blocked</h3>
        <span class="result-badge result-badge--error">Blocked</span>
      </div>
      <div class="result-card-body">
        <p class="error-msg">${esc(message)}</p>
      </div>`;
    messagesInner.appendChild(card);
    scrollBottom();
  }

  // ── Result renderers ───────────────────────────────────────────────────────
  function renderResult(result) {
    const t = result?.analysis_type;
    if (t === "frequency")          return buildFrequencyTable(result);
    if (t === "kwic")               return buildKwicTable(result);
    if (t === "ngram_collocation")  return buildNgramTable(result);
    if (t === "keyword_comparison") return buildKeywordTable(result);
    return `<p class="no-result">No displayable result for type: ${esc(t)}</p>`;
  }

  function metricGrid(...pairs) {
    const items = pairs.map(([label, value]) =>
      `<div class="metric">
        <span class="metric-label">${esc(label)}</span>
        <span class="metric-value">${esc(String(value))}</span>
      </div>`
    ).join("");
    return `<div class="metric-grid">${items}</div>`;
  }

  function buildFrequencyTable(r) {
    const rows  = r.rows || [];
    const maxF  = Math.max(...rows.map(x => x.frequency || 0), 1);
    const stats = metricGrid(
      ["Total Tokens", fmt(r.total_tokens)],
      ["Rows Shown",   rows.length],
    );
    if (!rows.length) return stats + `<p class="no-result">No frequency data returned.</p>`;
    const body = rows.map(row => {
      const w = ((row.frequency || 0) / maxF * 100).toFixed(1);
      return `<tr>
        <td class="num">${esc(row.rank)}</td>
        <td class="word"><strong>${esc(row.word)}</strong></td>
        <td class="num">${fmt(row.frequency)}</td>
        <td class="num">${fmt((row.relative_frequency || 0) * 100, 2)}%</td>
        <td class="bar-cell">
          <div class="bar-track"><div class="bar-fill" style="width:${w}%"></div></div>
        </td>
      </tr>`;
    }).join("");
    return stats + `<div class="table-wrap"><table>
      <thead><tr>
        <th>Rank</th><th>Word</th><th>Frequency</th><th>Relative %</th><th>Scale</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
  }

  function buildKwicTable(r) {
    const rows  = r.matches || [];
    const stats = metricGrid(
      ["Keyword", r.keyword || "—"],
      ["Window",  `±${r.window_size || 0} words`],
      ["Matches", rows.length],
    );
    if (!rows.length) return stats + `<p class="no-result">No concordance lines found for this keyword.</p>`;
    const body = rows.map(row => `<tr>
      <td class="num">${esc(row.position)}</td>
      <td class="kwic-ctx right">${esc(row.left_context)}</td>
      <td class="kwic-kw">${esc(row.keyword)}</td>
      <td class="kwic-ctx">${esc(row.right_context)}</td>
    </tr>`).join("");
    return stats + `<div class="table-wrap"><table>
      <thead><tr>
        <th>Pos.</th>
        <th style="text-align:right">Left Context</th>
        <th style="text-align:center">Keyword</th>
        <th>Right Context</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
  }

  function buildNgramTable(r) {
    const rows   = r.rows || [];
    const maxPMI = Math.max(...rows.map(x => x.pmi_score || 0), 1);
    const stats  = metricGrid(
      ["N-gram Size", r.ngram_size || "—"],
      ["Rows Shown",  rows.length],
    );
    if (!rows.length) return stats + `<p class="no-result">No n-grams met the minimum frequency threshold.</p>`;
    const body = rows.map(row => {
      const pct = Math.min(((row.pmi_score || 0) / maxPMI * 100), 100).toFixed(1);
      return `<tr>
        <td class="word"><strong>${esc(row.ngram_text)}</strong></td>
        <td class="num">${fmt(row.frequency)}</td>
        <td class="num">${fmt(row.pmi_score || 0, 2)}</td>
        <td class="bar-cell">
          <div class="bar-track pmi-track"><div class="bar-fill pmi-fill" style="width:${pct}%"></div></div>
        </td>
      </tr>`;
    }).join("");
    return stats + `<div class="table-wrap"><table>
      <thead><tr>
        <th>N-gram</th><th>Frequency</th><th>PMI Score</th><th>Association Strength</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
  }

  function buildKeywordTable(r) {
    const rows = r.rows || [];
    const maxK = Math.max(...rows.map(x => x.keyness_ratio || 0), 1);
    const stats = metricGrid(["Rows Shown", rows.length]);
    if (!rows.length) return stats + `<p class="no-result">No distinctive keywords found between the two corpora.</p>`;
    const body = rows.map(row => {
      const w = ((row.keyness_ratio || 0) / maxK * 100).toFixed(1);
      return `<tr>
        <td class="word"><strong>${esc(row.word)}</strong></td>
        <td class="num">${fmt(row.target_frequency)}</td>
        <td class="num">${fmt(row.reference_frequency)}</td>
        <td class="num">${fmt(row.keyness_ratio || 0, 2)}</td>
        <td class="bar-cell">
          <div class="bar-track"><div class="bar-fill" style="width:${w}%"></div></div>
        </td>
      </tr>`;
    }).join("");
    return stats + `<div class="table-wrap"><table>
      <thead><tr>
        <th>Word</th><th>Target Freq.</th><th>Ref. Freq.</th><th>Keyness Ratio</th><th>Scale</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table></div>`;
  }

  // ── Corpus library ─────────────────────────────────────────────────────────
  async function refreshCorpora() {
    try {
      const r = await fetch("/api/corpora");
      const d = await r.json();
      renderCorpora(d.corpora || []);
    } catch (e) {
      console.error("Failed to load corpora:", e);
    }
  }

  function renderCorpora(corpora) {
    corpusList.innerHTML  = "";
    corpusSelect.innerHTML = `<option value="">— target corpus (required for analysis) —</option>`;
    refSelect.innerHTML    = `<option value="">No reference corpus</option>`;

    if (!corpora.length) {
      corpusList.innerHTML = `<li class="sb-empty">No corpora uploaded yet.</li>`;
      return;
    }

    for (const c of corpora) {
      const name = typeof c === "string" ? c : c.name;
      const docs = typeof c === "object" ? (c.document_count ?? null) : null;

      const li = document.createElement("li");
      li.className = "sb-project-item";
      li.innerHTML = `<span class="sb-item-name">${esc(name)}</span>`
        + (docs !== null ? `<span class="sb-badge">${docs} doc${docs !== 1 ? "s" : ""}</span>` : "");
      li.addEventListener("click", () => {
        corpusSelect.value = name;
        document.querySelectorAll("#corpusList .sb-project-item").forEach(el => el.classList.remove("active"));
        li.classList.add("active");
      });
      corpusList.appendChild(li);

      corpusSelect.appendChild(new Option(name, name));
      refSelect.appendChild(new Option(name, name));
    }

    // Auto-select first corpus
    const first = corpusList.querySelector(".sb-project-item");
    if (first) first.classList.add("active");
  }

  // ── Conversation history ───────────────────────────────────────────────────
  async function refreshConversations() {
    try {
      const r = await fetch("/api/conversations");
      const d = await r.json();
      renderConversations(d.conversations || []);
    } catch (e) {
      console.error("Failed to load conversations:", e);
    }
  }

  function renderConversations(convs) {
    convList.innerHTML = "";
    if (!convs.length) {
      convList.innerHTML = `<li class="sb-empty">No conversations yet.</li>`;
      return;
    }
    for (const c of convs) {
      const li = document.createElement("li");
      li.className = "sb-conv-item" + (c.conversation_id === currentConversationId ? " active" : "");
      li.dataset.id = c.conversation_id;
      li.textContent = c.title || `Conversation ${c.conversation_id}`;
      li.title = c.updated_at ? new Date(c.updated_at).toLocaleString() : "";
      li.addEventListener("click", () => loadConversation(c.conversation_id));
      convList.appendChild(li);
    }
  }

  async function loadConversation(id) {
    currentConversationId = id;
    messagesInner.innerHTML = "";
    try {
      const r = await fetch(`/api/conversations/${id}/messages`);
      if (!r.ok) { addBubble("Failed to load conversation.", "assistant"); return; }
      const d = await r.json();
      for (const msg of d.messages || []) {
        if (msg.role === "user") {
          addBubble(msg.content, "user");
        } else if (msg.analysis_type === "conversational" || !msg.result_data) {
          addBubble(msg.content || "", "assistant");
        } else {
          addResultCard({ safe: true, result: msg.result_data });
        }
      }
    } catch (e) {
      addBubble("Error loading conversation: " + e.message, "assistant");
    }
    document.querySelectorAll(".sb-conv-item").forEach(el => {
      el.classList.toggle("active", parseInt(el.dataset.id) === id);
    });
  }

  // ── Send / query ───────────────────────────────────────────────────────────
  async function doSend() {
    const question = msgInput.value.trim();
    if (!question) return;

    const corpus_id           = corpusSelect.value || null;
    const reference_corpus_id = refSelect.value    || null;

    addBubble(question, "user");
    msgInput.value = "";
    msgInput.style.height = "auto";
    setSending(true);

    try {
      const r = await fetch("/api/query", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          question,
          corpus_id,
          reference_corpus_id,
          conversation_id: currentConversationId,
        }),
      });
      const d = await r.json();
      if (d.conversation_id) {
        currentConversationId = d.conversation_id;
        await refreshConversations();
      }
      addResultCard(d);
    } catch (e) {
      addErrorCard("Request failed: " + e.message);
    } finally {
      setSending(false);
    }
  }

  function setSending(loading) {
    sendBtn.disabled = loading;
    sendBtn.innerHTML = loading ? SPIN_SVG : SEND_SVG;
    if (loading) sendBtn.classList.add("spinning");
    else sendBtn.classList.remove("spinning");
  }

  // ── Upload ─────────────────────────────────────────────────────────────────
  uploadBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    fileInput.value = "";

    addBubble(`Uploading "${file.name}"…`, "user");
    const fd = new FormData();
    fd.append("file", file);

    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      const d = await r.json();
      if (r.ok) {
        const note = d.rag_warning ? ` Note: ${d.rag_warning}` : "";
        addBubble(`Corpus "${d.corpus_id}" is ready for analysis.${note}`, "assistant");
        await refreshCorpora();
        if (d.corpus_id) corpusSelect.value = d.corpus_id;
      } else {
        addErrorCard(d.error || "Upload failed.");
      }
    } catch (e) {
      addErrorCard("Upload request failed: " + e.message);
    }
  });

  // ── Input events ───────────────────────────────────────────────────────────
  msgInput.addEventListener("input", () => {
    msgInput.style.height = "auto";
    msgInput.style.height = Math.min(msgInput.scrollHeight, 130) + "px";
  });

  msgInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); doSend(); }
  });

  sendBtn.addEventListener("click", doSend);

  newConvBtn.addEventListener("click", () => {
    currentConversationId = null;
    messagesInner.innerHTML = "";
    addWelcome();
    document.querySelectorAll(".sb-conv-item").forEach(el => el.classList.remove("active"));
  });

  // ── Welcome message ────────────────────────────────────────────────────────
  function addWelcome() {
    const row = document.createElement("div");
    row.className = "msg-row assistant";
    row.innerHTML = `
      <div class="msg-avatar bot-av">${BOT_SVG}</div>
      <div class="msg-body">
        <div class="msg-bubble welcome-bubble">
          <strong>Welcome to ACAS.</strong> Upload a corpus from the sidebar, then try:
          <ul class="welcome-list">
            <li>Show the top 20 most frequent words</li>
            <li>Generate KWIC for "language" with a 10-word window</li>
            <li>Find bigram collocations with PMI scores</li>
            <li>Compare my corpus against a reference corpus</li>
            <li>What are the main themes in this text?</li>
          </ul>
        </div>
      </div>`;
    messagesInner.appendChild(row);
    scrollBottom();
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  sendBtn.innerHTML = SEND_SVG;
  addWelcome();
  refreshCorpora();
  refreshConversations();
})();
