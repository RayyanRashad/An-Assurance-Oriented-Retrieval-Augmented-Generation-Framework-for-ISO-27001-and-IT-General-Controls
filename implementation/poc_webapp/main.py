"""
main.py — Audit RAG Application
Single-file app: FastAPI backend + embedded HTML/CSS/JS frontend

Requirements (pip install):
    fastapi uvicorn openai sentence-transformers faiss-cpu rank-bm25 python-dotenv

Run:
    python main.py

Then open: http://localhost:8000
"""

import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
LLM_MODEL       = "gpt-4o-mini"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
BGE_PREFIX      = "Represent this sentence for searching relevant passages: "

DOCS_PATH   = Path("indexed_documents.json")
FAISS_PATH  = Path("faiss_index.bin")
BM25_PATH   = Path("bm25_index.pkl")

# ── Boot: load artifacts once at startup ─────────────────────────────────────

print("Loading artifacts...")

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

with open(DOCS_PATH) as f:
    documents = json.load(f)

faiss_index = faiss.read_index(str(FAISS_PATH))

with open(BM25_PATH, "rb") as f:
    bm25_payload = pickle.load(f)
bm25_index       = bm25_payload["bm25"]
tokenized_corpus = bm25_payload["tokenized_corpus"]

embed_model = SentenceTransformer(EMBED_MODEL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print(f"✅ Ready — {len(documents)} documents indexed")

# ── Retrieval ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hybrid_search(query: str, top_k: int = 6, source_filter: str = None) -> list[dict]:
    n = len(documents)
    q_emb = embed_model.encode(
        [BGE_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    _, faiss_indices = faiss_index.search(q_emb, k=n)
    faiss_ranks = {int(idx): rank for rank, idx in enumerate(faiss_indices[0])}

    bm25_scores = bm25_index.get_scores(tokenize(query))
    bm25_order  = np.argsort(bm25_scores)[::-1]
    bm25_ranks  = {int(idx): rank for rank, idx in enumerate(bm25_order)}

    rrf = {
        idx: (1 / (60 + faiss_ranks.get(idx, n))) + (1 / (60 + bm25_ranks.get(idx, n)))
        for idx in range(n)
    }
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        doc = documents[idx]
        if source_filter and doc["source"] != source_filter:
            continue
        results.append({**doc, "rrf_score": round(score, 6)})
        if len(results) >= top_k:
            break
    return results


def dual_source_search(query: str, k_each: int = 4) -> list[dict]:
    itgc = hybrid_search(query, top_k=k_each, source_filter="ITGC")
    iso  = hybrid_search(query, top_k=k_each, source_filter="ISO27001")
    merged = itgc + iso
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


# ── LLM Generation ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "general": (
        "You are a senior IT audit specialist with expertise in ISO 27001:2022 and ITGC.\n"
        "Answer strictly from the context documents provided. Cite every claim using [N] notation.\n"
        "If context is insufficient, explicitly state what is missing — never fabricate.\n"
        "FORMAT:\nAnswer: (with inline [N] citations)\nEvidence Sources: (numbered list)\nGaps: (missing info)"
    ),
    "compliance_check": (
        "You are a senior IT audit specialist conducting compliance gap analysis.\n"
        "Evaluate whether ITGC controls satisfy ISO 27001 requirements using only provided context.\n"
        "Partial coverage is NOT full compliance. Cite every finding with [N] notation.\n"
        "FORMAT:\nVerdict: COVERED / PARTIALLY COVERED / NOT COVERED\nReasoning: (with [N] citations)\nCovered By: (list)\nGaps: (list)"
    ),
    "cross_mapping": (
        "You are a senior IT audit specialist performing cross-framework control mapping.\n"
        "Map ITGC controls to ISO 27001 controls using only the provided context.\n"
        "Strong = shared objective AND method. Partial = shared objective, different scope.\n"
        "FORMAT:\nMappings: (table: ITGC → ISO → Alignment)\nReasoning: (with [N] citations)\nITGC Gaps:\nISO Gaps:"
    ),
}


def format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        title = (
            doc["metadata"].get("clause_title")
            or doc["metadata"].get("control_area")
            or doc["domain"]
        )
        parts.append(
            f"[{i+1}] SOURCE: {doc['source']} | ID: {doc['control_id']} | "
            f"DOMAIN: {doc['domain']} | TITLE: {title}\n{doc['raw_text'].strip()}"
        )
    return "\n\n---\n\n".join(parts)


def generate(query: str, mode: str, docs: list[dict]) -> str:
    context = format_context(docs)
    user_msg = f"CONTEXT DOCUMENTS:\n{context}\n\n---\n\nQUESTION: {query}"
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user",   "content": user_msg},
        ],
    )
    return response.choices[0].message.content


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Audit RAG")


class QueryRequest(BaseModel):
    query: str
    mode: str = "general"


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    if req.mode not in SYSTEM_PROMPTS:
        raise HTTPException(400, f"mode must be one of {list(SYSTEM_PROMPTS.keys())}")

    use_dual = req.mode in ("compliance_check", "cross_mapping")
    retrieved = dual_source_search(req.query) if use_dual else hybrid_search(req.query, top_k=6)

    answer = generate(req.query, req.mode, retrieved)

    # Build clean retrieval cards for the frontend
    cards = []
    for i, doc in enumerate(retrieved):
        title = (
            doc["metadata"].get("clause_title")
            or doc["metadata"].get("control_area")
            or doc["domain"]
        )
        snippet = doc["raw_text"].strip()[:220].rstrip() + "..."
        cards.append({
            "rank":       i + 1,
            "control_id": doc["control_id"],
            "source":     doc["source"],
            "domain":     doc["domain"],
            "title":      title,
            "snippet":    snippet,
            "score":      doc["rrf_score"],
        })

    return JSONResponse({"answer": answer, "retrievals": cards})


# ── Frontend (single HTML file embedded) ─────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Audit RAG — ISO 27001 & ITGC</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #0d1117;
  --surface:    #161b22;
  --surface2:   #1c2330;
  --border:     #30363d;
  --border2:    #21262d;
  --text:       #e6edf3;
  --muted:      #7d8590;
  --accent:     #f0a500;
  --accent2:    #58a6ff;
  --itgc:       #3fb950;
  --iso:        #58a6ff;
  --danger:     #f85149;
  --radius:     10px;
  --mono:       'DM Mono', monospace;
  --sans:       'DM Sans', sans-serif;
  --serif:      'DM Serif Display', serif;
}

html, body {
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.6;
}

/* ── Layout ── */
.shell {
  display: grid;
  grid-template-rows: 64px 1fr;
  grid-template-columns: 1fr 380px;
  grid-template-areas:
    "header header"
    "main   sidebar";
  height: 100vh;
  overflow: hidden;
}

/* ── Header ── */
header {
  grid-area: header;
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 28px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}
.logo {
  font-family: var(--serif);
  font-size: 22px;
  color: var(--accent);
  letter-spacing: -0.5px;
}
.logo span { color: var(--text); font-style: italic; }
.badge {
  font-family: var(--mono);
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 20px;
  border: 1px solid var(--border);
  color: var(--muted);
}
header .spacer { flex: 1; }
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--itgc);
  box-shadow: 0 0 6px var(--itgc);
}
.status-label { font-size: 12px; color: var(--muted); }

/* ── Main chat area ── */
main {
  grid-area: main;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
}

.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 28px 32px;
  display: flex;
  flex-direction: column;
  gap: 28px;
  scroll-behavior: smooth;
}
.chat-history::-webkit-scrollbar { width: 4px; }
.chat-history::-webkit-scrollbar-track { background: transparent; }
.chat-history::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* Welcome state */
.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  gap: 12px;
  color: var(--muted);
  text-align: center;
  padding: 40px;
  animation: fadeIn .6s ease;
}
.welcome h2 {
  font-family: var(--serif);
  font-size: 32px;
  color: var(--text);
  font-weight: normal;
}
.welcome p { max-width: 440px; font-size: 14px; line-height: 1.7; }
.example-queries {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-top: 8px;
}
.eq-btn {
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 7px 14px;
  border-radius: 20px;
  font-size: 12px;
  cursor: pointer;
  font-family: var(--sans);
  transition: all .2s;
}
.eq-btn:hover { border-color: var(--accent); color: var(--accent); }

/* Message bubbles */
.msg { display: flex; flex-direction: column; gap: 6px; animation: slideUp .3s ease; }
.msg-meta { font-size: 11px; color: var(--muted); font-family: var(--mono); }
.msg-meta .mode-tag {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 10px;
  font-size: 10px;
  margin-left: 6px;
  text-transform: uppercase;
  letter-spacing: .5px;
}
.mode-general       { background: rgba(88,166,255,.15); color: var(--iso); }
.mode-compliance_check { background: rgba(240,165,0,.15); color: var(--accent); }
.mode-cross_mapping { background: rgba(63,185,80,.15); color: var(--itgc); }

.bubble {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  padding: 16px 20px;
  font-size: 14px;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
}
.bubble.user {
  background: var(--surface2);
  border-color: var(--border);
  color: var(--text);
  align-self: flex-end;
  max-width: 75%;
  border-radius: var(--radius) var(--radius) 4px var(--radius);
}
.bubble.assistant {
  border-radius: 4px var(--radius) var(--radius) var(--radius);
  border-left: 3px solid var(--accent);
}

/* Loading */
.thinking {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
  padding: 12px 0;
  animation: fadeIn .3s ease;
}
.dots span {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: bounce .9s infinite;
}
.dots span:nth-child(2) { animation-delay: .15s; }
.dots span:nth-child(3) { animation-delay: .30s; }

/* ── Input bar ── */
.input-bar {
  padding: 16px 24px 20px;
  border-top: 1px solid var(--border);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.mode-row {
  display: flex;
  gap: 6px;
  align-items: center;
}
.mode-label { font-size: 11px; color: var(--muted); margin-right: 4px; font-family: var(--mono); }
.mode-btn {
  padding: 4px 12px;
  border-radius: 16px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-size: 11px;
  cursor: pointer;
  font-family: var(--sans);
  transition: all .2s;
}
.mode-btn.active { border-color: var(--accent); color: var(--accent); background: rgba(240,165,0,.1); }
.mode-btn:hover:not(.active) { border-color: var(--border2); color: var(--text); }

.input-row {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
textarea {
  flex: 1;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  padding: 12px 16px;
  resize: none;
  outline: none;
  min-height: 48px;
  max-height: 140px;
  transition: border-color .2s;
  line-height: 1.5;
}
textarea:focus { border-color: var(--accent); }
textarea::placeholder { color: var(--muted); }

.send-btn {
  width: 48px; height: 48px;
  border-radius: var(--radius);
  border: none;
  background: var(--accent);
  color: #000;
  cursor: pointer;
  font-size: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .2s;
  flex-shrink: 0;
}
.send-btn:hover { background: #ffbb33; transform: translateY(-1px); }
.send-btn:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; transform: none; }

/* ── Sidebar ── */
aside {
  grid-area: sidebar;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
}
.sidebar-header {
  padding: 16px 20px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.sidebar-title {
  font-family: var(--mono);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
}
.retrieval-count {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--accent);
}

.sidebar-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.sidebar-body::-webkit-scrollbar { width: 3px; }
.sidebar-body::-webkit-scrollbar-thumb { background: var(--border); }

/* Empty state */
.sidebar-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 10px;
  color: var(--muted);
  text-align: center;
  padding: 20px;
}
.sidebar-empty svg { opacity: .3; }
.sidebar-empty p { font-size: 12px; line-height: 1.6; }

/* Retrieval cards */
.ret-card {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: 12px 14px;
  cursor: default;
  transition: border-color .2s, transform .15s;
  animation: fadeIn .3s ease both;
}
.ret-card:hover { border-color: var(--border); transform: translateX(2px); }

.ret-card-top {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 8px;
}
.ret-rank {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 6px;
  flex-shrink: 0;
  margin-top: 1px;
}
.ret-id {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  flex: 1;
  word-break: break-all;
}
.ret-id.itgc { color: var(--itgc); }
.ret-id.iso  { color: var(--iso);  }

.source-pill {
  font-size: 9px;
  font-family: var(--mono);
  padding: 2px 6px;
  border-radius: 10px;
  flex-shrink: 0;
  text-transform: uppercase;
  letter-spacing: .5px;
}
.source-pill.itgc { background: rgba(63,185,80,.15); color: var(--itgc); border: 1px solid rgba(63,185,80,.3); }
.source-pill.iso  { background: rgba(88,166,255,.15); color: var(--iso);  border: 1px solid rgba(88,166,255,.3); }

.ret-title {
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
  margin-bottom: 5px;
  line-height: 1.4;
}
.ret-domain {
  font-size: 10px;
  color: var(--muted);
  font-family: var(--mono);
  margin-bottom: 6px;
}
.ret-snippet {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.55;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.ret-score {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.score-bar-bg {
  flex: 1;
  height: 3px;
  background: var(--surface2);
  border-radius: 2px;
  overflow: hidden;
}
.score-bar {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
  transition: width .5s ease;
}
.score-val {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  flex-shrink: 0;
}

/* ── Animations ── */
@keyframes fadeIn  { from { opacity: 0 } to { opacity: 1 } }
@keyframes slideUp { from { opacity: 0; transform: translateY(10px) } to { opacity: 1; transform: translateY(0) } }
@keyframes bounce  {
  0%, 80%, 100% { transform: translateY(0); opacity: .4 }
  40%           { transform: translateY(-6px); opacity: 1 }
}
</style>
</head>
<body>

<div class="shell">

  <!-- Header -->
  <header>
    <div class="logo">Audit<span>RAG</span></div>
    <span class="badge">ISO 27001 + ITGC</span>
    <div class="spacer"></div>
    <div class="status-dot"></div>
    <span class="status-label">158 documents indexed</span>
  </header>

  <!-- Main chat -->
  <main>
    <div class="chat-history" id="chatHistory">
      <div class="welcome" id="welcomeState">
        <h2>Ask anything about your audit</h2>
        <p>Grounded responses from ISO 27001:2022 and ITGC with full source citations. Every answer traces back to a specific clause or control.</p>
        <div class="example-queries">
          <button class="eq-btn" onclick="setExample(this)">What evidence for user access control?</button>
          <button class="eq-btn" onclick="setExample(this)">Map ITGC Access Management to ISO 27001</button>
          <button class="eq-btn" onclick="setExample(this)">Does ITGC cover ISO 27001 risk assessment?</button>
          <button class="eq-btn" onclick="setExample(this)">Evidence for change management audit?</button>
          <button class="eq-btn" onclick="setExample(this)">What is ISO 8.24 about?</button>
        </div>
      </div>
    </div>

    <div class="input-bar">
      <div class="mode-row">
        <span class="mode-label">Mode:</span>
        <button class="mode-btn active" data-mode="general" onclick="setMode(this)">General</button>
        <button class="mode-btn" data-mode="compliance_check" onclick="setMode(this)">Compliance Check</button>
        <button class="mode-btn" data-mode="cross_mapping" onclick="setMode(this)">Cross Mapping</button>
      </div>
      <div class="input-row">
        <textarea id="queryInput" placeholder="Ask an audit question..." rows="1"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <button class="send-btn" id="sendBtn" onclick="submitQuery()">&#x27A4;</button>
      </div>
    </div>
  </main>

  <!-- Sidebar -->
  <aside>
    <div class="sidebar-header">
      <span class="sidebar-title">Retrieved Sources</span>
      <span class="retrieval-count" id="retCount">—</span>
    </div>
    <div class="sidebar-body" id="sidebarBody">
      <div class="sidebar-empty">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/>
          <rect x="9" y="3" width="6" height="4" rx="1"/>
          <path d="M9 12h6M9 16h4"/>
        </svg>
        <p>Retrieved documents will appear here after you submit a query.</p>
      </div>
    </div>
  </aside>

</div>

<script>
let currentMode = 'general';
let isLoading = false;
let maxScore = 1;

function setMode(btn) {
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentMode = btn.dataset.mode;
}

function setExample(btn) {
  document.getElementById('queryInput').value = btn.textContent;
  autoResize(document.getElementById('queryInput'));
  document.getElementById('queryInput').focus();
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
}

function hideWelcome() {
  const w = document.getElementById('welcomeState');
  if (w) w.remove();
}

function appendMessage(role, content, mode) {
  hideWelcome();
  const history = document.getElementById('chatHistory');

  const div = document.createElement('div');
  div.className = 'msg';

  if (role === 'user') {
    div.innerHTML = `
      <div class="msg-meta">You</div>
      <div class="bubble user">${escHtml(content)}</div>`;
  } else {
    const modeTag = mode ? `<span class="mode-tag mode-${mode}">${mode.replace('_',' ')}</span>` : '';
    div.innerHTML = `
      <div class="msg-meta">AuditRAG ${modeTag}</div>
      <div class="bubble assistant">${escHtml(content)}</div>`;
  }

  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

function showThinking() {
  hideWelcome();
  const history = document.getElementById('chatHistory');
  const div = document.createElement('div');
  div.id = 'thinking';
  div.className = 'thinking';
  div.innerHTML = `<div class="dots"><span></span><span></span><span></span></div><span>Retrieving and reasoning...</span>`;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
}

function hideThinking() {
  const t = document.getElementById('thinking');
  if (t) t.remove();
}

function renderSidebar(retrievals) {
  const body = document.getElementById('sidebarBody');
  const count = document.getElementById('retCount');

  if (!retrievals || retrievals.length === 0) {
    body.innerHTML = '<div class="sidebar-empty"><p>No documents retrieved.</p></div>';
    count.textContent = '0 docs';
    return;
  }

  count.textContent = `${retrievals.length} docs`;
  maxScore = retrievals[0].score || 0.034;

  body.innerHTML = '';
  retrievals.forEach((r, i) => {
    const srcClass = r.source === 'ITGC' ? 'itgc' : 'iso';
    const barWidth  = Math.round((r.score / maxScore) * 100);

    const card = document.createElement('div');
    card.className = 'ret-card';
    card.style.animationDelay = (i * 0.06) + 's';
    card.innerHTML = `
      <div class="ret-card-top">
        <span class="ret-rank">#${r.rank}</span>
        <span class="ret-id ${srcClass}">${escHtml(r.control_id)}</span>
        <span class="source-pill ${srcClass}">${r.source}</span>
      </div>
      <div class="ret-title">${escHtml(r.title)}</div>
      <div class="ret-domain">${escHtml(r.domain)}</div>
      <div class="ret-snippet">${escHtml(r.snippet)}</div>
      <div class="ret-score">
        <div class="score-bar-bg">
          <div class="score-bar" style="width:${barWidth}%"></div>
        </div>
        <span class="score-val">${r.score.toFixed(5)}</span>
      </div>`;
    body.appendChild(card);
  });
}

async function submitQuery() {
  if (isLoading) return;
  const input = document.getElementById('queryInput');
  const query = input.value.trim();
  if (!query) return;

  isLoading = true;
  document.getElementById('sendBtn').disabled = true;
  input.value = '';
  input.style.height = 'auto';

  appendMessage('user', query);
  showThinking();

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode: currentMode }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();

    hideThinking();
    appendMessage('assistant', data.answer, currentMode);
    renderSidebar(data.retrievals);

  } catch (err) {
    hideThinking();
    appendMessage('assistant', `Error: ${err.message}. Make sure the server is running and your API key is set.`);
  }

  isLoading = false;
  document.getElementById('sendBtn').disabled = false;
  input.focus();
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")