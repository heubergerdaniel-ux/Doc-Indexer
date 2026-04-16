import os
import logging
from urllib.parse import quote

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_HOST  = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT  = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION   = os.environ.get("QDRANT_COLLECTION", "documents")
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "172.17.0.1")
OLLAMA_PORT  = int(os.environ.get("OLLAMA_PORT", 11434))
EMBED_MODEL      = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL        = os.environ.get("LLM_MODEL", "llama3.2:3b")
SYNOLOGY_IP      = os.environ.get("SYNOLOGY_IP", "192.168.1.216")
SMB_SHARE        = os.environ.get("SMB_SHARE_NAME", "documents")
SCORE_THRESHOLD  = float(os.environ.get("SCORE_THRESHOLD", "0.50"))

EMBED_URL    = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
GENERATE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = FastAPI(title="NAS Document Search")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def build_smb_link(rel_path: str) -> str:
    win_path = rel_path.replace("/", "\\")
    return f"\\\\NAS\\{SMB_SHARE}\\{win_path}"


def build_web_link(rel_path: str) -> str:
    encoded = quote(rel_path, safe="/")
    return (
        f"http://{SYNOLOGY_IP}:5000/webman/entry.cgi?"
        f"SynoPhotoAlbum_index.cgi&launchApp=SYNO.SDS.FileStation.Application"
        f"&launchParam=openfile%3D%2F{SMB_SHARE}%2F{encoded}"
    )


def hit_to_dict(hit, text_limit: int = 600) -> dict:
    p = hit.payload
    return {
        "score": round(hit.score, 4),
        "filename": p.get("filename", ""),
        "rel_path": p.get("rel_path", ""),
        "folder": p.get("folder", ""),
        "page": p.get("page", 0),
        "text": (p.get("text", ""))[:text_limit],
        "source": p.get("source", ""),
        "smb_link": build_smb_link(p.get("rel_path", "")),
        "web_link": build_web_link(p.get("rel_path", "")),
    }


def deduplicate(hits: list, limit: int) -> list:
    seen: set[tuple] = set()
    results = []
    for hit in hits:
        key = (hit.payload.get("path", ""), hit.payload.get("page", 0))
        if key not in seen:
            seen.add(key)
            results.append(hit)
            if len(results) >= limit:
                break
    return results


def build_filter(folder: str | None) -> Filter | None:
    if not folder:
        return None
    return Filter(
        must=[FieldCondition(key="folder", match=MatchValue(value=folder))]
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=WEB_UI_HTML)


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(8, ge=1, le=50),
    folder: str = Query(None),
    dedupe: bool = Query(True),
    score_threshold: float = Query(SCORE_THRESHOLD, ge=0.0, le=1.0),
):
    try:
        embedding = get_embedding(q)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding error: {exc}")

    fetch_limit = limit * 3 if dedupe else limit
    try:
        hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            limit=fetch_limit,
            query_filter=build_filter(folder),
            with_payload=True,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")

    if dedupe:
        hits = deduplicate(hits, limit)
    else:
        hits = hits[:limit]

    results = [hit_to_dict(h) for h in hits]
    return {"results": results, "total": len(results)}


@app.get("/ask")
async def ask(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    folder: str = Query(None),
    score_threshold: float = Query(SCORE_THRESHOLD, ge=0.0, le=1.0),
):
    try:
        embedding = get_embedding(q)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding error: {exc}")

    try:
        hits = qdrant.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            limit=limit * 2,
            query_filter=build_filter(folder),
            with_payload=True,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")

    hits = deduplicate(hits, limit)
    sources = [hit_to_dict(h, text_limit=600) for h in hits]

    if not sources:
        return {"answer": "Keine relevanten Dokumente gefunden.", "sources": []}

    context = "\n\n---\n\n".join(
        f"[Quelle: {s['filename']}, Seite {s['page']}]\n{s['text']}"
        for s in sources
    )
    prompt = (
        "Du bist ein hilfreicher Assistent. Beantworte die folgende Frage ausschließlich auf Deutsch, "
        "basierend nur auf den bereitgestellten Dokumentenausschnitten. "
        "Wenn die Antwort nicht in den Ausschnitten enthalten ist, sage das klar.\n\n"
        f"Dokumentenausschnitte:\n{context}\n\n"
        f"Frage: {q}\n\nAntwort:"
    )

    try:
        resp = requests.post(
            GENERATE_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    return {"answer": answer, "sources": sources}


@app.get("/stats")
async def stats():
    try:
        info = qdrant.get_collection(COLLECTION)
        return {
            "collection": COLLECTION,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")


# ---------------------------------------------------------------------------
# Embedded Web UI
# ---------------------------------------------------------------------------

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NAS Dokumentensuche</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5;
    color: #1a1a2e;
    min-height: 100vh;
  }
  header {
    background: #16213e;
    color: #e0e0ff;
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  header h1 { font-size: 1.3rem; font-weight: 600; }
  header .subtitle { font-size: 0.8rem; opacity: 0.6; }
  .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }

  .tabs { display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 2px solid #d0d4e8; }
  .tab {
    padding: 0.6rem 1.4rem;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: 500;
    color: #555;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    transition: color 0.2s, border-color 0.2s;
  }
  .tab.active { color: #0f3460; border-color: #0f3460; }
  .tab:hover:not(.active) { color: #0f3460; }

  .panel { display: none; }
  .panel.active { display: block; }

  .search-bar {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
  }
  .search-bar input[type="text"] {
    flex: 1;
    padding: 0.65rem 1rem;
    border: 1.5px solid #c8cde0;
    border-radius: 8px;
    font-size: 1rem;
    outline: none;
    transition: border-color 0.2s;
  }
  .search-bar input[type="text"]:focus { border-color: #0f3460; }
  .search-bar button {
    padding: 0.65rem 1.3rem;
    background: #0f3460;
    color: #fff;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.95rem;
    font-weight: 500;
    transition: background 0.2s;
  }
  .search-bar button:hover { background: #1a4a80; }
  .search-bar button:disabled { background: #aaa; cursor: not-allowed; }

  .filters {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1.25rem;
    font-size: 0.875rem;
  }
  .filters input[type="text"] {
    padding: 0.4rem 0.75rem;
    border: 1.5px solid #c8cde0;
    border-radius: 6px;
    font-size: 0.875rem;
    width: 180px;
    outline: none;
  }
  .filters input[type="text"]:focus { border-color: #0f3460; }
  .filters label { display: flex; align-items: center; gap: 0.4rem; cursor: pointer; }

  #results-count, #ask-count {
    font-size: 0.82rem;
    color: #666;
    margin-bottom: 0.75rem;
  }

  .result-card {
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #0f3460;
  }
  .result-card .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    align-items: center;
  }
  .result-card .filename {
    font-weight: 600;
    font-size: 0.95rem;
    color: #0f3460;
  }
  .badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
  }
  .badge-folder { background: #e8f0fe; color: #1a56db; }
  .badge-page   { background: #fef3c7; color: #92400e; }
  .badge-score  { background: #dcfce7; color: #166534; }
  .badge-ocr    { background: #fce7f3; color: #9d174d; }
  .badge-native { background: #ede9fe; color: #5b21b6; }

  .result-card .snippet {
    font-size: 0.875rem;
    color: #333;
    line-height: 1.6;
    margin-bottom: 0.7rem;
  }
  .result-card .links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
  }
  .link-web {
    font-size: 0.8rem;
    color: #0f3460;
    text-decoration: none;
    padding: 0.25rem 0.6rem;
    border: 1px solid #0f3460;
    border-radius: 5px;
    transition: background 0.15s;
  }
  .link-web:hover { background: #0f3460; color: #fff; }
  .smb-path {
    font-size: 0.78rem;
    color: #555;
    font-family: monospace;
    background: #f4f4f4;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    user-select: all;
    title: "Klicken zum Kopieren";
  }
  .smb-path:hover { background: #e0e4f0; }

  .answer-box {
    background: #fff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #059669;
    font-size: 0.95rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }
  .sources-heading {
    font-size: 0.85rem;
    font-weight: 600;
    color: #555;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 3px solid #ccc;
    border-top-color: #0f3460;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 0.4rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-msg { color: #c0392b; font-size: 0.9rem; margin-top: 0.5rem; }
  .empty-msg { color: #888; font-size: 0.9rem; text-align: center; padding: 2rem; }
</style>
</head>
<body>
<header>
  <div>
    <h1>NAS Dokumentensuche</h1>
    <div class="subtitle">Semantische Suche &amp; RAG Q&amp;A</div>
  </div>
</header>

<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('search')">Suche</div>
    <div class="tab" onclick="switchTab('ask')">Frage stellen</div>
    <div class="tab" onclick="switchTab('stats')" id="stats-tab">Info</div>
  </div>

  <!-- Search Panel -->
  <div class="panel active" id="panel-search">
    <div class="search-bar">
      <input type="text" id="search-q" placeholder="Suchbegriff eingeben..." onkeydown="if(event.key==='Enter')doSearch()">
      <button id="search-btn" onclick="doSearch()">Suchen</button>
    </div>
    <div class="filters">
      <input type="text" id="search-folder" placeholder="Ordner-Filter (optional)">
      <label><input type="checkbox" id="search-dedupe" checked> Duplikate ausblenden</label>
    </div>
    <div id="results-count"></div>
    <div id="search-results"></div>
  </div>

  <!-- Ask Panel -->
  <div class="panel" id="panel-ask">
    <div class="search-bar">
      <input type="text" id="ask-q" placeholder="Frage eingeben..." onkeydown="if(event.key==='Enter')doAsk()">
      <button id="ask-btn" onclick="doAsk()">Fragen</button>
    </div>
    <div class="filters">
      <input type="text" id="ask-folder" placeholder="Ordner-Filter (optional)">
    </div>
    <div id="ask-answer"></div>
    <div id="ask-sources"></div>
  </div>

  <!-- Stats Panel -->
  <div class="panel" id="panel-stats">
    <div id="stats-content" style="font-size:0.9rem; color:#444; line-height:2;"></div>
  </div>
</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  const tabs = ['search','ask','stats'];
  document.querySelectorAll('.tab')[tabs.indexOf(name)].classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'stats') loadStats();
}

function badge(cls, text) {
  return `<span class="badge ${cls}">${escHtml(String(text))}</span>`;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderResultCard(r) {
  const smb = escHtml(r.smb_link);
  return `
    <div class="result-card">
      <div class="meta">
        <span class="filename">${escHtml(r.filename)}</span>
        ${r.folder ? badge('badge-folder', r.folder) : ''}
        ${badge('badge-page', 'S. ' + r.page)}
        ${badge('badge-score', (r.score * 100).toFixed(1) + '%')}
        ${badge(r.source === 'ocr' ? 'badge-ocr' : 'badge-native', r.source.toUpperCase())}
      </div>
      <div class="snippet">${escHtml(r.text)}</div>
      <div class="links">
        <a class="link-web" href="${escHtml(r.web_link)}" target="_blank">Im Browser öffnen</a>
        <span class="smb-path" title="Netzwerkpfad – klicken zum Markieren">${smb}</span>
      </div>
    </div>`;
}

async function doSearch() {
  const q = document.getElementById('search-q').value.trim();
  if (!q) return;
  const folder = document.getElementById('search-folder').value.trim();
  const dedupe = document.getElementById('search-dedupe').checked;
  const btn = document.getElementById('search-btn');
  const resDiv = document.getElementById('search-results');
  const countDiv = document.getElementById('results-count');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Suche...';
  resDiv.innerHTML = '';
  countDiv.innerHTML = '';

  try {
    const params = new URLSearchParams({ q, dedupe });
    if (folder) params.set('folder', folder);
    const data = await fetch('/search?' + params).then(r => r.json());
    countDiv.textContent = data.total + ' Ergebnisse';
    if (!data.results.length) {
      resDiv.innerHTML = '<div class="empty-msg">Keine Ergebnisse gefunden.</div>';
    } else {
      resDiv.innerHTML = data.results.map(renderResultCard).join('');
    }
  } catch (e) {
    resDiv.innerHTML = `<div class="error-msg">Fehler: ${escHtml(String(e))}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Suchen';
  }
}

async function doAsk() {
  const q = document.getElementById('ask-q').value.trim();
  if (!q) return;
  const folder = document.getElementById('ask-folder').value.trim();
  const btn = document.getElementById('ask-btn');
  const answerDiv = document.getElementById('ask-answer');
  const sourcesDiv = document.getElementById('ask-sources');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Denkt nach...';
  answerDiv.innerHTML = '';
  sourcesDiv.innerHTML = '';

  try {
    const params = new URLSearchParams({ q });
    if (folder) params.set('folder', folder);
    const data = await fetch('/ask?' + params).then(r => r.json());

    answerDiv.innerHTML = `<div class="answer-box">${escHtml(data.answer)}</div>`;

    if (data.sources && data.sources.length) {
      sourcesDiv.innerHTML = '<div class="sources-heading">Quellen</div>' +
        data.sources.map(renderResultCard).join('');
    }
  } catch (e) {
    answerDiv.innerHTML = `<div class="error-msg">Fehler: ${escHtml(String(e))}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Fragen';
  }
}

async function loadStats() {
  const div = document.getElementById('stats-content');
  div.innerHTML = '<span class="spinner"></span> Lade...';
  try {
    const d = await fetch('/stats').then(r => r.json());
    div.innerHTML = `
      <b>Collection:</b> ${escHtml(d.collection)}<br>
      <b>Status:</b> ${escHtml(d.status)}<br>
      <b>Vektoren:</b> ${d.vectors_count ?? '–'}<br>
      <b>Indizierte Vektoren:</b> ${d.indexed_vectors_count ?? '–'}<br>
      <b>Punkte:</b> ${d.points_count ?? '–'}
    `;
  } catch (e) {
    div.innerHTML = `<div class="error-msg">Fehler: ${escHtml(String(e))}</div>`;
  }
}
</script>
</body>
</html>"""
