# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from collections import Counter
import math

app = FastAPI(title="BigQueryGPT-slim")

# Allow CORS during dev / preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------- super-slim "TF-IDF-ish" (no sklearn) -------
DOCS: List[str] = []
INDEX: Dict[str, float] = {}   # idf
DOC_TF: List[Dict[str, float]] = []  # per-doc tf

def _tokens(s: str):
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if t]

def build_index(texts: List[str]):
    global DOCS, INDEX, DOC_TF
    DOCS = [t for t in texts if t and t.strip()]
    N = len(DOCS)
    if N == 0:
        INDEX, DOC_TF = {}, []
        return

    # term frequencies
    DOC_TF = []
    df = Counter()
    for t in DOCS:
        tok = _tokens(t)
        c = Counter(tok)
        DOC_TF.append({k: v/len(tok) for k, v in c.items() if len(tok) > 0})
        df.update(set(tok))

    # idf
    INDEX = {}
    for term, d in df.items():
        INDEX[term] = math.log((N+1) / (1 + d)) + 1.0   # smooth idf

def search(query: str, k: int = 5):
    if not DOCS:
        return []
    qtok = _tokens(query)
    if not qtok:
        return []
    q_tf = Counter(qtok)
    # tf-idf cosine manually
    def score(doc_tf: Dict[str, float]) -> float:
        # dot
        dot = 0.0
        for term, qtf in q_tf.items():
            if term in doc_tf:
                dot += (qtf * INDEX.get(term, 0.0)) * (doc_tf[term] * INDEX.get(term, 0.0))
        # norms
        qn = math.sqrt(sum((q_tf[t] * INDEX.get(t, 0.0))**2 for t in q_tf))
        dn = math.sqrt(sum((doc_tf[t] * INDEX.get(t, 0.0))**2 for t in doc_tf))
        if qn == 0 or dn == 0:
            return 0.0
        return dot / (qn * dn)

    sims = [(i, score(DOC_TF[i])) for i in range(len(DOCS))]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [{"text": DOCS[i], "score": float(s)} for (i, s) in sims[:k]]

# ------------- Routes (all under /api) -----------------

@app.get("/api/health")
def health():
    return {"ok": True, "n_docs": len(DOCS)}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    texts = []
    for f in files:
        # read small text-like files (csv / txt / json lines)
        raw = await f.read()
        try:
            s = raw.decode("utf-8", errors="ignore")
        except Exception:
            s = str(raw)
        # naive line split
        for line in s.splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    if not texts:
        raise HTTPException(400, detail="No text could be parsed from uploads.")
    # stash in a global temp; index is built when user clicks "Build Index"
    app.state._pending_texts = texts
    return {"status": "ok", "n_lines": len(texts)}

@app.post("/api/build_index")
def build_index_endpoint():
    texts = getattr(app.state, "_pending_texts", None)
    if not texts:
        raise HTTPException(400, detail="No files uploaded yet.")
    build_index(texts)
    return {"status": "ok", "n_docs": len(DOCS)}

@app.get("/api/ask")
def ask(q: str, k: int = 5):
    if not q or not q.strip():
        raise HTTPException(400, detail="Empty query")
    if not DOCS:
        raise HTTPException(400, detail="Index not built")
    hits = search(q, k)
    # super-short answer = just show top 1 line
    ans = hits[0]["text"] if hits else "No match."
    return {"answer": ans, "retrieved": hits}

# ----------------- Static UI ---------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>BigQueryGPT-slim</title>
<style>
body{font-family:system-ui,Arial;background:#0b1220;color:#eaf6ff;margin:0}
.wrap{max-width:980px;margin:24px auto;padding:0 16px}
.zone{margin-top:16px;border:2px dashed #39c0ff;padding:18px;border-radius:12px;text-align:center}
.btn{background:#39c0ff;border:none;color:#001522;padding:8px 12px;border-radius:10px;cursor:pointer}
.row{display:flex;gap:8px;align-items:center;margin-top:14px}
input[type=text]{flex:1;border-radius:10px;border:1px solid #224; background:#0c1a2a;color:#eaf6ff;padding:10px}
.msg{margin:6px 0;background:#0c1a2a;border:1px solid #223;border-radius:10px;padding:10px}
small{color:#8bd3ff}
</style>
</head>
<body>
<div class="wrap">
  <h2>BigQueryGPT-slim</h2>
  <p><small>1) Upload small CSV/TXT/JSON → 2) Build Index → 3) Ask</small></p>

  <div class="zone" ondrop="dropHandler(event)" ondragover="dragOverHandler(event)" onclick="fileInput.click()">
    Drop/click to select files
    <input id="fileInput" type="file" multiple style="display:none" onchange="uploadFiles(this.files)"/>
  </div>
  <pre id="out" class="msg"></pre>

  <button class="btn" onclick="buildIndex()">Build Index</button>
  <span id="buildOut" class="msg" style="display:inline-block"></span>

  <div class="row">
    <input id="q" type="text" placeholder="Ask your data..." />
    <button class="btn" onclick="ask()">Ask</button>
  </div>
  <pre id="ans" class="msg"></pre>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const out = document.getElementById('out');
const ans = document.getElementById('ans');
const qInput = document.getElementById('q');

function dragOverHandler(e){ e.preventDefault(); }
function dropHandler(e){ e.preventDefault(); uploadFiles(e.dataTransfer.files); }

async function uploadFiles(fs){
  try{
    let fd = new FormData();
    for (let f of fs) fd.append('files', f);
    const r = await fetch('/api/upload', { method:'POST', body: fd });
    const j = await r.json();
    out.textContent = JSON.stringify(j);
  }catch(e){
    out.textContent = 'Upload error: ' + e;
  }
}

async function buildIndex(){
  try{
    document.getElementById('buildOut').textContent = 'Building...';
    const r = await fetch('/api/build_index', { method:'POST' });
    const j = await r.json();
    document.getElementById('buildOut').textContent = JSON.stringify(j);
  }catch(e){
    document.getElementById('buildOut').textContent = 'Build error: ' + e;
  }
}

qInput.addEventListener('keydown', (e)=>{
  if (e.key === 'Enter'){ e.preventDefault(); ask(); }
});

async function ask(){
  const q = qInput.value.trim();
  if (!q) return;
  ans.textContent = 'Thinking...';
  try{
    const r = await fetch('/api/ask?q=' + encodeURIComponent(q));
    const j = await r.json();
    ans.textContent = typeof j === 'string' ? j : JSON.stringify(j, null, 2);
  }catch(e){
    ans.textContent = 'Ask error: ' + e;
  }
}
</script>
</body>
</html>"""
    return HTMLResponse(html)

# optional plain root check
@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "pong"
