# api/index.py
import os
import re
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
from math import log, sqrt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Optional .env (tiny dep). If it's missing we simply continue.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Runtime-safe paths on Vercel ---
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Pure-Python TF-IDF ----------------
_token_re = re.compile(r"[a-zA-Z0-9_]+")

def tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower()) if text else []

def build_tfidf(texts: List[str]):
    """Return (vocab, idf, normed_vectors)."""
    docs = [tokenize(t) for t in texts]
    N = len(docs)
    df: Dict[str,int] = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    vocab = {t:i for i,t in enumerate(sorted(df))}
    idf = [log((N + 1) / (df[t] + 1)) + 1.0 for t in sorted(df)]
    vectors: List[Dict[int,float]] = []
    for d in docs:
        tf = Counter(d)
        vec = {vocab[t]: tf[t] * idf[vocab[t]] for t in tf if t in vocab}
        # L2 normalize
        norm = sqrt(sum(v*v for v in vec.values())) or 1.0
        vec = {k: v / norm for k, v in vec.items()}
        vectors.append(vec)
    return vocab, idf, vectors

def vec_for_query(q: str, vocab, idf):
    tf = Counter(tokenize(q))
    qvec = {}
    for t, cnt in tf.items():
        if t in vocab:
            i = vocab[t]
            val = cnt * idf[i]
            qvec[i] = val
    norm = sqrt(sum(v*v for v in qvec.values())) or 1.0
    return {k: v / norm for k, v in qvec.items()}

def cosine_sparse(a: Dict[int,float], b: Dict[int,float]) -> float:
    # iterate over smaller
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for i, va in a.items():
        vb = b.get(i)
        if vb is not None:
            s += va * vb
    return float(s)

# ---------------- App State ----------------
class Store:
    def __init__(self):
        self.texts: List[str] = []
        self.vocab = None
        self.idf = None
        self.vectors: List[Dict[int,float]] = []

STATE = Store()

# ---------------- FastAPI ----------------
app = FastAPI(title="BigQueryGPT-slim")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -------- utils --------
def read_simple_table(path: Path) -> List[str]:
    """
    Reads .csv/.txt/.json lines very simply to keep runtime tiny.
    Produces one big string per row/line for indexing.
    """
    ext = path.suffix.lower()
    rows: List[str] = []
    if ext in {".csv", ".txt"}:
        # read small/medium files; for big files you should pre-process offline
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line)
    elif ext == ".json":
        # line-delimited or array
        data = path.read_text(encoding="utf-8", errors="ignore")
        try:
            obj = json.loads(data)
            if isinstance(obj, list):
                for item in obj:
                    rows.append(json.dumps(item, ensure_ascii=False))
            else:
                rows.append(json.dumps(obj, ensure_ascii=False))
        except Exception:
            # maybe it's JSONL
            for ln in data.splitlines():
                ln = ln.strip()
                if ln:
                    rows.append(ln)
    else:
        # Fallback: read as lines
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(line)
    return rows

# -------- endpoints --------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>BigQueryGPT-slim</title>
<style>
body{font-family:system-ui,Arial;margin:0;background:#0b1220;color:#e7f1ff}
.wrap{max-width:900px;margin:24px auto;padding:0 16px}
.hint{color:#a9c2ff}
.zone{margin-top:14px;border:2px dashed #4da3ff;padding:18px;border-radius:12px;text-align:center}
.btn{background:#4da3ff;border:none;padding:8px 12px;border-radius:8px;color:#00122b;cursor:pointer}
.row{display:flex;gap:8px;align-items:center}
input[type=text]{flex:1;border-radius:8px;border:1px solid #355a8a;background:#071427;color:#e7f1ff;padding:8px}
.msg{margin:8px 0;padding:8px;border-radius:8px;background:#0f1c33}
pre{white-space:pre-wrap}
</style>
</head>
<body>
<div class="wrap">
  <h2>BigQueryGPT-slim</h2>
  <p class="hint">1) Upload small CSV/TXT/JSON → 2) Build Index → 3) Ask</p>
  <div class="zone" onclick="fileInput.click()">
    Drop/click to select files
    <input id="fileInput" type="file" multiple style="display:none" onchange="uploadFiles(this.files)"/>
  </div>
  <p id="files" class="hint"></p>
  <div style="margin-top:8px">
    <button class="btn" onclick="buildIndex()">Build Index</button>
    <span id="buildOut" class="hint"></span>
  </div>
  <div class="msg">
    <div class="row">
      <input id="q" type="text" placeholder="Ask your data..."/>
      <button class="btn" onclick="ask()">Ask</button>
    </div>
    <pre id="ans"></pre>
  </div>
</div>
<script>
const fileInput = document.getElementById('fileInput');
const filesOut  = document.getElementById('files');
const buildOut  = document.getElementById('buildOut');
const ansOut    = document.getElementById('ans');
document.getElementById('q').addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ ask(); }});

async function uploadFiles(fs){
  let fd = new FormData();
  for (let f of fs) fd.append('files', f);
  const r = await fetch('/upload', {method:'POST', body: fd});
  const j = await r.json();
  filesOut.textContent = j.message || JSON.stringify(j);
}
async function buildIndex(){
  buildOut.textContent = 'Building...';
  const r = await fetch('/build', { method:'POST' });
  const j = await r.json();
  buildOut.textContent = j.status ? ('Index ready ('+j.n_docs+' docs)') : (j.error||JSON.stringify(j));
}
async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q){ return; }
  ansOut.textContent = 'Thinking...';
  const r = await fetch('/ask?q='+encodeURIComponent(q));
  const j = await r.json();
  ansOut.textContent = j.answer || j.error || JSON.stringify(j,null,2);
}
</script>
</body></html>
        """
    )

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    count = 0
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with dest.open("wb") as out:
            out.write(await f.read())
        count += 1
    return {"status": "ok", "message": f"{count} file(s) uploaded to {UPLOAD_DIR}."}

@app.post("/build")
def build():
    # collect texts from everything in /tmp/uploads
    paths = [p for p in UPLOAD_DIR.glob("*") if p.is_file()]
    if not paths:
        raise HTTPException(400, "No files uploaded yet.")
    texts: List[str] = []
    for p in paths:
        try:
            rows = read_simple_table(p)
            texts.extend(rows)
        except Exception:
            # ignore unreadable file types to avoid crashes
            pass
    if not texts:
        raise HTTPException(400, "No readable rows found.")
    STATE.texts = texts
    STATE.vocab, STATE.idf, STATE.vectors = build_tfidf(texts)
    return {"status": "ok", "n_docs": len(texts)}

@app.get("/ask")
def ask(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "Empty question")
    if not STATE.vectors:
        raise HTTPException(400, "Index not built.")
    qv = vec_for_query(q, STATE.vocab, STATE.idf)
    sims = [(i, cosine_sparse(qv, v)) for i, v in enumerate(STATE.vectors)]
    sims.sort(key=lambda x: x[1], reverse=True)
    hits = sims[:max(1, min(top_k, 10))]
    # very small answer heuristic: echo best lines
    best_lines = [STATE.texts[i] for i,_ in hits[:2]]
    answer = " • " + "\n • ".join(best_lines) if best_lines else "No match."
    return {"answer": answer, "k": len(hits)}
