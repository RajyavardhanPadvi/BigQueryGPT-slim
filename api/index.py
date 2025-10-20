# api/index.py
# BigQueryGPT-slim: FastAPI + in-memory TF-IDF (no heavy deps)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from collections import Counter, defaultdict
from math import log, sqrt
import os, io, csv, json

app = FastAPI(title="BigQueryGPT-slim")

# CORS (handy for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ---------- simple storage (Vercel-friendly) ----------
ROOT_TMP = "/tmp"
UPLOAD_DIR = os.path.join(ROOT_TMP, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory index structures
DOCS: List[Dict] = []          # [{"text": "..."}]
VOCAB = set()                  # all tokens
DF = Counter()                 # document frequency
IDF = {}                       # token -> idf
DOC_VECS = []                  # list[dict token->tfidf]
TOKENIZER_SPLIT = " \t\r\n,.;:!?/\\|()[]{}\"'`~@#$%^&*-_=+<>"

def _tokens(s: str) -> List[str]:
    s = (s or "").lower()
    out, buf = [], []
    for ch in s:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return [t for t in out if t]

def _read_csv_bytes(b: bytes) -> List[Dict]:
    try:
        txt = b.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(txt))
        rows = list(reader)
        if rows: 
            return rows
        # fallback if no header
        sio = io.StringIO(txt)
        reader2 = csv.reader(sio)
        rows2 = [{"row": " | ".join(r)} for r in reader2]
        return rows2
    except Exception:
        return [{"row": b.decode('utf-8', errors='ignore')}]

def _read_json_bytes(b: bytes) -> List[Dict]:
    try:
        data = json.loads(b.decode("utf-8", errors="ignore"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return [{"value": str(data)}]
    except Exception:
        return [{"raw": b.decode("utf-8", errors="ignore")}]

def _read_txt_bytes(b: bytes) -> List[Dict]:
    txt = b.decode("utf-8", errors="ignore")
    # chunk by ~800 chars with overlap
    chunks, step, k = [], 600, 800
    i, n = 0, len(txt)
    while i < n:
        j = min(i+k, n)
        chunks.append({"text": txt[i:j]})
        if j >= n: break
        i = max(0, j - step)
    return chunks

def _row_to_text(row: Dict) -> str:
    if isinstance(row, dict):
        parts = []
        for k, v in row.items():
            if v is None: 
                continue
            s = str(v).strip()
            if s:
                parts.append(f"{k}: {s}")
        return " | ".join(parts)
    return str(row)

def _build_index():
    global VOCAB, DF, IDF, DOC_VECS
    VOCAB.clear()
    DF.clear()
    IDF.clear()
    DOC_VECS.clear()

    # collect DF
    for d in DOCS:
        toks = set(_tokens(d["text"]))
        VOCAB |= toks
        for t in toks:
            DF[t] += 1

    N = max(1, len(DOCS))
    for t in VOCAB:
        # +1 smoothing to avoid div by zero
        IDF[t] = log((N + 1) / (DF[t] + 1)) + 1.0

    for d in DOCS:
        toks = _tokens(d["text"])
        tf = Counter(toks)
        vec = {}
        for t, c in tf.items():
            if t in IDF:
                vec[t] = (c / max(1, len(toks))) * IDF[t]
        DOC_VECS.append(vec)

def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2: 
        return 0.0
    # dot
    dot = 0.0
    # iterate smaller
    keys = v1.keys() if len(v1) < len(v2) else v2.keys()
    for k in keys:
        if k in v1 and k in v2:
            dot += v1[k] * v2[k]
    # norms
    n1 = sqrt(sum(x*x for x in v1.values()))
    n2 = sqrt(sum(x*x for x in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

def _query_vec(q: str) -> Dict[str, float]:
    toks = _tokens(q)
    tf = Counter(toks)
    vec = {}
    L = max(1, len(toks))
    for t, c in tf.items():
        if t in IDF:
            vec[t] = (c / L) * IDF[t]
    return vec

# ------------------------ ROUTES ------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    # UI calls /api/* (important for Vercel)
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>BigQueryGPT-slim</title>
<style>
  body{font-family:Inter,Arial;background:#0b1220;color:#e6f7fb;margin:0}
  .wrap{max-width:900px;margin:28px auto;padding:0 16px}
  .hint{color:#9fd6e6;font-size:0.9rem}
  .btn{background:#2ec4ff;border:none;padding:8px 12px;border-radius:8px;cursor:pointer}
  .zone{margin-top:14px;border:2px dashed #2ec4ff;padding:18px;border-radius:12px;text-align:center}
  .row{display:flex;gap:8px;align-items:center}
  input[type=text]{flex:1;border-radius:8px;border:1px solid #2b4d66;background:#0b1b2b;color:#e6f7fb;padding:8px}
  pre{white-space:pre-wrap;color:#bfefff}
</style>
</head>
<body>
<div class="wrap">
  <h2>BigQueryGPT-slim</h2>
  <p class="hint">1) Upload small CSV/TXT/JSON → 2) Build Index → 3) Ask</p>

  <div class="zone" onclick="fileInput.click()" ondrop="dropHandler(event)" ondragover="event.preventDefault()">
    Drop/click to select files
    <input id="fileInput" type="file" multiple style="display:none" onchange="uploadFiles(this.files)"/>
  </div>
  <pre id="files" class="hint"></pre>

  <div style="margin-top:10px">
    <button class="btn" onclick="buildIndex()">Build Index</button>
    <span id="buildOut" class="hint"></span>
  </div>

  <div class="row" style="margin-top:16px">
    <input id="q" type="text" placeholder="Ask your data...">
    <button class="btn" onclick="ask()">Ask</button>
  </div>
  <pre id="ans" class="hint"></pre>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const filesOut  = document.getElementById('files');
const buildOut  = document.getElementById('buildOut');
const ansOut    = document.getElementById('ans');

async function uploadFiles(fs){
  let fd = new FormData();
  for (let f of fs) fd.append('files', f);
  try{
    const r = await fetch('/api/upload', { method:'POST', body: fd });
    const j = await r.json();
    filesOut.textContent = JSON.stringify(j);
  }catch(e){
    filesOut.textContent = 'Upload failed: ' + e;
  }
}

async function buildIndex(){
  buildOut.textContent = 'Building...';
  try{
    const r = await fetch('/api/build', { method:'POST' });
    const j = await r.json();
    buildOut.textContent = JSON.stringify(j);
  }catch(e){
    buildOut.textContent = 'Build failed: ' + e;
  }
}

async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  ansOut.textContent = 'Thinking...';
  try{
    const r = await fetch('/api/ask?q=' + encodeURIComponent(q));
    const j = await r.json();
    ansOut.textContent = typeof j.answer === 'string' ? j.answer : JSON.stringify(j, null, 2);
  }catch(e){
    ansOut.textContent = 'Ask failed: ' + e;
  }
}
</script>
</body>
</html>"""
    return HTMLResponse(html)

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Accepts multipart/form-data. Requires `python-multipart` to be installed.
    Saves files into /tmp/uploads on Vercel.
    """
    if not files:
        raise HTTPException(400, "No files attached")
    saved = []
    for f in files:
        if not f.filename:
            continue
        # allow small CSV/TXT/JSON
        name = f.filename
        data = await f.read()
        if len(data) == 0:
            continue
        dest = os.path.join(UPLOAD_DIR, name)
        with open(dest, "wb") as out:
            out.write(data)
        saved.append(name)
    if not saved:
        raise HTTPException(400, "No non-empty files saved")
    return {"ok": True, "saved": saved}

@app.get("/api/health")
def health():
    return {"ok": True, "docs": len(DOCS)}

@app.post("/api/build")
def build():
    """
    Reads all files in /tmp/uploads, converts to docs, builds TF-IDF.
    """
    global DOCS
    DOCS.clear()

    entries = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    if not entries:
        raise HTTPException(400, "No files uploaded yet.")

    for name in entries:
        path = os.path.join(UPLOAD_DIR, name)
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
            lower = name.lower()
            if lower.endswith(".csv"):
                rows = _read_csv_bytes(raw)
            elif lower.endswith(".json"):
                rows = _read_json_bytes(raw)
            else:
                rows = _read_txt_bytes(raw)
            # push into DOCS
            for r in rows:
                DOCS.append({"text": _row_to_text(r)})
        except Exception:
            # skip unreadables
            continue

    if not DOCS:
        raise HTTPException(400, "No readable content from uploaded files.")
    _build_index()
    return {"ok": True, "n_docs": len(DOCS)}

@app.get("/api/ask")
def ask(q: str):
    if not q or not q.strip():
        raise HTTPException(400, "Empty question")
    if not DOCS or not DOC_VECS:
        raise HTTPException(400, "Index not built yet.")
    qv = _query_vec(q)
    # scores
    scored = []
    for i, dv in enumerate(DOC_VECS):
        s = _cosine(qv, dv)
        scored.append((s, i))
    scored.sort(reverse=True)
    top = scored[:5]
    snippets = []
    for s, i in top:
        if s <= 0: 
            continue
        text = DOCS[i]["text"]
        snippets.append({"score": round(s, 4), "text": text[:400]})
    if not snippets:
        return {"answer": "I couldn't find a strong match in the uploaded data."}
    # compose short answer
    best = snippets[0]["text"]
    return {"answer": best, "matches": snippets}
