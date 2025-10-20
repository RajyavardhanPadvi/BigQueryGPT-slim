# api/index.py
#!/usr/bin/env python3
import os, json, tempfile, re
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib, requests, difflib

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote

# ---------------- Env ----------------
try:
    from dotenv import load_dotenv
    load_dotenv("GPT.env")
except Exception:
    pass

def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\u200b\u200c\u200d\u2060\uFEFF]", "", s)
    return s.strip()

HF_MODEL = _clean(os.getenv("HF_MODEL") or "google/flan-t5-small")
HF_USE_TOKEN = (_clean(os.getenv("HF_USE_TOKEN") or "0") == "1")
HF_TOKEN = _clean(os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "")

MLFLOW_TRACKING_URI = _clean(os.getenv("MLFLOW_TRACKING_URI") or "")
IS_VERCEL = os.getenv("VERCEL", "") or os.getenv("VERCEL_ENV", "")
UPLOAD_ROOT = "/tmp" if IS_VERCEL else "."
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", f"{UPLOAD_ROOT}/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Performance knobs
FAST_MODE = (_clean(os.getenv("FAST_MODE") or "1") == "1")
HF_TIMEOUT_SEC = int(_clean(os.getenv("HF_TIMEOUT") or "4"))
LOCAL_FALLBACK = (_clean(os.getenv("LOCAL_FALLBACK") or "0") == "1")
MAX_NEW_TOKENS = int(_clean(os.getenv("HF_MAX_NEW_TOKENS") or ("64" if FAST_MODE else "128")))
RETRIEVAL_K = int(_clean(os.getenv("RETRIEVAL_K") or ("3" if FAST_MODE else "5")))
MAX_CONTEXT_CHARS = int(_clean(os.getenv("MAX_CONTEXT_CHARS") or ("1200" if FAST_MODE else "2400")))

# Turnstile (optional)
TURNSTILE_SITEKEY = _clean(os.getenv("TURNSTILE_SITEKEY") or "")
TURNSTILE_SECRET  = _clean(os.getenv("TURNSTILE_SECRET") or "")

# ---------------- Optional MLflow ----------------
HAS_MLFLOW = False
try:
    if MLFLOW_TRACKING_URI:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

# ---------------- Optional readers for DOCX/PDF ----------------
try:
    import docx as _docx
except Exception:
    _docx = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ---------------- Fallback readers ----------------
def read_generic(path: str, fmt: str = "csv") -> pd.DataFrame:
    p = Path(path)
    fmt = fmt.lower().lstrip(".")
    if fmt == "csv":     return pd.read_csv(p, low_memory=False)
    if fmt == "json":    return pd.read_json(p)
    if fmt == "xlsx":    return pd.read_excel(p)
    if fmt == "parquet": return pd.read_parquet(p)
    if fmt == "docx":
        if _docx is None:
            raise ImportError("python-docx not installed. pip install python-docx")
        D = _docx.Document(str(p))
        paras = [para.text.strip() for para in D.paragraphs if para.text and para.text.strip()]
        return pd.DataFrame({"text": paras})
    if fmt == "pdf":
        if fitz is None:
            raise ImportError("PyMuPDF not installed. pip install pymupdf")
        doc = fitz.open(str(p))
        pages = [pg.get_text().strip() for pg in doc]
        return pd.DataFrame({"text": pages})
    return pd.read_csv(p, low_memory=False)

def _chunk_text(text: str, chunk_size=800, overlap=200):
    if not text: return []
    out=[]; i=0; L=len(text)
    while i < L:
        j=min(i+chunk_size, L); out.append(text[i:j])
        if j>=L: break
        i=max(0, j-overlap)
    return out

def dataframe_to_docs(df: pd.DataFrame):
    docs=[]
    if list(df.columns) == ["text"]:
        for _, t in df["text"].fillna("").astype(str).items():
            for ch in _chunk_text(t):
                docs.append({"text": ch})
        return docs
    for _, row in df.iterrows():
        parts=[f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]) and str(row[c]).strip()]
        row_text = " | ".join(parts)
        for ch in _chunk_text(row_text):
            docs.append({"text": ch})
    return docs

# ---------------- Turnstile helpers ----------------
def _is_localhost(host_header: str) -> bool:
    host = (host_header or "").split(":")[0].lower().strip()
    return host in {"localhost", "127.0.0.1"}

def turnstile_required(request: Request) -> bool:
    return bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET and not _is_localhost(request.headers.get("host")))

def verify_turnstile_token(token: Optional[str]) -> bool:
    if not TURNSTILE_SECRET:
        return True
    token = (token or "").strip()
    if not token:
        return False
    try:
        res = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={"secret": TURNSTILE_SECRET, "response": token},
            timeout=5,
        )
        return bool(res.json().get("success"))
    except Exception:
        return False

# ---------------- Vector store ----------------
class TfidfVectorStore:
    def __init__(self, docs: List[Dict]):
        self.docs  = docs
        self.texts = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors    = self.vectorizer.fit_transform(self.texts) if self.texts else None
    def search(self, q: str, top_k: int = 5):
        if self.vectors is None or not self.texts: return []
        qv   = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.vectors).flatten()
        idxs = sims.argsort()[-top_k:][::-1]
        return [{"text": self.texts[i], "score": float(sims[i])} for i in idxs]

# ---------------- LLMs: API → Local → Extractive ----------------
_LOCAL_PIPE = None

def _local_generate(prompt: str, max_new_tokens: int = 256) -> str:
    global _LOCAL_PIPE
    try:
        if _LOCAL_PIPE is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
            mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            _LOCAL_PIPE = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
        out = _LOCAL_PIPE(prompt, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1)
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return out[0]["generated_text"]
        return str(out)
    except Exception as e:
        return f"[LOCAL ERROR] {e}"

def _extractive_fallback(context: str) -> str:
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    if not lines:
        return "No matching context."
    return "Top matches:\n- " + "\n- ".join(lines[:2])

def _hf_api_generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    model_clean = HF_MODEL.replace("\\", "/").replace(" ", "")
    url = f"https://api-inference.huggingface.co/models/{quote(model_clean, safe='/-_.')}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        "options": {"wait_for_model": True},
    }
    headers = {"Content-Type": "application/json"}
    if HF_USE_TOKEN and HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT_SEC)
    except Exception as e:
        return f"[HF ERROR] Request failed: {e}"
    if r.status_code != 200:
        mode = "with-token" if ("Authorization" in headers) else "unauth"
        return f"[HF ERROR] {r.status_code} at {url} · {mode} · resp='{r.text[:300].replace(chr(10),' ')}'"
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data[0]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)

def answer_with_fallbacks(context: str, question: str, concise: bool = True) -> str:
    prompt = (f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 1–2 short sentences."
              if concise else f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly.")
    ans = _hf_api_generate(prompt)
    if not ans.startswith("[HF ERROR]"):
        if concise:
            parts = [s.strip() for s in ans.split(".") if s.strip()]
            ans = (". ".join(parts[:2]) + ('.' if parts else '')) or ans
        return ans
    if LOCAL_FALLBACK:
        local = _local_generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        if not local.startswith("[LOCAL ERROR]"):
            if concise:
                parts = [s.strip() for s in local.split(".") if s.strip()]
                local = (". ".join(parts[:2]) + ('.' if parts else '')) or local
            return local
    return _extractive_fallback(context)

# ---------------- Optional MLflow ----------------
def mlflow_log_index(input_files: str, docs: List[Dict], vstore) -> Optional[str]:
    if not HAS_MLFLOW: return None
    try:
        tmp = tempfile.mkdtemp(prefix="dataspark_art_")
        vec_path  = os.path.join(tmp, "tfidf.joblib")
        docs_path = os.path.join(tmp, "docs.json")
        try: joblib.dump(getattr(vstore, "vectorizer", None), vec_path)
        except Exception: open(vec_path, "wb").close()
        import mlflow
        with mlflow.start_run(run_name="dataspark_index_build") as run:
            mlflow.log_param("input_files", input_files)
            mlflow.log_metric("n_docs", len(docs))
            try:
                mlflow.log_artifact(vec_path,  artifact_path="artifacts")
                mlflow.log_artifact(docs_path, artifact_path="artifacts")
            except Exception: pass
            return run.info.run_id
    except Exception:
        return None

# ---------------- Server state ----------------
SERVER_STATE = {"vstore": None, "docs": None, "df": None, "input_files": None}

# ---------------- FastAPI ----------------
app = FastAPI(title="BigQueryGPT (no-auth)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# NOTE: these run under /api/* on Vercel

@app.post("/upload")
async def upload(
    request: Request,
    files: List[UploadFile] = File(...),
    cf_token: Optional[str] = Form(None),
):
    if turnstile_required(request):
        if not verify_turnstile_token(cf_token):
            raise HTTPException(403, detail="CAPTCHA failed. Please retry.")
    count = 0
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with open(dest, "wb") as out:
            out.write(await f.read())
        count += 1
    return {"status": "ok", "message": f"{count} file(s) uploaded."}

@app.post("/build_index")
def build_index(payload: Dict):
    path = payload.get("path", str(UPLOAD_DIR))
    p = Path(path)
    if not p.exists(): raise HTTPException(400, detail="Path not found")

    dfs=[]
    for f in p.glob("*.*"):
        fmt = f.suffix.lstrip(".").lower()
        try:
            df = read_generic(str(f), fmt=fmt)
            if not df.empty: dfs.append(df)
        except Exception:
            pass
    if not dfs: raise HTTPException(400, detail="No supported data files found.")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    docs = dataframe_to_docs(df)
    vstore = TfidfVectorStore(docs)

    SERVER_STATE["vstore"] = vstore
    SERVER_STATE["docs"] = docs
    SERVER_STATE["df"] = df
    SERVER_STATE["input_files"] = ",".join([x.name for x in p.glob("*")])
    run_id = mlflow_log_index(SERVER_STATE["input_files"], docs, vstore)
    return {"status": "ok", "n_docs": len(docs), "mlflow_run": run_id}

@app.get("/ask")
def ask(q: str, request: Request, top_k: int = 5, concise: bool = True, cf_token: str = ""):
    if turnstile_required(request):
        if not verify_turnstile_token(cf_token):
            raise HTTPException(403, detail="CAPTCHA failed. Please retry.")
    if not q or not q.strip(): raise HTTPException(400, detail="Empty question")
    vstore = SERVER_STATE["vstore"]
    if vstore is None: raise HTTPException(400, detail="Index not built — upload files and click Build Index.")

    df = SERVER_STATE.get("df")
    if df is not None and not df.empty:
        # quick data-first answers omitted for brevity; you can keep your version here if desired
        pass

    hits = vstore.search(q, top_k=min(top_k, RETRIEVAL_K))
    context = "\n".join([h["text"] for h in hits])[:MAX_CONTEXT_CHARS]
    ans = answer_with_fallbacks(context, q, concise=concise)
    return {"answer": ans, "retrieved": hits[:top_k]}

# HEALTH AT ROOT (so it shows up at /api on Vercel)
@app.get("/")
def health_api(request: Request):
    ready = SERVER_STATE.get("vstore") is not None
    model_clean = HF_MODEL.replace("\\", "/").replace(" ", "")
    hf_url = f"https://api-inference.huggingface.co/models/{quote(model_clean, safe='/-_.')}"
    using_auth = bool(HF_USE_TOKEN and HF_TOKEN)
    return {
        "ok": "up",
        "index_ready": bool(ready),
        "hf_model": HF_MODEL,
        "hf_url": hf_url,
        "using_auth": using_auth,
        "mlflow": "on" if HAS_MLFLOW else "off",
        "turnstile_enabled": bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET and not _is_localhost(request.headers.get("host"))),
        "turnstile_sitekey": TURNSTILE_SITEKEY if (TURNSTILE_SITEKEY and TURNSTILE_SECRET) else None,
    }

# ---------------- Inline UI at /ui (NO f-string!) ----------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>BigQueryGPT</title>
<style>
body{font-family:Inter,Arial;background:#071027;color:#e6f7fb;margin:0}
.wrap{max-width:960px;margin:24px auto;padding:0 16px}
.hint{color:#9fd6e6;font-size:0.9rem}
.btn{background:#2ec4ff;border:none;padding:8px 12px;border-radius:8px;cursor:pointer}
.zone{margin-top:14px;border:2px dashed #2ec4ff;padding:18px;border-radius:12px;text-align:center}
.chat{margin-top:20px;background:#071827;border:1px solid #123245;border-radius:10px;padding:12px}
.msg{margin:8px 0;padding:8px;border-radius:8px;max-width:80%}
.user{margin-left:auto;background:#0e2f4a}
.bot{margin-right:auto;background:#0b1f33}
.row{display:flex;gap:8px;align-items:center}
input[type=text]{flex:1;border-radius:8px;border:1px solid #2b4d66;background:#0b1b2b;color:#e6f7fb;padding:8px}
pre{white-space:pre-wrap;color:#bfefff}
.ts-wrap{margin-top:14px}
</style>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
</head>
<body>
<div class="wrap">
  <h2>BigQueryGPT</h2>
  <p class="hint" id="cfg">Loading…</p>

  <div id="tswrap" class="ts-wrap" style="display:none">
    <div class="cf-turnstile" data-sitekey="__SITEKEY__" data-theme="dark" data-callback="onTsSolved"></div>
  </div>

  <div class="zone" ondrop="dropHandler(event)" ondragover="dragOverHandler(event)" onclick="fileInput.click()">
    Drop files here or click to select
    <input id="fileInput" type="file" multiple style="display:none" onchange="uploadFiles(this.files)"/>
  </div>
  <pre id="files" class="hint"></pre>

  <div style="margin-top:8px">
    <button class="btn" onclick="buildIndex()">Build Index</button>
    <span id="buildOut" class="hint"></span>
  </div>

  <div class="chat" id="chatBox">
    <div id="msgs"></div>
    <div class="row" style="margin-top:8px">
      <input id="q" type="text" placeholder="Ask something about your uploaded data..."/>
      <button class="btn" onclick="ask()">Ask</button>
    </div>
  </div>
</div>

<script>
let TURNSTILE_ENABLED = false;
let TS_TOKEN = "";
function onTsSolved(token) { TS_TOKEN = token; }

const qInput = document.getElementById('q');
qInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); ask(); }
});

const cfg = document.getElementById('cfg');
const fileInput = document.getElementById('fileInput');
const filesOut  = document.getElementById('files');
const msgs      = document.getElementById('msgs');

async function loadCfg(){
  try {
    const r = await fetch('/api');
    const j = await r.json();
    cfg.textContent = '1) Drop your files → 2) Build Index → 3) Ask your data anything.';
    if (j.turnstile_enabled && j.turnstile_sitekey) {
      TURNSTILE_ENABLED = true;
      document.getElementById('tswrap').style.display = 'block';
    }
  } catch(e) {
    cfg.textContent = 'Server not ready.';
  }
}
loadCfg();

function dragOverHandler(event){ event.preventDefault(); }
function dropHandler(event){ event.preventDefault(); uploadFiles(event.dataTransfer.files); }

async function getTurnstileToken(timeoutMs=2000) {
  if (!TURNSTILE_ENABLED) return "";
  const step=100; let waited=0;
  while (!TS_TOKEN && waited < timeoutMs) {
    await new Promise(r => setTimeout(r, step));
    waited += step;
  }
  return TS_TOKEN || "";
}

async function uploadFiles(fs){
  let fd = new FormData();
  for (let f of fs) fd.append('files', f);
  const token = await getTurnstileToken();
  if (TURNSTILE_ENABLED && !token) {
    alert('Please complete the CAPTCHA first.');
    return;
  }
  fd.append('cf_token', token);
  const r = await fetch('/api/upload', { method:'POST', body: fd });
  const j = await r.json();
  filesOut.textContent = j.message || JSON.stringify(j);
}

async function buildIndex(){
  document.getElementById('buildOut').textContent = "Building...";
  const r = await fetch('/api/build_index', {
    method:'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: './uploads' })
  });
  const j = await r.json();
  document.getElementById('buildOut').textContent = (j.status ? "Index ready ("+j.n_docs+" docs)" : (j.detail||j.error||JSON.stringify(j)));
}

function addMsg(txt, who) {
  const d = document.createElement('div');
  d.className = 'msg ' + (who==='user' ? 'user' : 'bot');
  d.textContent = txt;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}

async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  addMsg(q, 'user');
  addMsg('Thinking...', 'bot');
  const token = await getTurnstileToken();
  const url = '/api/ask?q=' + encodeURIComponent(q) + (TURNSTILE_ENABLED ? '&cf_token=' + encodeURIComponent(token) : '');
  const r = await fetch(url);
  const j = await r.json();
  const a = j.answer;
  msgs.lastChild.textContent = (typeof a === 'string') ? a : JSON.stringify(a, null, 2);
}
</script>
</body></html>"""
    return HTMLResponse(html.replace("__SITEKEY__", TURNSTILE_SITEKEY or ""))
