#!/usr/bin/env python3
# BigQueryGPT — Render FastAPI, tiny TF-IDF, with debug endpoints + faster timeouts

import os, re, json, math
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import pandas as pd
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from urllib.parse import quote

# ---------- config ----------
def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\u200b\u200c\u200d\u2060\uFEFF]", "", s)
    return s.strip()

HF_MODEL = _clean(os.getenv("HF_MODEL") or "google/flan-t5-small")
HF_USE_TOKEN = (_clean(os.getenv("HF_USE_TOKEN") or "0") == "1")
HF_TOKEN = _clean(os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = UPLOAD_DIR / "index.json"

# tight + predictable
HF_TIMEOUT_SEC = int(_clean(os.getenv("HF_TIMEOUT") or "3"))
MAX_NEW_TOKENS = int(_clean(os.getenv("HF_MAX_NEW_TOKENS") or "64"))
RETRIEVAL_K = int(_clean(os.getenv("RETRIEVAL_K") or "3"))
MAX_CONTEXT_CHARS = int(_clean(os.getenv("MAX_CONTEXT_CHARS") or "1200"))

# ---------- readers ----------
def read_generic(path: str, fmt: str = "csv") -> pd.DataFrame:
    p = Path(path)
    fmt = fmt.lower().lstrip(".")
    try:
        if fmt == "csv":     return pd.read_csv(p, low_memory=False)
        if fmt == "json":    return pd.read_json(p)
        if fmt == "xlsx":    return pd.read_excel(p)       # optional
        if fmt == "parquet": return pd.read_parquet(p)     # optional
        if fmt in {"txt", "md"}:
            return pd.DataFrame({"text": [p.read_text(encoding="utf-8", errors="ignore")]})
    except Exception:
        pass
    # fallback
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame({"text": [p.read_text(encoding="utf-8", errors="ignore")]})

def _chunk_text(text: str, chunk_size=500, overlap=100):
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
        for ch in _chunk_text(" | ".join(parts)):
            docs.append({"text": ch})
    return docs

# ---------- tiny TF-IDF ----------
def _tokenize(s: str) -> List[str]:
    return [t for t in re.sub(r"[^a-zA-Z0-9]+", " ", str(s)).lower().split() if t]

class TinyTfidf:
    def __init__(self, docs: List[str]):
        self.docs = docs
        toks = [_tokenize(d) for d in docs]
        self.N = len(toks)
        df = Counter()
        for ts in toks: df.update(set(ts))
        self.idf = {t: math.log((1 + self.N) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = []
        for ts in toks:
            tf = Counter(ts); L = max(1, len(ts))
            self.vecs.append({t: (tf[t]/L) * self.idf.get(t, 0.0) for t in tf})

    def _qvec(self, q: str) -> Dict[str, float]:
        ts = _tokenize(q)
        if not ts: return {}
        tf = Counter(ts); L = len(ts)
        return {t: (tf[t]/L) * self.idf.get(t, 0.0) for t in tf}

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b: return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k,0.0)*b.get(k,0.0) for k in keys)
        na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0.0

    def search(self, q: str, k: int = 5):
        qv = self._qvec(q)
        scores = [(i, self._cos(qv, self.vecs[i])) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"text": self.docs[i], "score": s} for i, s in scores[:max(1,k)]]

# ---------- LLM fallback ----------
def _hf_api_generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    model_clean = HF_MODEL.replace("\\", "/").replace(" ", "")
    url = f"https://api-inference.huggingface.co/models/{quote(model_clean, safe='/-_.')}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        "options": {"wait_for_model": False}
    }
    headers = {"Content-Type": "application/json"}
    if HF_USE_TOKEN and HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT_SEC)
    except Exception as e:
        return f"[HF ERROR] Request failed: {e}"
    if r.status_code != 200:
        return f"[HF ERROR] {r.status_code} · {r.text[:160].replace(chr(10),' ')}"
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]: return data[0]["generated_text"]
    if isinstance(data, list) and data and isinstance(data[0], str): return data[0]
    if isinstance(data, dict) and "generated_text" in data: return data["generated_text"]
    return str(data)

def answer_with_fallbacks(context: str, question: str, concise: bool = True, fast: bool = True) -> str:
    prompt = (f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 1–2 short sentences."
              if concise else f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly.")
    if not fast:
        ans = _hf_api_generate(prompt)
        if ans and not ans.startswith("[HF ERROR]"):
            if concise:
                parts = [s.strip() for s in ans.split(".") if s.strip()]
                ans = (". ".join(parts[:2]) + ('.' if parts else '')) or ans
            return ans
    # Extractive fallback
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    return "Top matches:\n- " + "\n- ".join(lines[:2]) if lines else "No matching context."

# ---------- cache helpers ----------
def save_index(docs: List[Dict], vstore: TinyTfidf):
    try:
        obj = {"docs": docs}
        INDEX_PATH.write_text(json.dumps(obj), encoding="utf-8")
    except Exception:
        pass

def load_index():
    if not INDEX_PATH.exists(): return None
    try:
        obj = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        docs = obj.get("docs") or []
        v = TinyTfidf([d["text"] for d in docs])
        return docs, v
    except Exception:
        return None

# ---------- app ----------
app = FastAPI(title="BigQueryGPT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=512)

@app.on_event("startup")
def _load_cached_index():
    cached = load_index()
    if cached:
        docs, vstore = cached
        SERVER_STATE["docs"] = docs
        SERVER_STATE["vstore"] = vstore
        print(f"[startup] Loaded cached index with {len(docs)} docs")

SERVER_STATE: Dict[str, Optional[object]] = {"vstore": None, "docs": None}

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {
        "ok": "up",
        "index_ready": SERVER_STATE.get("vstore") is not None,
        "n_docs": len(SERVER_STATE.get("docs") or []),
        "upload_dir": str(UPLOAD_DIR),
    }

# --- DEBUG: list files server can see ---
@app.get("/debug/uploads")
def debug_uploads():
    files = [p.name for p in UPLOAD_DIR.glob("*")]
    return {"dir": str(UPLOAD_DIR), "files": files}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    count = 0
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with open(dest, "wb") as out:
            out.write(await f.read())
        count += 1
    print(f"[upload] saved {count} file(s)")
    return {"status": "ok", "message": f"{count} file(s) uploaded."}

@app.post("/build_index")
def build_index(payload: Dict = Body(...)):
    path = payload.get("path", str(UPLOAD_DIR))
    p = Path(path)
    if not p.exists():
        raise HTTPException(400, detail="Path not found")

    files = list(p.glob("*.*"))
    if not files:
        # crisp error for UI instead of hanging
        return JSONResponse({"status": "error", "detail": "No files in ./uploads. Upload first."}, status_code=400)

    dfs=[]
    for f in files:
        try:
            df = read_generic(str(f), fmt=f.suffix.lstrip(".").lower())
            if not df.empty: dfs.append(df)
        except Exception as e:
            print(f"[build_index] skip {f.name}: {e}")
    if not dfs:
        return JSONResponse({"status":"error","detail":"No supported data files found."}, status_code=400)

    df = pd.concat(dfs, ignore_index=True, sort=False)
    docs = dataframe_to_docs(df)
    vstore = TinyTfidf([d["text"] for d in docs])

    SERVER_STATE["vstore"] = vstore
    SERVER_STATE["docs"] = docs
    save_index(docs, vstore)

    print(f"[build_index] built {len(docs)} docs from {len(files)} file(s)")
    return {"status": "ok", "n_docs": len(docs)}

@app.get("/ask")
def ask(
    q: str = Query(..., description="Your question"),
    top_k: int = Query(3, ge=1, le=10),
    concise: bool = Query(True),
    fast: int = Query(1, description="1 = skip slow LLM if needed")
):
    if not q or not q.strip():
        raise HTTPException(400, detail="Empty question")
    vstore: Optional[TinyTfidf] = SERVER_STATE["vstore"]  # type: ignore
    if vstore is None:
        raise HTTPException(400, detail="Index not built — upload files and click Build Index.")
    hits = vstore.search(q, k=min(top_k, RETRIEVAL_K))
    context = "\n".join([h["text"] for h in hits])[:MAX_CONTEXT_CHARS]
    ans = answer_with_fallbacks(context, q, concise=concise, fast=(fast == 1))
    return {"answer": ans, "retrieved": hits[:top_k]}
