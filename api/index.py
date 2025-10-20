#!/usr/bin/env python3
# BigQueryGPT-slim — Vercel-ready FastAPI + tiny TF-IDF (no heavy deps)

import os, re, json, math
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import pandas as pd
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote

# ---------------- Env ----------------
def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\u200b\u200c\u200d\u2060\uFEFF]", "", s)
    return s.strip()

HF_MODEL = _clean(os.getenv("HF_MODEL") or "google/flan-t5-small")
HF_USE_TOKEN = (_clean(os.getenv("HF_USE_TOKEN") or "0") == "1")
HF_TOKEN = _clean(os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "")

IS_VERCEL = os.getenv("VERCEL", "") or os.getenv("VERCEL_ENV", "")
UPLOAD_ROOT = "/tmp" if IS_VERCEL else "."
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", f"{UPLOAD_ROOT}/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Perf knobs
FAST_MODE = (_clean(os.getenv("FAST_MODE") or "1") == "1")
HF_TIMEOUT_SEC = int(_clean(os.getenv("HF_TIMEOUT") or "4"))
MAX_NEW_TOKENS = int(_clean(os.getenv("HF_MAX_NEW_TOKENS") or ("64" if FAST_MODE else "128")))
RETRIEVAL_K = int(_clean(os.getenv("RETRIEVAL_K") or ("3" if FAST_MODE else "5")))
MAX_CONTEXT_CHARS = int(_clean(os.getenv("MAX_CONTEXT_CHARS") or ("1200" if FAST_MODE else "2400")))

# Optional Turnstile
TURNSTILE_SITEKEY = _clean(os.getenv("TURNSTILE_SITEKEY") or "")
TURNSTILE_SECRET  = _clean(os.getenv("TURNSTILE_SECRET") or "")

# ---------------- Readers ----------------
def read_generic(path: str, fmt: str = "csv") -> pd.DataFrame:
    p = Path(path)
    fmt = fmt.lower().lstrip(".")
    try:
        if fmt == "csv":     return pd.read_csv(p, low_memory=False)
        if fmt == "json":    return pd.read_json(p)
        if fmt == "xlsx":    return pd.read_excel(p)       # optional: openpyxl
        if fmt == "parquet": return pd.read_parquet(p)     # optional: pyarrow
    except Exception:
        pass
    # Fallback to raw text
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame({"text": [p.read_text(encoding="utf-8", errors="ignore")]})

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
            for ch in _chunk_text(t): docs.append({"text": ch})
        return docs
    for _, row in df.iterrows():
        parts=[f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]) and str(row[c]).strip()]
        for ch in _chunk_text(" | ".join(parts)): docs.append({"text": ch})
    return docs

# ---------------- Tiny TF-IDF (pure Python) ----------------
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
        ts = _tokenize(q); 
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

# ---------------- LLM fallback chain ----------------
def _hf_api_generate(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> str:
    model_clean = HF_MODEL.replace("\\", "/").replace(" ", "")
    url = f"https://api-inference.huggingface.co/models/{quote(model_clean, safe='/-_.')}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}, "options": {"wait_for_model": True}}
    headers = {"Content-Type": "application/json"}
    if HF_USE_TOKEN and HF_TOKEN: headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT_SEC)
    except Exception as e:
        return f"[HF ERROR] Request failed: {e}"
    if r.status_code != 200:
        mode = "with-token" if ("Authorization" in headers) else "unauth"
        return f"[HF ERROR] {r.status_code} {mode} · {r.text[:180].replace(chr(10),' ')}"
    data = r.json()
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]: return data[0]["generated_text"]
    if isinstance(data, list) and data and isinstance(data[0], str): return data[0]
    if isinstance(data, dict) and "generated_text" in data: return data["generated_text"]
    return str(data)

def answer_with_fallbacks(context: str, question: str, concise: bool = True) -> str:
    prompt = (f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 1–2 short sentences."
              if concise else f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly.")
    ans = _hf_api_generate(prompt)
    if ans and not ans.startswith("[HF ERROR]"):
        if concise:
            parts = [s.strip() for s in ans.split(".") if s.strip()]
            ans = (". ".join(parts[:2]) + ('.' if parts else '')) or ans
        return ans
    lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
    return "Top matches:\n- " + "\n- ".join(lines[:2]) if lines else "No matching context."

# ---------------- Server state ----------------
SERVER_STATE: Dict[str, Optional[object]] = {"vstore": None, "docs": None, "df": None, "input_files": None}

# ---------------- CAPTCHA helpers ----------------
def _is_localhost(host_header: str) -> bool:
    host = (host_header or "").split(":")[0].lower().strip()
    return host in {"localhost", "127.0.0.1"}

def turnstile_required(request: Request) -> bool:
    return bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET and not _is_localhost(request.headers.get("host")))

def verify_turnstile_token(token: Optional[str]) -> bool:
    if not TURNSTILE_SECRET:
        return True
    token = (token or "").strip()
    if not token: return False
    try:
        res = requests.post("https://challenges.cloudflare.com/turnstile/v0/siteverify",
                            data={"secret": TURNSTILE_SECRET, "response": token}, timeout=5)
        return bool(res.json().get("success"))
    except Exception:
        return False

# ---------------- FastAPI ----------------
app = FastAPI(title="BigQueryGPT-slim")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# NOTE: On Vercel this function is mounted at /api/*

@app.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...), cf_token: Optional[str] = Form(None)):
    if turnstile_required(request) and not verify_turnstile_token(cf_token):
        raise HTTPException(403, detail="CAPTCHA failed. Please retry.")
    count = 0
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with open(dest, "wb") as out: out.write(await f.read())
        count += 1
    return {"status": "ok", "message": f"{count} file(s) uploaded."}

@app.post("/build_index")
def build_index(payload: Dict = Body(...)):
    path = payload.get("path", str(UPLOAD_DIR))
    p = Path(path)
    if not p.exists(): raise HTTPException(400, detail="Path not found")

    dfs=[]
    for f in p.glob("*.*"):
        try:
            df = read_generic(str(f), fmt=f.suffix.lstrip(".").lower())
            if not df.empty: dfs.append(df)
        except Exception:
            pass
    if not dfs: raise HTTPException(400, detail="No supported data files found.")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    docs = dataframe_to_docs(df)
    vstore = TinyTfidf([d["text"] for d in docs])

    SERVER_STATE["vstore"] = vstore
    SERVER_STATE["docs"] = docs
    SERVER_STATE["df"] = df
    SERVER_STATE["input_files"] = ",".join([x.name for x in p.glob("*")])
    return {"status": "ok", "n_docs": len(docs)}

# extra aliases (covers any front-end path you try)
@app.post("/build")                 # /api/build
def build_alias(payload: Dict = Body(...)): return build_index(payload)

@app.post("/api/build_index")       # /api/api/build_index (if double prefixed by mistake)
def build_compat(payload: Dict = Body(...)): return build_index(payload)

@app.get("/ask")
def ask(q: str, request: Request, top_k: int = 5, concise: bool = True, cf_token: str = ""):
    if turnstile_required(request) and not verify_turnstile_token(cf_token):
        raise HTTPException(403, detail="CAPTCHA failed. Please retry.")
    if not q or not q.strip(): raise HTTPException(400, detail="Empty question")
    vstore: Optional[TinyTfidf] = SERVER_STATE["vstore"]  # type: ignore
    if vstore is None: raise HTTPException(400, detail="Index not built — upload files and click Build Index.")
    hits = vstore.search(q, k=min(top_k, RETRIEVAL_K))
    context = "\n".join([h["text"] for h in hits])[:MAX_CONTEXT_CHARS]
    ans = answer_with_fallbacks(context, q, concise=concise)
    return {"answer": ans, "retrieved": hits[:top_k]}

# Health at function root (exposed at /api on Vercel)
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
        "turnstile_enabled": bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET and not _is_localhost(request.headers.get("host"))),
        "turnstile_sitekey": TURNSTILE_SITEKEY if (TURNSTILE_SITEKEY and TURNSTILE_SECRET) else None,
    }

# Route inspector (debug)
@app.get("/routes")
def routes():
    return sorted([f"{getattr(r,'methods',{'GET'})} {getattr(r,'path','')}" for r in app.routes])
