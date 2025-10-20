from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from upstash_redis import Redis
from collections import Counter
import math, os, json

app = FastAPI(title="BigQueryGPT-slim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel KV (Upstash Redis) – add the Vercel KV integration
redis = Redis(
    url=os.environ.get("KV_REST_API_URL", ""),
    token=os.environ.get("KV_REST_API_TOKEN", "")
)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in "".join(c if c.isalnum() else " " for c in text).split() if t]

def build_tfidf(docs: List[str]) -> Dict[str, Any]:
    tokenized = [tokenize(d) for d in docs]
    N = len(tokenized)
    df = Counter()
    for terms in tokenized:
        df.update(set(terms))
    idf = {term: math.log((1 + N) / (1 + dfreq)) + 1.0 for term, dfreq in df.items()}
    vectors = []
    for terms in tokenized:
        tf = Counter(terms)
        vec = {term: (tf[term] / max(1, len(terms))) * idf.get(term, 0.0) for term in tf}
        vectors.append(vec)
    return {"docs": docs, "idf": idf, "vectors": vectors, "vocab": list(idf.keys())}

def cosine_sim(qvec: Dict[str, float], dvec: Dict[str, float]) -> float:
    keys = set(qvec) | set(dvec)
    dot = sum(qvec.get(k, 0.0) * dvec.get(k, 0.0) for k in keys)
    qn = math.sqrt(sum(v*v for v in qvec.values()))
    dn = math.sqrt(sum(v*v for v in dvec.values()))
    return dot / (qn * dn) if qn and dn else 0.0

def vectorize_query(q: str, idf: Dict[str, float]) -> Dict[str, float]:
    terms = tokenize(q)
    if not terms: return {}
    tf = Counter(terms)
    return {t: (tf[t] / len(terms)) * idf.get(t, 0.0) for t in tf}

def key_docs(project_id: str) -> str: return f"bqgpt:docs:{project_id}"
def key_index(project_id: str) -> str: return f"bqgpt:index:{project_id}"
def key_hist(project_id: str, session_id: str) -> str: return f"bqgpt:hist:{project_id}:{session_id}"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload_files(project_id: str = Form(...), files: List[UploadFile] = File(...)):
    texts = []
    for f in files:
        content = await f.read()
        texts.append(content.decode("utf-8", errors="ignore"))
    existing = await redis.get(key_docs(project_id))
    docs = (json.loads(existing) if existing else []) + texts
    await redis.set(key_docs(project_id), json.dumps(docs))
    await redis.del_(key_index(project_id))  # invalidate previous index
    return {"uploaded": len(texts), "total_docs": len(docs)}

@app.post("/build")
async def build_index(project_id: str = Form(...)):
    stored = await redis.get(key_docs(project_id))
    docs = json.loads(stored) if stored else []
    if not docs:
        raise HTTPException(status_code=400, detail="No documents uploaded.")
    index = build_tfidf(docs)
    await redis.set(key_index(project_id), json.dumps(index))
    return {"ok": True, "docs_indexed": len(docs), "vocab_size": len(index["vocab"])}

@app.post("/chat")
async def chat(project_id: str = Form(...), session_id: str = Form(...), message: str = Form(...), top_k: int = Form(4)):
    raw = await redis.get(key_index(project_id))
    if not raw:
        raise HTTPException(status_code=400, detail="Index not built. Call /api/build first.")
    index = json.loads(raw)
    qvec = vectorize_query(message, index["idf"])
    scores = [(i, cosine_sim(qvec, index["vectors"][i])) for i in range(len(index["docs"]))]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [{"doc_id": i, "score": s, "snippet": index["docs"][i][:500]} for i, s in scores[:max(1, top_k)]]
    # (optional) keep a simple session history
    hist = json.loads(await redis.get(key_hist(project_id, session_id)) or "[]")
    hist.append({"role": "user", "content": message})
    hist.append({"role": "context", "content": "\n\n".join(t["snippet"] for t in top)})
    await redis.set(key_hist(project_id, session_id), json.dumps(hist))
    return {"results": top, "history_len": len(hist)}
