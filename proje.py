# proje.py — Fırat Üniversitesi Chatbot (sağlam sürüm)
# FastAPI + pdfplumber + sentence-transformers + FAISS + BM25 + Cross-Encoder + MMR
# Kısa ve net cevaplar, sağlam başlatma ve yeniden indexleme

import os, re, pickle, threading
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pdfplumber
import faiss

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------- Ortam ----------
load_dotenv()
BOT_TITLE   = os.getenv("BOT_TITLE", "Fırat Üniversitesi Chatbot")
DOCS_DIR    = Path(os.getenv("DOCS_DIR", "docs"))
INDEX_PATH  = Path(os.getenv("INDEX_PATH", "faiss.index"))
META_PATH   = Path(os.getenv("META_PATH", "meta.pkl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL= os.getenv("RERANK_MODEL","cross-encoder/ms-marco-MiniLM-L-12-v2")
RERANK_ON   = (os.getenv("RERANK_ON","1") != "0")  # istersen kapat: RERANK_ON=0
DOCS_DIR.mkdir(exist_ok=True)

# ---------- Modeller ----------
from sentence_transformers import SentenceTransformer, CrossEncoder
_SBERT = SentenceTransformer(EMBED_MODEL)
_RANK  = CrossEncoder(RERANK_MODEL) if RERANK_ON else None

def get_embedding(text: str) -> np.ndarray:
    v = _SBERT.encode([text], normalize_embeddings=True)[0]
    return np.array(v, dtype=np.float32)

# ---------- PDF -> metin -> chunk ----------
_SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+')
_TR_TOKEN   = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")

def extract_text_from_pdf(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            if txt.strip():
                out.append((txt, f"{path.name} s:{i}"))
    return out

def chunk_text(text: str, source: str, max_chars=850, overlap=140) -> List[Tuple[str, str]]:
    sents = _SENT_SPLIT.split(text)
    blocks, cur = [], ""
    for s in sents:
        s = (s or "").strip()
        if not s: continue
        if len(cur) + len(s) <= max_chars:
            cur += (" " if cur else "") + s
        else:
            if cur: blocks.append(cur)
            cur = s
    if cur: blocks.append(cur)

    final: List[Tuple[str, str]] = []
    for i, c in enumerate(blocks):
        if i > 0:
            prev = blocks[i-1]
            tail = prev[-overlap:] if len(prev)>overlap else prev
            final.append(((tail + " " + c).strip(), source))
        else:
            final.append((c, source))
    return final

# ---------- BM25 ----------
from rank_bm25 import BM25Okapi
def tokenize_tr(s: str) -> List[str]:
    return [w.lower() for w in _TR_TOKEN.findall(s)]

# ---------- İndeks ----------
_index = None
_chunks: List[Tuple[str,str]] | None = None
_lock = threading.Lock()

def list_pdfs() -> List[Path]:
    return [p for p in DOCS_DIR.rglob("*.pdf")]

def build_index():
    pdf_files = list_pdfs()
    if not pdf_files:
        raise RuntimeError(f"'{DOCS_DIR}' klasöründe PDF bulunamadı.")

    chunks: List[Tuple[str, str]] = []
    for pdf in sorted(pdf_files):
        try:
            for page_txt, src in extract_text_from_pdf(pdf):
                chunks.extend(chunk_text(page_txt, src))
        except Exception as e:
            print(f"[WARN] {pdf} okunamadı: {e}")

    if not chunks:
        raise RuntimeError("PDF'lerden metin çıkarılamadı.")

    embs = [get_embedding(t) for t,_ in chunks]
    dim  = embs[0].shape[0]
    arr  = np.stack(embs)

    index = faiss.IndexFlatIP(dim)  # normalize_embeddings=True → cosine = IP
    index.add(arr)

    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("wb") as f:
        pickle.dump(chunks, f)

    print(f"[INDEX] {len(chunks)} parça işlendi | PDF sayısı: {len(pdf_files)}")

def load_index():
    if not (INDEX_PATH.exists() and META_PATH.exists()):
        raise RuntimeError("Index dosyaları yok. build_index() gerekli.")
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("rb") as f:
        chunks: List[Tuple[str,str]] = pickle.load(f)
    return index, chunks

# ---------- Hibrit alma + Rerank + MMR ----------
def rrf_fuse(ranks_lists: List[List[int]], k: int = 60, topk: int = 50) -> List[int]:
    scores: dict[int,float] = {}
    for ranks in ranks_lists:
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx,0.0) + 1.0/(k+(r+1))
    return [i for i,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)][:topk]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a,b))

def mmr_select(query_vec: np.ndarray, cand_vecs: List[np.ndarray], cand_ids: List[int], lam: float=0.70, k: int=8) -> List[int]:
    selected: List[int] = []
    pool = set(range(len(cand_ids)))
    sims_q = np.array([cosine(query_vec, v) for v in cand_vecs])
    while pool and len(selected) < k:
        best_i, best = None, -1e9
        for i in list(pool):
            div = 0.0
            if selected:
                div = max(cosine(cand_vecs[i], cand_vecs[j]) for j in selected)
            score = lam * sims_q[i] - (1-lam) * div
            if score > best:
                best, best_i = score, i
        selected.append(best_i); pool.remove(best_i)
    return [cand_ids[i] for i in selected]

def retrieve(query: str, index, chunks: List[Tuple[str,str]], topk: int=8) -> List[Tuple[str,str]]:
    docs = [t for t,_ in chunks]

    # 1) FAISS
    qv = get_embedding(query)
    D,I = index.search(np.expand_dims(qv, axis=0), 60)
    vec_ids = list(I[0])

    # 2) BM25
    bm25 = BM25Okapi([tokenize_tr(d) for d in docs])
    kw_scores = bm25.get_scores(tokenize_tr(query))
    kw_ids = list(np.argsort(-kw_scores))[:120]

    # 3) RRF
    fused = rrf_fuse([vec_ids, kw_ids], topk=80)

    # 4) Rerank (opsiyonel)
    if RERANK_ON and _RANK is not None:
        pairs = [(query, docs[i]) for i in fused]
        scores = _RANK.predict(pairs)
        fused = [fid for _, fid in sorted(zip(scores, fused), key=lambda x:x[0], reverse=True)][:24]

    # 5) MMR çeşitlilik
    cand_vecs = [get_embedding(docs[i]) for i in fused]
    mmr_ids = mmr_select(qv, cand_vecs, fused, lam=0.70, k=topk)

    return [chunks[i] for i in mmr_ids]

# ---------- Özetleyici (kısa & net) ----------
DOMAIN_HINTS = [
    "sınav","vize","final","bütünleme","yarıyıl","mazeret","devam","yoklama",
    "başarı","not","bağıl","katkı","yüzde","ödev","proje","ders","devamsızlık",
    "harf","puan","baraj","telafi","süre","kredi","geçme","ortalama"
]
def _split_sents(text: str) -> List[str]:
    sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]

def _score_sent(q: str, s: str) -> int:
    low = s.lower()
    score = 0
    for w in set(tokenize_tr(q)):
        if w in low: score += 3
    for dom in DOMAIN_HINTS:
        if dom in low: score += 1
    if re.search(r'(%|\byüzde\b|\b\d{1,3}\b|\b[0-3]?[0-9]\s*(?:gün|hafta|ay)\b)', low):
        score += 1
    if re.search(r'(?:Madde|MADDE|Md\.?)\s*\d+', s):  # madde ipucu
        score += 1
    return score

def _dedupe_keep_order(sents: List[str], thr: float=0.60) -> List[str]:
    kept, seen = [], []
    for s in sents:
        ws = set(tokenize_tr(s))
        if not ws: continue
        dup = any((len(ws & w2)/len(ws | w2) >= thr) for w2 in seen)
        if not dup: kept.append(s); seen.append(ws)
    return kept

def summarize(question: str, contexts: List[str], mode: str="kisa", short_n: int=3, long_n: int=6, line_max: int=140) -> str:
    if not contexts:
        return "Uygun bir hüküm bulunamadı. Resmî duyurulara/bölüm sayfasına bakın."
    sents = []
    for c in contexts: sents.extend(_split_sents(c))
    scored = sorted(sents, key=lambda s: _score_sent(question, s), reverse=True)
    scored = _dedupe_keep_order(scored)
    limit = short_n if (mode or "kisa").lower()=="kisa" else long_n
    top = scored[:limit]
    if not top:
        return "Net bir maddeye rastlanmadı; bölüm duyurularını kontrol edin."
    out = []
    for s in top:
        s = s.replace("\n"," ").strip()
        if len(s) > line_max: s = s[:line_max-3].rstrip()+"..."
        out.append(f"• {s}")
    return "\n".join(out)

def extract_maddeler(texts: List[str], max_items: int=10) -> List[str]:
    found: list[str] = []
    for t in texts:
        for m in re.findall(r'(?:Madde|MADDE|Md\.?)\s*(\d+)', t):
            item = f"Madde {m}"
            if item not in found:
                found.append(item)
                if len(found) >= max_items: return found
    return found

# ---------- FastAPI ----------
app = FastAPI(title=BOT_TITLE, version="4.0.0")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

class Query(BaseModel):
    question: str
    mode: str | None = None

class ReindexReq(BaseModel):
    token: str | None = None  # istersen env tabanlı doğrulama ekleyebilirsin

@app.on_event("startup")
def on_start():
    global _index, _chunks
    with _lock:
        if not (INDEX_PATH.exists() and META_PATH.exists()):
            print("[STARTUP] Index yok; oluşturuluyor...")
            build_index()
        _index, _chunks = load_index()
    print("[READY] Chatbot hazır (Hibrit + Rerank + MMR).")

@app.get("/health")
def health():
    try:
        pdf_count = len(list_pdfs())
    except Exception:
        pdf_count = 0
    return {"status": "ok", "pdf_count": pdf_count, "rerank": RERANK_ON}

@app.post("/ask")
def ask(q: Query):
    global _index, _chunks
    if _index is None or _chunks is None:
        raise HTTPException(500, "Index hazır değil.")
    hits = retrieve(q.question, _index, _chunks, topk=8)
    contexts = [t for t,_ in hits]
    sources: list[str] = []
    for _, src in hits:
        if src not in sources:
            sources.append(src)
        if len(sources) >= 4: break

    # güven eşiği
    weak = sum(len(_split_sents(t)) for t in contexts) < 3
    answer = summarize(q.question, contexts, mode=(q.mode or "kisa"))
    if not contexts or weak:
        answer = "Sorduğunuz konuya dair net bir madde bulamadım. Resmî duyurular/bölüm sayfasını kontrol edin."
    maddeler = extract_maddeler(contexts, max_items=10)
    return {"answer": answer, "sources": sources, "maddeler": maddeler}

@app.post("/reindex")
def reindex(_: ReindexReq):
    global _index, _chunks
    with _lock:
        build_index()
        _index, _chunks = load_index()
    return {"status": "ok", "message": "Index yeniden oluşturuldu."}
