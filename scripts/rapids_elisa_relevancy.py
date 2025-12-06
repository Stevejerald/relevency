#!/usr/bin/env python3
"""
rapids_elisa_relevancy.py  (improved)
- Loads rapids_elisa_embeddings.npy and rapids_elisa_index.json
- Provides predict_rapids_elisa(query, top_k=5)
- Improvements:
  * title / product_code exact-match boosting
  * numeric token emphasis
  * alias/synonym mapping
  * JSON-safe outputs (no float32 issue)
  * optional faiss usage
"""
import os
import json
import math
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse

# optional faiss
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
EMB_FILE = os.path.join(EMB_DIR, "rapids_elisa_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "rapids_elisa_index.json")
FAISS_FILE = os.path.join(EMB_DIR, "rapids_elisa_faiss.index")

# ---------- Tunables (change as you test) ----------
EMB_WEIGHT = 1.0        # keep main weight on embeddings
TOKEN_WEIGHT = 0.35     # higher token weight improves exact token matches for small catalogs
TITLE_BOOST = 0.55      # additional boost added when query strongly overlaps title or product_code
EXACT_PRODUCT_CODE_BOOST = 1.2  # big multiplier when product code appears exactly in query

# logistic mapping: tune to map raw_score -> (0..1)
LOGISTIC_CENTER = 0.75
LOGISTIC_SCALE = 0.12
RELEVANCY_THRESHOLD = 0.50

# retrieval
RETRIEVE_CANDIDATES = 30  # how many candidates to score after initial retrieval

# small alias/synonym map to improve token matches (extend as needed)
ALIASES = {
    "ns1": "dengue ns1",
    "hiv4": "hiv 4th gen",
    "hiv3": "hiv 3rd gen",
    "hcv4": "hcv 4th gen",
    "crp": "crp",
    "hb a1c": "hba1c",
    "hba1c": "hba1c"
}

# ---------- Utils ----------
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def logistic_map(score, center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE):
    z = (score - center) / scale
    return sigmoid(z)

def normalize_query(q):
    if not q:
        return ""
    q = q.lower().strip()
    # apply alias expansions
    for a, b in ALIASES.items():
        q = re.sub(r'\b' + re.escape(a) + r'\b', b, q)
    # collapse spaces
    q = re.sub(r'\s+', ' ', q)
    return q

def tokenize_keep_numbers(s):
    # keep numeric tokens (e.g. 1x40, 48, 96) as tokens and longer words only
    tokens = re.split(r"[^\w\d]+", (s or "").lower())
    tokens = [t for t in tokens if t and (len(t) > 2 or re.search(r'\d', t))]
    return set(tokens)

def token_overlap_score(q, c):
    q_tokens = tokenize_keep_numbers(q)
    c_tokens = tokenize_keep_numbers(c)
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)

def to_py(val):
    # convert numpy types -> python native for json serialization
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

# ---------- Load resources ----------
print("Loading Rapids & ELISA model...")
if not os.path.exists(META_FILE):
    raise SystemExit(f"Missing metadata file: {META_FILE}")
if not os.path.exists(EMB_FILE):
    raise SystemExit(f"Missing embeddings file: {EMB_FILE}")

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

emb = np.load(EMB_FILE)
# ensure 2D array
if emb.ndim == 1:
    emb = emb.reshape(1, -1)

# If you generated non-normalized embeddings, you can L2-normalize here:
# emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

# faiss index if available and built
use_faiss = False
if HAS_FAISS and os.path.exists(FAISS_FILE):
    index = faiss.read_index(FAISS_FILE)
    use_faiss = True
else:
    index = None

# sentence-transformer for query encoding
smodel = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def predict_rapids_elisa(query, top_k=5):
    q_orig = query
    query = normalize_query(query)

    # handle quick product_code exact match
    q_lc = query.lower()
    for item in meta:
        if item.get("product_code") and item["product_code"].lower() in q_lc:
            # immediate high-confidence return (product code explicitly mentioned)
            return {
                "query": q_orig,
                "relevancy_score": 0.99,
                "relevant": True,
                "best_match": {
                    "title": item.get("title"),
                    "product_code": item.get("product_code"),
                    "specification": item.get("specification"),
                    "emb_score": None,
                    "token_score": 1.0,
                    "raw_score": None,
                    "relevancy_local": 0.99
                },
                "top_matches": []
            }

    # encode
    q_emb = smodel.encode(query, normalize_embeddings=True)

    # retrieval
    if use_faiss:
        D, I = index.search(np.array([q_emb], dtype="float32"), RETRIEVE_CANDIDATES)
        idxs = [int(i) for i in I[0] if i != -1]
    else:
        sims = emb @ q_emb
        idxs = list(np.argsort(-sims)[: RETRIEVE_CANDIDATES])

    candidates = []
    for i in idxs:
        item = meta[i]
        # emb similarity (if embeddings normalized, dot product ~ cosine)
        emb_score = float(np.dot(emb[i], q_emb))
        combined_text = " ".join(filter(None, [ item.get("title",""), item.get("specification",""), item.get("sub_type",""), item.get("main_type","") ]))
        tok_score = float(token_overlap_score(query, combined_text))

        # title overlap strong match if title tokens cover majority of query tokens
        title_tok = tokenize_keep_numbers(item.get("title",""))
        q_tok = tokenize_keep_numbers(query)
        title_overlap = 0.0
        if q_tok:
            title_overlap = len(title_tok & q_tok) / len(q_tok)

        raw_score = float(EMB_WEIGHT * emb_score + TOKEN_WEIGHT * tok_score)

        # apply title boost (if query is mostly title tokens)
        if title_overlap >= 0.6:
            raw_score += TITLE_BOOST

        # product_code mention in original query (exact) -> multiply strong
        if item.get("product_code") and item["product_code"].lower() in q_lc:
            raw_score *= EXACT_PRODUCT_CODE_BOOST

        candidates.append({
            "index": int(i),
            "title": item.get("title"),
            "product_code": item.get("product_code"),
            "main_type": item.get("main_type"),
            "sub_type": item.get("sub_type"),
            "specification": item.get("specification"),
            "emb_score": emb_score,
            "token_score": tok_score,
            "title_overlap": title_overlap,
            "raw_score": raw_score,
            "relevancy_local": float(logistic_map(raw_score))
        })

    # sort and pick top_k
    candidates.sort(key=lambda x: x["raw_score"], reverse=True)
    top = candidates[:top_k]
    best = top[0] if top else None
    relevancy_score = best["relevancy_local"] if best else 0.0
    relevant = bool(relevancy_score >= RELEVANCY_THRESHOLD)

    # prepare json-safe output
    def make_safe(c):
        return {k: to_py(v) for k, v in c.items()}

    result = {
        "query": q_orig,
        "relevancy_score": float(relevancy_score),
        "relevant": bool(relevant),
        "best_match": make_safe(best) if best else None,
        "top_matches": [make_safe(c) for c in top]
    }
    return result

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=None, help="Query to test")
    parser.add_argument("--top-k", type=int, default=3, help="Top K matches to return")
    parser.add_argument("--print-meta", action="store_true", help="Print metadata count")
    args = parser.parse_args()

    if args.print_meta:
        print("Metadata items:", len(meta))

    test_queries = [
        "CRP 1x40 kit",
        "Dengue NS1 48 Test",
    ]
    queries = [args.query] if args.query else test_queries

    for q in queries:
        out = predict_rapids_elisa(q, top_k=args.top_k)
        print(json.dumps(out, indent=2, ensure_ascii=False))
