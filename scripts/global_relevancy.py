#!/usr/bin/env python3
"""
global_relevancy.py
- Single entrypoint for relevancy search across the global catalog
- Uses embedding similarity + token overlap + category-specific boosts
- CLI: python global_relevancy.py "your query here"
"""
import os
import json
import numpy as np
import re
import math
import argparse
import unicodedata
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
#  ADD ANALYSER MODEL IMPORT (SAFE IMPORT)
# ------------------------------------------------------------------
try:
    from analyser_relevancy import predict_relevancy as analyser_predict
    HAS_ANALYSER_MODEL = True
except Exception as e:
    print("Warning: analyser_relevancy.py not loaded:", e)
    HAS_ANALYSER_MODEL = False
# ------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "embeddings", "global_index.json")
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "global_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Tunable weights
EMB_WEIGHT = 1.0
TOKEN_WEIGHT = 0.35
TITLE_WEIGHT = 0.5
CATEGORY_BOOST = 0.25
TOP_K = 5

# keyword->category map (extendable)
CATEGORY_KEYWORDS = {
    "pipette": "Pipettes",
    "pipettes": "Pipettes",
    "fixed volume": "Pipettes",
    "variable": "Pipettes",
    "dengue": "Elisa",
    "ns1": "Elisa",
    "hiv": "Elisa",
    "hbsag": "Elisa",
    "crp": "Turbidimetry",
    "rf": "Nephelometry",
    "aso": "Nephelometry",
    "control": "Controls",
    "control kit": "Controls",
    "system pack": "System Packs",
    "albumin": "System Packs",
    "anti a": "BloodGroup",
    "anti b": "BloodGroup",
    "anti d": "BloodGroup",
    "anti ab": "BloodGroup",
    "blood grouping": "BloodGroup",
    "reagent": "Reagents",
    "reagents": "Reagents",
    "analyser": "Analyser",
    "analyzer": "Analyser",
    "hematology": "Analyser",
    "hb": "Analyser",
    "meriscreen": "Meriscreen",
    "rapid": "Rapids",
    "elisa": "Elisa",
    "nephelometry": "Nephelometry",
    "turbidimetry": "Turbidimetry",
    "5 part": "Analyser",
    "3 part": "Analyser",
    "cbc": "Analyser",
    "celquant": "Analyser",
    "autoloader": "Analyser"
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", " ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip()
    return s

def norm_token_list(s: str):
    s = normalize_text(s).lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    return [t for t in tokens if len(t) > 2]

def token_set(s: str):
    return set(norm_token_list(s))

def token_overlap(query: str, target: str) -> float:
    q = token_set(query)
    t = token_set(target)
    if not q:
        return 0.0
    return len(q & t) / len(q)

def detect_category_from_query(q: str, index_items=None):
    ql = normalize_text(q).lower()

    hits = []
    for kw, cat in CATEGORY_KEYWORDS.items():
        if kw in ql:
            hits.append((len(kw), kw, cat))

    if hits:
        hits.sort(reverse=True)
        return hits[0][2]

    if index_items:
        q_tokens = token_set(q)
        best = None
        best_score = 0
        for it in index_items:
            combined = " ".join([
                it.get("title") or "",
                it.get("category") or "",
                it.get("type") or "",
                it.get("merged_text") or ""
            ])
            score = len(q_tokens & token_set(combined))
            if score > best_score:
                best_score = score
                best = it.get("category") or it.get("type")
        if best_score > 0:
            return best

    return None

def safe_product_code(item):
    candidates = []
    for k in ("product_code", "product code", "productcode", "code", "product"):
        v = item.get(k)
        if v:
            candidates.append(str(v).strip())

    v0 = str(item.get("product_code") or "").strip()
    if v0:
        candidates.insert(0, v0)

    for c in candidates:
        if re.search(r"[A-Za-z]", c) and re.search(r"[0-9]", c):
            return c

    for c in candidates:
        low = c.lower()
        if low in ("regular", "no slab") or low.startswith("slab"):
            continue
        if c:
            return c
    return ""

print("Loading index and embeddings...")
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    INDEX_RAW = json.load(f)

INDEX = []
for it in INDEX_RAW:
    title = normalize_text(it.get("title") or it.get("Title") or "")
    prod = safe_product_code(it)
    spec = it.get("specification") or it.get("spec") or it.get("specification_text") or ""
    spec = normalize_text(spec)

    if "SLABS" in spec:
        if "kit_price" not in spec:
            spec += " kit_price: —"
        if "test_price" not in spec:
            spec += " test_price: —"

    merged = normalize_text(it.get("merged_text") or it.get("mergedText") or title or spec)

    item = {
        "index": int(it.get("index")) if it.get("index") not in (None, "") else None,
        "product_code": prod,
        "title": title,
        "type": normalize_text(it.get("type") or it.get("Type") or it.get("category") or ""),
        "category": normalize_text(it.get("category") or it.get("Category") or ""),
        "specification": spec,
        "merged_text": merged,
    }
    INDEX.append(item)

EMB = np.load(EMB_PATH)
MODEL = SentenceTransformer(MODEL_NAME)

# ------------------------------------------------------------------
#                  MAIN PREDICT FUNCTION
# ------------------------------------------------------------------
def predict(query, top_k=TOP_K):

    detected_category = detect_category_from_query(query, INDEX)

    # ------------------------------------------------------------------
    #  SPECIAL HANDLING: ROUTE ANALYSER QUERIES TO analyser_relevancy.py
    # ------------------------------------------------------------------
    if detected_category and detected_category.lower() == "analyser" and HAS_ANALYSER_MODEL:
        print("Routing query to analyser_relevancy.py ...")
        result = analyser_predict(query, top_k=top_k)

        return {
            "query": query,
            "detected_category": "Analyser",
            "relevancy_score": result.get("relevancy_score", 0),
            "relevant": result.get("relevant", False),
            "best_match": result.get("best_match"),
            "top_matches": result.get("top_matches"),
            "model_used": "analyser_relevancy"
        }
    # ------------------------------------------------------------------

    q_emb = MODEL.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(EMB, q_emb)

    results = []
    q_lower = normalize_text(query).lower()

    for i, item in enumerate(INDEX):
        emb_score = float(sims[i])
        tok = float(token_overlap(query, item.get("merged_text", "")))
        title_tok = float(token_overlap(query, item.get("title", "")))

        raw = EMB_WEIGHT * emb_score + TOKEN_WEIGHT * tok + TITLE_WEIGHT * title_tok

        if detected_category:
            item_cat = (item.get("category") or "").lower()
            item_type = (item.get("type") or "").lower()
            if detected_category.lower() in item_cat or detected_category.lower() in item_type:
                raw += CATEGORY_BOOST

        pc = (item.get("product_code") or "").lower()
        if pc and re.search(r"\b" + re.escape(pc) + r"\b", q_lower):
            raw += 0.5

        results.append({
            "index": int(item.get("index")) if item.get("index") is not None else int(i),
            "product_code": item.get("product_code") or "",
            "title": item.get("title"),
            "type": item.get("type"),
            "category": item.get("category"),
            "specification": item.get("specification"),
            "emb_score": emb_score,
            "token_score": tok,
            "title_overlap": title_tok,
            "raw_score": raw,
            "relevancy": float(1 / (1 + math.exp(-raw)))
        })

    results.sort(key=lambda x: x["raw_score"], reverse=True)

    top = results[:top_k]
    best = top[0] if top else None
    final_score = float(best["relevancy"]) if best else 0.0

    return {
        "query": query,
        "detected_category": detected_category,
        "relevancy_score": final_score,
        "relevant": bool(final_score >= 0.5),
        "best_match": best,
        "top_matches": top
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global relevancy search")
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("--top", type=int, default=5, help="Top K results")
    args = parser.parse_args()
    q = " ".join(args.query) if args.query else "5 Part Automated Hematology Analyser (V2) (Q2)"
    res = predict(q, top_k=args.top)
    print(json.dumps(res, indent=2, ensure_ascii=False))
