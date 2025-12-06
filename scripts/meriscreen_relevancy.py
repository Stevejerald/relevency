#!/usr/bin/env python3
import os, json, numpy as np
from sentence_transformers import SentenceTransformer
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data/embeddings/meriscreen_index.json")
EMB_PATH = os.path.join(ROOT, "data/embeddings/meriscreen_embeddings.npy")

model = SentenceTransformer("all-mpnet-base-v2")

# -------------------------
# Universal JSON Sanitizer
# -------------------------
def to_py(obj):
    """Convert numpy data types recursively into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_py(v) for v in obj]

    # numpy scalars → python scalars
    if hasattr(obj, "item"):
        return obj.item()

    return obj


def tokenize(q):
    return re.findall(r"[a-zA-Z0-9]+", q.lower())


def score_match(query, item):
    tokens = tokenize(query)
    text = item["merged_text"].lower()

    token_hits = sum(1 for t in tokens if t in text)
    token_score = token_hits / max(len(tokens), 1)

    title_hits = sum(1 for t in tokens if t in item["title"].lower())
    title_overlap = title_hits / max(len(tokens), 1)

    return token_score, title_overlap


def predict(query):
    # Load data + embeddings
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    emb = np.load(EMB_PATH)

    # Encode query embedding
    q_emb = model.encode([query])[0]

    # Cosine similarity
    sims = np.dot(emb, q_emb) / (
        np.linalg.norm(emb, axis=1) * np.linalg.norm(q_emb)
    )

    scored = []
    for i, item in enumerate(data):
        token_score, title_overlap = score_match(query, item)
        raw = sims[i] + token_score + (0.5 * title_overlap)

        scored.append((raw, i, item, sims[i], token_score, title_overlap))

    scored.sort(reverse=True)

    # Best match
    best_raw, idx, best_item, emb_s, tok_s, title_s = scored[0]
    final_score = 1 / (1 + np.exp(-best_raw))  # sigmoid

    # Build top matches (3)
    top_matches = []
    for raw, i, it, emb_val, tok_val, title_val in scored[:3]:
        top_matches.append({
            "index": int(i),
            "title": it["title"],
            "product_code": it["product_code"],
            "specification": it["specification"],
            "emb_score": float(emb_val),
            "token_score": float(tok_val),
            "title_overlap": float(title_val),
            "raw_score": float(raw),
            "relevancy_local": float(1 / (1 + np.exp(-raw)))
        })

    # Final dict (will be sanitized)
    result = {
        "query": query,
        "relevancy_score": float(final_score),
        "relevant": bool(final_score >= 0.5),
        "best_match": top_matches[0],
        "top_matches": top_matches
    }

    # Make everything JSON-safe
    return to_py(result)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "MeriScreen HIV 50 Test"
    safe_output = to_py(predict(q))
    print(json.dumps(safe_output, indent=2, ensure_ascii=False))
