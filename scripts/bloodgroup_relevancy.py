#!/usr/bin/env python3
import os, json, math, sys, re
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "embeddings", "bloodgroup_index.json")
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "bloodgroup_embeddings.npy")

EMB_WEIGHT = 1.0
TOKEN_WEIGHT = 0.35
TITLE_WEIGHT = 0.5

LOGISTIC_CENTER = 0.8
LOGISTIC_SCALE = 0.12

TOP_K = 3

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def logistic_map(score, center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE):
    z = (score - center) / scale
    return sigmoid(z)

def tokenize(s):
    return [t for t in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(t) > 1]

def token_overlap(query, text):
    qtokens = set(tokenize(query))
    ttokens = set(tokenize(text))
    if not qtokens:
        return 0.0
    return len(qtokens & ttokens) / len(qtokens)

def title_overlap(query, title):
    qtokens = set(tokenize(query))
    ttokens = set(tokenize(title or ""))
    if not qtokens:
        return 0.0
    return len(qtokens & ttokens) / len(qtokens)


# NEW: exact antigen match logic
def antigen_match_boost(query, title):
    q = query.lower()
    t = title.lower()

    antigens = ["anti a", "anti b", "anti ab", "anti d"]

    for ag in antigens:
        if ag in q:

            if ag == "anti a":
                if "anti a" in t and "anti ab" not in t:
                    return 1.0
                return -0.4

            if ag == "anti b":
                if "anti b" in t and "anti ab" not in t:
                    return 1.0
                return -0.4

            if ag == "anti ab":
                if "anti ab" in t:
                    return 1.0
                return -0.4

            if ag == "anti d":
                if "anti d" in t:
                    return 1.0
                return -0.4

    return 0.0


def predict(query, top_k=TOP_K):
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    emb = np.load(EMB_PATH)

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    sims = np.dot(emb, q_emb)

    scored = []
    for i, item in enumerate(data):
        emb_score = float(sims[i])
        tok_score = token_overlap(query, item.get("merged_text") or item.get("specification") or "")
        t_overlap = title_overlap(query, item.get("title") or "")

        boost = antigen_match_boost(query, item.get("title") or "")

        raw = (
            EMB_WEIGHT * emb_score +
            TOKEN_WEIGHT * tok_score +
            TITLE_WEIGHT * t_overlap +
            boost
        )

        scored.append({
            "index": int(i),
            "item": item,
            "emb_score": emb_score,
            "token_score": float(tok_score),
            "title_overlap": float(t_overlap),
            "boost": float(boost),
            "raw": float(raw),
            "relevancy_local": float(logistic_map(raw))
        })

    scored.sort(key=lambda x: x["raw"], reverse=True)
    top = scored[:top_k]

    best_raw = top[0]["raw"]
    final_relevancy = float(logistic_map(best_raw))
    is_relevant = final_relevancy >= 0.5

    top_out = []
    for s in top:
        it = s["item"]
        top_out.append({
            "index": s["index"],
            "title": it.get("title"),
            "product_code": it.get("product_code"),
            "specification": it.get("specification"),
            "emb_score": float(s["emb_score"]),
            "token_score": float(s["token_score"]),
            "title_overlap": float(s["title_overlap"]),
            "boost": float(s["boost"]),
            "raw_score": float(s["raw"]),
            "relevancy_local": float(s["relevancy_local"])
        })

    out = {
        "query": query,
        "relevancy_score": final_relevancy,
        "relevant": bool(is_relevant),
        "best_match": top_out[0],
        "top_matches": top_out
    }
    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ANTI A PACK 1x10"
    print(json.dumps(predict(q), indent=2, ensure_ascii=False))
