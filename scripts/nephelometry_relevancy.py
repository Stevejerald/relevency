# scripts/nephelometry_relevancy.py
import os, json, numpy as np, re, math
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")

EMB_FILE = os.path.join(EMB_DIR, "nephelometry_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "nephelometry_index.json")

# ------------------ Hyperparameters ------------------
TOKEN_WEIGHT = 0.50
EMB_WEIGHT = 1.0
LOGISTIC_CENTER = 0.50
LOGISTIC_SCALE = 0.20
RELEVANCY_THRESHOLD = 0.40

# ------------------ Utility Functions ------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_map(x):
    return sigmoid((x - LOGISTIC_CENTER) / LOGISTIC_SCALE)

def token_overlap(q, t):
    q_tokens = {w for w in re.split(r"\W+", q.lower()) if len(w) > 2}
    t_tokens = {w for w in re.split(r"\W+", t.lower()) if len(w) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)

# ------------------ Load Resources ------------------
print("Loading Nephelometry model...")

emb = np.load(EMB_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# ------------------ Core Predictor ------------------
def predict_nephelometry(query, top_k=3):
    q_emb = model.encode(query, normalize_embeddings=True)
    sims = emb @ q_emb
    idxs = list(np.argsort(-sims)[:top_k])

    results = []

    for i in idxs:
        item = meta[i]
        emb_score = float(sims[i])                      # <-- FIXED: convert to float
        tok_score = float(token_overlap(query, item["title"] + " " + item["specification"]))
        raw = float(EMB_WEIGHT * emb_score + TOKEN_WEIGHT * tok_score)

        results.append({
            "title": item["title"],
            "product_code": item["product_code"],
            "specification": item["specification"],
            "emb_score": emb_score,
            "token_score": tok_score,
            "raw_score": raw,
            "relevancy": float(logistic_map(raw))       # <-- FIXED
        })

    results.sort(key=lambda x: x["raw_score"], reverse=True)
    best = results[0]
    relevant = best["relevancy"] >= RELEVANCY_THRESHOLD

    return {
        "query": query,
        "relevancy_score": float(best["relevancy"]),    # <-- FIXED
        "relevant": relevant,
        "best_match": best,
        "top_matches": results
    }


# ------------------ Test Run ------------------
if __name__ == "__main__":
    import sys, json

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "CRP nephelometry 1x40 kit"  # default fallback

    result = predict_nephelometry(query)
    print(json.dumps(result, indent=2))
