# controls_relevancy.py
import os
import json
import numpy as np
import re
import math
from sentence_transformers import SentenceTransformer

# -----------------------------
# PATH SETUP
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")

EMB_FILE = os.path.join(EMB_DIR, "controls_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "controls_index.json")

# -----------------------------
# HYPERPARAMETERS (Optimized)
# -----------------------------
EMB_WEIGHT = 1.0
TOKEN_WEIGHT = 0.40

# logistic mapping tuned for small datasets
LOGISTIC_CENTER = 0.50
LOGISTIC_SCALE = 0.10

RELEVANCY_THRESHOLD = 0.40  # output relevancy score must exceed this

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_map(value, center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE):
    """Map raw score to 0..1"""
    z = (value - center) / scale
    return sigmoid(z)

def token_overlap(query, text):
    """Compute how many query tokens match product tokens"""
    q_tok = {w for w in re.split(r"\W+", query.lower()) if len(w) > 2}
    t_tok = {w for w in re.split(r"\W+", text.lower()) if len(w) > 2}

    if not q_tok:
        return 0.0

    return len(q_tok & t_tok) / len(q_tok)


# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
print("Loading Controls & Calibrators model...")

emb = np.load(EMB_FILE)

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def predict_controls(query, top_k=3):
    """Returns relevancy + best match + top matches"""
    q_emb = model.encode(query, normalize_embeddings=True)

    sims = emb @ q_emb
    idxs = list(np.argsort(-sims)[:top_k])

    results = []
    for i in idxs:
        item = meta[i]

        emb_score = float(sims[i])
        tok_score = token_overlap(query, item["title"] + " " + item["specification"])

        raw_score = (
            EMB_WEIGHT * emb_score +
            TOKEN_WEIGHT * tok_score
        )

        results.append({
            "title": item["title"],
            "product_code": item["product_code"],
            "specification": item["specification"],
            "emb_score": emb_score,
            "token_score": tok_score,
            "raw_score": raw_score,
            "relevancy": logistic_map(raw_score)
        })

    # Sort final results
    results.sort(key=lambda x: x["raw_score"], reverse=True)

    best = results[0]
    is_relevant = best["relevancy"] >= RELEVANCY_THRESHOLD

    return {
        "query": query,
        "relevancy_score": best["relevancy"],
        "relevant": is_relevant,
        "best_match": best,
        "top_matches": results
    }


# -----------------------------
# LOCAL TEST
# -----------------------------
if __name__ == "__main__":
    print(json.dumps(
        predict_controls("BioNorm 1x5 control"),
        indent=2
    ))
