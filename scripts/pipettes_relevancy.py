# pipettes_relevancy.py
import os, json, numpy as np, re, math
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_FILE = os.path.join(ROOT, "data", "embeddings", "pipettes_embeddings.npy")
META_FILE = os.path.join(ROOT, "data", "embeddings", "pipettes_index.json")

# Hyperparameters tuned for high accuracy
TOKEN_WEIGHT = 0.70
EMB_WEIGHT = 1.0
LOGISTIC_CENTER = 0.65
LOGISTIC_SCALE = 0.12
RELEVANCY_THRESHOLD = 0.50

def sigmoid(x): return 1/(1+math.exp(-x))
def logistic_map(v, c=LOGISTIC_CENTER, s=LOGISTIC_SCALE): return sigmoid((v-c)/s)

def token_overlap(q, t):
    q_t = {w for w in re.split(r"\W+", q.lower()) if len(w)>2}
    t_t = {w for w in re.split(r"\W+", t.lower()) if len(w)>2}
    if not q_t: return 0.0
    return len(q_t & t_t) / len(q_t)

print("Loading Pipettes / Merilettes model...")

emb = np.load(EMB_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def predict_pipettes(query, top_k=3):
    q_emb = model.encode(query, normalize_embeddings=True)
    sims = emb @ q_emb
    idxs = list(np.argsort(-sims)[: top_k])

    results = []
    for i in idxs:
        item = meta[i]
        combined_text = item["title"] + " " + item["specification"]

        emb_score = float(sims[i])
        tok_score = token_overlap(query, combined_text)

        raw = EMB_WEIGHT * emb_score + TOKEN_WEIGHT * tok_score

        results.append({
            "title": item["title"],
            "product_code": item["product_code"],
            "type": item["type"],
            "volume": item["volume"],
            "specification": item["specification"],
            "emb_score": emb_score,
            "token_score": tok_score,
            "raw_score": raw,
            "relevancy": logistic_map(raw)
        })

    results.sort(key=lambda x: x["raw_score"], reverse=True)
    best = results[0]
    relevant = best["relevancy"] >= RELEVANCY_THRESHOLD

    return {
        "query": query,
        "relevancy_score": best["relevancy"],
        "relevant": relevant,
        "best_match": best,
        "top_matches": results
    }

# Local Test
if __name__ == "__main__":
    print(json.dumps(predict_pipettes("variable pipette 100-1000"), indent=2))
