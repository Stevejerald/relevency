import os, json, numpy as np, re, math
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")

EMB_FILE = os.path.join(EMB_DIR, "systempacks_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "systempacks_index.json")

TOKEN_WEIGHT = 0.40
EMB_WEIGHT = 1.0
LOGISTIC_CENTER = 0.70
LOGISTIC_SCALE = 0.15
RELEVANCY_THRESHOLD = 0.50

def sigmoid(x): 
    return 1/(1+math.exp(-x))

def logistic_map(v, c=LOGISTIC_CENTER, s=LOGISTIC_SCALE):
    return sigmoid((v - c) / s)

def token_overlap(q, t):
    q_t = {w for w in re.split(r"\W+", q.lower()) if len(w) > 2}
    t_t = {w for w in re.split(r"\W+", t.lower()) if len(w) > 2}
    if not q_t:
        return 0.0
    return len(q_t & t_t) / len(q_t)

print("Loading System Packs model...")

emb = np.load(EMB_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def predict_systempacks(query, top_k=3):
    q_emb = model.encode(query, normalize_embeddings=True)
    sims = emb @ q_emb
    idxs = np.argsort(-sims)[:top_k]

    results = []
    for i in idxs:
        item = meta[int(i)]
        emb_score = float(sims[i])
        tok_score = token_overlap(query, item["title"] + " " + item["specification"])
        raw = (EMB_WEIGHT * emb_score) + (TOKEN_WEIGHT * tok_score)

        results.append({
            "title": item["title"],
            "product_code": item["product_code"],
            "type": item["type"],
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

if __name__ == "__main__":
    print(json.dumps(predict_systempacks("CREATINE KINASE NAC	4 x 60 / 4 x 15"), indent=2))
