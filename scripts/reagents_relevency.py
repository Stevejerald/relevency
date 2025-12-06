# reagents_relevency.py
import os, json, numpy as np, joblib, re, math
from sentence_transformers import SentenceTransformer

# Optional FAISS
try:
    import faiss
    HAS_FAISS = True
except:
    HAS_FAISS = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")

# --- reagent-specific model files ---
EMB_FILE        = os.path.join(EMB_DIR, "reagents_embeddings.npy")
META_FILE       = os.path.join(EMB_DIR, "reagents_index.json")
FAISS_FILE      = os.path.join(EMB_DIR, "reagents_faiss.index")
TYPE_CLF_FILE   = os.path.join(EMB_DIR, "reagents_type_clf.joblib")
TFIDF_FILE      = os.path.join(EMB_DIR, "reagents_tfidf.joblib")

# -------------------------------------------------------------
# 🔧 Tuned Hyperparameters (replace with auto_tune outputs)
# -------------------------------------------------------------
TYPE_WEIGHT = 0.20
TOKEN_WEIGHT = 0.30
EMB_WEIGHT = 1.0
LOGISTIC_CENTER = 0.75
LOGISTIC_SCALE = 0.12
RELEVANCY_THRESHOLD = 0.40

REQUIRE_TYPE_MATCH = True
STRICT_TYPE_FILTER = True
STRICT_CONF_THRESHOLD = 0.50

# Keyword → Reagent Type mapping
REAGENT_TYPE_KEYWORDS = {
    "fsr": "Fluid Stable Reagents",
    "fluid stable": "Fluid Stable Reagents",
    "ck": "Fluid Stable Reagents",
    "ldh": "Fluid Stable Reagents",
    "bilirubin": "Fluid Stable Reagents",
    "albumin": "Fluid Stable Reagents",
    "tp": "Fluid Stable Reagents",
}

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_map(raw, center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE):
    return sigmoid((raw - center) / scale)

def token_overlap(q, c):
    q_t = {t for t in re.split(r"\W+", q.lower()) if len(t) > 2}
    c_t = {t for t in re.split(r"\W+", c.lower()) if len(t) > 2}
    if not q_t:
        return 0.0
    return len(q_t & c_t) / len(q_t)

def detect_reagent_type(query):
    q = query.lower()
    for kw, t in REAGENT_TYPE_KEYWORDS.items():
        if kw in q:
            return t
    return None

# -------------------------------------------------------------
# Load model & resources
# -------------------------------------------------------------
print("Loading Reagent Model...")

emb = np.load(EMB_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
clf = joblib.load(TYPE_CLF_FILE)
tfidf = joblib.load(TFIDF_FILE)

if HAS_FAISS and os.path.exists(FAISS_FILE):
    index = faiss.read_index(FAISS_FILE)
    use_faiss = True
else:
    index = None
    use_faiss = False

# -------------------------------------------------------------
# Core Prediction
# -------------------------------------------------------------
def predict_reagent(query, top_k=5):
    q_emb = model.encode(query, normalize_embeddings=True)

    # Retrieval
    if use_faiss:
        D, I = index.search(np.array([q_emb], dtype="float32"), top_k * 5)
        idxs = [int(i) for i in I[0] if i != -1]
    else:
        sims = emb @ q_emb
        idxs = list(np.argsort(-sims)[: top_k * 5])

    # Type prediction
    vec = tfidf.transform([query])
    t_pred = clf.predict(vec)[0]
    t_proba = clf.predict_proba(vec)[0]
    labels = list(clf.classes_)
    t_conf = float(t_proba[labels.index(t_pred)])

    kw_type = detect_reagent_type(query)

    # Strict filtering
    filtered = idxs
    final_type = kw_type if kw_type else (t_pred if t_conf >= STRICT_CONF_THRESHOLD else None)

    if STRICT_TYPE_FILTER and final_type:
        filtered = [i for i in idxs if meta[i]["type"] == final_type] or \
                   [i for i in range(len(meta)) if meta[i]["type"] == final_type] or idxs

    # Scoring
    scored = []
    for i in filtered:
        item = meta[i]

        emb_s = float(emb[i] @ q_emb)
        tok_s = token_overlap(query, item["title"] + " " + item["specification"])

        cand_type = item["type"]
        cand_tconf = float(t_proba[labels.index(cand_type)]) if cand_type in labels else 0.0

        raw = emb_s + TYPE_WEIGHT * cand_tconf + TOKEN_WEIGHT * tok_s

        scored.append({
            "title": item["title"],
            "product_code": item["product_code"],
            "type": item["type"],
            "specification": item["specification"],
            "emb_score": emb_s,
            "token_score": tok_s,
            "type_conf": cand_tconf,
            "raw_score": raw,
            "relevancy": logistic_map(raw)
        })

    scored.sort(key=lambda x: x["raw_score"], reverse=True)
    best = scored[0] if scored else None

    relevant = best and best["relevancy"] >= RELEVANCY_THRESHOLD

    return {
        "query": query,
        "keyword_type": kw_type,
        "predicted_type": t_pred,
        "predicted_type_conf": t_conf,
        "relevancy_score": best["relevancy"] if best else 0.0,
        "relevant": bool(relevant),
        "best_match": best,
        "top_matches": scored[:top_k],
    }

# Quick test
if __name__ == "__main__":
    print(json.dumps(predict_reagent("Electrode Deprotein. Washing Solution"), indent=2))
