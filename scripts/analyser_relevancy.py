# analyser_relevancy.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib, re, math

# Optional faiss
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
EMB_FILE = os.path.join(EMB_DIR, "analyser_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "analyser_index.json")
FAISS_FILE = os.path.join(EMB_DIR, "analyser_faiss.index")
TYPE_CLF_FILE = os.path.join(EMB_DIR, "type_clf.joblib")
VECT_FILE = os.path.join(EMB_DIR, "type_tfidf.joblib")

# ---------- TUNABLE HYPERPARAMETERS ----------
TYPE_WEIGHT = 0.3        # contribution of type confidence to final_score
TOKEN_WEIGHT = 0.2       # contribution of token overlap
EMB_WEIGHT = 1.0          # embedding weight (kept 1.0 for readability)
# Logistic mapping parameters (maps raw final_score to 0..1)
LOGISTIC_CENTER = 0.85    # center (mu) of logistic; tune on validation
LOGISTIC_SCALE = 0.08     # scale (sigma); smaller -> steeper mapping

# Relevancy thresholds
RELEVANCY_SCORE_THRESHOLD = 0.6  # relevancy_score >= this -> candidate considered relevant
# Hybrid rule:
# require_type_match: if True, require predicted type or keyword-detected type to match candidate type
REQUIRE_TYPE_MATCH = True

# Strict-type-mode behavior (reuse your previous strict detection logic)
STRICT_TYPE_FILTER = True
STRICT_CONF_THRESHOLD = 0.50

# manual keyword-to-type mapping (same as we used before)
TYPE_KEYWORDS = {
    "semi": "Semi Automated Biochemistry Analyzers",
    "semiauto": "Semi Automated Biochemistry Analyzers",
    "semi-automated": "Semi Automated Biochemistry Analyzers",
    "fully": "Fully Automated Biochemistry Analyzer",
    "full": "Fully Automated Biochemistry Analyzer",
    "fully automated": "Fully Automated Biochemistry Analyzer",
    "3 part": "3-Part Hematology Analyzer",
    "3-part": "3-Part Hematology Analyzer",
    "5 part": "5-Part Hematology Analyzer",
    "5-part": "5-Part Hematology Analyzer",
    "hplc": "HPLC Analyzer",
    "electrolyte": "Electrolyte Analyzer",
    "immunofluorescence": "Immunofluorescence Analyzer",
    "elisa": "ELISA Reader & Washer"
}


# ---------- Utilities ----------
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def logistic_map(score, center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE):
    # map raw score -> (-inf, +inf) then sigmoid to [0,1]
    # x -> (score - center) / scale
    z = (score - center) / scale
    return sigmoid(z)

def token_overlap_score(q, c):
    q_tokens = set(re.split(r"\W+", q.lower()))
    c_tokens = set(re.split(r"\W+", c.lower()))
    q_tokens = {t for t in q_tokens if len(t) > 2}
    c_tokens = {t for t in c_tokens if len(t) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)

def detect_type_from_keywords(query):
    q = query.lower()
    for kw, t in TYPE_KEYWORDS.items():
        if kw in q:
            return t
    return None

# ---------- Load resources ----------
print("Loading embeddings / metadata / models...")
emb = np.load(EMB_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

# sentence-transformer (same model used to create vectors)
smodel = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# type classifier and TF-IDF for soft predictions
clf = joblib.load(TYPE_CLF_FILE)
tf = joblib.load(VECT_FILE)

if HAS_FAISS and os.path.exists(FAISS_FILE):
    index = faiss.read_index(FAISS_FILE)
    use_faiss = True
else:
    index = None
    use_faiss = False

# ---------- Core prediction function ----------
def predict_relevancy(query, top_k=5):
    """
    Returns:
      {
        "query": query,
        "keyword_type_detected": str or None,
        "predicted_type": str,
        "predicted_type_conf": float,
        "relevancy_score": float (0..1),
        "relevant": bool,
        "best_match": {...},
        "top_matches": [...]
      }
    """
    # 1) embedding
    q_emb = smodel.encode(query, normalize_embeddings=True)

    # 2) candidate retrieval (FAISS or linear)
    if use_faiss:
        D, I = index.search(np.array([q_emb], dtype="float32"), top_k * 5)
        idxs = [int(i) for i in I[0] if i != -1]
    else:
        sims = emb @ q_emb
        idxs = list(np.argsort(-sims)[: top_k * 5])

    # 3) classifier & keyword detection
    q_vec = tf.transform([query])
    type_pred = clf.predict(q_vec)[0]
    type_proba = clf.predict_proba(q_vec)[0]
    type_labels = list(clf.classes_)
    type_conf = float(type_proba[type_labels.index(type_pred)]) if type_pred in type_labels else 0.0

    keyword_type = detect_type_from_keywords(query)
    

    # 4) strict-type filtering (if enabled)
    filtered_idxs = idxs
    if STRICT_TYPE_FILTER:
        final_type = None
        if keyword_type:
            final_type = keyword_type
        elif type_conf >= STRICT_CONF_THRESHOLD:
            final_type = type_pred

        if final_type:
            # prefer intersection to preserve retrieval ranking; fallback to all if none
            filtered_idxs = [i for i in idxs if meta[i]["type"] == final_type]
            if not filtered_idxs:
                # fallback to scanning entire catalog for that type
                filtered_idxs = [i for i, m in enumerate(meta) if m["type"] == final_type]

            # if still empty, fall back to original candidate list
            if not filtered_idxs:
                filtered_idxs = idxs

    # 5) score candidates and build top list
    candidates = []
    for i in filtered_idxs:
        item = meta[i]
        emb_score = float(emb[i] @ q_emb)
        tok_score = token_overlap_score(query, item.get("title", "") + " " + item.get("specification", ""))
        cand_type = item.get("type", "")
        # candidate's type_conf: how much classifier supports this type
        cand_type_conf = 0.0
        if cand_type in type_labels:
            cand_type_conf = float(type_proba[type_labels.index(cand_type)])
        # If keyword_type detected ⇒ strong confidence in type
            if keyword_type and keyword_type == cand_type:
                boosted_type_conf = 1.0     # FULL CONFIDENCE
            else:
                boosted_type_conf = cand_type_conf

            raw_score = (EMB_WEIGHT * emb_score
                + TYPE_WEIGHT * boosted_type_conf
                + TOKEN_WEIGHT * tok_score)


        candidates.append({
            "index": int(i),
            "title": item.get("title"),
            "product_code": item.get("product_code"),
            "type": cand_type,
            "segment": item.get("segment"),
            "specification": item.get("specification"),
            "emb_score": emb_score,
            "type_conf": cand_type_conf,
            "token_score": tok_score,
            "raw_score": raw_score
        })

    # sort by raw_score descending
    candidates.sort(key=lambda x: x["raw_score"], reverse=True)
    top_candidates = candidates[:top_k]

    # 6) compute relevancy score for best match (or aggregated)
    # We'll compute relevancy_score using the top candidate's raw_score mapped via logistic,
    # and also compute an aggregated confidence (optional).
    if top_candidates:
        best = top_candidates[0]
        best_raw = best["raw_score"]
    else:
        best = None
        best_raw = 0.0

    relevancy_score = logistic_map(best_raw, center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE)

    # 7) Hybrid rule for boolean 'relevant'
    # If REQUIRE_TYPE_MATCH is True, we require that:
    #  - (keyword detected type OR predicted_type equals candidate.type) AND relevancy_score >= RELEVANCY_SCORE_THRESHOLD
    # If REQUIRE_TYPE_MATCH is False: relevant if relevancy_score >= threshold
    is_type_match = False
    if best:
        if keyword_type and best["type"] == keyword_type:
            is_type_match = True
        elif type_pred == best["type"]:
            is_type_match = True

    if REQUIRE_TYPE_MATCH:
        relevant = bool(is_type_match and (relevancy_score >= RELEVANCY_SCORE_THRESHOLD))
    else:
        relevant = bool(relevancy_score >= RELEVANCY_SCORE_THRESHOLD)

    # 8) Prepare output (include top matches)
    out_top_matches = []
    for c in top_candidates:
        # map raw score -> local relevancy estimate too
        c_relevancy = logistic_map(c["raw_score"], center=LOGISTIC_CENTER, scale=LOGISTIC_SCALE)
        out_top_matches.append({
            "title": c["title"],
            "product_code": c["product_code"],
            "type": c["type"],
            "segment": c["segment"],
            "specification": c["specification"],
            "emb_score": c["emb_score"],
            "type_conf": c["type_conf"],
            "token_score": c["token_score"],
            "raw_score": c["raw_score"],
            "relevancy_estimate": c_relevancy
        })

    result = {
        "query": query,
        "keyword_type_detected": keyword_type,
        "predicted_type": type_pred,
        "predicted_type_conf": type_conf,
        "relevancy_score": float(relevancy_score),
        "relevant": bool(relevant),
        "best_match": out_top_matches[0] if out_top_matches else None,
        "top_matches": out_top_matches
    }
    return result

# ---------- Quick local test ----------
if __name__ == "__main__":
    tests = [
        "5 Part Automated Hematology Analyser (V2) (Q2)"
    ]
    for t in tests:
        print(json.dumps(predict_relevancy(t, top_k=3), indent=2))
