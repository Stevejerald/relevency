# scripts/auto_tune_reagents.py
import os, json, itertools, numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, math, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT, "data", "raw_catalog", "Reagents.csv")
EMB_FILE = os.path.join(ROOT, "data", "embeddings", "reagents_embeddings.npy")
META_FILE = os.path.join(ROOT, "data", "embeddings", "reagents_index.json")
TYPE_CLF_FILE = os.path.join(ROOT, "data", "embeddings", "reagents_type_clf.joblib")
TFIDF_FILE = os.path.join(ROOT, "data", "embeddings", "reagents_tfidf.joblib")

print("Loading reagent catalog:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

# Build "title" from Material Description
df["title"] = df["Material Description"].astype(str)

# Type column is "Type"
df["type"] = df["Type"]

# Create metadata JSON list
meta = []
for _, row in df.iterrows():
    meta.append({
        "title": row["title"],
        "product_code": row["Product Code"],
        "type": row["Type"],
        "specification": (
            f"Pack Size (ml): {row['Pack Size (ml)']} | "
            f"MRP: {row['New MRP / Kit (with GST)']} | "
            f"CTP: {row['New CTP']} | "
            f"SAARTHI: {row['SAARTHI Price']} | "
            f"TP (1-50): {row['New TP (1-50)']} | "
            f"Slab-2: {row['New Slab-2 (51-100)']} | "
            f"Slab-3: {row['New Slab-3 (101-200)']}"
        )
    })

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Build embedding dataset
print("Encoding reagent descriptions...")
texts = [m["title"] + " " + m["specification"] for m in meta]
emb = model.encode(texts, normalize_embeddings=True)
np.save(EMB_FILE, emb)

# Train type classifier (TF-IDF + Logistic)
print("Training type classifier...")
tfidf = TfidfVectorizer(stop_words="english", min_df=1)
X = tfidf.fit_transform(df["title"].tolist())
y = df["Type"].tolist()

clf = LogisticRegression(max_iter=400)
clf.fit(X, y)

joblib.dump(clf, TYPE_CLF_FILE)
joblib.dump(tfidf, TFIDF_FILE)

# -----------------------------
# SCORING UTILS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def logistic_map(score, center, scale):
    return sigmoid((score - center) / scale)

def token_overlap(q, c):
    q_t = set(re.split(r"\W+", q.lower()))
    c_t = set(re.split(r"\W+", c.lower()))
    q_t = {t for t in q_t if len(t) > 2}
    c_t = {t for t in c_t if len(t) > 2}
    if not q_t: return 0.0
    return len(q_t & c_t) / len(q_t)

# -----------------------------
# AUTO-TUNE GRID
# -----------------------------
TYPE_WEIGHTS = [0.1, 0.2, 0.3, 0.4]
TOKEN_WEIGHTS = [0.1, 0.2, 0.3]
LOGISTIC_CENTERS = [0.70, 0.75, 0.80]
LOGISTIC_SCALES = [0.10, 0.12, 0.15]
RELEVANCY_THRESHOLDS = [0.35, 0.40, 0.45]

print("Evaluating parameter grid...")

best_acc = -1
best_params = None

# Use each reagent as its own query for validation
queries = df["title"].tolist()

for TW, KW, LC, LS, TH in itertools.product(
    TYPE_WEIGHTS, TOKEN_WEIGHTS, LOGISTIC_CENTERS, LOGISTIC_SCALES, RELEVANCY_THRESHOLDS
):
    correct = 0
    total = len(queries)

    for idx, q in enumerate(queries):
        q_emb = model.encode(q, normalize_embeddings=True)
        sims = emb @ q_emb
        best_i = int(np.argmax(sims))
        best_match = meta[best_i]

        # Get scores
        emb_score = sims[best_i]
        tok = token_overlap(q, best_match["title"] + " " + best_match["specification"])

        Xq = tfidf.transform([q])
        type_pred = clf.predict(Xq)[0]
        type_proba = clf.predict_proba(Xq)[0]
        labels = clf.classes_
        type_conf = type_proba[list(labels).index(type_pred)]

        raw = emb_score + TW * type_conf + KW * tok
        rel = logistic_map(raw, LC, LS)

        is_rel = rel >= TH

        # Correct if matched product_code
        if is_rel and best_match["product_code"] == df.iloc[idx]["Product Code"]:
            correct += 1

    acc = correct / total

    if acc > best_acc:
        best_acc = acc
        best_params = {
            "TYPE_WEIGHT": TW,
            "TOKEN_WEIGHT": KW,
            "LOGISTIC_CENTER": LC,
            "LOGISTIC_SCALE": LS,
            "RELEVANCY_THRESHOLD": TH,
            "accuracy": acc
        }

print("\nBest Params:")
print(json.dumps(best_params, indent=2))
