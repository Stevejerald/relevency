import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_FILE = os.path.join(ROOT, "data", "unified", "reagents_products_enriched.json")
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

EMB_FILE = os.path.join(EMB_DIR, "reagents_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "reagents_index.json")
TYPE_CLF_FILE = os.path.join(EMB_DIR, "reagents_type_clf.joblib")
VECT_FILE = os.path.join(EMB_DIR, "reagents_tfidf.joblib")

with open(IN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
types = []
meta = []

for d in data:
    txt = f"{d['title']} {d['product_code']} {d['specification']}"
    texts.append(txt)
    types.append(d["type"])
    meta.append(d)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
np.save(EMB_FILE, np.array(embs, dtype="float32"))

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Embeddings saved:", EMB_FILE)
print("Meta saved:", META_FILE)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = tfidf.fit_transform(texts)

clf = LogisticRegression(max_iter=2000)
clf.fit(X, types)

joblib.dump(clf, TYPE_CLF_FILE)
joblib.dump(tfidf, VECT_FILE)

print("Type classifier saved:", TYPE_CLF_FILE)
print("TFIDF saved:", VECT_FILE)
