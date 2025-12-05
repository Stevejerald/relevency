# generate_embeddings_enriched.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# optional FAISS
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_FILE = os.path.join(ROOT, "data", "unified", "analyser_products_enriched.json")
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
EMB_FILE = os.path.join(EMB_DIR, "analyser_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "analyser_index.json")
FAISS_FILE = os.path.join(EMB_DIR, "analyser_faiss.index")
TYPE_CLF_FILE = os.path.join(EMB_DIR, "type_clf.joblib")
VECT_FILE = os.path.join(EMB_DIR, "type_tfidf.joblib")

os.makedirs(EMB_DIR, exist_ok=True)

print("Loading enriched JSON:", IN_FILE)
with open(IN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["search_text"] for d in data]
meta = [{
    "title": d["title"],
    "product_code": d["product_code"],
    "type": d["type"],
    "segment": d["segment"],
    "specification": d["enriched_description"],
    "price_unit": d.get("price_unit",""),
    "slab_qty": d.get("slab_qty","")
} for d in data]

print("Encoding embeddings with sentence-transformers (all-mpnet-base-v2)...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
embs = np.array(embs, dtype="float32")

print("Saving embeddings and metadata...")
np.save(EMB_FILE, embs)
with open(META_FILE, "w", encoding="utf-8") as fo:
    json.dump(meta, fo, indent=2, ensure_ascii=False)

# build FAISS index if available
if HAS_FAISS:
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (embeddings normalized => cosine)
    index.add(embs)
    faiss.write_index(index, FAISS_FILE)
    print("Saved FAISS index:", FAISS_FILE)
else:
    print("faiss not available — skipping FAISS index (linear search fallback)")

# Train a lightweight type classifier (title -> type) for soft filtering/weighting
print("Training type classifier (TF-IDF -> LogisticRegression)...")
titles = [d["title"] for d in data]
types = [d["type"] for d in data]

tf = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
X = tf.fit_transform(titles)
clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
clf.fit(X, types)

joblib.dump(clf, TYPE_CLF_FILE)
joblib.dump(tf, VECT_FILE)
print("Saved type classifier:", TYPE_CLF_FILE)
print("Saved TF-IDF vectorizer:", VECT_FILE)

print("Done. Total products:", len(meta))
