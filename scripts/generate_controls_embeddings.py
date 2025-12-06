import os, json, numpy as np
from sentence_transformers import SentenceTransformer
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_JSON = os.path.join(ROOT, "data", "unified", "controls_products.json")

EMB_DIR = os.path.join(ROOT, "data", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

EMB_FILE = os.path.join(EMB_DIR, "controls_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "controls_index.json")

with open(IN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["title"] + " " + item["specification"] for item in data]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
emb = np.array(emb, dtype="float32")

np.save(EMB_FILE, emb)
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Saved:", EMB_FILE, META_FILE)
