import os, json, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")

META_FILE = os.path.join(EMB_DIR, "systempacks_index.json")
OUT_FILE = os.path.join(EMB_DIR, "systempacks_embeddings.npy")

print("Loading metadata...")

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

texts = []
for item in meta:
    spec = item.get("specification", "")
    text = f"{item['title']} {item['product_code']} {spec}"
    texts.append(text)

print("Generating embeddings...")
emb = model.encode(texts, normalize_embeddings=True)

np.save(OUT_FILE, emb)

print("Saved embeddings:", OUT_FILE)
print("Shape:", emb.shape)
