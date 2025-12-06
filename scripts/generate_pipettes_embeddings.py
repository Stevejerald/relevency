# generate_pipettes_embeddings.py
import os, json, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META_FILE = os.path.join(ROOT, "data", "embeddings", "pipettes_index.json")
OUT_FILE = os.path.join(ROOT, "data", "embeddings", "pipettes_embeddings.npy")

print("Loading metadata...")
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

texts = []
for item in meta:
    t = f"{item['title']} {item['type']} volume {item['volume']} {item['specification']}"
    texts.append(t)

print("Generating embeddings...")
emb = model.encode(texts, normalize_embeddings=True)

np.save(OUT_FILE, emb)
print(f"Saved embeddings: {OUT_FILE}")
print("Shape:", emb.shape)
