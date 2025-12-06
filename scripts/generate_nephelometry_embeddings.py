# scripts/generate_nephelometry_embeddings.py
import os, json, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "data/embeddings/nephelometry_index.json")
OUT = os.path.join(ROOT, "data/embeddings/nephelometry_embeddings.npy")

print("Loading metadata...")
with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

texts = [
    m["title"] + " " + m["specification"]
    for m in meta
]

print("Generating embeddings...")
emb = model.encode(texts, normalize_embeddings=True, batch_size=16)

np.save(OUT, emb)
print(f"Saved embeddings: {OUT}\nShape: {emb.shape}")
