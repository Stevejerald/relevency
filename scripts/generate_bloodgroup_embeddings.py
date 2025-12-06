#!/usr/bin/env python3
"""
Generate embeddings for bloodgroup_index.json and save as numpy file.
Usage:
  python scripts/generate_bloodgroup_embeddings.py
"""
import os, json, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "embeddings", "bloodgroup_index.json")
OUT_EMB = os.path.join(ROOT, "data", "embeddings", "bloodgroup_embeddings.npy")

print("Loading metadata...")
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
for item in data:
    # use merged_text if present else fallback to title + specification
    t = item.get("merged_text") or " ".join(filter(None, [item.get("title"), item.get("specification")]))
    texts.append(t)

print(f"Items: {len(texts)}")
print("Loading sentence-transformer model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("Generating embeddings...")
# encode with normalized embeddings for cosine via dot product
embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

embs = np.asarray(embs, dtype="float32")
os.makedirs(os.path.dirname(OUT_EMB), exist_ok=True)
np.save(OUT_EMB, embs)
print("Saved embeddings:", OUT_EMB)
print("Shape:", embs.shape)
