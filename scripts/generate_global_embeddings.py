#!/usr/bin/env python3
"""
generate_global_embeddings.py
- Reads data/embeddings/global_index.json and generates global_embeddings.npy
- Uses sentence-transformers/all-mpnet-base-v2 (same model used elsewhere)
- Saves as float32 and normalized embeddings
"""

import os, json, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "embeddings", "global_index.json")
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "global_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print("Loading index:", INDEX_PATH)
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [ (d.get("merged_text") or "") for d in data ]

print("Using model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings for", len(texts), "items...")
# encode (batching done by model.encode internally)
emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
emb = np.array(emb, dtype=np.float32)  # ensure float32

os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
np.save(EMB_PATH, emb)
print("Saved embeddings:", EMB_PATH)
print("Shape:", emb.shape)
