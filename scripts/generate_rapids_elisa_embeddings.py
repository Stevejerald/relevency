#!/usr/bin/env python3
"""
generate_rapids_elisa_embeddings.py
- Loads data/embeddings/rapids_elisa_index.json (produced by build_rapids_elisa_index.py)
- Generates sentence-transformer embeddings and saves:
    - data/embeddings/rapids_elisa_embeddings.npy
    - optionally data/embeddings/rapids_elisa_faiss.index (if faiss is installed)
"""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
META_FILE = os.path.join(EMB_DIR, "rapids_elisa_index.json")
EMB_FILE = os.path.join(EMB_DIR, "rapids_elisa_embeddings.npy")
FAISS_FILE = os.path.join(EMB_DIR, "rapids_elisa_faiss.index")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                    help="SentenceTransformer model to use")
parser.add_argument("--normalize", action="store_true",
                    help="Normalize embeddings (L2) before saving")
parser.add_argument("--use-faiss", action="store_true",
                    help="Build faiss index (requires faiss installed)")
args = parser.parse_args()

print("Loading metadata:", META_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

texts = []
for it in meta:
    # merged_text is the main text used for embedding creation
    txt = it.get("merged_text") or " ".join(filter(None, [it.get("title"), it.get("specification")]))
    texts.append(txt)

print(f"Generating embeddings for {len(texts)} items using model {args.model} ...")
model = SentenceTransformer(args.model)
emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

if args.normalize:
    # L2-normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

os.makedirs(EMB_DIR, exist_ok=True)
np.save(EMB_FILE, emb)
print("Saved embeddings:", EMB_FILE)
print("Shape:", emb.shape)

# Optionally build and save FAISS index
if args.use_faiss or (HAS_FAISS and args.use_faiss):
    if not HAS_FAISS:
        print("faiss requested but not installed. Skipping faiss index build.")
    else:
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product; use L2 index if embedding not normalized
        if args.normalize:
            # embeddings are normalized -> IP is cosine
            pass
        index.add(emb.astype("float32"))
        faiss.write_index(index, FAISS_FILE)
        print("Saved faiss index:", FAISS_FILE)

print("Done.")
