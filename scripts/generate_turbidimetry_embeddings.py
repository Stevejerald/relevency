import os, json
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_DIR = os.path.join(ROOT, "data", "embeddings")

JSON_FILE = os.path.join(EMB_DIR, "turbidimetry_index.json")
EMB_FILE = os.path.join(EMB_DIR, "turbidimetry_embeddings.npy")

print("Loading metadata...")
with open(JSON_FILE, "r", encoding="utf-8") as f:
    items = json.load(f)

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

texts = [
    f"{x['title']} {x['specification']}"
    for x in items
]

print("Generating embeddings...")
emb = model.encode(texts, normalize_embeddings=True)

np.save(EMB_FILE, emb)
print("Saved embeddings:", EMB_FILE)
print("Shape:", emb.shape)
