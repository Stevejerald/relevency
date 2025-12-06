import os, json, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "embeddings", "meriscreen_index.json")
OUT_EMB = os.path.join(ROOT, "data", "embeddings", "meriscreen_embeddings.npy")

print("Loading metadata...")
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer("all-mpnet-base-v2")

texts = [item["merged_text"] for item in data]

print("Generating embeddings...")
emb = model.encode(texts, show_progress_bar=True)

np.save(OUT_EMB, emb)

print("Saved embeddings:", OUT_EMB)
print("Shape:", emb.shape)
