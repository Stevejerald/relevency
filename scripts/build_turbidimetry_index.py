import pandas as pd
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")

os.makedirs(EMB_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "Turbidimetry_Immuno_Assay.csv")
OUT_JSON = os.path.join(EMB_DIR, "turbidimetry_index.json")

df = pd.read_csv(CSV_FILE)

items = []
for _, r in df.iterrows():
    spec = (
        f"Pack Size: {r['Pack Size']} | "
        f"MRP: {r['New MRP / Kit (with GST)']} | "
        f"CTP: {r['CTP']} | "
        f"Saarthi: {r['Saarthi Price']} | "
        f"New TP (1-50): {r['New TP (1-50)']} | "
        f"Assorted 50: {r['Assorted 50']} | "
        f"Assorted 100: {r['Assorted 100']}"
    )

    items.append({
        "title": r["Material Description"],
        "product_code": r["Product Code"],
        "specification": spec
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2)

print("Saved:", OUT_JSON)
print("Total items:", len(items))
