# scripts/build_nephelometry_index.py
import pandas as pd
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT, "data","Proviso_Nephelometry.csv")
OUT_FILE = os.path.join(ROOT, "data", "embeddings", "nephelometry_index.json")

df = pd.read_csv(DATA_FILE)

index = []
for _, r in df.iterrows():
    spec = (
        f"Pack Size: {r['Pack Size']} | "
        f"MRP: {r['New MRP / Kit (with GST)']} | "
        f"CTP: {r['CTP']} | "
        f"Saarthi: {r['Saarthi Price']} | "
        f"TP(1-50): {r['New TP (1-50)']} | "
        f"Assorted 50: {r['Assorted 50']} | "
        f"Assorted 100: {r['Assorted 100']}"
    )

    index.append({
        "title": r["Material Description"],
        "product_code": r["Product Code"],
        "specification": spec
    })

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)

print(f"Saved: {OUT_FILE}\nTotal items: {len(index)}")
