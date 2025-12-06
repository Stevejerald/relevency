import os, json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Reagents.csv"
OUT_FILE = ROOT / "data" / "unified" / "reagents_products_enriched.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def build_spec(row):
    parts = []
    for col, val in row.items():
        if pd.isna(val): 
            continue
        if col.lower() in ["type", "type shortform", "product code", "material description"]:
            continue
        parts.append(f"{col}: {val}")
    return " | ".join(parts)

df = pd.read_csv(DATA_FILE)

records = []
for _, row in df.iterrows():
    rec = {
        "type": str(row["Type"]).strip(),
        "type_short": str(row["Type Shortform"]).strip(),
        "product_code": str(row["Product Code"]).strip(),
        "title": str(row["Material Description"]).strip(),
        "pack_size": str(row["Pack Size (ml)"]).strip(),
        "specification": build_spec(row),
        "search_text": f"{row['Material Description']} {row['Product Code']} {row['Pack Size (ml)']}",
    }
    records.append(rec)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print("Saved:", OUT_FILE)
print("Total reagent products:", len(records))
