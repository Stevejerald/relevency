import os, json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "ControlsCalibrators.csv"
OUT_DIR = ROOT / "data" / "unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "controls_products.json"

def build_spec(row):
    return " | ".join([
        f"{col}: {row[col]}"
        for col in row.index
        if col not in ["Material Code", "Material Description"] and pd.notna(row[col])
    ])

df = pd.read_csv(INPUT_FILE)

records = []
for _, r in df.iterrows():
    records.append({
        "title": r["Material Description"],
        "product_code": r["Material Code"],
        "type": "Controls & Calibrators",
        "specification": build_spec(r),
        "search_text": f"{r['Material Description']} {r['Material Code']} {r['Pack Size (ml)']}"
    })

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print("Saved:", OUT_JSON)
print("Total:", len(records))
