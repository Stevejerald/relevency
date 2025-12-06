# build_pipettes_index.py
import os, csv, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT, "data", "Pipettes.csv")
OUT_FILE = os.path.join(ROOT, "data", "embeddings", "pipettes_index.json")

def normalize_key(k):
    return k.strip().lower()

print(f"Reading CSV: {DATA_FILE}")

rows = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = [normalize_key(h) for h in reader.fieldnames]

    print("\nNormalized Headers:")
    for h in headers:
        print(" -", h)

    for raw in reader:
        row = {normalize_key(k): v.strip() for k, v in raw.items()}

        title = row["material description"]
        volume = row["volume (µl)"]

        spec = (
            f"Type: {row['type']} | Volume: {volume} | "
            f"MRP: {row['mrp']} | CTP: {row['ctp']} | "
            f"Saarthi: {row['saarthi']} | TP: {row['tp']}"
        )

        rows.append({
            "title": title,
            "product_code": row["product code"],
            "type": row["type"],
            "volume": volume,
            "specification": spec
        })

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

print(f"\nSaved: {OUT_FILE}")
print(f"Total items: {len(rows)}")
