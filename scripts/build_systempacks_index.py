import os, csv, json, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "embeddings", "systempacks_index.json")
CSV_FILE = os.path.join(ROOT, "data", "SystemPacks.csv")


def normalize(s: str):
    """Convert header to a safe key."""
    s = s.replace("\xa0", " ")       # remove NBSP
    s = re.sub(r"\s+", " ", s)       # collapse spaces
    return s.strip().lower()         # normalize


print("Reading CSV:", CSV_FILE)

items = []

with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    raw_headers = next(reader)

    # Normalize headers
    headers = [normalize(h) for h in raw_headers]

    # Debug print
    print("\nNormalized Headers:")
    for h in headers:
        print(" -", h)

    # Create dict reader using normalized headers
    reader = csv.DictReader(f, fieldnames=headers)

    for row in reader:
        # Access normalized keys safely
        title = row.get("material description", "").strip()
        product_code = row.get("product code", "").strip()
        pack = row.get("pack size", "").strip()

        mrp = row.get("new mrp / kit (with gst)", "").strip()
        ctp = row.get("ctp", "").strip()
        saarthi = row.get("saarthi price", "").strip()
        tp = row.get("new tp (1-50)", "").strip()
        a50 = row.get("assorted 50", "").strip()
        a100 = row.get("assorted 100", "").strip()

        type_val = row.get("type", "").strip()

        spec = (
            f"Pack Size: {pack} | MRP: {mrp} | CTP: {ctp} | "
            f"Saarthi: {saarthi} | TP(1-50): {tp} | Assorted 50: {a50} | Assorted 100: {a100}"
        )

        items.append({
            "title": title,
            "product_code": product_code,
            "type": type_val,
            "specification": spec
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2)

print("\nSaved:", OUT)
print("Total items:", len(items))
