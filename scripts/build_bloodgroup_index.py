#!/usr/bin/env python3
import csv, os, json, re, unicodedata
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "BloodGrouping.csv")
OUT_PATH = os.path.join(ROOT, "data/embeddings/bloodgroup_index.json")


# --------------------
# Helpers
# --------------------
def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", "", s)
    s = s.strip().lower()
    return re.sub(r"\s+", " ", s)


def clean(s):
    return re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else s


print("Reading CSV:", CSV_PATH)

items = OrderedDict()

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    header_map = {norm(k): k for k in reader.fieldnames}

    def get(row, *names):
        for n in names:
            k = norm(n)
            if k in header_map:
                return clean(row[header_map[k]])
        return None

    for row in reader:

        code = get(row, "material code", "code", "product code")
        if not code:
            continue

        title = get(row, "material description")
        pack = get(row, "pack size")
        slab = get(row, "slab") or "Regular"
        qty = get(row, "qty in slab", "qty in slab (tests)", "qty") or "No Slab"
        mrp = get(row, "mrp (₹)", "mrp")
        ctp = get(row, "ctp (₹)", "ctp")
        saarthi = get(row, "saarthi (₹)", "saarthi")
        price_kit = get(row, "price / kit", "price / kit as per slab (₹)", "kit price")

        # Initialize product entry
        if code not in items:
            items[code] = {
                "product_code": code,
                "title": title,
                "pack_size": pack,
                "mrp": mrp,
                "ctp": ctp,
                "saarthi": saarthi,
                "slabs": []
            }

        # Append slab info
        items[code]["slabs"].append({
            "slab": slab,
            "qty_in_slab": qty,
            "price_kit": price_kit
        })


# --------------------
# Build final merged output
# --------------------
out = []
for code, it in items.items():

    slabs_text = [
        f"{s['slab']} qty:{s['qty_in_slab']} kit_price:{s['price_kit']}"
        for s in it["slabs"]
    ]

    spec = " | ".join(filter(None, [
        f"Pack Size: {it['pack_size']}",
        f"MRP: {it['mrp']}",
        f"CTP: {it['ctp']}",
        f"Saarthi: {it['saarthi']}",
        "SLABS: " + "; ".join(slabs_text)
    ]))

    merged_text = " ".join(filter(None, [
        it["title"],
        it["pack_size"],
        spec
    ]))

    out.append({
        "product_code": code,
        "title": it["title"],
        "pack_size": it["pack_size"],
        "specification": spec,
        "merged_text": merged_text
    })


os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as wf:
    json.dump(out, wf, indent=2, ensure_ascii=False)

print("Saved:", OUT_PATH)
print("Total items:", len(out))
