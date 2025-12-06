#!/usr/bin/env python3
"""
build_rapids_elisa_index.py
- Reads Rapids & ELISA CSV (with repeated Product Code / slab rows)
- Merges rows by Product Code into one unique item per product_code
- Produces metadata JSON to be used for embedding generation
"""
import csv, os, json, sys, re
from collections import OrderedDict
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "Rapids_Elisa.csv")
OUT_PATH = os.path.join(ROOT, "data", "embeddings", "rapids_elisa_index.json")


# -------------------------------------------------
# Normalization Helper
# -------------------------------------------------
def normalize(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


print("Reading CSV:", CSV_PATH)

items = OrderedDict()

with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # Normalize header names
    normalized_headers = {normalize(k): k for k in reader.fieldnames}

    for r in reader:
        # Build normalized row
        row = {}
        for orig_key, value in r.items():
            nkey = normalize(orig_key)
            row[nkey] = value.strip() if isinstance(value, str) else value

        # ------------------------------------------------------------------
        # Mandatory key: product code
        # ------------------------------------------------------------------
        pc = (
            row.get("product code")
            or row.get("product_code")
            or row.get("productcode")
        )
        if not pc:
            continue
        pc = pc.strip()

        # ------------------------------------------------------------------
        # Extract base item fields (only on first occurrence)
        # ------------------------------------------------------------------
        if pc not in items:
            items[pc] = {
                "product_code": pc,
                "title": row.get("material description"),
                "main_type": row.get("main type"),
                "sub_type": row.get("sub type")
                    or row.get("category")
                    or row.get("full type"),
                "pack_size": row.get("pack size / volume"),
                "mrp": row.get("mrp"),
                "ctp": row.get("ctp"),
                "saarthi": row.get("saarthi"),
                "tp": row.get("tp"),
                "slabs": []
            }

        # ------------------------------------------------------------------
        # Append slab details
        # ------------------------------------------------------------------
        slab = row.get("slab") or "Regular"
        qty = row.get("qty in slab (tests)") or row.get("qty in slab") or ""
        price_kit = row.get("price / kit (₹)") or row.get("price / kit")
        price_test = row.get("price / test (₹)") or row.get("price / test")

        items[pc]["slabs"].append({
            "slab": slab,
            "qty_in_slab": qty,
            "price_kit": price_kit,
            "price_test": price_test
        })


# -------------------------------------------------
# Build output list (merged) for embeddings
# -------------------------------------------------
out_list = []

for pc, it in items.items():
    slab_parts = [
        f"{s['slab']} qty:{s['qty_in_slab']} kit_price:{s['price_kit']} test_price:{s['price_test']}"
        for s in it["slabs"]
    ]

    spec = " | ".join(filter(None, [
        f"Pack Size: {it['pack_size']}" if it.get("pack_size") else None,
        f"MRP: {it['mrp']}" if it.get("mrp") else None,
        f"CTP: {it['ctp']}" if it.get("ctp") else None,
        f"TP: {it['tp']}" if it.get("tp") else None,
        "SLABS: " + "; ".join(slab_parts)
    ]))

    merged_text = " ".join(filter(None, [
        it.get("title"),
        it.get("main_type"),
        it.get("sub_type"),
        it.get("pack_size"),
        spec
    ]))

    out_list.append({
        "product_code": pc,
        "title": it["title"],
        "main_type": it["main_type"],
        "sub_type": it["sub_type"],
        "pack_size": it["pack_size"],
        "specification": spec,
        "merged_text": merged_text
    })


# -------------------------------------------------
# Save JSON
# -------------------------------------------------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as wf:
    json.dump(out_list, wf, indent=2, ensure_ascii=False)

print("Saved:", OUT_PATH)
print("Total items:", len(out_list))
