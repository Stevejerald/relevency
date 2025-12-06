#!/usr/bin/env python3
import csv, os, json, re, unicodedata
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "data", "Meriscreen.csv")
OUT_PATH = os.path.join(ROOT, "data/embeddings/meriscreen_index.json")

def norm(s):
    """Normalize header/text to consistent lookup key"""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", "", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def clean(s):
    return re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else s


# ------------------------------------------------------
# NEW IMPORTANT FIX — CLEAN PRODUCT CODE COMPLETELY
# Removes hidden unicode, NBSP, spaces inside code, etc.
# ------------------------------------------------------
def clean_product_code(pc):
    if not isinstance(pc, str):
        return ""
    # Normalize unicode
    pc = unicodedata.normalize("NFKD", pc)

    # Remove ALL invisible characters even if inside text
    pc = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", "", pc)

    # Remove everything except A–Z, 0–9, and hyphen
    pc = re.sub(r"[^A-Za-z0-9\-]", "", pc)

    # Final strip
    return pc.strip()


print("Reading CSV:", CSV_PATH)
items = OrderedDict()

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    header_map = {norm(k): k for k in reader.fieldnames}

    def get(row, *names):
        for n in names:
            key = norm(n)
            if key in header_map:
                original = header_map[key]
                val = row.get(original)
                return clean(val) if val is not None else None
        return None


    for row in reader:
        # --------------------------
        # FETCH + CLEAN PRODUCT CODE
        # --------------------------
        raw_pc = get(row, "product code", "product_code", "code")
        pc = clean_product_code(raw_pc)

        if not pc:
            continue

        title = get(row, "material description")
        pack = get(row, "pack size / volume", "pack size", "pack", "size")

        main_type = get(row, "type")
        master_cat = get(row, "master category")
        category = get(row, "category")

        slab = get(row, "slab") or "Regular"
        qty = get(row, "qty in slab (tests)", "qty in slab", "qty") or "No Slab"
        price_kit = get(row, "price / kit (₹)", "price / kit", "kit price")
        price_test = get(row, "price / test (₹)", "price / test", "test price")

        mrp = get(row, "mrp")
        ctp = get(row, "ctp")
        saarthi = get(row, "saarthi")
        tp = get(row, "tp")

        # Initialize unique product entry
        if pc not in items:
            items[pc] = {
                "product_code": pc,
                "title": title,
                "type": main_type,
                "master_category": master_cat,
                "category": category,
                "pack_size": pack,
                "mrp": mrp,
                "ctp": ctp,
                "saarthi": saarthi,
                "tp": tp,
                "slabs": []
            }

        # Store slab information
        items[pc]["slabs"].append({
            "slab": slab,
            "qty_in_slab": qty,
            "price_kit": price_kit,
            "price_test": price_test
        })


# ------------------------------------------------------
# Build merged output
# ------------------------------------------------------
out = []
for pc, it in items.items():
    slabs_text = [
        f"{s['slab']} qty:{s['qty_in_slab']} kit_price:{s['price_kit']} test_price:{s['price_test']}"
        for s in it["slabs"]
    ]

    spec = " | ".join(filter(None, [
        f"Pack Size: {it['pack_size']}",
        f"MRP: {it['mrp']}",
        f"CTP: {it['ctp']}",
        f"Saarthi: {it['saarthi']}",
        f"TP: {it['tp']}",
        "SLABS: " + "; ".join(slabs_text)
    ]))

    merged_text = " ".join(filter(None, [
        it["title"], it["type"], it["master_category"], it["category"],
        it["pack_size"], spec
    ]))

    out.append({
        "product_code": pc,
        "title": it["title"],
        "type": it["type"],
        "master_category": it["master_category"],
        "category": it["category"],
        "pack_size": it["pack_size"],
        "specification": spec,
        "merged_text": merged_text
    })


os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as wf:
    json.dump(out, wf, indent=2, ensure_ascii=False)

print("Saved:", OUT_PATH)
print("Total items:", len(out))
