#!/usr/bin/env python3
"""
build_global_index.py
- Reads multiple CSV files (Rapids_Elisa.csv, Meriscreen.csv, SystemPacks.csv, Controls.csv, Pipettes.csv, Turbidimetry.csv,
  Nephelometry.csv, BloodGroup.csv, Analyser.csv, Reagents.csv, etc.)
- Normalizes headers, merges repeated product_code rows (slabs) into single product records
- Outputs: data/embeddings/global_index.json
"""

import csv, os, json, re, unicodedata
from collections import OrderedDict, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "embeddings")
OUT_PATH = os.path.join(OUT_DIR, "global_index.json")

# List CSV files to include (adjust filenames if different)
CSV_FILES = [
    "SystemPacks.csv",
    "Rapids_Elisa.csv",
    "Meriscreen.csv",
    "Pipettes.csv",
    "Turbidimetry_Immuno_Assay.csv",
    "Proviso_Nephelometry.csv",
    "BloodGrouping.csv",
    "ControlsCalibrators.csv",
    "Analyser.csv",
    "Reagents.csv"
]

# Keyword columns mapping attempts — each CSV may use different headers; we'll try many variants
POSSIBLE_FIELDS = {
    "product_code": ["product code", "product_code", "productcode", "product", "code", "Material Code"],
    "title": ["material description", "material", "title", "description", "material description"],
    "pack_size": ["pack size", "pack size / volume", "pack_size", "pack", "volume", "pack size/volume"],
    "mrp": ["mrp", "mrp (rs.)", "new mrp / kit (with gst)", "mrp (₹)"],
    "ctp": ["ctp", "ctp (₹)"],
    "saarthi": ["saarthi", "saarthi price"],
    "tp": ["tp", "new tp (1-50)", "tp (1-50)"],
    "type": ["type", "main type", "full type", "main_type"],
    "category": ["category", "sub type", "sub_type", "master category", "master_category"],
    "slab": ["slab", "slabs", "slab type"],
    "qty_in_slab": ["qty in slab (tests)", "qty in slab", "qty_in_slab", "qty"],
    "price_kit": ["price / kit (₹)", "price / kit", "price / kit (rs.)", "kit price"],
    "price_test": ["price / test (₹)", "price / test", "price / test (rs.)", "test price"]
}

def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", " ", s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def clean(s):
    if s is None:
        return None
    return re.sub(r"\s+", " ", str(s).strip())

def find_header_key(header_map, names):
    for n in names:
        k = norm(n)
        if k in header_map:
            return header_map[k]
    return None

os.makedirs(OUT_DIR, exist_ok=True)
items = OrderedDict()
seen_files = []

print("Scanning CSV files in:", DATA_DIR)
for fname in CSV_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        # skip missing files gracefully
        continue
    seen_files.append(fname)
    print("Processing:", fname)
    with open(fpath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header_map = { norm(h): h for h in reader.fieldnames if h is not None }

        # helper to get canonical field value from row
        def get_field(row, canonical):
            key = find_header_key(header_map, POSSIBLE_FIELDS.get(canonical, []))
            if key:
                return clean(row.get(key))
            return None

        for row in reader:
            pc = get_field(row, "product_code")
            if not pc:
                # try other columns
                pc = clean(row.get(next(iter(row.keys())))) if row else None
                if not pc:
                    continue

            title = get_field(row, "title") or get_field(row, "category") or ""
            pack = get_field(row, "pack_size") or ""
            mrp = get_field(row, "mrp") or ""
            ctp = get_field(row, "ctp") or ""
            saarthi = get_field(row, "saarthi") or ""
            tp = get_field(row, "tp") or ""
            main_type = get_field(row, "type") or ""
            category = get_field(row, "category") or main_type or ""

            slab = get_field(row, "slab") or "Regular"
            qty = get_field(row, "qty_in_slab") or "No Slab"
            price_kit = get_field(row, "price_kit") or ""
            price_test = get_field(row, "price_test") or ""

            # initialize
            if pc not in items:
                items[pc] = {
                    "product_code": pc,
                    "title": title,
                    "pack_size": pack,
                    "mrp": mrp,
                    "ctp": ctp,
                    "saarthi": saarthi,
                    "tp": tp,
                    "type": main_type,
                    "category": category,
                    "source_files": [fname],
                    "slabs": []
                }
            else:
                if fname not in items[pc]["source_files"]:
                    items[pc]["source_files"].append(fname)
                # Update fields if blank
                if not items[pc].get("title") and title:
                    items[pc]["title"] = title
                if not items[pc].get("pack_size") and pack:
                    items[pc]["pack_size"] = pack
                if not items[pc].get("type") and main_type:
                    items[pc]["type"] = main_type
                if not items[pc].get("category") and category:
                    items[pc]["category"] = category

            # append slab record (avoid exact duplicates)
            slab_rec = {
                "slab": slab,
                "qty_in_slab": qty,
                "price_kit": price_kit,
                "price_test": price_test
            }
            if slab_rec not in items[pc]["slabs"]:
                items[pc]["slabs"].append(slab_rec)

# produce merged_text / specification for embeddings
out = []
for pc, it in items.items():
    slabs_text = []
    for s in it["slabs"]:
        slabs_text.append(f"{s.get('slab')} qty:{s.get('qty_in_slab')} kit_price:{s.get('price_kit')} test_price:{s.get('price_test')}")
    spec_parts = []
    if it.get("pack_size"): spec_parts.append(f"Pack Size: {it.get('pack_size')}")
    if it.get("mrp"): spec_parts.append(f"MRP: {it.get('mrp')}")
    if it.get("ctp"): spec_parts.append(f"CTP: {it.get('ctp')}")
    if it.get("saarthi"): spec_parts.append(f"Saarthi: {it.get('saarthi')}")
    if it.get("tp"): spec_parts.append(f"TP: {it.get('tp')}")
    if slabs_text:
        spec_parts.append("SLABS: " + "; ".join(slabs_text))
    specification = " | ".join(spec_parts)

    merged_text = " ".join(filter(None, [
        it.get("title"),
        it.get("type"),
        it.get("category"),
        it.get("pack_size"),
        specification
    ]))

    out.append({
        "product_code": it["product_code"],
        "title": it.get("title"),
        "type": it.get("type"),
        "category": it.get("category"),
        "pack_size": it.get("pack_size"),
        "specification": specification,
        "merged_text": merged_text,
        "source_files": it.get("source_files"),
        "slabs": it.get("slabs")
    })

with open(OUT_PATH, "w", encoding="utf-8") as wf:
    json.dump(out, wf, indent=2, ensure_ascii=False)

print("Saved:", OUT_PATH)
print("Files processed:", len(seen_files))
print("Total items:", len(out))
