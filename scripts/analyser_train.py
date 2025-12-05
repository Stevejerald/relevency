import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
from openai import OpenAI

# -------------------------
# CONFIG
# -------------------------
DATA_FILE = r"D:\Tender System\Model\relevency\data\Analyser.csv"

EMB_DIR = r"D:\Tender System\Model\relevency\data\embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

EMB_OUT = os.path.join(EMB_DIR, "analyser_embeddings.npy")
INDEX_OUT = os.path.join(EMB_DIR, "analyser_index.json")
TYPE_MODEL_OUT = os.path.join(EMB_DIR, "analyser_type_model.pkl")
TYPE_LABELS_OUT = os.path.join(EMB_DIR, "analyser_type_labels.json")

client = OpenAI()


# ----------------------------------------
# STEP 1 — Load CSV and clean entries
# ----------------------------------------
df = pd.read_csv(DATA_FILE)

def clean_record(row):
    title = str(row["Name of Instrument"]).strip()
    product_code = str(row["Product Code"]).strip()

    type_name = str(row["Type"]).strip()
    segment = str(row["Segment.1"]).strip()

    # Build enriched specification text
    spec_parts = [
        f"Type: {type_name}",
        f"Segment: {segment}",
        f"Warranty: {row['Warranty']}",
        f"FOC Reagents: {row['FOC Reagents on MRP Rs.']}",
        f"Payment Terms: {row['Payment Terms']}"
    ]

    specification = " | ".join([p for p in spec_parts if p and p != "nan"])

    return {
        "title": title,
        "product_code": product_code,
        "type": type_name,
        "segment": segment,
        "specification": specification
    }

products = [clean_record(r) for _, r in df.iterrows()]

print(f"Loaded products: {len(products)}")


# ----------------------------------------
# STEP 2 — Build classifier training data
# ----------------------------------------
train_texts = []
train_labels = []

for p in products:
    train_texts.append(p["title"] + " " + p["specification"])
    train_labels.append(p["type"])

lbl = LabelEncoder()
y = lbl.fit_transform(train_labels)

# Save label encoder classes for inference
with open(TYPE_LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump(list(lbl.classes_), f, indent=2)

print("Type classes:", lbl.classes_)


# ----------------------------------------
# STEP 3 — Generate embeddings for classifier training
# ----------------------------------------
def embed_texts(text_list):
    CHUNK = 50
    out = []

    for i in range(0, len(text_list), CHUNK):
        batch = text_list[i:i+CHUNK]

        res = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )

        for item in res.data:
            out.append(item.embedding)

    return np.array(out, dtype=np.float32)


print("Generating classifier embeddings...")
X = embed_texts(train_texts)

print("Classifier embedding shape:", X.shape)


# ----------------------------------------
# STEP 4 — Train logistic regression classifier
# ----------------------------------------
clf = LogisticRegression(max_iter=2000)
clf.fit(X, y)

# Save the classifier
with open(TYPE_MODEL_OUT, "wb") as f:
    pickle.dump(clf, f)

print("Saved type classifier →", TYPE_MODEL_OUT)


# ----------------------------------------
# STEP 5 — Semantic Embeddings for product matching
# ----------------------------------------
semantic_corpus = [
    p["title"] + " | " + p["specification"]
    for p in products
]

print("Generating product semantic embeddings...")
semantic_embeddings = embed_texts(semantic_corpus)

np.save(EMB_OUT, semantic_embeddings)
print("Saved product embeddings →", EMB_OUT)


# ----------------------------------------
# STEP 6 — Save metadata index
# ----------------------------------------
with open(INDEX_OUT, "w", encoding="utf-8") as f:
    json.dump(products, f, indent=2)

print("Saved product index →", INDEX_OUT)

print("\n🎉 TRAINING COMPLETE!")
print("Classifier + Embeddings + Index are all ready.")
