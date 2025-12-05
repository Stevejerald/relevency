import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EMB_DIR = os.path.join(ROOT, "data", "embeddings")

ANALYSER_EMB_FILE = os.path.join(EMB_DIR, "analyser_embeddings.npy")
ANALYSER_META_FILE = os.path.join(EMB_DIR, "analyser_index.json")
ANALYSER_TYPE_CLF = os.path.join(EMB_DIR, "type_clf.joblib")
ANALYSER_TFIDF = os.path.join(EMB_DIR, "type_tfidf.joblib")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
