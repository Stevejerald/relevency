import numpy as np
import json
import joblib
from sentence_transformers import SentenceTransformer
from .config import (
    ANALYSER_EMB_FILE,
    ANALYSER_META_FILE,
    ANALYSER_TYPE_CLF,
    ANALYSER_TFIDF,
    MODEL_NAME
)
from scripts.analyser_relevancy import predict_relevancy

class AnalyserModel:
    def __init__(self):
        print("[Analyser] Loading embeddings, metadata, classifier, vectorizer...")

        self.emb = np.load(ANALYSER_EMB_FILE)
        with open(ANALYSER_META_FILE, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.clf = joblib.load(ANALYSER_TYPE_CLF)
        self.tfidf = joblib.load(ANALYSER_TFIDF)
        self.encoder = SentenceTransformer(MODEL_NAME)

        print("[Analyser] Model loaded successfully.")

    def predict(self, query: str, top_k: int = 3):
        return predict_relevancy(query, top_k=top_k)


# Single global instance for all FastAPI sessions (best practice)
analyser_model = AnalyserModel()
