from .loader import analyser_model

def analyse_query(query: str, top_k: int = 3):
    return analyser_model.predict(query, top_k)
