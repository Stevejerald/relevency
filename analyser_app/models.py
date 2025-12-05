from pydantic import BaseModel

class AnalyseRequest(BaseModel):
    query: str
    top_k: int = 3

class AnalyseResponse(BaseModel):
    query: str
    relevancy_score: float
    relevant: bool
    best_match: dict | None
    top_matches: list
    predicted_type: str | None
    predicted_type_conf: float | None
    keyword_type_detected: str | None
