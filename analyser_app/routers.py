from fastapi import APIRouter
from .models import AnalyseRequest, AnalyseResponse
from .service import analyse_query

router = APIRouter(prefix="/analyser", tags=["Analyser Model"])

@router.post("/predict", response_model=AnalyseResponse)
async def predict(req: AnalyseRequest):
    res = analyse_query(req.query, req.top_k)
    return res

@router.get("/health")
async def health():
    return {"status": "ok", "service": "Analyser Model API"}
