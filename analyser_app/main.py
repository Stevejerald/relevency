from fastapi import FastAPI
from .routers import router as analyser_router

app = FastAPI(
    title="Analyser Model API",
    version="1.0.0",
    description="Production-grade semantic product-matching engine (Analyser)",
)

app.include_router(analyser_router)

@app.get("/")
async def root():
    return {"message": "Analyser Model API Running"}
