from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="Multimodal RAG (Chroma) — Codespaces")

# Mount API routes
app.include_router(api_router, prefix="")

# Root ping
@app.get("/")
def root():
    return {"hello": "world", "docs": "/docs"}
