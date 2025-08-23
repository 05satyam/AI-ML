import os
from typing import Dict, Any
import numpy as np
from fastapi import APIRouter, UploadFile

from app.services.retriever import (
    load_text_documents,
    build_text_retriever,
    build_image_index,
    top_k_similar_images,
    default_top_k_images,
)
from app.services.embeddings import generate_image_embedding
from app.services.rag_pipeline import multimodal_rag
from app.core.config import DOCS_DIR, IMAGES_DIR

router = APIRouter()

# Build assets at import time (simple & fine for workshop)
_docs = load_text_documents(str(DOCS_DIR))
_text_retriever = build_text_retriever(_docs)
_img_paths, _img_matrix = build_image_index(str(IMAGES_DIR))

@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@router.post("/rag")
def rag_endpoint(query: str) -> Dict[str, Any]:
    answer = multimodal_rag(query, _text_retriever)
    return {"query": query, "answer": answer, "docs_indexed": len(_docs)}

@router.post("/search-image")
def search_image_endpoint(file: UploadFile) -> Dict[str, Any]:
    if not file or not file.filename:
        return {"matches": [], "message": "No file uploaded."}

    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(file.file.read())

    q_vec = generate_image_embedding(tmp)
    if _img_matrix.shape[0] == 0:
        return {"matches": [], "message": "No images indexed in data/images."}

    idxs = top_k_similar_images(q_vec, _img_matrix, default_top_k_images())
    matches = [{"path": _img_paths[i], "rank": r + 1} for r, i in enumerate(idxs)]
    try:
        os.remove(tmp)
    except Exception:
        pass
    return {"matches": matches, "total_indexed": int(_img_matrix.shape[0])}
