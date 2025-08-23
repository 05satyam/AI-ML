import os
from typing import List, Tuple
import numpy as np

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from chromadb.config import Settings

from app.services.embeddings import get_text_embeddings, generate_image_embedding
from app.core.config import DOCS_DIR, IMAGES_DIR, CHROMA_DIR, TOP_K_DOCS, TOP_K_IMAGES

# --------- Load text documents (PDF/TXT/MD as plain text) ----------
def load_text_documents(path: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(path):
        return docs

    for fname in sorted(os.listdir(path)):
        full = os.path.join(path, fname)
        if not os.path.isfile(full):
            continue
        try:
            if fname.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(full).load())
            elif fname.lower().endswith(".md") or fname.lower().endswith(".txt"):
                # Simple text loader for MD/TXT to avoid heavy deps
                docs.extend(TextLoader(full, encoding="utf-8").load())
        except Exception as e:
            # Skip bad files but don't crash the app
            print(f"[loader] Skipping {fname}: {e}")
    return docs

# --------- Build Chroma text retriever ----------
def build_text_retriever(docs: List[Document]):
    if not docs:
        # Return a tiny adapter retriever that yields nothing
        class _EmptyRetriever:
            def get_relevant_documents(self, _q): return []
        return _EmptyRetriever()

    embedder = get_text_embeddings()
    client_settings = Settings(
        anonymized_telemetry=False,   # <— kill telemetry
        is_persistent=True
    )
    vectorstore = Chroma.from_texts(
        [d.page_content for d in docs],
        embedding=embedder,
        persist_directory=str(CHROMA_DIR),
        client_settings=client_settings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_DOCS})

# --------- Build image "index">  Note: [in memory] ----------
def build_image_index(image_dir: str) -> Tuple[List[str], np.ndarray]:
    paths: List[str] = []
    if os.path.isdir(image_dir):
        for fname in sorted(os.listdir(image_dir)):
            full = os.path.join(image_dir, fname)
            if os.path.isfile(full) and fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                paths.append(full)

    if not paths:
        return [], np.zeros((0, 512), dtype="float32")

    embs = []
    for p in paths:
        try:
            embs.append(generate_image_embedding(p))
        except Exception as e:
            print(f"[images] Skipping {p}: {e}")
    if not embs:
        return [], np.zeros((0, 512), dtype="float32")

    img_matrix = np.vstack(embs)  # [N, 512]
    return paths, img_matrix

# --------- Image search by cosine similarity (on normalized vectors) ----------
def top_k_similar_images(q_vec: np.ndarray, img_matrix: np.ndarray, k: int) -> List[int]:
    if img_matrix.shape[0] == 0:
        return []
    # Cosine similarity since vectors are L2-normalized: sim = dot(q, X.T)
    sims = img_matrix @ q_vec  # [N]
    order = np.argsort(-sims)[:k]
    return order.tolist()

def default_top_k_images() -> int:
    return TOP_K_IMAGES
