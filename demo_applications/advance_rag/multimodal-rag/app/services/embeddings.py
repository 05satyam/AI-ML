from typing import Optional
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_openai import OpenAIEmbeddings
from app.core.config import TEXT_EMBED_MODEL

# ---------- Text Embeddings (OpenAI) ----------
def get_text_embeddings() -> OpenAIEmbeddings:
    # Uses OPENAI_API_KEY from env automatically via langchain-openai
    return OpenAIEmbeddings(model=TEXT_EMBED_MODEL)

# ---------- Image Embeddings (CLIP) ----------
# Loaded once per process
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, ord=2)
    return x / (denom + 1e-12)

def generate_image_embedding(image_path: str) -> np.ndarray:
    """
    Returns a 512-dim CLIP image embedding (float32, L2-normalized).
    """
    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)  # [1, 512]
    vec = feats[0].cpu().numpy().astype("float32")
    return _l2_normalize(vec)
