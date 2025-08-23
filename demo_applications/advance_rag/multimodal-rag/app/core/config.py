import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ==== Base Paths ====
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_store")

# Ensure folders exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

# ==== API Keys ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==== Model Configurations ====
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "text-embedding-3-large")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "clip")  # placeholder for future switches

# ==== Retrieval Settings ====
TOP_K_DOCS = int(os.getenv("TOP_K_DOCS", 1))
TOP_K_IMAGES = int(os.getenv("TOP_K_IMAGES", 1))

# ==== Validate Required Keys ====
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")
