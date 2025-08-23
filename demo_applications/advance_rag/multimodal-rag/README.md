# Multimodal RAG with FastAPI + Chroma + CLIP + OpenAI

This project demonstrates a **Multimodal Retrieval-Augmented Generation (RAG)** system.  
It retrieves **text and image data**, augments queries with context, and generates grounded answers using **OpenAI GPT-4o**.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a method that combines:

1. **Retriever** → finds relevant content from a knowledge base using vector search.  
2. **Augmenter** → adds the retrieved content to the user’s query.  
3. **Generator** → an LLM produces an answer grounded in retrieved data.  

**Why RAG?**
- Reduces hallucinations.  
- Makes LLMs up-to-date with external data.  
- Allows multimodal search (text + images).

---

## Why Multimodal?

Traditional RAG is **text-only**.  
This project extends it to support **images**, so you can:

- Ask questions about **documents** (PDFs, TXT, Markdown).  
- Upload **images** and retrieve visually similar ones.  
- Combine both into one **assistant that understands text + images**.

---

## Tech Stack

- **FastAPI** → API framework  
- **LangChain** → Retriever + LLM orchestration  
- **ChromaDB** → Vector database for text & image embeddings  
- **OpenAI GPT-4o** → LLM + text embeddings  
- **CLIP (OpenAI)** → Image embeddings  
- **Transformers (Hugging Face)** → Model integration  
- **Uvicorn** → ASGI server  

---

## Why CLIP for Images?

We use **CLIP (Contrastive Language–Image Pretraining)** because it:

- Trained on **400M image–text pairs**, aligning both into the same vector space.  
- Works in **zero-shot** mode → no fine-tuning needed.  
- Open-source and easy to use.  
- Great for demos and workshops.  

Other options: **ALIGN, CoCa, VLM2Vec, Cohere Embed-3, Voyage, Nomic Embed Vision**.  
For simplicity, **CLIP is the best tradeoff between performance and ease of use**.

---

## Why L2 Normalization?

After generating embeddings, we apply **L2 normalization**:

- Converts **cosine similarity** into a dot product → faster retrieval.  
- Ensures **fair comparisons** across embeddings.  
- Prevents bias from vector magnitude.  

This is standard in vector search systems.

---

## How Data is Stored

### Text (ChromaDB)
- **Ingest** → Documents are split into passages.  
- **Embed** → Each passage encoded using OpenAI `text-embedding-3-large`.  
- **Store** → Embeddings + text chunks + metadata stored in **ChromaDB**.  
- **Query** → A question is embedded and compared → top-k results returned.  

### Images (CLIP)
- I did in-memory due to compuatation-limitation.

#### Option 1: In-Memory (Workshop Default)
- Images embedded with CLIP → 512-dim vectors.  
- Stored in memory:  
  - List of file paths.  
  - Matrix of embeddings.  
- Query → Upload image → embed → cosine similarity → return top-k matches.  

#### Option 2: Persist in ChromaDB
- Generate CLIP embeddings for each image.  
- Insert into **ChromaDB collection** with metadata:
  ```json
  {
    "id": "123",
    "embedding": [0.12, 0.34, ...],
    "metadata": {"filename": "dog.png", "path": "/data/images/dog.png"}
  }
````

* Query → embed new image → Chroma similarity search → return stored image paths.

**Pro**: Persistence across restarts.
**Con**: Requires precomputation and metadata management.

---

## 🔄 Workflow

```mermaid
flowchart TD

A[User Query] --> B{Type?}
B -->|Text| C[Embed with OpenAI text-embedding-3-large]
B -->|Image| D[Embed with CLIP vision encoder]

C --> E[ChromaDB (Text Collection)]
D --> F[ChromaDB (Image Collection or In-Memory Index)]

E --> G[Retrieve top-k text chunks]
F --> H[Retrieve top-k similar images]

G --> I[Augment Query with Context]
H --> I

I --> J[LLM (GPT-4o) Generates Final Answer]
J --> K[Response to User]
```

---

## 🚀 How to Run

1. **Clone and setup**

   ```bash
   git clone <your-repo>
   cd multimodal-rag
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Configure `.env`**

   ```env
   OPENAI_API_KEY=sk-xxx
   TEXT_MODEL=gpt-4o
   TEXT_EMBED_MODEL=text-embedding-3-large
   IMAGE_MODEL=clip
   CHROMA_DIR=./chroma_store
   TOP_K_DOCS=3
   TOP_K_IMAGES=3
   ```

4. **Prepare data**

   ```bash
   mkdir -p data/documents data/images
   # Add PDFs/TXT/MD into data/documents/
   # Add PNG/JPG into data/images/
   ```

5. **Run the server**

   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Test via Swagger**

   * Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * Use:

     * `/rag` → Ask text-based questions.
     * `/search-image` → Upload an image for similarity search.

---

## Example Queries

### Text

```json
POST /rag
query = "Who wrote 'Attention Is All You Need'?"
```

### Answer: 
```{
  "query": "tell me about the authors of attentional all u need",
  "answer": "The authors of \"Attention Is All You Need\" are:\n\n1. Ashish Vaswani (Google Brain)\n2. Noam Shazeer (Google Brain)\n3. Niki Parmar (Google Research)\n4. Jakob Uszkoreit (Google Research)\n5. Llion Jones (Google Research)\n6. Aidan N. Gomez (University of Toronto, work performed while at Google Brain)\n7. Łukasz Kaiser (Google Brain)\n8. Illia Polosukhin (work performed while at Google Research)\n\nThe authors contributed equally to the work, and the listing order is random.",
  "docs_indexed": 15
}
```

### Image

* Upload a cat image → returns top-3 closest matches from `data/images/`.

---

## Some tips on troubleshooting which I encountered during implementations: 

* **`python-multipart not installed`**
  → Run: `pip install python-multipart`

* **Torch install fails (macOS)**
  → Run: `pip install torch==2.2.2`

* **Chroma telemetry warnings**
  → Set: `export CHROMA_TELEMETRY_DISABLED=1`

* **LangChain warning `get_relevant_documents`**
  → Use `.invoke()` instead (already patched in pipeline).

---

## ✅ Summary

* **RAG** grounds LLMs with external knowledge.
* **CLIP** enables multimodal embeddings (text + images).
* **L2 normalization** ensures robust similarity search.
* **Chroma** provides persistent storage.
* This project = **simple, extensible, and workshop-ready multimodal RAG system**.


💡```
This is a small project considering how an industry setup can look like. 
With this, you can now build assistants that understand **both documents and images**.
Additionally we can add docker to run container and publish containerized image to deploy anywhere.

```
