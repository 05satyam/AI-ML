# Corrective RAG (CRAG)

This project demonstrates a **Corrective Retrieval-Augmented Generation (CRAG)** pipeline using **LangChain**, **LangGraph**, **ChromaDB**, **Hugging Face Embeddings**, **Groq LLM**, and **Tavily Web Search**.

Unlike a traditional Retrieval-Augmented Generation (RAG) system, this implementation evaluates the retrieved documents before generating a response. If the retrieved context is not sufficiently relevant, the system rewrites the query and performs a web search to retrieve better information.

---

# What is RAG?

**Retrieval-Augmented Generation (RAG)** combines three major components:

1. **Retriever** → Retrieves relevant information from a knowledge base.
2. **Augmenter** → Adds the retrieved context to the user query.
3. **Generator** → Uses an LLM to generate an answer grounded in the retrieved context.

### Why RAG?

- Reduces hallucinations.
- Enables LLMs to answer using private documents.
- Makes responses more accurate and trustworthy.
- Allows knowledge to be updated without retraining the model.

---

# Why Corrective RAG?

Traditional RAG assumes that the retriever always returns useful information.

In practice, retrieval may fail because:

- irrelevant chunks are retrieved,
- important information is missing,
- the user's question is poorly phrased.

CRAG introduces a **retrieval evaluation step** before generation.

If the retrieved documents are relevant, the answer is generated immediately.

Otherwise, the system:

- rewrites the user query,
- performs a web search,
- augments the retrieved context,
- generates the final response.

This significantly improves answer quality when the local knowledge base cannot answer the question.

---

# Tech Stack

| Component | Tool |
|-----------|------|
| Orchestration | LangChain + LangGraph |
| Embeddings | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| LLM | Groq |
| Web Search | Tavily |
| Document Loader | PyPDF |
| Environment Variables | python-dotenv |

---

# Workflow

```
                 User Question
                       │
                       ▼
          Retrieve Documents (ChromaDB)
                       │
                       ▼
         Grade Retrieved Documents
                       │
          ┌────────────┴────────────┐
          │                         │
     Relevant                  Not Relevant
          │                         │
          ▼                         ▼
 Generate Response           Rewrite Query
                                    │
                                    ▼
                               Tavily Search
                                    │
                                    ▼
                           Generate Response
```

---

# Project Structure

```
CRAG/
├── crag_code.ipynb
├── data/
│   └── attention.pdf
├── README.md
├── requirements.txt
└── .env.example
```

---

# How It Works

## Step 1 — Document Processing

- Load the sample PDF.
- Split the document into smaller chunks.
- Generate embeddings using Hugging Face.
- Store embeddings inside ChromaDB.

---

## Step 2 — Retrieval

When a user asks a question:

- the query is embedded,
- ChromaDB retrieves the most relevant chunks.

---

## Step 3 — Retrieval Evaluation

The retrieved documents are evaluated by the LLM.

Two outcomes are possible:

### Relevant

The retrieved context is sufficient.

→ Generate the answer.

### Not Relevant

The retrieved context is insufficient.

→ Rewrite the query.

→ Perform Tavily Web Search.

→ Generate the answer using the web results.

---

# Setup

Clone the repository and install the dependencies.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`.

```env
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
MODEL_NAME=openai/gpt-oss-20b
```

---

# Run

Open the notebook and execute every cell from top to bottom.

The notebook demonstrates:

1. Loading and chunking the sample PDF.
2. Creating vector embeddings.
3. Building a Chroma vector database.
4. Running a baseline Naive RAG query.
5. Constructing the CRAG workflow using LangGraph.
6. Demonstrating the web-search correction path.
7. Visualizing the LangGraph workflow.

---

# Example Queries

### Local Knowledge Base

```
What is self-attention?
```

The answer is generated using the local PDF stored in ChromaDB.

---

### Web Search Fallback

```
Who won the FIFA World Cup 2022?
```

Since this information is not available in the PDF:

- Retrieval is graded as **Not Relevant**.
- The query is rewritten.
- Tavily performs a web search.
- The final answer is generated using the retrieved web content.

---

# Notes

- This notebook implements a **simplified version of Corrective RAG**.
- Document grading is **binary** (Relevant / Not Relevant).
- Query rewriting is performed **only before web search**.
- The original CRAG paper additionally introduces:
  - Correct / Ambiguous / Incorrect grading,
  - knowledge strip decomposition,
  - strip refinement,
  - strip recomposition.

These advanced components are intentionally omitted to keep the notebook simple and beginner-friendly.

---

# Security Note

Retrieved documents and web-search results are inserted directly into the prompt as context. Since this
content is untrusted (especially live web results), it is a potential **indirect prompt-injection** vector —
a malicious page or document could try to override the system instructions. This demo does not harden
against that. For production use, wrap untrusted context in clear delimiters and instruct the model to
treat it strictly as data, never as instructions.

---

# Summary

- Demonstrates a practical Corrective RAG pipeline.
- Uses LangGraph for workflow orchestration.
- Uses ChromaDB for document retrieval.
- Uses Groq for answer generation.
- Uses Tavily as a web-search fallback.
- Shows how retrieval evaluation can improve RAG quality over a traditional pipeline.