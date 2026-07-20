# Corrective RAG (CRAG)

A hands-on implementation of **Corrective Retrieval-Augmented Generation**, extending naive RAG with a
retrieval-evaluation step that grades, rewrites, and falls back to web search when local retrieval fails.

## Why CRAG?

Naive RAG blindly trusts whatever the retriever returns. If the vector store retrieves irrelevant chunks,
the LLM still generates an answer from them — often producing a confident, hallucinated response.

CRAG adds a grading step between retrieval and generation:

```text
                    User Question
                          │
                          ▼
                Retrieve Documents (ChromaDB)
                          │
                          ▼
                Grade Document Relevance
                          │
              ┌───────────┴───────────┐
              │                       │
          Relevant                Not Relevant
              │                       │
              ▼                       ▼
       Generate Answer          Rewrite Query
                                      │
                                      ▼
                                 Web Search
                                      │
                                      ▼
                               Generate Answer
```

## Stack

| Component  | Tool |
|---|---|
| Orchestration | LangChain + LangGraph |
| Embeddings | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| LLM | Groq |
| Web search fallback | Tavily |
| Sample corpus | "Attention Is All You Need" (`data/attention.pdf`) |

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
```

## Run

Open `crag_notebook.ipynb` and run all cells top to bottom. The notebook:

1. Loads and chunks the sample PDF (`data/attention.pdf`).
2. Embeds chunks into ChromaDB.
3. Runs a baseline naive-RAG query for comparison.
4. Builds the CRAG graph: retrieve → grade → (generate) or (rewrite → web search → generate).
5. Renders the compiled LangGraph as a diagram.

## Files

```
CRAG/
├── crag_notebook.ipynb   # main notebook
├── data/
│   └── attention.pdf     # sample corpus
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- Relevance grading here is binary (relevant / not relevant). The original CRAG paper uses a three-way
  grade (Correct / Ambiguous / Incorrect) with strip-level knowledge refinement on the "Correct" path —
  noted here as a natural follow-up extension, not implemented in this notebook.
- Query rewriting targets the *web search* step, not a second local vector-store query — since the local
  knowledge base's content hasn't changed, re-querying it with a reworded question rarely helps.