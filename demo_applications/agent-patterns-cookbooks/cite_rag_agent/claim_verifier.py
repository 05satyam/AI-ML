from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool


SYSTEM_PROMPT = """
You are a policy QA assistant.

Rules:
- Use the tool `search_policy` to find evidence in the PDF before answering.
- Answer ONLY using retrieved evidence.
- Always include citations like (page X) in your final answer.
- If the PDF does not contain the answer, say: "Not found in the provided PDF."
""".strip()


def _format_docs_with_citations(docs: List) -> str:
    lines = []
    for d in docs:
        page = d.metadata.get("page", None)
        src = d.metadata.get("source", "pdf")
        snippet = d.page_content.replace("\n", " ").strip()
        snippet = snippet[:900] + ("..." if len(snippet) > 900 else "")
        lines.append(f"[source={src} page={page}] {snippet}")
    return "\n\n".join(lines)


def build_retriever(
    pdf_path: str,
    persist_dir: str = "chroma_claim_db",
    collection_name: str = "claim_pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    k: int = 5,
):
    """
    Load PDF -> chunk -> embed -> persist into Chroma (once) -> return retriever.
    Re-runs will reuse persisted embeddings/index for fast startup.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    emb = OpenAIEmbeddings()

    # Create/load persistent Chroma collection
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=str(persist_path),
    )

    # Determine whether the DB already has docs (safe probe)
    try:
        existing = vs.similarity_search("probe", k=1)
        has_data = len(existing) > 0
    except Exception:
        has_data = False

    # If empty, ingest + persist
    if not has_data:
        loader = PyPDFLoader(str(p))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        vs.add_documents(chunks)
        vs.persist()

    return vs.as_retriever(search_kwargs={"k": k})


def make_search_tool(retriever):
    @tool
    def claim_verifier(query: str) -> str:
        """Search the PDF claim for relevant passages. Returns cited snippets with page numbers."""
        hits = retriever.get_relevant_documents(query)
        if not hits:
            return "No relevant passages found."
        return _format_docs_with_citations(hits)

    return claim_verifier
