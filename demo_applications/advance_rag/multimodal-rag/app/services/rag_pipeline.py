from typing import Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.config import TEXT_MODEL, TOP_K_DOCS

# Single shared LLM client
_llm = ChatOpenAI(model=TEXT_MODEL, temperature=0)

_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful, precise assistant.\n\n"
    "User Query:\n{query}\n\n"
    "Retrieved Context (may be partial):\n{context}\n\n"
    "Using ONLY the context when relevant, answer clearly and concisely. "
    "If the answer is not in the context, say so briefly."
)

def multimodal_rag(query: str, retriever: Any) -> str:
    docs = retriever.get_relevant_documents(query) if retriever else []
    context = "\n\n".join([d.page_content for d in docs][:TOP_K_DOCS]) if docs else ""
    out = (_PROMPT | _llm).invoke({"query": query, "context": context})
    # langchain-openai ChatOpenAI returns an object with `.content`
    return getattr(out, "content", str(out))
