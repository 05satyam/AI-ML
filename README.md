<h1 align="center">AI, ML, Deep Learning & LLMs — Content & Cookbooks</h1>

<p align="center">
  A curated, hands-on library of notebooks, demos, and resources for AI/ML, Deep Learning, Generative AI, RAG, agents, fine-tuning, and modern tooling.
</p>

<p align="center">
  <a href="https://github.com/05satyam/AI-ML" target="_blank">
    <img src="https://img.shields.io/badge/Explore_Repository-black?style=for-the-badge&logo=github" alt="AI-ML Repository"/>
  </a>
  <a href="https://www.linkedin.com/in/satyam-sm" target="_blank">
    <img src="https://img.shields.io/badge/Connect_on_LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/>
  </a>
</p>

<p align="center">
  <img src="ai_ml_repository.png" alt="AI Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/05satyam/AI-ML?style=flat-square" alt="Last Commit">
  <img src="https://img.shields.io/github/stars/05satyam/AI-ML?style=flat-square" alt="Stars">
  <img src="https://img.shields.io/github/repo-size/05satyam/AI-ML?style=flat-square" alt="Repo Size">
</p>

---

## ✨ Start Here (Pick your path)

If you're new, start with the path that matches your goal:

1. **New to AI/ML (Foundations)**
   - Start with **Tokens & Embeddings → Fine-tuning basics → Small demos**
2. **🤖 GenAI / RAG learner**
   - Go to **RAG Systems table → Multimodal/Graph RAG → Agents → Observability**
3. **🛠️ Want handson notebooks**
   - Go to **LangChain + LlamaIndex + RAG Systems tables**
4. **🎯 Interview prep**
   - Go to **Interview Experiences**
5. **📚 Need tools / courses / blogs**
   - Go to **External Resources**

---

## 📌 Table of Contents

1. [Mission & Scope](#-mission--scope)
2. [Foundations (AI/ML Core)](#-foundations-aiml-core)
3. [LangChain (All Notebooks)](#-langchain-all-notebooks)
4. [LlamaIndex (All Notebooks)](#-llamaindex-all-notebooks)
5. [RAG Systems (All Variants)](#-rag-systems-all-variants)
6. [Agents & Orchestration](#-agents--orchestration)
7. [Graph & Multimodal](#-graph--multimodal)
8. [MCP (Model Context Protocol)](#-mcp-model-context-protocol)
9. [LLM Observability](#-llm-observability)
10. [Interview Experiences](#-interview-experiences)
11. [External Resources](#-external-resources)
12. [Utils](#-utils)
13. [Repository Rules](#-repository-rules)
14. [Contributing & Support](#-contributing--support)
15. [License & Citation](#-license--citation)

---

## 🎯 Mission & Scope

This repository is a **living library** of practical AI/ML and Generative AI knowledge.  
The focus is on **learning by doing** — notebooks and guides are reproducible, intuitive, and easy to extend.

You’ll find content across:
- Classical ML & Deep Learning fundamentals  
- LLMs, embeddings, fine-tuning  
- RAG systems (naive → hybrid → graph → multimodal)  
- Agentic AI patterns and orchestration  
- Production evaluation and observability  

---

## Foundations (AI/ML Core)

| Notebook | What you’ll learn | Level |
|---|---|---|
| [Tokens in GenAI](https://github.com/05satyam/AI-ML/blob/main/topcis_and_handson/Tokens_in_AI(GenAI).ipynb) | Tokenization intuition + cost/latency impact | Beginner |
| [ML Word Embeddings](https://github.com/05satyam/AI-ML/blob/main/topcis_and_handson/ML_WordEmbeddings.ipynb) | Word2Vec/GloVe/CBOW intuition | Beginner |
| [Simple LoRA Fine-Tuning](https://github.com/05satyam/AI-ML/blob/main/topcis_and_handson/finetuning/Simple_LoRA.ipynb) | PEFT/LoRA fine-tuning end-to-end | Intermediate |

---

## 🔗 LangChain (All Notebooks)

Everything that uses **LangChain / LangGraph / LCEL** lives here.

| Notebook | What it does | Level | Tags |
|---|---|---|---|
| [LangChain Prompt Chains](https://github.com/05satyam/AI-ML/blob/main/demo_applications/langchain_langgraph/lanchain-openai-prompt-chains.ipynb) | Prompt chaining + LCEL patterns | Beginner | #prompting #lcel |
| [Plan & Execute (LangGraph)](https://github.com/05satyam/AI-ML/blob/main/topcis_and_handson/agentic_ai_design_patterns/plan_and_execute_langgraph.ipynb) | Multi-step planning + execution | Intermediate | #langgraph #agents |
| [Reflexion Pattern](https://github.com/05satyam/AI-ML/blob/main/topcis_and_handson/agentic_ai_design_patterns/reflexion_pattern.ipynb) | Self-critique agent loops | Intermediate | #agents #reasoning |
| [LangGraph Agents](https://github.com/05satyam/AI-ML/blob/main/demo_applications/langchain_langgraph/AI_Agents_and_Agent_LangGraph.ipynb) | Tool-calling agents with graphs | Intermediate | #langgraph #tools |

---

## 🦙 LlamaIndex (All Notebooks)

Everything that uses **LlamaIndex** lives here.

| Notebook | What it does | Level | Tags |
|---|---|---|---|
| [Text-to-SQL w/ LlamaIndex](https://github.com/05satyam/AI-ML/blob/main/demo_applications/text-to-sql/Text_To_SQL_LlamaIndex.ipynb) | Natural language → SQL over DB | Intermediate | text2sql, llamaindex |
| [LlamaExtract (LlamaIndex)](https://github.com/05satyam/AI-ML/blob/main/demo_applications/llama_index/llama_extract/llama_extract.ipynb) | Structured extraction from invoices using LlamaIndex | Intermediate | llamaindex, extraction |

---

## 🔎 RAG Systems (All Variants)

All Retrieval-Augmented Generation notebooks, grouped by type.

| Notebook | RAG Type | What it does | Level |
|---|---|---|---|
| [Hybrid Search RAG](https://github.com/05satyam/AI-ML/blob/main/demo_applications/simple-rag/HybridSearch.ipynb) | Hybrid RAG | BM25 + vectors + reranking | Intermediate |
| [Semantic Search (Pinecone)](https://github.com/05satyam/AI-ML/blob/main/demo_applications/simple-rag/semantic_search_vec_pinecone.ipynb) | Vector RAG | Simple embedding retrieval | Beginner |
| [GraphRAG](https://github.com/05satyam/AI-ML/blob/main/demo_applications/graph_based_applications/graph_rag.ipynb) | Graph RAG | Graph retrieval + LLM answering | Advanced |
| [Multimodal RAG: Text + Images](https://github.com/05satyam/AI-ML/blob/main/demo_applications/advance_rag/multimodal-rag/README.md) | Multimodal RAG | Retrieve across text & images | Intermediate |

---

## 🤖 Agents & Orchestration

| Notebook / Resource | What it does | Level |
|---|---|---|
| [LLM Query Router](https://github.com/05satyam/AI-ML/tree/main/topcis_and_handson/query_router) | Route queries to best chain/tool | Intermediate |
| [PydanticAI Agents And Tools](https://github.com/05satyam/AI-ML/blob/main/demo_applications/pydnatic_ai/pydantic_ai_agents_and_tools.ipynb) | Typed agents + strict tool schemas | Intermediate |
| [PydanticAI Agentic Lib](https://github.com/05satyam/AI-ML/blob/main/demo_applications/pydnatic_ai/pydantic_ai_agentic_lib.ipynb) | Agentic patterns using PydanticAI | Intermediate |
| [Crew AI Agents](https://github.com/05satyam/AI-ML/blob/main/demo_applications/crewai/crewai_agents_basics.ipynb) | Multi-agent teams + roles | Beginner–Intermediate |
| [Agentic Webcrawler Chatbot](https://github.com/05satyam/AI-ML/blob/main/demo_applications/agentic_webcrawler_chatbot.ipynb) | Crawl web + answer with agents | Intermediate |

---

## 🕸️ Graph & Multimodal

| Notebook | What it does | Level |
|---|---|---|
| [GraphMyDoc](https://github.com/05satyam/AI-ML/blob/main/demo_applications/graph_based_applications/graph_my_doc.ipynb) | Build doc knowledge graphs | Intermediate |
| [GraphNavAI](https://github.com/05satyam/AI-ML/blob/main/demo_applications/graph_based_applications/graph_nav.ipynb) | Navigate knowledge as graph | Intermediate |

---

## 🧩 MCP (Model Context Protocol)

| Demo | What it does | Level |
|---|---|---|
| [MCP Server Demo](https://github.com/05satyam/AI-ML/tree/main/demo_applications/mcp_server_demo) | MCP server-client tooling end-to-end | Intermediate |

---

## 📈 LLM Observability

| Notebook | What it does | Level |
|---|---|---|
| [LlamaTrace — LLM Observability](https://github.com/05satyam/AI-ML/blob/main/demo_applications/llm-observability/LlamaTrace_(Hosted_Arize_Phoenix).ipynb) | Tracing, evals, monitoring with Phoenix | Intermediate |

---

## 🎯 Interview Experiences

| Doc | Focus Area |
|---|---|
| [LLM Architecture Comparison](https://github.com/05satyam/AI-ML/blob/main/concepts-interview-experience/comparison_of_major_llms_architectures(2017-2025).md) | Evolution of LLM architectures (2017–2025) |
| [Interview Q&A](https://github.com/05satyam/AI-ML/blob/main/concepts-interview-experience/interview-expereince/AI-ML-QnA.md) | Common AI/ML/LLM interview questions |
| [Contextual & GPT Embeddings](https://github.com/05satyam/AI-ML/blob/main/concepts-interview-experience/Contexual%20And%20GPT%20Embeddings.md) | Embedding types + intuition |
| [AI Agent Memory Types](https://github.com/05satyam/AI-ML/blob/main/concepts-interview-experience/ai_agents_memory_types.md) | Memory patterns for agents |
| [Stanford LLM Cheatsheet](https://github.com/05satyam/AI-ML/blob/main/concepts-interview-experience/standford_transformer_llm_cheatsheet.pdf) | Compact transformer/LLM summary |

---

## 📚 External Resources

### Free Open Source Learning Resources

| 🧠 Provider | 📚 Resource | 🔍 Focus Area |
|---|---|---|
| **OpenSource Book: Agentic Design Patterns** | [Agentic Design Patterns](https://docs.google.com/document/u/0/d/1rsaK53T3Lg5KoGwvf8ukOUvbELRtH-V0LnOIFDxBryE/mobilebasic) | Hands-on agentic systems |
| **LangChain** | [Chat LangChain](https://chat.langchain.com/) | Chat with LangChain docs |
|  | [LangChain for LLM App Dev](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | Prompting, chains, memory |
|  | [Functions, Tools & Agents](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/) | Tool calling, agents |
|  | [LangGraph Intro Course](https://academy.langchain.com/courses/intro-to-langgraph) | Agentic execution |
|  | [LangChain Tutorials](https://python.langchain.com/docs/tutorials/) | End-to-end apps |
| **LlamaIndex** | [Chat LlamaIndex](https://chat.llamaindex.ai/) | Chat with LlamaIndex docs |
|  | [Advanced RAG Certification](https://learn.activeloop.ai/courses/rag) | Production RAG |
|  | [Agentic RAG Course](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) | Agentic RAG |
|  | [LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/) | Indexing & ingestion |
| **Hugging Face** | [LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Transformers & tokenizers |
|  | [AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | Agent architectures |
|  | [Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1) | Image diffusion |
|  | [Open Source Models](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) | Discovery & eval |
| **Microsoft** | [Generative AI for Beginners](https://microsoft.github.io/generative-ai-for-beginners/) | GenAI foundations |
|  | [AI for Beginners](https://microsoft.github.io/AI-For-Beginners/) | Classical AI/ML |
|  | [AI Agents for Beginners](https://microsoft.github.io/ai-agents-for-beginners/) | Agent systems |
| **AWS** | [Intro to GenAI](https://www.aboutamazon.com/news/aws/7-free-and-low-cost-aws-courses-that-can-help-you-use-generative-ai) | Enterprise GenAI |
|  | [Prompt Engineering Essentials](https://www.aboutamazon.com/news/aws/7-free-and-low-cost-aws-courses-that-can-help-you-use-generative-ai) | Prompting |
|  | [Responsible AI](https://www.aboutamazon.com/news/aws/7-free-and-low-cost-aws-courses-that-can-help-you-use-generative-ai) | Governance |
|  | [AWS PartyRock](https://www.aws.training/) | No-code GenAI apps |
| **Meta (LLaMA)** | [Building with Llama 4](https://www.deeplearning.ai/short-courses/building-with-llama-4/) | Llama models |

### AI & ML Tools
- [LLM Visualization](https://bbycroft.net/llm)
- [Chatbot Arena LLM Leaderboard](https://lmarena.ai/)

### Technical Blogs
- [DeepLearning.AI — The Batch](https://www.deeplearning.ai/the-batch/)
- [Uber AI — LLM Training](https://www.uber.com/en-GB/blog/open-source-and-in-house-how-uber-optimizes-llm-training/)
- [Netflix ML Recommendations](https://netflixtechblog.com/)
- [Sebastian Raschka](https://magazine.sebastianraschka.com/)

### Industry AI & ML Talks
- [Deep Dive into LLMs — Andrej Karpathy](https://youtu.be/7xTGNNLPyMI?t=1052)
- [Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?t=797)
- [Software Is Changing (Again)](https://youtu.be/LCEmiRjPEtQ)
- [Making AI accessible](https://youtu.be/c3b-JASoPi0?t=1950)
- [What is Agentic AI?](https://youtu.be/kJLiOGle3Lw)
- [Discover AI — Code4AI](https://www.youtube.com/@code4AI)
- [The AI GRID](https://www.youtube.com/@TheAiGrid)
- [Krish Naik](https://www.youtube.com/@krishnaik06)

### Technical Newsletters
- [AI by Hand — Dr. Tom Yeh](https://aibyhand.substack.com/)
- [LLM Watch — Pascal Biese](https://www.llmwatch.com/)
- [LangChain Blog](https://blog.langchain.dev/)
- [LlamaIndex Blog](https://www.llamaindex.ai/blog)
- [MLOps Architect Mindset](https://www.linkedin.com/newsletters/mlops-architect-mindset-7015185399367012352/)
- [ByteByteGo Newsletter](https://www.linkedin.com/newsletters/bytebytego-newsletter-7144012310280359936/)

---

## 🧰 Utils

- [script_to_update_packages.py](https://github.com/05satyam/AI-ML/blob/main/script_to_update_packages.py)
- [Steps2CreateEnvFile.MD](https://github.com/05satyam/AI-ML/blob/main/Steps2CreateEnvFile.MD)
- [requirements.txt](https://github.com/05satyam/AI-ML/blob/main/requirements.txt)

---

## 📜 Repository Rules

- [Contributing Guide](https://github.com/05satyam/AI-ML/blob/main/repository_rules/contributing.md)
- [Code of Conduct](https://github.com/05satyam/AI-ML/blob/main/repository_rules/code_of_conduct.md)
- [Security Policy](https://github.com/05satyam/AI-ML/blob/main/repository_rules/security.md)
- [MIT License](https://github.com/05satyam/AI-ML/blob/main/repository_rules/license)
- [Citation File](https://github.com/05satyam/AI-ML/blob/main/repository_rules/citation.cff)

---

## 🤝 Contributing & Support

Contributions are welcome!  
If you spot an error, want a new notebook, or have an improvement idea:
- Read the **Contributing Guide**
- Open a PR / issue with a clear description

Security issues should be reported privately (see **SECURITY.md**).

---

## 📜 License
This project is licensed under the **MIT License** — see the [LICENSE](repository_rules/license) file for details.

---

### ⭐ Final Note

These notebooks reflect personal learnings and experiments.  
Mistakes are part of the journey — use this repo as a starting point and adapt freely.

If this helps you, consider giving it a ⭐ on GitHub — it helps others find it too.
