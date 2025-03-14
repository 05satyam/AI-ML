**GenAI**

    Q1:  Interview Experience: RAG in Production!  How would you optimize a Retrieval-Augmented Generation (RAG) pipeline to reduce latency while maintaining retrieval accuracy?
    Ans: I shared my thoughts on: 
      ✅ Index Optimization – Using dense embeddings with FAISS for faster retrieval 
      ✅ Chunking Strategy – Finding the right balance to improve retrieval precision 
      ✅ Caching Mechanisms – Reducing redundant retrieval calls 
      ✅ Efficient Query Expansion – Enhancing recall without adding unnecessary overhead 
      ✅ Hybrid Search Techniques – Combining sparse and dense retrieval for better results
      
    Q2:How would you optimize a RAG system for low-latency responses while maintaining high retrieval accuracy?
    Ans: RAG latency isn’t just an engineering problem—it’s a tradeoff between speed, accuracy, and cost. The key is balancing all components efficiently.
    The retriever plays a crucial role in fetching relevant documents fast. To reduce latency: 
        ✅ Vector Index Optimization – Use efficient indexing methods like FAISS HNSW (Hierarchical Navigable Small World) or ScaNN instead of brute-force search. These approximate nearest neighbor (ANN) methods can speed up retrieval significantly. 
        ✅ Hybrid Retrieval – Combine BM25 (keyword-based retrieval) with dense embeddings to get better results in fewer retrieval iterations. 
        ✅ Query Expansion & Preprocessing – Precompute expanded queries or leverage embeddings from a smaller LLM to refine search before hitting the vector store.
    Reducing Unnecessary Calls: Each call to the retriever or LLM adds delay. So, I mentioned: 
        ✅ Caching – Cache frequent queries and responses at the retrieval layer or even at the API level.
    Once retrieval is fast, the next bottleneck is LLM inference:
        ✅ Use a smaller model (e.g., Llama-3-8B) or fine-tune or quantized a model on domain-specific data. 
        ✅ Streaming Tokens: Stream partial responses in chat applications rather than waiting for the full answer.
    Other methods:
        ✅ Scaling Hardware – Deploying inference on GPUs (A100/H100) or TPUs with optimized tensor parallelism. 
        ✅ Multi-Stage RAG – First, use a fast retriever (BM25) and only use expensive reranking when necessary. 


---

**Q.** What happens if you remove hidden layers from a neural network - which logical operator cannot be used? <br />
**Ans.** 
- Without hidden layers, a neural network becomes just a simple linear classifier. 
- i.e. it can handle straightforward, linearly separable tasks like AND and OR but it cannot handle XOR (exclusive OR), which needs at least one hidden layer to capture its non-linear pattern.
- And, removing the non linear activation functions from hidden nodes will cause in linearity. Having hidden nodes without non linear activations is equivalent to having a linear model.
