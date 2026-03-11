# ADR-002: Exhaustive In-Memory Identity Matching

## Status
Accepted

## Context
Face recognition requires comparing a 512-dimensional vector against a database of registered vectors using Cosine Similarity. 

## Decision
For the current version, we will load all registered `Person` entities into memory and perform an exhaustive (linear) search using Dot Product/Cosine Similarity at runtime.

## Consequences
- **Pros**: Implementation simplicity, no external vector database dependency (like Pinecone or Milvus), works extremely well for databases < 10,000 users.
- **Cons**: Becomes a performance bottleneck (O(N)) as the registrations grow to hundreds of thousands. Future versions might require Approximate Nearest Neighbor (ANN) indexing.
