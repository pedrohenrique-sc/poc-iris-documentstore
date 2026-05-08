---
title: Quick Start
---

# Quick Start

This guide shows a complete working example: indexing documents, performing semantic search, and keyword search — in under 40 lines.

**Prerequisites:** IRIS running via Docker and the package installed. See [Docker Setup](docker.md) and [Installation](installation.md).

---

## 1. Set credentials

```bash
export IRIS_CONNECTION_STRING="localhost:1972/USER"
export IRIS_USERNAME="_system"
export IRIS_PASSWORD="SYS"
```

---

## 2. Complete example

```python title="quickstart.py"
from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.iris import IRISDocumentStore
from haystack_integrations.components.retrievers.iris import (
    IRISBm25Retriever,
    IRISEmbeddingRetriever,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Initialize the store ──────────────────────────────────────────────────
store = IRISDocumentStore(embedding_dim=384)
print(f"Documents before indexing: {store.count_documents()}")

# ── Indexing pipeline ─────────────────────────────────────────────────────
indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component("writer", DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE))
indexing.connect("embedder.documents", "writer.documents")

documents = [
    Document(content="IRIS is a high-performance multimodel database.", meta={"category": "db"}),
    Document(content="Haystack is a framework for building LLM applications.", meta={"category": "ai"}),
    Document(content="Vector search finds semantically similar documents.", meta={"category": "ai"}),
    Document(content="IRIS supports SQL, JSON, vectors, and globals.", meta={"category": "db"}),
]
indexing.run({"embedder": {"documents": documents}})
print(f"Documents after indexing: {store.count_documents()}")

# ── Semantic search ───────────────────────────────────────────────────────
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
query_pipeline.add_component("retriever", IRISEmbeddingRetriever(store, top_k=2))
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

result = query_pipeline.run({"embedder": {"text": "how does similarity search work?"}})
print("\n── Semantic search ──────────")
for doc in result["retriever"]["documents"]:
    print(f"  [{doc.score:.4f}] {doc.content[:60]}...")

# ── BM25 keyword search ───────────────────────────────────────────────────
bm25 = IRISBm25Retriever(store, top_k=2)
result = bm25.run(query="database SQL JSON")
print("\n── BM25 keyword search ──────")
for doc in result["documents"]:
    print(f"  [{doc.score:.4f}] {doc.content[:60]}...")

# ── Metadata filter ───────────────────────────────────────────────────────
ai_docs = store.filter_documents({"category": "ai"})
print(f"\n── Filter category=ai ───────")
for doc in ai_docs:
    print(f"  {doc.content[:60]}...")

store.close()
```

---

## 3. Expected output

```
Documents before indexing: 0
Documents after indexing:  4

── Semantic search ──────────
  [0.6123] Vector search finds semantically similar documents....
  [0.4891] IRIS is a high-performance multimodel database....

── BM25 keyword search ──────
  [1.4320] IRIS supports SQL, JSON, vectors, and globals....
  [0.8741] IRIS is a high-performance multimodel database....

── Filter category=ai ───────
  Haystack is a framework for building LLM applications....
  Vector search finds semantically similar documents....
```

---

## Next steps

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **IRISDocumentStore**

    All initialization options, table schema, and connection management.

    [:octicons-arrow-right-24: DocumentStore guide](../user-guide/document-store.md)

-   :material-filter:{ .lg .middle } **Metadata Filtering**

    Legacy format, official Haystack operator/conditions, all operators.

    [:octicons-arrow-right-24: Filtering guide](../user-guide/filtering.md)

-   :material-code-tags:{ .lg .middle } **API Reference**

    Complete parameter reference for all classes.

    [:octicons-arrow-right-24: API docs](../api/index.md)

</div>