---
title: iris-haystack
description: A production-ready Haystack 2.x DocumentStore backed by InterSystems IRIS
hide:
  - navigation
  - toc
---

# iris-haystack

<div class="grid cards" markdown>

-   :material-database-arrow-right:{ .lg .middle } **DocumentStore**

    ---
    Full Haystack 2.x protocol implementation backed by InterSystems IRIS native `VECTOR(DOUBLE, N)` type.

    [:octicons-arrow-right-24: Getting Started](getting-started/index.md)

-   :material-magnify:{ .lg .middle } **Semantic Search**

    ---
    `IRISEmbeddingRetriever` uses IRIS native `VECTOR_COSINE` with SIMD optimisation — no external vector DB required.

    [:octicons-arrow-right-24: Embedding Retriever](user-guide/embedding-retriever.md)

-   :material-text-search:{ .lg .middle } **Keyword Search**

    ---
    `IRISBm25Retriever` implements Okapi BM25 in-memory over the filtered document set.

    [:octicons-arrow-right-24: BM25 Retriever](user-guide/bm25-retriever.md)

-   :material-filter:{ .lg .middle } **Metadata Filtering**

    ---
    Uses Haystack's official `document_matches_filter` — identical behaviour to `InMemoryDocumentStore`, all Python types supported.

    [:octicons-arrow-right-24: Filtering](user-guide/filtering.md)

</div>

---

## What is iris-haystack?

`iris-haystack` integrates **InterSystems IRIS** as a `DocumentStore` for the [Haystack 2.x](https://haystack.deepset.ai/) framework. It lets you store documents, embeddings, and metadata in IRIS and retrieve them with semantic or keyword-based search — without maintaining a separate vector database.

```python
from haystack import Document
from haystack_integrations.document_stores.iris import IRISDocumentStore

store = IRISDocumentStore(embedding_dim=384)

store.write_documents([
    Document(content="IRIS is a multimodel database.", meta={"category": "db"}),
    Document(content="Haystack builds LLM pipelines.",  meta={"category": "ai"}),
])

print(store.count_documents())
```

---

## Why IRIS?

| Capability | What it means for you |
|---|---|
| `VECTOR(DOUBLE, N)` column | Embeddings stored natively in SQL — no separate vector DB |
| `VECTOR_COSINE` function | SIMD-optimised cosine similarity computed by the database engine |
| ANN / HNSW index | Approximate nearest-neighbour search for large collections |
| SQL + JSON + globals | One platform for relational data and vectors |

---

## Stack

```
intersystems-iris-haystack
├── IRISDocumentStore          ← Haystack 2.x protocol
├── IRISEmbeddingRetriever     ← VECTOR_COSINE via SQL
└── IRISBm25Retriever          ← Okapi BM25 in-memory
```

Credentials are managed via Haystack [`Secret`](user-guide/credentials.md) — never hardcoded.

---

## Quick install

```bash
pip install #Lib-iris-haystack (development)
```

!!! tip "New here?"
    Start with the [Prerequisites](getting-started/prerequisites.md) page, then follow [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md).