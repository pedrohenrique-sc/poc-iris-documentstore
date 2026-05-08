---
title: IRISEmbeddingRetriever
---

# IRISEmbeddingRetriever

`IRISEmbeddingRetriever` is a Haystack component that performs **semantic similarity search**
by comparing a query embedding against all stored document embeddings using IRIS's native
`VECTOR_COSINE` function.

---

## How it works

When you call `.run(query_embedding=...)`, the retriever executes this SQL against IRIS:

```sql
SELECT TOP ?
    id, content, meta, score,
    VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) AS similarity
FROM SQLUser.HaystackDocuments
WHERE embedding IS NOT NULL
ORDER BY similarity DESC
```

`VECTOR_COSINE` is computed by the IRIS engine with SIMD (Single Instruction, Multiple Data)
hardware acceleration. The result is a float between **-1** and **1**, where **1** means the
vectors are identical.

Documents without an embedding are automatically excluded by the `WHERE embedding IS NOT NULL` clause.

---

## Basic usage

### Standalone

```python
from haystack_integrations.document_stores.iris import IRISDocumentStore
from haystack_integrations.components.retrievers.iris import IRISEmbeddingRetriever

store = IRISDocumentStore(embedding_dim=384)
retriever = IRISEmbeddingRetriever(document_store=store, top_k=5)

# query_embedding must have the same number of dimensions
# as the embeddings stored in the document store
result = retriever.run(query_embedding=[0.1, 0.2, ...])  # 384 floats
for doc in result["documents"]:
    print(f"[{doc.score:.4f}] {doc.content}")
```

### In a pipeline

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.iris import IRISEmbeddingRetriever

pipeline = Pipeline()
pipeline.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
pipeline.add_component(
    "retriever",
    IRISEmbeddingRetriever(document_store=store, top_k=5),
)
pipeline.connect("embedder.embedding", "retriever.query_embedding")

result = pipeline.run({"embedder": {"text": "what is vector search?"}})
for doc in result["retriever"]["documents"]:
    print(f"[{doc.score:.4f}] {doc.content[:60]}...")
```

!!! tip "Embedding model consistency"
    The model used to embed the query **must be the same** as the model used to embed
    the documents at indexing time. Mixing models produces nonsensical similarity scores.

---

## Parameters

### `__init__` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_store` | `IRISDocumentStore` | **required** | The store instance to query |
| `top_k` | `int` | `10` | Maximum number of documents to return |
| `filters` | `dict \| None` | `None` | Filters applied to results at initialization time. See [FilterPolicy](#filterpolicy). |
| `filter_policy` | `FilterPolicy \| str` | `FilterPolicy.REPLACE` | How init-time and runtime filters interact |

### `run()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_embedding` | `list[float]` | **required** | The query vector — must have `embedding_dim` floats |
| `filters` | `dict \| None` | `None` | Runtime filter (interaction with init filter depends on `filter_policy`) |
| `top_k` | `int \| None` | `None` | Override the `top_k` set at initialization |

### Output

```python
{"documents": [Document, Document, ...]}
```

Each returned `Document` has its `score` field set to the `VECTOR_COSINE` similarity value.

---

## FilterPolicy

`FilterPolicy` controls how filters set at retriever initialization interact with filters
passed at query time via `retriever.run(filters=...)`.

```python
from haystack.document_stores.types import FilterPolicy
```

| Policy | Behaviour |
|---|---|
| `REPLACE` (default) | Runtime filters completely **replace** init-time filters. If you pass `filters=None` at runtime, no filter is applied regardless of what was set at init. |
| `MERGE` | Runtime filters are **AND-merged** with init-time filters. Both sets of conditions must be satisfied. |

### `REPLACE` example

```python
retriever = IRISEmbeddingRetriever(
    document_store=store,
    top_k=5,
    filters={"category": "database"},   # init-time filter
    filter_policy=FilterPolicy.REPLACE,
)

# This call IGNORES {"category": "database"} and applies {"year": 2024} instead
result = retriever.run(
    query_embedding=my_vector,
    filters={"year": 2024},
)

# This call applies NO filter (runtime None replaces init filter)
result = retriever.run(
    query_embedding=my_vector,
    filters=None,
)
```

### `MERGE` example

```python
retriever = IRISEmbeddingRetriever(
    document_store=store,
    top_k=5,
    filters={"language": "en"},         # always require English docs
    filter_policy=FilterPolicy.MERGE,
)

# Effective filter: {"language": "en"} AND {"category": "database"}
result = retriever.run(
    query_embedding=my_vector,
    filters={"category": "database"},
)

# Effective filter: {"language": "en"} only (runtime None adds nothing)
result = retriever.run(
    query_embedding=my_vector,
    filters=None,
)
```

---

## Understanding similarity scores

`VECTOR_COSINE` returns the cosine similarity between two vectors:

| Score | Meaning |
|---|---|
| `1.0` | Vectors are identical |
| `0.8–0.95` | Highly similar — semantically close |
| `0.5–0.8` | Moderately similar |
| `0.0–0.5` | Weakly related |
| `< 0.0` | Pointing in opposite directions — semantically opposite |

!!! note "Low scores with small collections"
    With fewer than ~20 short documents, scores between 0.3 and 0.6 are normal.
    The absolute score value depends on the variance between documents — with more
    diverse documents in a larger collection, high-relevance matches score 0.8+.

---

## Post-search filtering

When `filters` are provided, the retriever fetches `top_k × 4` documents from IRIS first,
then applies the filter in Python. This guarantees that exactly `top_k` documents are
returned even when many candidates are filtered out.

```python
# Internally:
# 1. SELECT TOP (top_k * 4) ... ORDER BY VECTOR_COSINE DESC
# 2. for each row: if document_matches_filter(filters, doc): keep
# 3. stop when len(results) == top_k
```

If you need strict guarantees with aggressive filtering (many documents filtered out),
increase `top_k` accordingly.

---

## Serialization

`IRISEmbeddingRetriever` is fully serializable:

```python
# Serialize
d = retriever.to_dict()

# Restore — the nested IRISDocumentStore is also restored
restored = IRISEmbeddingRetriever.from_dict(d)
```

This works transparently in Haystack YAML pipelines.

---

## Using with HNSW for large collections

For collections with more than ~100k documents, `VECTOR_COSINE` performs a full table
scan which can be slow. You can create an HNSW (Approximate Nearest Neighbour) index
in IRIS to speed it up:

```sql
-- Run this once in the IRIS Management Portal (SQL editor)
CREATE INDEX HaystackHNSW
ON TABLE SQLUser.HaystackDocuments (embedding)
AS HNSW(Distance='Cosine')
```

Once created, IRIS uses the HNSW index automatically for `VECTOR_COSINE` queries.
This trades a small amount of accuracy for significantly faster retrieval.

!!! info "HNSW requires IRIS 2024.1+"
    The HNSW index type is available from IRIS version 2024.1 onwards.
    IRIS Community Edition 2024.1+ supports it.