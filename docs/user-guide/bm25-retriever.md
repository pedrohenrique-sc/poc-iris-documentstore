---
title: IRISBm25Retriever
---

# IRISBm25Retriever

`IRISBm25Retriever` performs **keyword-based search** using the
[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) probabilistic ranking algorithm,
computed in-memory over the (optionally filtered) document set loaded from IRIS.

---

## How it works

BM25 is the ranking function behind most traditional search engines. It scores documents
by how often query terms appear in them, with two adjustments:

1. **Term frequency saturation** — a term appearing 100 times is not 100× more relevant
   than one appearing once. The `k1` parameter controls this saturation.
2. **Length normalization** — a 10-word document where your term appears twice is more
   relevant than a 1000-word document with the same term twice. The `b` parameter controls this.

The BM25 index is built from scratch on every `run()` call to always reflect the current
state of the document store.

### Algorithm (Okapi BM25)

For each document $d$ and query $q$ with terms $q_1 \ldots q_n$:

$$
\text{BM25}(d, q) = \sum_{i=1}^{n}
\text{IDF}(q_i) \cdot
\frac{f(q_i, d) \cdot (k_1 + 1)}
     {f(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

Where:

- $f(q_i, d)$ = frequency of term $q_i$ in document $d$
- $|d|$ = document length in tokens
- $\text{avgdl}$ = average document length across the corpus
- $k_1$ = term frequency saturation parameter (default: 1.5)
- $b$ = length normalization parameter (default: 0.75)

---

## Basic usage

### Standalone

```python
from haystack_integrations.document_stores.iris import IRISDocumentStore
from haystack_integrations.components.retrievers.iris import IRISBm25Retriever

store = IRISDocumentStore(embedding_dim=384)
retriever = IRISBm25Retriever(document_store=store, top_k=5)

result = retriever.run(query="multimodel database SQL JSON")
for doc in result["documents"]:
    print(f"[{doc.score:.4f}] {doc.content}")
```

### In a pipeline

```python
from haystack import Pipeline
from haystack_integrations.components.retrievers.iris import IRISBm25Retriever

pipeline = Pipeline()
pipeline.add_component(
    "retriever",
    IRISBm25Retriever(document_store=store, top_k=5),
)

result = pipeline.run({"retriever": {"query": "vector search embeddings"}})
for doc in result["retriever"]["documents"]:
    print(f"[{doc.score:.4f}] {doc.content[:60]}...")
```

---

## Parameters

### `__init__` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_store` | `IRISDocumentStore` | **required** | The store instance to query |
| `top_k` | `int` | `10` | Maximum number of documents to return |
| `filters` | `dict \| None` | `None` | Pre-filters applied before BM25 ranking. Reduces the candidate set. |
| `filter_policy` | `FilterPolicy \| str` | `FilterPolicy.REPLACE` | How init-time and runtime filters interact |

### `run()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | **required** | Natural language search text |
| `filters` | `dict \| None` | `None` | Runtime filter |
| `top_k` | `int \| None` | `None` | Override the `top_k` set at initialization |

### Output

```python
{"documents": [Document, Document, ...]}
```

Each returned `Document` has its `score` field set to the BM25 score.
Documents with a score of 0 (no query term matched) are excluded.

---

## BM25 parameters

The `k1` and `b` parameters are set on the **DocumentStore**, not the retriever:

```python
store = IRISDocumentStore(
    embedding_dim=384,
    bm25_k1=1.2,   # lower = faster saturation (good for short docs)
    bm25_b=0.75,   # 0.0 = no length normalization, 1.0 = full
)
```

### Tuning guidelines

| Scenario | Recommended settings |
|---|---|
| Short documents (tweets, titles) | `k1=1.2`, `b=0.3` — less length normalization |
| Long documents (articles, books) | `k1=1.5`, `b=0.75` — standard |
| Very long documents with many repetitions | `k1=2.0`, `b=0.75` — higher saturation |
| Uniform document lengths | `k1=1.5`, `b=0.0` — disable length normalization |

---

## FilterPolicy

Same as `IRISEmbeddingRetriever` — see the [FilterPolicy section](embedding-retriever.md#filterpolicy).

When `filters` are provided, the retriever first calls `filter_documents(filters)` to narrow
the candidate set, then builds the BM25 index over those candidates only:

```python
# Only index and rank documents matching the filter
retriever = IRISBm25Retriever(
    document_store=store,
    top_k=5,
    filters={"language": "en"},
    filter_policy=FilterPolicy.REPLACE,
)
result = retriever.run(query="machine learning", filters={"category": "ai"})
# Effective filter: {"category": "ai"} (REPLACE)
# BM25 built over those documents only
```

---

## Tokenization

The BM25 tokenizer uses a simple regex that extracts alphanumeric tokens and normalizes
to lowercase. It supports accented characters (UTF-8):

```python
# Input:  "InterSystems IRIS is a high-performance, multimodel database."
# Tokens: ["intersystems", "iris", "is", "a", "high", "performance", "multimodel", "database"]
```

This tokenizer does **not** perform stemming or stop-word removal. The query
`"database"` will match documents containing `"database"` but not `"databases"`.

!!! tip "For multilingual content"
    The tokenizer works for any UTF-8 text including Portuguese, Spanish, French,
    German, and other Latin-script languages. Non-Latin scripts (Chinese, Japanese,
    Arabic) are not tokenized correctly — consider using the embedding retriever for
    multilingual content.

---

## Comparing BM25 and embedding retrieval

| Aspect | BM25 (`IRISBm25Retriever`) | Embedding (`IRISEmbeddingRetriever`) |
|---|---|---|
| **Match type** | Exact keyword overlap | Semantic meaning |
| **Model required** | No | Yes (embedding model) |
| **Cost** | Free — computed in Python | Embedding inference cost |
| **Strength** | Exact product names, codes, IDs | Synonyms, paraphrases, cross-lingual |
| **Weakness** | Misses synonyms | Misses exact rare terms |
| **Best for** | Technical docs, code, logs | General text, Q&A, multilingual |

### Hybrid retrieval

For the best of both worlds, run both retrievers and merge their results:

```python
from haystack.components.rankers import TransformersSimilarityRanker

pipeline = Pipeline()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="..."))
pipeline.add_component("embedding_retriever", IRISEmbeddingRetriever(store, top_k=20))
pipeline.add_component("bm25_retriever", IRISBm25Retriever(store, top_k=20))
pipeline.add_component("ranker", TransformersSimilarityRanker(top_k=5))

pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
pipeline.connect("embedding_retriever.documents", "ranker.documents")
pipeline.connect("bm25_retriever.documents", "ranker.documents")

result = pipeline.run({
    "text_embedder": {"text": "my query"},
    "bm25_retriever": {"query": "my query"},
    "ranker": {"query": "my query"},
})
```

---

## Performance notes

The BM25 index is rebuilt from scratch on every `run()` call. For a store with:

- **< 10k documents** — rebuilding takes < 50 ms, imperceptible
- **10k–100k documents** — rebuilding takes 0.5–5 s, noticeable
- **> 100k documents** — rebuilding takes > 5 s, consider caching

To reduce rebuild time for large collections, always use `filters` to narrow the
candidate set before ranking:

```python
# Instead of ranking all 500k documents:
retriever.run(query="database performance")

# Rank only the 5k English documents:
retriever.run(query="database performance", filters={"language": "en"})
```