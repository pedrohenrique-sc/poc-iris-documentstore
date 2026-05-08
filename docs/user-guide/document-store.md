---
title: IRISDocumentStore
---

# IRISDocumentStore

`IRISDocumentStore` is the core class of `iris-haystack`. It implements the full
[Haystack 2.x DocumentStore protocol](https://docs.haystack.deepset.ai/docs/document-store)
and manages the connection to an InterSystems IRIS instance.

---

## Initialization

The simplest initialization reads all credentials from environment variables:

```bash
export IRIS_CONNECTION_STRING="localhost:1972/USER"
export IRIS_USERNAME="_system"
export IRIS_PASSWORD="SYS"
```

```python
from haystack_integrations.document_stores.iris import IRISDocumentStore

store = IRISDocumentStore(embedding_dim=384)
print(store)
# IRISDocumentStore(table='HaystackDocuments', embedding_dim=384)
```

### All parameters

```python
from haystack.utils import Secret

store = IRISDocumentStore(
    connection_string=Secret.from_env_var("IRIS_CONNECTION_STRING"),
    username=Secret.from_env_var("IRIS_USERNAME"),
    password=Secret.from_env_var("IRIS_PASSWORD"),
    table_name="HaystackDocuments",   # SQL table name (SQLUser schema prepended)
    embedding_dim=384,                # must match your embedding model
    bm25_k1=1.5,                      # BM25 term frequency saturation
    bm25_b=0.75,                      # BM25 length normalization
    recreate_table=False,             # True = drop all data and recreate table
)
```

### Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_string` | `Secret` | `$IRIS_CONNECTION_STRING` | DB-API string: `host:port/namespace` |
| `username` | `Secret` | `$IRIS_USERNAME` | IRIS username |
| `password` | `Secret` | `$IRIS_PASSWORD` | IRIS password |
| `table_name` | `str` | `"HaystackDocuments"` | Table name without schema. `SQLUser.` is prepended automatically. |
| `embedding_dim` | `int` | `384` | Number of dimensions of the embedding vectors. **Must match the model used at indexing time.** |
| `bm25_k1` | `float` | `1.5` | BM25 term-frequency saturation. Typical range: 1.2–2.0. |
| `bm25_b` | `float` | `0.75` | BM25 length normalization. 0.0 = none, 1.0 = full. |
| `recreate_table` | `bool` | `False` | Drop and re-create the table on startup. **All existing data is lost.** Useful in tests. |

!!! warning "`recreate_table=True` in production"
    Setting `recreate_table=True` permanently deletes all indexed documents. Never use it
    in a production deployment. It is intended for test fixtures that need a clean slate.

---

## Table schema

The DocumentStore creates the following table in IRIS automatically on first use:

```sql
CREATE TABLE IF NOT EXISTS SQLUser.HaystackDocuments (
    id        VARCHAR(128)  NOT NULL PRIMARY KEY,
    content   LONGVARCHAR,
    meta      LONGVARCHAR,   -- JSON, always serialized with sort_keys=True
    score     DOUBLE,
    embedding VECTOR(DOUBLE, 384)
)
```

### Column details

| Column | Type | Notes |
|---|---|---|
| `id` | `VARCHAR(128)` | Haystack-generated hash of the document content. Primary key. |
| `content` | `LONGVARCHAR` | Full document text. No upper size limit. |
| `meta` | `LONGVARCHAR` | JSON string serialized with `json.dumps(sort_keys=True)`. |
| `score` | `DOUBLE` | Optional source score. Often `NULL`. |
| `embedding` | `VECTOR(DOUBLE, N)` | Native IRIS vector type. Populated via `TO_VECTOR(?, DOUBLE)`. |

!!! info "Why `sort_keys=True`?"
    Serializing `meta` with `sort_keys=True` ensures that `{"b": 1, "a": 2}` and
    `{"a": 2, "b": 1}` always produce the same string. This matters for two reasons:

    1. **Deterministic document IDs** — Haystack generates IDs from a hash of the content,
       and the meta is included in that hash.
    2. **Reliable LIKE-pattern filtering** — even though filtering is done in-memory
       via `document_matches_filter`, the deterministic ordering makes the stored data
       consistent and auditable.

---

## Protocol methods

### `count_documents()`

Returns the total number of documents in the store.

```python
store.count_documents()
# 42
```

Internally executes:

```sql
SELECT COUNT(*) FROM SQLUser.HaystackDocuments
```

---

### `filter_documents(filters=None)`

Returns all documents that satisfy the provided filters.
When `filters=None`, all documents are returned.

```python
# All documents
all_docs = store.filter_documents()

# Simple equality (legacy format)
db_docs = store.filter_documents({"category": "database"})

# Official Haystack format
recent_docs = store.filter_documents({
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "database"},
        {"field": "meta.year",     "operator": ">=", "value": 2023},
    ],
})
```

See the [Metadata Filtering guide](filtering.md) for the full filter syntax reference.

---

### `write_documents(documents, policy=DuplicatePolicy.NONE)`

Persists a list of `Document` objects to IRIS.

```python
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

docs = [
    Document(
        content="IRIS is a multimodel database.",
        meta={"category": "database", "year": 2024},
    ),
    Document(
        content="Haystack builds LLM pipelines.",
        meta={"category": "ai", "year": 2024},
        embedding=[0.1, 0.2, ...]  # 384 floats
    ),
]

written = store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
print(written)  # 2
```

#### Duplicate policies

| Policy | Behaviour |
|---|---|
| `NONE` | Defaults to `FAIL` |
| `FAIL` | Raises `DuplicateDocumentError` if a document with the same ID already exists |
| `SKIP` | Silently ignores documents whose ID already exists — returns 0 for those |
| `OVERWRITE` | Deletes the existing document and inserts the new one |

#### How embeddings are stored

Documents that have an `embedding` are inserted using IRIS's `TO_VECTOR(?, DOUBLE)`:

```sql
INSERT INTO SQLUser.HaystackDocuments (id, content, meta, score, embedding)
VALUES (?, ?, ?, ?, TO_VECTOR(?, DOUBLE))
```

The embedding list is first converted to a string in the format `[v1,v2,...,vN]`
before being passed to `TO_VECTOR`.

Documents without an embedding are inserted without the `embedding` column —
the field remains `NULL` and the document is excluded from vector search results.

---

### `delete_documents(document_ids)`

Deletes documents by their ID. Accepts an empty list without error (idempotent).
IDs that do not exist are silently ignored by IRIS.

```python
store.delete_documents(["id-1", "id-2", "id-3"])

# Empty list — no-op
store.delete_documents([])
```

---

## Connection management

### Automatic reconnection

Before every SQL operation, the store pings IRIS with `SELECT 1`. If the connection
has been dropped (e.g., IRIS restarted, idle timeout reached), the store reconnects
automatically with exponential backoff:

| Attempt | Wait before retry |
|---|---|
| 1st failure | 0.5 s |
| 2nd failure | 1.0 s |
| 3rd failure | 2.0 s |
| 4th failure | raises `ConnectionError` |

This makes the store resilient to transient network failures and IRIS restarts
without any application-level intervention.

### Context manager

```python
with IRISDocumentStore(embedding_dim=384) as store:
    store.write_documents([...])
    results = store.filter_documents()
# Connection is closed automatically when exiting the `with` block
```

Using the context manager is the recommended pattern for short-lived scripts.
For long-running services (e.g., a FastAPI app), create one store instance at
startup and reuse it — the reconnection logic handles transient failures.

### Manual close

```python
store = IRISDocumentStore(embedding_dim=384)
# ... use the store ...
store.close()  # idempotent — safe to call multiple times
```

---

## Serialization

The store is fully serializable for use in Haystack YAML pipelines:

```python
# Serialize
d = store.to_dict()
print(d["type"])
# haystack_integrations.document_stores.iris.document_store.IRISDocumentStore
print("password" in d["init_parameters"])
# False — password is never serialized

# Deserialize (password is read from env var at runtime)
restored = IRISDocumentStore.from_dict(d)
```

!!! note "Password is intentionally omitted"
    `to_dict()` serializes the `Secret` objects by their env var name, not the
    resolved value. When `from_dict()` restores the store, it reads the password
    from the environment variable at that moment. This prevents credentials from
    appearing in committed YAML pipeline files.

---

## Common patterns

### Using in an indexing pipeline

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

store = IRISDocumentStore(embedding_dim=384)

pipeline = Pipeline()
pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
pipeline.add_component(
    "writer",
    DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE),
)
pipeline.connect("embedder.documents", "writer.documents")

pipeline.run({"embedder": {"documents": my_documents}})
print(f"Total indexed: {store.count_documents()}")
```

### Checking what is stored

```python
# Count
print(store.count_documents())

# Inspect a sample
sample = store.filter_documents()[:5]
for doc in sample:
    print(doc.id, doc.meta, doc.content[:50])
```

### Deleting all documents

```python
ids = [doc.id for doc in store.filter_documents()]
store.delete_documents(ids)
print(store.count_documents())  # 0
```