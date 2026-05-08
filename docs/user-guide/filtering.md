---
title: Metadata Filtering
---

# Metadata Filtering

`iris-haystack` uses Haystack's official `document_matches_filter` utility for all in-memory filtering. This guarantees **identical behavior to `InMemoryDocumentStore`** and full compatibility with every Haystack filter mix-in.

---

## Why in-memory filtering?

Filtering is performed in Python after loading all documents from IRIS, not via SQL `WHERE` clauses. This was a deliberate design decision for three reasons:

1. **IRIS Community Edition compatibility** — `JSON_VALUE` is not available as a native SQL function in the free edition.
2. **Type safety** — `json.dumps(sort_keys=True)` serializes all Python types consistently, but parsing them back for SQL comparison requires extra logic for each type.
3. **Operator completeness** — `in`, `not in`, nested `AND/OR/NOT` and all comparison operators work without any SQL-specific transformation.

!!! note "Performance at scale"
    For very large collections (>500k documents) with simple equality filters, consider adding a SQL `WHERE` clause as an optimization layer before the in-memory pass. See the [Contributing guide](../contributing/index.md) if you want to help implement this.

---

## Supported filter formats

### 1 — Legacy format (simple dict equality)

The simplest form: a flat dictionary where every key-value pair must match.

```python
# All documents where category equals "db"
store.filter_documents({"category": "db"})

# Multiple conditions (implicit AND)
store.filter_documents({"category": "db", "language": "en"})
```

**Supported Python types:**

| Type | Example |
|---|---|
| `str` | `{"category": "db"}` |
| `int` | `{"year": 2024}` |
| `float` | `{"confidence": 0.95}` |
| `bool` | `{"active": True}` |
| `None` | `{"deleted_at": None}` |

---

### 2 — Official Haystack format (operator/conditions)

The full format supports nested logic and comparison operators.

#### Leaf condition

```python
{"field": "meta.year", "operator": ">=", "value": 2023}
```

| Key | Description |
|---|---|
| `field` | Field path. Use `meta.` prefix for metadata fields: `meta.category`. |
| `operator` | One of the supported operators (see table below). |
| `value` | The value to compare against — any Python type. |

**Supported comparison operators:**

| Operator | Meaning | Example |
|---|---|---|
| `==` | Equal | `{"field": "meta.lang", "operator": "==", "value": "en"}` |
| `!=` | Not equal | `{"field": "meta.lang", "operator": "!=", "value": "fr"}` |
| `>` | Greater than | `{"field": "meta.year", "operator": ">", "value": 2022}` |
| `>=` | Greater or equal | `{"field": "meta.year", "operator": ">=", "value": 2023}` |
| `<` | Less than | `{"field": "meta.score", "operator": "<", "value": 0.5}` |
| `<=` | Less or equal | `{"field": "meta.score", "operator": "<=", "value": 0.9}` |
| `in` | Value is in list | `{"field": "meta.tag", "operator": "in", "value": ["a","b"]}` |
| `not in` | Value not in list | `{"field": "meta.tag", "operator": "not in", "value": ["x"]}` |

#### Logical node

```python
{
    "operator": "AND",   # or "OR" or "NOT"
    "conditions": [...]
}
```

**Supported logical operators:** `AND`, `OR`, `NOT`

---

## Examples

### AND — all conditions must be true

```python
store.filter_documents({
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "db"},
        {"field": "meta.year",     "operator": ">=", "value": 2023},
    ],
})
```

### OR — at least one condition must be true

```python
store.filter_documents({
    "operator": "OR",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "db"},
        {"field": "meta.category", "operator": "==", "value": "ai"},
    ],
})
```

Equivalent shorthand using `in`:

```python
store.filter_documents({
    "field": "meta.category",
    "operator": "in",
    "value": ["db", "ai"],
})
```

### NOT — inverts the condition

```python
store.filter_documents({
    "operator": "NOT",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "draft"},
    ],
})
```

### Nested — AND inside OR

```python
store.filter_documents({
    "operator": "OR",
    "conditions": [
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "db"},
                {"field": "meta.year",     "operator": ">=", "value": 2024},
            ],
        },
        {"field": "meta.language", "operator": "in", "value": ["pt", "es"]},
    ],
})
```

---

## Filters with Retrievers

Both retrievers accept the same filter format via `filters` at init time and at `run()` time:

```python
from haystack_integrations.components.retrievers.iris import IRISEmbeddingRetriever
from haystack.document_stores.types import FilterPolicy

retriever = IRISEmbeddingRetriever(
    document_store=store,
    top_k=5,
    filters={"category": "db"},          # init-time filter
    filter_policy=FilterPolicy.REPLACE,  # runtime filters REPLACE this (default)
)

# Runtime filter overrides init-time filter (REPLACE policy)
result = retriever.run(
    query_embedding=my_vector,
    filters={"category": "ai"},   # this replaces {"category": "db"}
)

# Or use MERGE to combine both filters
retriever_merge = IRISEmbeddingRetriever(
    document_store=store,
    top_k=5,
    filters={"language": "en"},
    filter_policy=FilterPolicy.MERGE,
)
result = retriever_merge.run(
    query_embedding=my_vector,
    filters={"category": "db"},   # AND'd with {"language": "en"}
)
```

### FilterPolicy

| Policy | Behaviour |
|---|---|
| `REPLACE` (default) | Runtime `filters` completely override init-time `filters` |
| `MERGE` | Runtime `filters` are AND-merged with init-time `filters` |

---

## Implementation note

Under the hood, `filter_documents` works as follows:

```python
from haystack.utils.filters import document_matches_filter

def filter_documents(self, filters=None):
    # 1. Load all documents from IRIS (SELECT id, content, meta, score)
    docs = [self._row_to_document(row) for row in cursor.fetchall()]

    # 2. Apply the filter in Python using the official Haystack utility
    if not filters:
        return docs
    return [d for d in docs if document_matches_filter(filters, d)]
```

`document_matches_filter` is the same function used internally by `InMemoryDocumentStore`, which means the filter behaviour is **guaranteed to be identical** to the Haystack reference implementation.