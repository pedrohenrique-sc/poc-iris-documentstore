# intersystems-iris-haystack

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Haystack 2.x](https://img.shields.io/badge/Haystack-2.x-orange)](https://haystack.deepset.ai/)

A [Haystack 2.x](https://haystack.deepset.ai/) `DocumentStore` backed by [InterSystems IRIS](https://www.intersystems.com/products/intersystems-iris/) — native vector search via `VECTOR_COSINE`, keyword search via Okapi BM25, and full Haystack filter protocol support.

---

## Installation

**Requires:** Python 3.10+, a running InterSystems IRIS instance.

Start IRIS locally with Docker:

```bash
docker run -d --name iris -p 1972:1972 -p 52773:52773 \
  intersystemsdc/iris-community:latest
```

Start an interactive terminal with the following:

```bash
docker exec -it my-iris iris session IRIS
```

Or login to the Mangement Portal at http://localhost:52773/csp/sys/%25CSP.Portal.Home.zen

The default username is ```_SYSTEM``` and password is ```SYS```; you will be prompted to change this password after logging in.

---

## Quick start

Create a ```.env``` file using ```.env.example``` template and import the default config credentials for IntersystemsIris.

```bash
IRIS_CONNECTION_STRING="localhost:1972/USER"
IRIS_USERNAME="_system"
IRIS_PASSWORD="SYS"
```

### Example (RAG)


```python
from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.iris import IRISDocumentStore
from haystack_integrations.components.retrievers.iris import (
    IRISEmbeddingRetriever,
    IRISBm25Retriever,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
store = IRISDocumentStore(embedding_dim=384)

# Indexing
indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component("writer", DocumentWriter(store, policy=DuplicatePolicy.OVERWRITE))
indexing.connect("embedder.documents", "writer.documents")
indexing.run({"embedder": {"documents": [
    Document(content="IRIS is a multimodel database.", meta={"category": "db"}),
    Document(content="Haystack builds LLM pipelines.",  meta={"category": "ai"}),
]}})

# Semantic search
query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
query_pipeline.add_component("retriever", IRISEmbeddingRetriever(store, top_k=3))
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

result = query_pipeline.run({"embedder": {"text": "what is vector search?"}})

# BM25 keyword search
bm25 = IRISBm25Retriever(store, top_k=3)
result = bm25.run(query="multimodel database")
```

---

## Documentation

The full documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and covers installation, all components, API reference, and a contributor guide.

### Serve locally

#### With hatch (recommended)

```bash
# Install hatch if you don't have it
pip install hatch

# Serve docs with live reload at http://127.0.0.1:8000
hatch run docs:serve
```

#### With pip

```bash
pip install mkdocs-material mkdocstrings[python] \
mkdocs-git-revision-date-localized-plugin \
mkdocs-minify-plugin pymdown-extensions mike

mkdocs serve
```

---

## Development

### Setup

```bash
git clone https://github.com/s-c-ai/iris-haystack.git
cd iris-haystack

# Start IRIS and example
cd examples/
docker-compose up -d
hatch run example:run

# Run all tests
hatch run test:all
```

### Test commands

| Command | Description |
|---|---|
| `hatch run test:unit` | Unit tests — no IRIS required |
| `hatch run test:integration` | Integration tests — IRIS must be running |
| `hatch run test:all` | All tests |
| `hatch run test:cov` | All tests with coverage report |

### Code quality

```bash
hatch run fmt          # format and fix lint issues
hatch run fmt-check    # check only (used in CI)
hatch run type-check   # mypy
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## References

- [Haystack 2.x — Custom DocumentStore](https://docs.haystack.deepset.ai/docs/creating-custom-document-stores)
- [Haystack — Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
- [InterSystems IRIS — Vector Search](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch)
- [intersystems-irispython — DB-API driver](https://pypi.org/project/intersystems-irispython/)
- [InterSystems Developer Community](https://community.intersystems.com/)
- [Haystack Integrations](https://haystack.deepset.ai/integrations)

