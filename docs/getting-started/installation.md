---
title: Installation
---

# Installation

---

## From PyPI (recommended)

```bash
pip install #Lib-iris-haystack (development)
```

This installs the core package with two required dependencies:

| Dependency | Purpose |
|---|---|
| `haystack-ai` | Haystack 2.x framework |
| `intersystems-irispython` | Official InterSystems DB-API driver |

---

## Optional extras

### Embedding model

The retrievers require embeddings. The recommended free, local model is `all-MiniLM-L6-v2` via `sentence-transformers`:

```bash
pip install #Lib-iris-haystack (development)
# Equivalent to: pip install sentence-transformers
```

You can use any Haystack-compatible embedder — the DocumentStore itself is model-agnostic.

### Development extras

For running tests and working on the source code:

```bash
pip install #Lib-iris-haystack (development)
```

This adds `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy`, and `python-dotenv`.

---

## With hatch (contributor workflow)

If you are contributing to the project, use [hatch](https://hatch.pypa.io/) to manage isolated environments automatically:

```bash
# Install hatch
pip install hatch

# Clone the repository
git clone https://github.com/s-c-ai/iris-haystack.git
cd iris-haystack

# Enter the default dev environment (installs all deps automatically)
hatch shell

# Or run a command directly in the test environment
hatch run test:all
```

See [Development Setup](../development/hatch.md) for a full breakdown of all hatch environments.

---

## Verify the installation

```python
from haystack_integrations.document_stores.iris import IRISDocumentStore
from haystack_integrations.components.retrievers.iris import (
    IRISEmbeddingRetriever,
    IRISBm25Retriever,
)
print("iris-haystack installed correctly ✓")
```

---

## Version pinning

For reproducible deployments, pin the exact version:

```bash
pip install "intersystems-iris-haystack==1.0.0"
```

Or in `pyproject.toml` / `requirements.txt`:

```toml title="pyproject.toml"
dependencies = [
    "intersystems-iris-haystack>=1.0.0,<2.0.0",
]
```

---

## Upgrading

```bash
pip install --upgrade #Lib-iris-haystack (development)
```

!!! tip "Check the Changelog"
    Before upgrading in production, review the [Changelog](../changelog.md) for breaking changes.