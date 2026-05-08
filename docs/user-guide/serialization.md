---
title: Pipeline Serialization
---

# Pipeline Serialization

Haystack pipelines can be saved to and loaded from YAML files.
`iris-haystack` fully supports this — both `IRISDocumentStore` and
the companion retrievers implement `to_dict()` and `from_dict()`.

---

## How it works

Haystack serializes a pipeline by calling `to_dict()` on each component.
The resulting dictionary contains the class path and initialization parameters.
On deserialization, `from_dict()` reconstructs each component.

For `IRISDocumentStore`, credentials are handled specially:

- The `connection_string`, `username`, and `password` `Secret` objects are serialized
  as their **env var names** (e.g., `"IRIS_PASSWORD"`), **never as resolved values**.
- On `from_dict()`, `deserialize_secrets_inplace` rebuilds the `Secret` objects,
  which then resolve the values from the environment at runtime.

---

## Saving a pipeline

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.iris import IRISDocumentStore
from haystack_integrations.components.retrievers.iris import IRISEmbeddingRetriever

store = IRISDocumentStore(embedding_dim=384)

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

# Save to YAML
with open("rag_pipeline.yaml", "w") as f:
    pipeline.dump(f)
```

The generated `rag_pipeline.yaml` will look like:

```yaml
components:
  embedder:
    init_parameters:
      model: sentence-transformers/all-MiniLM-L6-v2
    type: haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder

  retriever:
    init_parameters:
      document_store:
        init_parameters:
          connection_string:
            env_vars:
              - IRIS_CONNECTION_STRING  # env var name, not the value
            strict: true
            type: env_var
          embedding_dim: 384
          password:
            env_vars:
              - IRIS_PASSWORD
            strict: true
            type: env_var
          table_name: HaystackDocuments
          username:
            env_vars:
              - IRIS_USERNAME
            strict: true
            type: env_var
        type: haystack_integrations.document_stores.iris.document_store.IRISDocumentStore
      filter_policy: replace
      top_k: 5
    type: haystack_integrations.components.retrievers.iris.embedding_retriever.IRISEmbeddingRetriever

connections:
  - receiver: retriever.query_embedding
    sender: embedder.embedding
max_runs_per_component: 100
metadata: {}
```

---

## Loading a pipeline

```python
# Make sure credentials are set in the environment
import os
os.environ["IRIS_CONNECTION_STRING"] = "localhost:1972/USER"
os.environ["IRIS_USERNAME"]          = "_system"
os.environ["IRIS_PASSWORD"]          = "SYS"

from haystack import Pipeline

with open("rag_pipeline.yaml") as f:
    pipeline = Pipeline.load(f)

# Use it immediately
result = pipeline.run({"embedder": {"text": "what is vector search?"}})
```

---

## BM25 pipeline example

```python
from haystack_integrations.components.retrievers.iris import IRISBm25Retriever

bm25_pipeline = Pipeline()
bm25_pipeline.add_component(
    "retriever",
    IRISBm25Retriever(document_store=store, top_k=10),
)

with open("bm25_pipeline.yaml", "w") as f:
    bm25_pipeline.dump(f)

with open("bm25_pipeline.yaml") as f:
    restored = Pipeline.load(f)

result = restored.run({"retriever": {"query": "database SQL"}})
```

---

## Security considerations

!!! danger "Never commit `rag_pipeline.yaml` with hardcoded credentials"
    The YAML file is safe to commit **as long as you use `Secret.from_env_var`**
    (the default). The YAML will contain the env var name, not the resolved password.

    If you initialize the store with `Secret.from_token("my-password")` instead,
    the token value **will** appear in the YAML. Never do this in production.

### Using `.env` files in CI

```bash
# .env (never committed — in .gitignore)
IRIS_CONNECTION_STRING=prod-server:1972/PROD
IRIS_USERNAME=haystack_user
IRIS_PASSWORD=s3cur3P@ssw0rd
```

```python
from dotenv import load_dotenv
load_dotenv()  # loads from .env before constructing the pipeline

with open("rag_pipeline.yaml") as f:
    pipeline = Pipeline.load(f)
```