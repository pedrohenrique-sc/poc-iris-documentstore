---
title: User Guide
---

# User Guide

Detailed guides for every component and feature of `iris-haystack`.

---

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **IRISDocumentStore**

    ---
    Initialization options, table schema, connection management, and all four protocol methods.

    [:octicons-arrow-right-24: DocumentStore](document-store.md)

-   :material-vector-bezier:{ .lg .middle } **Embedding Retriever**

    ---
    Semantic search via IRIS native `VECTOR_COSINE`. Supports `FilterPolicy` and runtime filter overrides.

    [:octicons-arrow-right-24: IRISEmbeddingRetriever](embedding-retriever.md)

-   :material-text-search:{ .lg .middle } **BM25 Retriever**

    ---
    Keyword ranking via Okapi BM25 computed in-memory over the filtered document set.

    [:octicons-arrow-right-24: IRISBm25Retriever](bm25-retriever.md)

-   :material-filter-outline:{ .lg .middle } **Metadata Filtering**

    ---
    Legacy dict format, full Haystack `operator/conditions` format, all Python types and operators.

    [:octicons-arrow-right-24: Filtering](filtering.md)

-   :material-file-code:{ .lg .middle } **Pipeline Serialization**

    ---
    Save and restore Haystack pipelines that include an `IRISDocumentStore` to/from YAML files.

    [:octicons-arrow-right-24: Serialization](serialization.md)

-   :material-shield-key:{ .lg .middle } **Credentials & Secrets**

    ---
    How to supply IRIS credentials safely using Haystack `Secret` objects and environment variables.

    [:octicons-arrow-right-24: Credentials](credentials.md)

</div>