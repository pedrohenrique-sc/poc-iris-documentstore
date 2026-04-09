# SPDX-FileCopyrightText: 2026-present Pedro Henrique <pedro@s-c.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end RAG pipeline example with IRISDocumentStore.

This example shows:
  1. Indexing pipeline: documents → embeddings → IRIS
  2. Semantic search pipeline: query → embedding → VECTOR_COSINE retrieval
  3. BM25 keyword search
  4. Haystack filter formats (legacy and official)

Prerequisites:
    docker-compose up -d
    pip install iris-haystack sentence-transformers
    export IRIS_CONNECTION_STRING="localhost:1972/USER"
    export IRIS_USERNAME="_system"
    export IRIS_PASSWORD="SYS"
"""

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.intersystems_iris import (
    IRISBm25Retriever,
    IRISEmbeddingRetriever,
)
from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# 1. Initialize the store
# ---------------------------------------------------------------------------
store = IRISDocumentStore(embedding_dim=384)
print(f"Store: {store}")
print(f"Documents before indexing: {store.count_documents()}\n")

# ---------------------------------------------------------------------------
# 2. Indexing pipeline
# ---------------------------------------------------------------------------
indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component(
    "writer",
    DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE),
)
indexing.connect("embedder.documents", "writer.documents")

documents = [
    Document(
        content="InterSystems IRIS is a high-performance multimodel database.",
        meta={"category": "database", "year": 2024, "lang": "en"},
    ),
    Document(
        content="Haystack is an open-source framework for building LLM applications.",
        meta={"category": "framework", "year": 2024, "lang": "en"},
    ),
    Document(
        content="Vector search enables semantic similarity retrieval using embeddings.",
        meta={"category": "concept", "year": 2023, "lang": "en"},
    ),
    Document(
        content="IRIS supports SQL, JSON, vectors, and globals in a single platform.",
        meta={"category": "database", "year": 2024, "lang": "en"},
    ),
    Document(
        content="Haystack 2.x introduced a component-based, declarative pipeline API.",
        meta={"category": "framework", "year": 2024, "lang": "en"},
    ),
]

print("Indexing documents...")
indexing.run({"embedder": {"documents": documents}})
print(f"Documents after indexing: {store.count_documents()}\n")

# ---------------------------------------------------------------------------
# 3. Semantic search pipeline
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEMANTIC SEARCH (VECTOR_COSINE)")
print("=" * 60)

query_pipeline = Pipeline()
query_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
query_pipeline.add_component(
    "retriever", IRISEmbeddingRetriever(document_store=store, top_k=3)
)
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

for query in [
    "how does similarity search work?",
    "what are the features of IRIS?",
    "what is RAG with LLMs?",
]:
    result = query_pipeline.run({"embedder": {"text": query}})
    print(f"\n🔍 '{query}'")
    for i, doc in enumerate(result["retriever"]["documents"], 1):
        print(f"  {i}. [{doc.score:.4f}] {doc.content[:70]}...")

# ---------------------------------------------------------------------------
# 4. BM25 keyword search
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("KEYWORD SEARCH (BM25)")
print("=" * 60)

bm25_retriever = IRISBm25Retriever(document_store=store, top_k=3)
for query in ["database SQL JSON", "pipeline components"]:
    result = bm25_retriever.run(query=query)
    print(f"\n🔍 '{query}'")
    for i, doc in enumerate(result["documents"], 1):
        print(f"  {i}. [{doc.score:.4f}] {doc.content[:70]}...")

# ---------------------------------------------------------------------------
# 5. Filters — legacy format
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FILTERS — LEGACY FORMAT")
print("=" * 60)

print("\ncategory = 'database':")
for doc in store.filter_documents({"category": "database"}):
    print(f"  - {doc.content[:70]}...")

print("\nyear = 2023:")
for doc in store.filter_documents({"year": 2023}):
    print(f"  - {doc.content[:70]}...")

# ---------------------------------------------------------------------------
# 6. Filters — official Haystack format
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FILTERS — OFFICIAL HAYSTACK FORMAT (operator/conditions)")
print("=" * 60)

print("\nAND: category == framework AND year >= 2024:")
docs = store.filter_documents({
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "framework"},
        {"field": "meta.year",     "operator": ">=", "value": 2024},
    ],
})
for doc in docs:
    print(f"  - {doc.content[:70]}...")

print("\nOR: category == database OR category == concept:")
docs = store.filter_documents({
    "operator": "OR",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "database"},
        {"field": "meta.category", "operator": "==", "value": "concept"},
    ],
})
for doc in docs:
    print(f"  - {doc.content[:70]}...")

# ---------------------------------------------------------------------------
# 7. Embedding retrieval with filter
# ---------------------------------------------------------------------------
print("\nSemantic search filtered to 'database' only:")
result = query_pipeline.run({"embedder": {"text": "features of the platform"}})
emb = result["embedder"]["embedding"]
filtered = store._embedding_retrieval(
    query_embedding=emb, top_k=3, filters={"category": "database"}
)
for i, doc in enumerate(filtered, 1):
    print(f"  {i}. [{doc.score:.4f}] {doc.content[:70]}...")

print("\n✅ Example completed successfully!")
store.close()
