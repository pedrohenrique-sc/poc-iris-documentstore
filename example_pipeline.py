import os
from dotenv import load_dotenv

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from iris_document_store import IRISDocumentStore
from iris_retriever import IRISEmbeddingRetriever

load_dotenv()

store = IRISDocumentStore(
    host=os.getenv("IRIS_HOST", "localhost"),
    port=int(os.getenv("IRIS_PORT", "1972")),
    namespace=os.getenv("IRIS_NAMESPACE", "USER"),
    username=os.getenv("IRIS_USERNAME", "_system"),
    password=os.getenv("IRIS_PASSWORD", "SYS"),
    table_name="HaystackDocuments",
    embedding_dim=384,
)

print(f"DocumentStore: {store}")
print(f"Total de documentos antes da indexação: {store.count_documents()}")

indexing_pipeline = Pipeline()

indexing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(
        document_store=store,
        policy=DuplicatePolicy.OVERWRITE,
    ),
)
indexing_pipeline.connect("embedder.documents", "writer.documents")

# Documents - Exemplo
documents = [
    Document(
        content="InterSystems IRIS é um banco de dados multimodelo de alta performance.",
        meta={"categoria": "banco_de_dados", "idioma": "pt"},
    ),
    Document(
        content="Haystack é um framework open-source para construir aplicações com LLMs e RAG.",
        meta={"categoria": "framework", "idioma": "pt"},
    ),
    Document(
        content="Busca vetorial permite encontrar documentos semanticamente similares usando embeddings.",
        meta={"categoria": "conceito", "idioma": "pt"},
    ),
    Document(
        content="O Haystack 2.x introduziu uma API baseada em componentes e pipelines declarativos.",
        meta={"categoria": "framework", "idioma": "pt"},
    ),
    Document(
        content="IRIS suporta SQL padrão, JSON, vetores e globals na mesma plataforma.",
        meta={"categoria": "banco_de_dados", "idioma": "pt"},
    ),
]

print("\nIndexando documentos...")
indexing_pipeline.run({"embedder": {"documents": documents}})
print(f"Total após indexação: {store.count_documents()} documentos\n")

query_pipeline = Pipeline()

query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
query_pipeline.add_component(
    "retriever",
    IRISEmbeddingRetriever(document_store=store, top_k=3),
)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

queries = [
    "como funciona busca por similaridade?",
    "quais são os recursos do IRIS?",
    "o que é RAG com LLM?",
]

for query in queries:
    print(f"🔍 Consulta: '{query}'")
    result = query_pipeline.run({"text_embedder": {"text": query}})
    for i, doc in enumerate(result["retriever"]["documents"], 1):
        print(f"  {i}. [{doc.score:.4f}] {doc.content[:80]}...")
    print()

print("🔍 Buscando apenas documentos da categoria 'banco_de_dados':")
filtered_docs = store.filter_documents(filters={"categoria": "banco_de_dados"})
for doc in filtered_docs:
    print(f"  - {doc.content[:80]}...")

store.close()