import os
from pathlib import Path
from dotenv import load_dotenv

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore

def main():
    load_dotenv()
    
    print("Advanced Document Indexing (Multiple PDFs + Raw Text)")
    
    store = IRISDocumentStore(table_name="POC_Docs", embedding_dim=384, recreate_table=True)
    converter = PyPDFToDocument()
    splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=20)
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE)
    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf"))
    pdf_docs = []
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF(s) in the '{data_dir.name}' folder. Converting...")
        raw_docs = converter.run(sources=pdf_files)["documents"]
        pdf_docs = splitter.run(documents=raw_docs)["documents"]
    else:
        print(f"No PDFs found in '{data_dir.name}/', skipping PDF part.")

    print("Adding manual technical knowledge base...")
    manual_docs = [
        Document(
            content="Vector Embeddings are numerical representations of concepts. In a Vector Store like InterSystems IRIS, similar meanings are stored close to each other mathematically.",
            meta={"topic": "AI Concepts", "source": "manual"}
        ),
        Document(
            content="AI Agents are autonomous systems that use Large Language Models (LLMs) to reason, use tools, and perform tasks to achieve a specific goal without step-by-step programming.",
            meta={"topic": "Agents", "source": "manual"}
        ),
        Document(
            content="Agentic RAG goes beyond simple search by allowing the model to decide if it needs more information or if it should use a specific tool (like a calculator or API) before answering.",
            meta={"topic": "Agents", "source": "manual"}
        ),
        Document(
            content="Haystack 2.x is a modular framework where every step (embedding, retrieving, generating) is a 'Component'. These components are connected via 'Pipelines'.",
            meta={"topic": "Haystack", "source": "manual"}
        ),
        Document(
            content="The DocumentWriter component in Haystack is responsible for taking processed documents and saving them into a DocumentStore, such as the InterSystems IRIS integration.",
            meta={"topic": "Haystack", "source": "manual"}
        ),        
        Document(
            content="The company wifi password for the main meeting room is Pineaple123.",
            meta={"source": "manual", "security": "confidential"}
        )
    ]
    all_docs = pdf_docs + manual_docs

    # Pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    print(f"Embedding and indexing a total of {len(all_docs)} document chunks into IRIS...")
    indexing_pipeline.run({"embedder": {"documents": all_docs}})
    
    print(f"\nDone! Total documents stored in InterSystems IRIS: {store.count_documents()}")

if __name__ == "__main__":
    main()