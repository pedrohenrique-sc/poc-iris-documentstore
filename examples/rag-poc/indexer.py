"""
Document Indexing Script for InterSystems IRIS Vector Search.
This script loads PDFs from a local directory, combines them with 
pre-defined technical examples, generates vector embeddings, 
and stores everything in InterSystems IRIS.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    load_dotenv()
    logger.info("Starting Advanced Document Ingestion Process (PDFs + Manual Text)...")

    try:
        store = IRISDocumentStore(table_name="POC_Docs", embedding_dim=384, recreate_table=False)
    except Exception as e:
        logger.error(f"Failed to connect to InterSystems IRIS: {e}")
        return

    converter = PyPDFToDocument()
    splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=20)
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    writer = DocumentWriter(document_store=store, policy=DuplicatePolicy.SKIP)
    data_dir = Path("data")
    pdf_docs = []
    
    if data_dir.exists() and data_dir.is_dir():
        pdf_files = list(data_dir.glob("*.pdf"))
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF(s) in '{data_dir.name}/'. Converting and splitting...")
            raw_docs = converter.run(sources=pdf_files)["documents"]
            pdf_docs = splitter.run(documents=raw_docs)["documents"]
        else:
            logger.info(f"No PDFs found in '{data_dir.name}/', skipping PDF extraction.")
    else:
        logger.warning(f"Directory '{data_dir}' not found. Skipping PDF extraction.")

    logger.info("Loading manual technical knowledge base examples...")
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
        ),
        Document(
            content="The security wifi password is P@ssw0rd!",
            meta={"source": "manual", "security": "confidential"}
        )
    ]

    all_docs = pdf_docs + manual_docs
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    logger.info(f"Embedding and indexing a total of {len(all_docs)} document chunks into IRIS...")
    
    try:
        indexing_pipeline.run({"embedder": {"documents": all_docs}})
        logger.info(f"Success! Total documents now stored in InterSystems IRIS: {store.count_documents()}")
    except Exception as e:
        logger.error(f"Error during indexing pipeline execution: {e}")

if __name__ == "__main__":
    main()