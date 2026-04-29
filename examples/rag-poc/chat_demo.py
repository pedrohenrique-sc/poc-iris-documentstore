import os
from dotenv import load_dotenv

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore
from haystack_integrations.components.retrievers.intersystems_iris import IRISEmbeddingRetriever

from haystack.components.generators import OpenAIGenerator

def main():
    load_dotenv()
    
    print("--- Phase 2: RAG Chatbot Demo ---")
    store = IRISDocumentStore(table_name="Mixed_POC_Docs", embedding_dim=384)
    
    if store.count_documents() == 0:
        print("Error: The database is empty. Please run 'index_documents.py' first.")
        return

    chat_template = """
    You are a helpful and professional corporate assistant. 
    Answer the user's question based ONLY on the provided context documents below.
    If the context does not contain the answer, politely say "I'm sorry, I cannot find this information in the provided documents."
    Do not hallucinate or use outside knowledge.

    Context Documents:
    {% for doc in documents %}
        - {{ doc.content }}
    {% endfor %}

    Question: {{ query }}
    Answer:
    """

    embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = IRISEmbeddingRetriever(document_store=store, top_k=3) 
    prompt_builder = PromptBuilder(template=chat_template)
    
    # Using Qwen 3 via Groq
    llm = OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model="qwen/qwen3-32b",
        generation_kwargs = {"max_tokens": 512}
    )
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("embedder", embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

    print("\n System Ready. Type 'exit' to quit.\n")
    print("="*50)

    while True:
        user_query = input("\n🧑‍💻 You: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_query.strip():
            continue

        print("🤖 Bot is thinking (Retrieving from IRIS & Generating)...")

        result = rag_pipeline.run({
            "embedder": {"text": user_query},
            "prompt_builder": {"query": user_query}
        })
        
        answer = result["llm"]["replies"][0]
        print(f"\n🤖 Assistant: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()