"""
Streamlit Web Application for the RAG Chatbot.
Demonstrates Retrieval-Augmented Generation using Haystack, Groq,
and InterSystems IRIS as the Vector Database.
Features custom parsing to hide <think> reasoning tags within an expander.
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret

from haystack_integrations.document_stores.intersystems_iris import IRISDocumentStore
from haystack_integrations.components.retrievers.intersystems_iris import IRISEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator


# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🤖", layout="centered")

def extract_think_tags(text: str) -> tuple[str | None, str]:
    """
    Parses the LLM response to separate the <think> reasoning block from the main answer.
    Returns a tuple: (think_content, main_answer).
    """
    # Search for everything between <think> and </think>, including line breaks (re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    
    if think_match:
        think_content = think_match.group(1).strip()
        # Remove the <think> block from the original text to get the clean answer
        main_answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return think_content, main_answer
    
    # If no <think> tag is found, return None for thinking and the original text
    return None, text

@st.cache_resource(show_spinner="Initializing AI Components...")
def initialize_rag_pipeline() -> Pipeline:
    """
    Initializes the InterSystems IRIS connection and builds the RAG Pipeline.
    Uses st.cache_resource to avoid reconnecting on every UI interaction.
    """
    store = IRISDocumentStore(table_name="POC_Docs", embedding_dim=384)
    
    chat_template = """
    You are a helpful and professional corporate assistant.
    Answer the user's question based ONLY on the provided context documents below.
    If the information is not in the documents, politely reply: "I'm sorry, I cannot find this information in our knowledge base."

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
    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    
    return pipeline

def main():
    """Main UI rendering function."""
    st.title("🤖 Enterprise AI Assistant")
    st.caption("Powered by Haystack, Groq, & InterSystems IRIS Vector Search")

    try:
        rag_pipeline = initialize_rag_pipeline()
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        st.stop()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything about our indexed documents!"}
        ]

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # If it's the assistant, we parse the message to hide the <think> tag in the history too
            if message["role"] == "assistant":
                think_content, main_answer = extract_think_tags(message["content"])
                
                if think_content:
                    with st.expander("🧠 View reasoning process (Think)"):
                        st.markdown(think_content)
                
                st.markdown(main_answer)
            else:
                st.markdown(message["content"])

    # User Input Handling
    if user_query := st.chat_input("Type your question here..."):
        
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # 2. Process and Display AI Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving vectors from InterSystems IRIS..."):
                try:
                    result = rag_pipeline.run({
                        "embedder": {"text": user_query},
                        "prompt_builder": {"query": user_query}
                    })
                    
                    raw_answer = result["llm"]["replies"][0]
                    source_documents = result.get("retriever", {}).get("documents", [])
                    
                    # Parse the raw answer to separate thinking from the final answer
                    think_content, main_answer = extract_think_tags(raw_answer)
                    
                    # Display the thinking block if it exists
                    if think_content:
                        with st.expander("🧠 View reasoning process (Think)"):
                            st.markdown(think_content)
                            
                    # Display the clean main answer
                    st.markdown(main_answer)
                    
                    # Display Source Evidence
                    if source_documents:
                        with st.expander("📚 View Retrieval Evidence"):
                            for i, doc in enumerate(source_documents, 1):
                                st.info(f"**Chunk {i} (Cosine Similarity: {doc.score:.2f})**\n\n{doc.content}")

                    # Save the FULL raw answer to history so it can be re-parsed if the user scrolls up
                    st.session_state.messages.append({"role": "assistant", "content": raw_answer})
                
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()