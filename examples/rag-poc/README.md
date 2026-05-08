# Enterprise RAG POC: Haystack & InterSystems IRIS 

This folder contains a complete Proof of Concept (POC) demonstrating a Retrieval-Augmented Generation (RAG) pipeline using **Haystack 2.x**, **Qwen 3** (via Groq), and **InterSystems IRIS** as the Vector Database.

## Important: InterSystems IRIS Docker Setup

To run this POC, you need a running instance of InterSystems IRIS. We use Docker Compose to spin up the Community Edition. Please pay attention to the following details:

1. **Initial Password Policy:** InterSystems IRIS enforces a strict security policy requiring a password change on the first login for the `_system` user. Ensure your `.env` file matches the password configured in your database setup.
2. **Network Resolution (Linux/WSL):** If you are running this on WSL or certain Linux distribution    ```Bash
    pip install -r requirements.txt
    ```s, using `localhost` might cause connection timeouts. **Always use `127.0.0.1`** in your connection string to avoid DNS resolution issues.
3. **Ports:** Ensure ports `1972` (SuperServer) and `52773` (Web Management Portal) are free on your host machine before starting the container.

---

## Prerequisites

- **Python 3.10+**
- **Docker** & **Docker Compose**
- A free API key from [Groq Console](https://console.groq.com/)

## Environment Setup

Create a `.env` file in the root of this POC directory (the same folder as `app.py`) with the following variables:

```ini
# Database Credentials
IRIS_CONNECTION_STRING="127.0.0.1:1972/USER"
IRIS_USERNAME="_system"
IRIS_PASSWORD="YourSecurePasswordHere" 

# LLM Provider
GROQ_API_KEY="gsk_your_api_key_here..."
```

## Execution Guide

Follow these steps in order to start the RAG environment:

1. **Start the Vector Database**
Navigate to the folder containing the docker-compose.yml and start the InterSystems IRIS container in the background:

    ```bash
    docker compose up -d
    ```

2. **Install Python Dependencies**
Activate your virtual environment and install the required packages:

    ```bash
    source venv/bin/activate
    ```
    ```
    pip install -r requirements.txt
    ```

3. **Add Knowledge Base Files**
If you want to add knowdlege by .pdf, create a folder named data/ in the same directory as the scripts and place your .pdf files inside it.

    ```Plaintext
    rag_poc/
    ├── data/
    │   ├── company_handbook.pdf
    │   └── HR_policies.pdf
    ├── indexer.py
    └── app.py
    ```
4. **Run the Indexing Pipeline**
Execute the ingestion script. This will read the PDFs, chunk the text, generate embeddings, and store them in InterSystems IRIS.

    ```bash
    python index.py
    
    Note: This script uses recreate_table=False and DuplicatePolicy.SKIP. You can run it multiple times as you add new PDFs without duplicating existing data.
    ```
5. **Launch the Chatbot UI**
Start the Streamlit application to interact with your data:

    ```bash
    streamlit run app.py
    The application will automatically open in your default web browser at http://localhost:8501.
    ```

*** 
