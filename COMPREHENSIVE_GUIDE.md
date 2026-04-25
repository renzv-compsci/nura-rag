# Nura RAG & Facility Unified API 🧠 🏥
**Comprehensive Architecture & Process Guide**

Welcome to the comprehensive deep-dive guide for the **Nura RAG** repository. This document explains the architecture, data flow, technologies used, and the step-by-step processes that power the Nura unified backend.

---

## 1. System Overview

Nura RAG is a smart, contextual backend system built with **FastAPI**. Its primary goal is to help citizens understand their health benefits (specifically PhilHealth) and find relevant healthcare facilities. 

The system is broken down into two main pillars:
1. **RAG (Retrieval-Augmented Generation) Pipeline**: Reads official PhilHealth PDFs, stores them as vector embeddings, and uses an AI model (Google Gemini) to answer user questions strictly based on the official guidelines.
2. **Unified Search & Translation**: Combines facility searches with contextual RAG answers, and translates the final AI response into local Philippine languages (Filipino, Cebuano, Hiligaynon, Ilocano) so it is accessible to everyone.

---

## 2. Tech Stack

*   **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **Vector Database**: [Supabase](https://supabase.com/) with the `pgvector` extension.
*   **Embeddings**: Google Cloud [Vertex AI](https://cloud.google.com/vertex-ai) (`text-embedding-004`).
*   **LLM (Large Language Model)**: Google [Gemini](https://aistudio.google.com/) API (`gemini-1.5-flash` / `gemini-2.5-flash`).
*   **Translation**: `deep-translator` (Google Translate API wrapper).
*   **Document Processing**: `PyMuPDF` (fitz) for PDF extraction, LangChain for chunking.

---

## 3. The Three Core Processes

### Process A: Document Ingestion (The "Knowledge" Phase)
**File:** `backend/app/scripts/ingest_philhealth.py`

This script runs entirely offline/ahead-of-time to populate the database with knowledge.
1. **Extraction**: Uses `fitz` to open the PhilHealth Benefits PDF and extract all raw text.
2. **Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` to break the massive text into smaller, digestible chunks (500 characters each, with 50 characters overlap to keep context).
3. **Embedding**: Sends each chunk to Google Cloud's Vertex AI (`text-embedding-004`), converting the text into a mathematical vector (an array of floating-point numbers).
4. **Storage**: Uploads the raw text chunk and its corresponding vector embedding into the Supabase database (`benefit_guides` table) in batches.

### Process B: Semantic Retrieval (The "Search" Phase)
**File:** `backend/app/infrastracture/retrieval/pgvector_client.py`

When a user asks a question, we don't send the entire PDF to the AI. Instead:
1. **Query Embedding**: The user's question (e.g., *"What are my PhilHealth benefits?"*) is sent to Vertex AI to get a query vector.
2. **Vector Search (Cosine Similarity)**: The system passes the vector to Supabase using the remote procedure call (`rpc`) named `match_benefits`.
3. **Threshold Filtering**: Supabase compares the query vector to all document vectors. We use a `match_threshold` of `0.1` to ensure we grab the closest matches.
4. **Return Context**: It returns the top 3 or 4 text chunks that are semantically most relevant to the user's question.

### Process C: AI Generation & Translation (The "Answer" Phase)
**Files:** `backend/app/scripts/test_rag.py`, `backend/app/services/translator.py`, `backend/app/main.py`

Once we have the relevant snippets of information:
1. **Prompt Construction**: We build a prompt for Gemini that looks like this:
   > *"You are an AI assistant. Use this context to answer the question. Context: [Supabase Chunks]. Question: [User Question]"*
2. **LLM Invocation**: The prompt is sent to the `google.generativeai` API. Because we passed explicit context, Gemini answers factually without "hallucinating."
3. **Translation**: If the user requested a language other than English (e.g., `fil`, `ceb`), the final English response from Gemini is intercepted and passed through `translator.translate_rag_response()`, which converts it to the preferred dialect.

---

## 4. API Endpoints (`main.py`)

The system is tied together via a FastAPI server, exposing these endpoints:

### `GET /health`
Returns the status of the API, checking if the RAG client and Gemini model successfully initialized on startup.

### `GET /supported-languages`
Returns a JSON list of supported languages dynamically generated from `SUPPORTED_LANGUAGES` in the translation service.

### `POST /search`
The flagship endpoint of the system. 
*   **Input (`SearchRequest`)**: Takes the user's `query`, `region`, `city`, boolean flags (`is_philhealth`, `is_malasakit`), and desired `language`.
*   **Execution**:
    1. Triggers the facility search function to find nearby clinics/hospitals.
    2. Triggers the PgVector client to retrieve relevant PDF chunks.
    3. Passes retrieved chunks to Gemini to formulate an answer.
    4. Translates the final answer.
*   **Output**: A JSON payload containing the intermediate facility data, the RAG retrieved context, and the final translated LLM response.

---

## 5. Environment & Configuration (.env)

To run the system, specific API credentials are required to route between Supabase, Google Cloud, and Google AI Studio:

```env
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-secret-key>   # Required for bypassing RLS during ingestion
GCP_PROJECT_ID=<google-cloud-project-id>      # Used for Vertex AI Embeddings
GCP_REGION=us-central1                        # Default Vertex region
GOOGLE_API_KEY=<google-ai-studio-key>         # Bypasses GCP routing, used directly for Gemini Chat
```
*(Crucial Note: Make sure there are no quotes around the values in your `.env` file to prevent parsing errors).*

---

## 6. How to Test locally

If you want to test the RAG without starting the FastAPI server, you can use the interactive CLI:

```powershell
python backend/app/scripts/test_rag.py
```
This script will:
1. Verify Supabase connection.
2. Verify Google Generative AI connection.
3. Allow you to select a translation language (Tagalog, Cebuano, etc.).
4. Let you type questions iteratively.

To run the production server:
```powershell
uvicorn backend.app.main:app --reload
```

---

## 7. Future Scalability

*   **Caching**: Translated RAG responses could be cached using Redis to save LLM tokens and translation calls for identical questions.
*   **Multi-Document Support**: Ingestion can be scaled to pull multiple PhilHealth, DOH, or LGU guidelines by adding metadata tags filtering searches to specific government sectors.
*   **Streaming**: The FastAPI endpoint can be upgraded to stream Gemini's response token-by-token for a better UI experience.