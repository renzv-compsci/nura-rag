import sys
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai

DOTENV_PATH = r"C:\Users\Renz\nura-rag\.env"
load_dotenv(dotenv_path=DOTENV_PATH)

if not os.environ.get("SUPABASE_KEY") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
    os.environ["SUPABASE_KEY"] = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Fix possible region mismatch for Vertex AI initialization
if not os.environ.get("GCP_REGION") and os.environ.get("GCP_LOCATION"):
    os.environ["GCP_REGION"] = os.environ["GCP_LOCATION"]

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.schemas.search import SearchRequest
from backend.app.infrastracture.retrieval.pgvector_client import PgVectorBenefitGuideRepo
from backend.app.services.translator import (
    translate_rag_response, 
    SUPPORTED_LANGUAGES
)

app = FastAPI(
    title="Nura RAG & Facility Unified API",
    description="Single server handling both contextual RAG queries and facility search.",
    version="1.0.0",
)

# Initialize the RAG component once on startup
print("Initializing PgVector RAG client...")
rag_repo = PgVectorBenefitGuideRepo()
print("RAG client initialized.")

# Initialize Gemini model on startup
print("Initializing Gemini model...")
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    # Remove quotes if present
    if api_key.startswith('"') and api_key.endswith('"'):
        api_key = api_key[1:-1]
    genai.configure(api_key=api_key)
    
    # Get first available model
    available_models = []
    try:
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                available_models.append(model.name.split("/")[-1])
        
        if available_models:
            gemini_model = genai.GenerativeModel(available_models[0])
            print(f"✓ Gemini model initialized: {available_models[0]}")
        else:
            gemini_model = None
            print("⚠️  No Gemini models available")
    except Exception as e:
        gemini_model = None
        print(f"⚠️  Failed to initialize Gemini: {e}")
else:
    gemini_model = None
    print("⚠️  GOOGLE_API_KEY not found in environment")

@app.get("/health", tags=["Meta"])
def health_check():
    return {
        "status": "ok",
        "message": "Unified API is running",
        "rag_ready": rag_repo is not None,
        "gemini_ready": gemini_model is not None,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }

@app.post("/search", tags=["Unified Search"])
def unified_search(request: SearchRequest):
    """
    Unified search endpoint combining RAG, Facility Search, and LLM generation with translation.
    
    Parameters:
    - query: User's question about PhilHealth benefits
    - region: User's region
    - city: User's city
    - is_philhealth: Filter for PhilHealth facilities
    - is_malasakit: Filter for Malasakit facilities
    - language: Response language (en, fil, ceb, hil, ilo)
    """
    
    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        return {
            "error": f"Unsupported language '{request.language}'. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
            "status": "error"
        }

    # 1. RAG Query
    try:
        retrieved_contexts = rag_repo.retrieve_guides(query_text=request.query, limit=3)
        rag_data = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No relevant guidelines found."
    except Exception as e:
        rag_data = f"RAG retrieval failed: {e}"
        retrieved_contexts = []

    # 2. LLM Generation with RAG Context
    llm_response = None
    if gemini_model and retrieved_contexts:
        try:
            system_prompt = """You are an AI assistant helping citizens understand their PhilHealth benefits.
Use the provided context to answer the user's question accurately.
If you don't know the answer based on the context, say that you don't have that information."""
            
            full_prompt = f"""{system_prompt}

Context:
{rag_data}

Question: {request.query}

Helpful Answer:"""
            
            response = gemini_model.generate_content(full_prompt)
            llm_response = response.text
        except Exception as e:
            llm_response = f"LLM generation failed: {e}"

    # 3. Translation
    translated_response = llm_response
    if llm_response and request.language != "en":
        try:
            translated_response = translate_rag_response(llm_response, request.language)
        except Exception as e:
            translated_response = f"Translation failed: {e}. Original: {llm_response}"

    # 4. Return unified response
    return {
        "status": "success",
        "query": request.query,
        "language_requested": SUPPORTED_LANGUAGES[request.language]["name"],
        "rag_context": rag_data[:500] + "..." if len(rag_data) > 500 else rag_data,  # Truncate for response size
        "llm_response": translated_response if request.language != "en" else llm_response,
        "original_response": llm_response if request.language != "en" else None,
        "retrieved_chunks_count": len(retrieved_contexts)
    }

@app.get("/supported-languages", tags=["Meta"])
def get_supported_languages():
    """Get list of supported languages for translation."""
    return {
        "supported_languages": [
            {"code": code, "name": config["name"]}
            for code, config in SUPPORTED_LANGUAGES.items()
        ]
    }