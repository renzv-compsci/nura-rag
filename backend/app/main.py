import sys
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai

from backend.app.services.translator import translate_rag_response, SUPPORTED_LANGUAGES

DOTENV_PATH = r"C:\Users\Renz\nura-rag\.env"
load_dotenv(dotenv_path=DOTENV_PATH)

if not os.environ.get("SUPABASE_KEY") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
    os.environ["SUPABASE_KEY"] = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Fix possible region mismatch for Vertex AI initialization
if not os.environ.get("GCP_REGION") and os.environ.get("GCP_LOCATION"):
    os.environ["GCP_REGION"] = os.environ["GCP_LOCATION"]

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.schemas.search import SearchRequest
from facility_search import get_facilities_in_city
from backend.app.infrastracture.retrieval.pgvector_client import PgVectorBenefitGuideRepo

app = FastAPI(
    title="Nura RAG & Facility Unified API",
    description="Single server handling both contextual RAG queries and facility search.",
    version="1.0.0",
)

# 1. Initialize RAG Client
print("Initializing PgVector RAG client...")
try:
    rag_repo = PgVectorBenefitGuideRepo()
    print("RAG client initialized.")
except Exception as e:
    rag_repo = None
    print(f"⚠️ Failed to initialize RAG client: {e}")

# 2. Initialize Gemini Model
print("Initializing Gemini model...")
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    # Remove quotes if present
    if api_key.startswith('"') and api_key.endswith('"'):
        api_key = api_key[1:-1]
    genai.configure(api_key=api_key)
    
    # Get first available model for generation
    available_models = []
    try:
        # Use a stable fast model by default instead of listing all
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        print("✓ Gemini model initialized: gemini-2.5-flash")
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
    """
    
    # Validate language
    if getattr(request, "language", "en") not in SUPPORTED_LANGUAGES:
        return {
            "error": f"Unsupported language. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
            "status": "error"
        }
    
    request_language = getattr(request, "language", "en")

    # 1. Facility Search
    try:
        from facility_search import get_facilities_in_city # locally import to ensure it works
        facility_response = get_facilities_in_city(
            region=request.region,
            city=request.city,
            is_philhealth=request.is_philhealth,
            is_malasakit=request.is_malasakit
        )
        facilities_data = getattr(facility_response, "dict", lambda: facility_response)() if hasattr(facility_response, "dict") else facility_response.model_dump()
        facilities_data["facilities"] = facilities_data.get("facilities", [])[:5]
    except HTTPException as e:
        facilities_data = {"error": e.detail, "facilities": []}
    except Exception as e:
        facilities_data = {"error": str(e), "facilities": []}

    # Format facilities for the LLM prompt
    fac_list_str = "No facilities found matching the criteria."
    if facilities_data.get("facilities"):
        fac_list_str = "\n".join([
            f"- {f.get('name_of_health_facility', 'Unknown Clinic')} ({f.get('street', 'No street')}, {f.get('municipality_city', '')})"
            for f in facilities_data["facilities"]
        ])

    # 2. RAG Query
    retrieved_contexts = []
    rag_data = "No relevant guidelines found."
    if rag_repo:
        try:
            retrieved_contexts = rag_repo.retrieve_guides(query_text=request.query, limit=3)
            rag_data = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No relevant guidelines found."
        except Exception as e:
            rag_data = f"RAG retrieval failed: {e}"

    # 3. LLM Generation with RAG Context & Facility Data
    llm_response = "AI Generation is currently unavailable."
    if gemini_model:
        try:
            # We strictly instruct the LLM to act as a bridge to healthcare, NOT a doctor.
            system_prompt = """You are Nura, an empathetic AI healthcare guide for PhilHealth members in the Philippines. 
YOUR CRITICAL RULES:
1. DO NOT DIAGNOSE. DO NOT prescribe home remedies (e.g., never say "drink water", "take over the counter medicine", or "rest").
2. DO NOT ACT AS A DOCTOR.
3. Validate their concern empathetically, but immediately encourage them to visit a clinic or hospital for proper assessment. 
4. Assure them that consulting a professional is the best step and that they can use their PhilHealth benefits at accredited facilities to lessen the burden.
5. ALWAYS list the provided accredited facilities naturally in your response as the best next step.

Your goal is to eliminate the patient's fear of going to the clinic by showing them accessible, accredited locations nearby."""
            
            full_prompt = f"""{system_prompt}

--- PHILHEALTH GUIDELINES CONTEXT ---
{rag_data}

--- NEARBY ACCREDITED FACILITIES ---
{fac_list_str}

User's Question: {request.query}
User's Location: {request.city}, {request.region}

Helpful Answer (incorporate the facility recommendations to encourage a visit):"""
            
            response = gemini_model.generate_content(full_prompt)
            llm_response = response.text
        except Exception as e:
            llm_response = f"LLM generation failed: {e}"

    # 4. Translation
    translated_response = llm_response
    if llm_response and request_language != "en" and not llm_response.startswith("LLM generation failed"):
        try:
            translated_response = translate_rag_response(llm_response, request_language)
        except Exception as e:
            translated_response = f"Translation failed: {e}. Original: {llm_response}"

    # 5. Return unified response
    return {
        "status": "success",
        "query": request.query,
        "language_requested": SUPPORTED_LANGUAGES[request_language]["name"],
        "facilities": facilities_data.get("facilities", []), 
        "llm_response": translated_response if request_language != "en" else llm_response,
        "original_response": llm_response if request_language != "en" else None
    }
