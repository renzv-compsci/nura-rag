import sys
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

DOTENV_PATH = r"C:\Users\Charles\Documents\nura-rag\nura-rag\.env"
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

# Initialize the RAG component once on startup
print("Initializing PgVector RAG client...")
rag_repo = PgVectorBenefitGuideRepo()
print("RAG client initialized.")

@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "ok", "message": "Unified API is running"}

@app.post("/search", tags=["Unified Search"])
def unified_search(request: SearchRequest):
    # 1. Facility Search
    try:
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

    # 2. RAG Query
    try:
        # Fetch top 3 relevant text chunks based on the user's question
        retrieved_contexts = rag_repo.retrieve_guides(query_text=request.query, limit=3)
        rag_data = "\n\n".join(retrieved_contexts) if retrieved_contexts else "No relevant guidelines found."
    except Exception as e:
        rag_data = f"RAG retrieval failed: {e}"

    # 3. LLM Post-Processing (Placeholder for Step 5)
    final_output = "LLM merge pending..."

    return {
        "status": "success",
        "intermediate_facilities": facilities_data,
        "rag_context": rag_data,
        "final_output_placeholder": final_output
    }