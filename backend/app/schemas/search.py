from pydantic import BaseModel, Field
from typing import Optional

class SearchRequest(BaseModel):
    query: str = Field(..., description="The user's question or context for the RAG pipeline.")
    region: str = Field(..., description="User's region abbreviation or full name (e.g., 'NCR').")
    city: str = Field(..., description="User's city or municipality.")
    barangay: Optional[str] = Field(None, description="User's barangay (optional).")
    is_philhealth: bool = Field(True, description="Filter for PhilHealth-accredited facilities.")
    is_malasakit: bool = Field(False, description="Filter for Malasakit-accredited facilities.")