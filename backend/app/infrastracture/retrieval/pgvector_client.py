import os
from typing import Sequence
from supabase import create_client, Client

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from ...services.interfaces import BenefitGuideRepository

class PgVectorBenefitGuideRepo(BenefitGuideRepository):
    def __init__(self):
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL or Key missing in environment.")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        gcp_project = os.environ.get("GCP_PROJECT_ID")
        gcp_region = os.environ.get("GCP_REGION", "us-central1")
        if not gcp_project:
            raise ValueError("GCP_PROJECT_ID missing in environment.")

        vertexai.init(project=gcp_project, location=gcp_region)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
    def retrieve_guides(self, *, query_text: str, limit: int = 4) -> Sequence[str]:
        inputs = [TextEmbeddingInput(query_text, "RETRIEVAL_QUERY")]
        embeddings = self.embedding_model.get_embeddings(inputs)
        query_vector = embeddings[0].values
        
        try:
            response = self.supabase.rpc(
                'match_benefits',
                {'query_embedding': query_vector, 'match_threshold': 0.1, 'match_count': limit}
            ).execute()
            
            chunks = []
            if response.data:
                for item in response.data:
                    chunks.append(item['content'])
                    
            return chunks
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            return []