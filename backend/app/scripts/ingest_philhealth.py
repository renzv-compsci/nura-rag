import os
import fitz # PyMuPDF
from supabase import create_client, Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
import vertexai
import time 
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from dotenv import load_dotenv


DOTENV_PATH = r"C:\Users\Renz\nura-rag\.env"
PDF_PATH = r"C:\Users\Renz\Nura\data\guidelines\philhealth_benefits.pdf"
load_dotenv(dotenv_path=DOTENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GCP_PROJECT = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

def main():
    print("1. Extracting text from PDF...")
    doc = fitz.open(PDF_PATH)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
        
    print(f"Extracted {len(full_text)} characters.")

    print("2. Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Created {len(chunks)} chunks.")

    print("3. Initializing Vertex AI & Supabase...")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("4. Generating Embeddings & Uploading to Supabase in Batches...")
    batch_size = 10 
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                inputs = [TextEmbeddingInput(chunk, task_type="RETRIEVAL_DOCUMENT") for chunk in batch_chunks]
                embeddings = model.get_embeddings(inputs)
                
                for j, embedding_obj in enumerate(embeddings):
                    data = {
                        "content": batch_chunks[j],
                        "embedding": embedding_obj.values,
                        "source": "PhilHealth Benefits Guidelines 2026"
                    }
                    supabase.table("benefit_guides").insert(data).execute()
                    
                print(f"Successfully processed chunks {i + 1} to {i + len(batch_chunks)} of {len(chunks)}")
                
                time.sleep(5) 
                break
                
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    print(f"\n[!] Hit quota limit on chunk {i + 1}. Waiting 65 seconds for quota to reset (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(65) 
                else:
                    print(f"Unknown error processing batch starting at chunk {i + 1}: {e}")
                    raise e 
            
    print("Ingestion complete!")

if __name__ == "__main__":
    main()