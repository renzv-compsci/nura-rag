import os
from dotenv import load_dotenv
import sys
import time
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.infrastracture.retrieval.pgvector_client import PgVectorBenefitGuideRepo

DOTENV_PATH = r"C:\Users\Renz\nura-rag\.env"
load_dotenv(dotenv_path=DOTENV_PATH)

def main():
    print("1. Initializing Retriever...")
    retriever = PgVectorBenefitGuideRepo()

    print("2. Listing available Gemini models...")
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    # List all available models
    print("\n📋 Available Models:")
    available_models = []
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            available_models.append(model.name.split("/")[-1])
            print(f"  ✓ {model.name.split('/')[-1]}")
    
    print(f"\nTotal: {len(available_models)} models available for generateContent")
    
    # Use the first available model
    if available_models:
        selected_model = available_models[0]
        print(f"\n3. Using model: {selected_model}")
        model = genai.GenerativeModel(selected_model)
    else:
        print("❌ No models available!")
        return

    system_prompt = """You are an AI assistant helping a citizen understand their PhilHealth benefits.
Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer."""

    while True:
        query = input("\nEnter your question (or type 'quit' to exit): ")
        if query.lower() in ['quit', 'exit']:
            break
            
        print("\nSearching for relevant context in Supabase...")
        retrieved_chunks = retriever.retrieve_guides(query_text=query, limit=4)
        
        context_text = "\n\n---\n\n".join(retrieved_chunks)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

        print("Generating answer with Gemini...\n")
        try:
            full_prompt = f"""{system_prompt}

Context:
{context_text}

Question: {query}

Helpful Answer:"""
            
            response = model.generate_content(full_prompt)
            
            print("RAG Answer:")
            print(response.text)
            print("-" * 50)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("⏳ Quota limit hit. Waiting 35 seconds before retrying...")
                time.sleep(35)
                print("Retrying now...")
                response = model.generate_content(full_prompt)
                print("RAG Answer:")
                print(response.text)
                print("-" * 50)
            else:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()