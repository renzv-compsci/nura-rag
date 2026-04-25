import os
from dotenv import load_dotenv
import sys
import time
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.infrastracture.retrieval.pgvector_client import PgVectorBenefitGuideRepo
from backend.app.services.translator import (
    translate_rag_response, 
    get_supported_languages, 
    SUPPORTED_LANGUAGES
)

DOTENV_PATH = r"C:\Users\Renz\nura-rag\.env"
load_dotenv(dotenv_path=DOTENV_PATH)

def display_language_menu():
    """Display available languages and get user selection."""
    print("\n🌐 Available Languages:")
    languages = list(SUPPORTED_LANGUAGES.keys())
    for i, lang_code in enumerate(languages, 1):
        lang_name = SUPPORTED_LANGUAGES[lang_code]["name"]
        print(f"  {i}. {lang_name} ({lang_code})")
    
    while True:
        try:
            choice = input("\nSelect language (1-5 or type code like 'fil'): ").strip().lower()
            
            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(languages):
                    return languages[idx]
            # Check if it's a language code
            elif choice in SUPPORTED_LANGUAGES:
                return choice
            
            print("❌ Invalid choice. Please try again.")
        except (ValueError, IndexError):
            print("❌ Invalid choice. Please try again.")


def main():
    print("1. Initializing Retriever...")
    retriever = PgVectorBenefitGuideRepo()

    print("2. Listing available Gemini models...")
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    
    # List all available models
    available_models = []
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            available_models.append(model.name.split("/")[-1])
    
    # Use the first available model
    if available_models:
        selected_model = available_models[0]
        print(f"3. Using model: {selected_model}")
        model = genai.GenerativeModel(selected_model)
    else:
        print("❌ No models available!")
        return

    # Language selection
    print("\n4. Selecting language preference...")
    current_language = display_language_menu()
    lang_name = SUPPORTED_LANGUAGES[current_language]["name"]
    print(f"✓ Language set to: {lang_name}")

    system_prompt = """You are an AI assistant helping a citizen understand their PhilHealth benefits.
Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer."""

    while True:
        # Show current language
        lang_name = SUPPORTED_LANGUAGES[current_language]["name"]
        
        query = input(f"\n[{lang_name}] Enter your question (or type 'quit' to exit, 'lang' to change language): ").strip()
        
        if query.lower() == 'quit' or query.lower() == 'exit':
            print("👋 Thank you for using PhilHealth RAG!")
            break
        
        # Allow language switching mid-session
        if query.lower() == 'lang':
            current_language = display_language_menu()
            lang_name = SUPPORTED_LANGUAGES[current_language]["name"]
            print(f"✓ Language changed to: {lang_name}")
            continue
        
        if not query:
            continue
            
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
            answer = response.text
            
            # Translate if needed
            if current_language != "en":
                print(f"Translating to {lang_name}...")
                answer = translate_rag_response(answer, current_language)
            
            print("=" * 70)
            print("RAG Answer:")
            print("=" * 70)
            print(answer)
            print("=" * 70)
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("⏳ Quota limit hit. Waiting 35 seconds before retrying...")
                time.sleep(35)
                print("Retrying now...")
                try:
                    response = model.generate_content(full_prompt)
                    answer = response.text
                    
                    if current_language != "en":
                        print(f"Translating to {lang_name}...")
                        answer = translate_rag_response(answer, current_language)
                    
                    print("=" * 70)
                    print("RAG Answer:")
                    print("=" * 70)
                    print(answer)
                    print("=" * 70)
                except Exception as retry_error:
                    print(f"❌ Retry failed: {retry_error}")
            else:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()