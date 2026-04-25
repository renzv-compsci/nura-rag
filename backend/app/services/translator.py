from deep_translator import GoogleTranslator
from typing import Optional

SUPPORTED_LANGUAGES = {
    "en":  {"name": "English",    "google_code": None},   
    "fil": {"name": "Filipino",   "google_code": "tl"},   
    "ceb": {"name": "Cebuano",    "google_code": "ceb"},
    "hil": {"name": "Hiligaynon", "google_code": "ceb"},  
    "ilo": {"name": "Ilocano",    "google_code": "ilo"},
}

def translate_text(text: str, google_code: str) -> str:
    """Translate a single text string to the target language."""
    if not text or not google_code:
        return text

    try:
        result = GoogleTranslator(source="en", target=google_code).translate(text)
        return result if result else text
    except Exception as e:
        print(f"⚠️  Translation error (target={google_code}): {e}")
        return text


def translate_rag_response(response_text: str, language: str) -> str:
    """Translate the RAG response to the user's preferred language."""
    if language == "en":
        return response_text

    config = SUPPORTED_LANGUAGES.get(language)
    if not config or not config["google_code"]:
        print(f"⚠️  Language '{language}' not supported. Returning English response.")
        return response_text

    google_code = config["google_code"]
    translated = translate_text(response_text, google_code)
    return translated

def get_supported_languages() -> dict:
    """Return list of supported languages."""
    return SUPPORTED_LANGUAGES

def get_language_code(language_name: str) -> Optional[str]:
    """Get the language code from the language name."""
    for code, config in SUPPORTED_LANGUAGES.items():
        if config["name"].lower() == language_name.lower():
            return code
    return None