# Nura RAG - Healthcare Assistant API

## Overview
**Nura RAG** is a highly capable FastAPI-based Retrieval-Augmented Generation (RAG) system designed to help users find relevant healthcare benefits and facilities in the Philippines (e.g., PhilHealth, Malasakit). 
The system is built on **Google Generative AI** (Gemini 2.5 Flash) and **Vertex AI** for embeddings, storing knowledge inside a **Supabase (pgvector)** database for semantic search. The LLM has strict medical guardrails implemented to ensure it acts as a healthcare guide bridging patients to clinics without ever diagnosing or providing direct medical advice. Additionally, the system features a multi-language translation pipeline via `deep_translator`.

---

## Tech Stack
* **Web Framework**: FastAPI (`0.115.0`), Uvicorn (`0.30.6`), Pydantic (`2.9.2`)
* **AI & Embeddings**: `google-generativeai` (Gemini), Vertex AI Embeddings
* **Database & Vector Search**: Supabase (PostgreSQL with `pgvector`)
* **Document Processing**: `PyMuPDF` (PDF extraction), `langchain-text-splitters` (text chunking)
* **Translation**: `deep_translator` (`GoogleTranslator`)
* **Environment Configuration**: `python-dotenv`

---

## Project Structure & File Contributions

The codebase is organized under `backend/app/` following a modular architecture:

### 🗂️ `backend/app/`
The root application directory housing all backend logic. 
* `main.py`: The core FastAPI application that orchestrates routes (e.g., `/search`, `/health`), sets up strict medical guardrails via prompt engineering, combines RAG outputs with facility retrieval, and handles translations.

### 🗂️ `backend/app/data/`
* `guidelines/`: Contains source PDFs and static knowledge documents (e.g., PhilHealth guidelines) used for text extraction, chunking, and embedding.

### 🗂️ `backend/app/infrastructure/`
* `retrieval/pgvector_client.py`: The database client connecting to Supabase. Handles connecting to `pgvector` to query embedded knowledge and retrieve the most semantically relevant text chunks via an RPC call.

### 🗂️ `backend/app/schemas/`
* `chat.py` (and search schemas): Pydantic models defining request/response API contracts. Ensures all incoming user queries and outgoing payloads adhere strictly to expected data types and structures.

### 🗂️ `backend/app/scripts/`
* `ingest_philhealth.py`: A standalone operational script. Uses `PyMuPDF` to parse PDF files, `langchain-text-splitters` to segment them into logical chunks, generates embeddings with Vertex AI, and ingests them into the Supabase vector store (`benefit_guides` table).
* `test_rag.py`: Development CLI script used to manually test the complete RAG loop, LLM generation, and Deep Translator routing without spinning up the entire FastAPI server.

### 🗂️ `backend/app/services/`
* `chat_service.py`: Contains the primary business logic interacting with Google Generative AI (Gemini). Handles prompt formatting, context injection, and ensuring the LLM adheres to the medical guardrails.
* `translator.py`: Wraps `deep_translator` (`GoogleTranslator`) to translate the final context-aware LLM response into regional dialects like Filipino (`fil`), Cebuano (`ceb`), Hiligaynon (`hil`), and Ilocano (`ilo`).
* `interfaces.py`: Defines the abstract base classes or protocols to keep services decoupled, modular, and easily testable.

---

## API Contract

The system exposes a primary search endpoint that combines semantic search (RAG) with geospatial facility lookup, wrapped with LLM generation and translation.

### POST `/search`

**Example Request:**
```json
{
  "query": "Meron akong ubo at sipon, 3 araw na itong nangyayari. Saan ako puwedeng pumunta",
  "region": "NCR",
  "city": "Taguig",
  "is_philhealth": true,
  "is_malasakit": false,
  "language": "fil"
}
```

**Example Response:**
```json
{
  "status": "success",
  "query": "Meron akong ubo at sipon, 3 araw na itong nangyayari. Saan ako puwedeng pumunta",
  "language_requested": "Filipino",
  "facilities": [
    {
      "name_of_health_facility": "AC CARE VENTURE CO.",
      "street": "UNIT 177B MANUEL L QUEZON ST PUROK\n3",
      "municipality_city": "Taguig",
      "region": "NCR - National Capital Region",
      "is_philhealth": true,
      "is_malasakit": false,
      "expire_date": "2027-12-31",
      "source": "YAKAP.pdf",
      "match_tier": "exact_city"
    },
    {
      "name_of_health_facility": "ARMY GENERAL HOSPITAL",
      "street": "FORT ANDRES BONIFACIO",
      "municipality_city": "Taguig",
      ...
    }
  ],
  "llm_response": "Kumusta ka, naiintindihan ko na nakakabahala ang pagkakaroon ng ubo at sipon sa loob ng tatlong araw. Mahalagang masuri ito ng isang healthcare professional para sa tamang pagtatasa at payo.\n\nHuwag kang mag-alala, ang pagkonsulta sa doktor ay ang pinakamainam na hakbang para masiguro ang iyong kalusugan. Magandang balita sa PhilHealth, maaari mong gamitin ang iyong mga miyembro sa mga accredited facility upang makatulong sa gastusin.\n\nPara sa iyong lokasyon sa Taguig, narito ang ilang PhilHealth accredited facilities na maaari mong puntahan para sa wastong assessment at pangangalaga:\n\n* **AC CARE VENTURE CO.** (UNIT 177B MANUEL L QUEZON ST PUROK 3, Taguig)\n* **ARMY GENERAL HOSPITAL** (FORT ANDRES BONIFACIO, Taguig)\n* **CITY GOVERNMENT OF TAGUIG / BAMBANG HEALTH CENTER** (#20 KENTUCKY ST., BAMBANG,, Taguig)\n* **CITY GOVERNMENT OF TAGUIG / CALZADA HEALTH CENTER** (1 RUHALE ST., CALZADA, TIPAS,, Taguig)\n* **CITY GOVERNMENT OF TAGUIG / CENTRAL BICUTAN HEALTH CENTER** (PUROK 1 CENTRAL BICUTAN,, Taguig)\n\nPumunta ka lang sa malapit sa iyo para matulungan ka.",
  "original_response": "Kumusta ka, naiintindihan ko na nakakabahala ang pagkakaroon ng ubo at sipon sa loob ng tatlong araw. Mahalagang masuri ito ng isang healthcare professional para sa tamang pagtatasa at payo..."
}
```

---

## Database Schemas (Supabase)

This project relies on the following database entities to manage RAG memory, healthcare facilities, and spatial mappings:

### `benefit_guides` (Vector Store for RAG)
| Name | Type | Key/Constraints | Description |
|------|------|-----------------|-------------|
| `id` | `uuid` | Primary Key | Unique document chunk ID |
| `content` | `text` | Nullable | Document chunk text block |
| `embedding`| `vector` | Nullable | Vertex AI embedding vector used for RAG retrieval |
| `source` | `text` | Nullable | Source document name (e.g. `Philhealth-Guide.pdf`) |

### `health_facilities`
| Name | Type | Key/Constraints | Description |
|------|------|-----------------|-------------|
| `id` | `int8` | Primary Identity| Unique internal facility ID |
| `name_of_health_facility` | `text` | | Facility name |
| `street` | `text` | Nullable | Street address |
| `municipality_city` | `text` | Nullable | City matching query |
| `region` | `text` | Nullable | Region of operation |
| `is_philhealth` | `bool` | | boolean toggle for PhilHealth accreditation |
| `is_malasakit` | `bool` | | boolean toggle for Malasakit Center presence|
| `expire_date` | `date` | Nullable | Accreditation expiry date |
| `source` | `text` | Nullable | Document data origin |
| `created_at` | `timestamptz` | | Metadata timestamp |
| `updated_at` | `timestamptz` | | Metadata timestamp |

### `facilities` (Expanded Facility Metadata)
| Name | Type | Key/Constraints | Description |
|------|------|-----------------|-------------|
| `id` | `uuid` | Primary Key | Main reference UUID (if unassociated with `health_facilities`) |
| `name` | `text` | | Facility identifier |
| `facility_type`| `text` | | Type of healthcare venue |
| `source_dataset`| `text` | | Datastore origin |
| `address_full`, `city`, `barangay` | `text` | Nullable | Geospatial address info |
| `latitude`, `longitude` | `float8` | Nullable | Map coordinate plotting |
| `geom` | `geography` | Nullable | PostGIS geographical formatting mapping |
| `accepts_yakap`, `has_malasakit`, `accepts_philcare`, `is_lgu_free`, `accepts_senior_discount`, `accepts_pwd_discount`, `supports_4ps`, `has_er` | `bool` | Nullable | Toggles for available social service implementations |
| `opening_hours`, `phone`, `notes`, `maps_url` | `text` | Nullable | Misc directory attributes |

### `sessions` (Chat History)
| Name | Type | Key/Constraints | Description |
|------|------|-----------------|-------------|
| `id` | `uuid` | Primary Key | Session identifier |
| `language` | `text` | Nullable | The selected response dialect |
| `location_city`, `location_raw` | `text` | Nullable | User's targeted demographic area |
| `benefits` | `_text` | | Extracted RAG benefits payload array |
| `conversation_history`| `jsonb` | | Complete RAG discussion trace |

### `spatial_ref_sys` (System Internal)
| Name | Type | Key/Constraints | Description |
|------|------|-----------------|-------------|
| `srid`, `auth_srid` | `int4` | Primary Key | Spatial formatting keys (PostGIS) |
| `auth_name`, `srtext`, `proj4text` | `varchar`| Nullable | Mapping boundaries configuration |
