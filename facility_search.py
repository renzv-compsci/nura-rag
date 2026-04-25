"""
Healthcare Facility Finder
───────────────────────────────────
Part of the benefits-RAG pipeline. Accepts a user's self-reported
Region + City and their accreditation type(s), returns ALL accredited
health facilities in that city.

Matching strategy (no GPS required):
  Tier 1 – Exact city match              → return ALL in that city
  Tier 2 – Fuzzy city match (≥ 80 sim.)  → return ALL fuzzy matches (fallback)
  Tier 3 – Same region only              → return ALL in region (last resort)
  Accreditation is a hard pre-filter before any matching.

Setup:
    pip install fastapi uvicorn supabase rapidfuzz python-dotenv

    .env file:
        SUPABASE_URL=https://your-project.supabase.co
        SUPABASE_KEY=your-service-role-or-anon-key

Run:
    uvicorn facility_search:app --reload
"""

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rapidfuzz import fuzz
from supabase import create_client, Client

load_dotenv()

# ── Supabase client ──────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare Facility Finder",
    description="Returns all accredited health facilities in the user's city.",
    version="2.0.0",
)

# ── Constants ────────────────────────────────────────────────────────────────

TABLE           = "health_facilities"
FUZZY_THRESHOLD = 80

# ── Region alias map ─────────────────────────────────────────────────────────

REGION_ALIASES: dict[str, str] = {
    "ncr":                              "National Capital",
    "national capital region":          "National Capital",
    "ncr - national capital region":    "National Capital",
    "metro manila":                     "National Capital",
    "car":                              "Cordillera",
    "cordillera":                       "Cordillera",
    "cordillera administrative region": "Cordillera",
    "car - cordillera administrative region": "Cordillera",
    "region i":                         "Ilocos",
    "region 1":                         "Ilocos",
    "ilocos":                           "Ilocos",
    "ilocos region":                    "Ilocos",
    "region i - ilocos region":         "Ilocos",
    "region ii":                        "Cagayan Valley",
    "region 2":                         "Cagayan Valley",
    "cagayan valley":                   "Cagayan Valley",
    "region ii - cagayan valley":       "Cagayan Valley",
    "region iii":                       "Central Luzon",
    "region 3":                         "Central Luzon",
    "central luzon":                    "Central Luzon",
    "region iii - central luzon":       "Central Luzon",
    "region iv-a":                      "CALABARZON",
    "region 4a":                        "CALABARZON",
    "calabarzon":                       "CALABARZON",
    "region iv-a - calabarzon":         "CALABARZON",
    "region iv-b":                      "MIMAROPA",
    "region 4b":                        "MIMAROPA",
    "mimaropa":                         "MIMAROPA",
    "region iv-b - mimaropa":           "MIMAROPA",
    "region v":                         "Bicol",
    "region 5":                         "Bicol",
    "bicol":                            "Bicol",
    "bicol region":                     "Bicol",
    "region v - bicol region":          "Bicol",
    "region vi":                        "Western Visayas",
    "region 6":                         "Western Visayas",
    "western visayas":                  "Western Visayas",
    "region vi - western visayas":      "Western Visayas",
    "region vii":                       "Central Visayas",
    "region 7":                         "Central Visayas",
    "central visayas":                  "Central Visayas",
    "region vii - central visayas":     "Central Visayas",
    "region viii":                      "Eastern Visayas",
    "region 8":                         "Eastern Visayas",
    "eastern visayas":                  "Eastern Visayas",
    "region viii - eastern visayas":    "Eastern Visayas",
    "region ix":                        "Zamboanga",
    "region 9":                         "Zamboanga",
    "zamboanga":                        "Zamboanga",
    "zamboanga peninsula":              "Zamboanga",
    "region ix - zamboanga peninsula":  "Zamboanga",
    "region x":                         "Northern Mindanao",
    "region 10":                        "Northern Mindanao",
    "northern mindanao":                "Northern Mindanao",
    "region x - northern mindanao":     "Northern Mindanao",
    "region xi":                        "Davao",
    "region 11":                        "Davao",
    "davao":                            "Davao",
    "davao region":                     "Davao",
    "region xi - davao region":         "Davao",
    "region xii":                       "SOCCSKSARGEN",
    "region 12":                        "SOCCSKSARGEN",
    "soccsksargen":                     "SOCCSKSARGEN",
    "region xii - soccsksargen":        "SOCCSKSARGEN",
    "region xiii":                      "Caraga",
    "region 13":                        "Caraga",
    "caraga":                           "Caraga",
    "region xiii - caraga":             "Caraga",
    "barmm":                            "Bangsamoro",
    "bangsamoro":                       "Bangsamoro",
    "armm":                             "Bangsamoro",
    "barmm - bangsamoro autonomous region in muslim mindanao": "Bangsamoro",
}


def _resolve_region(user_input: str) -> str:
    key = user_input.strip().lower()
    return REGION_ALIASES.get(key, user_input.strip())


# ── Schemas ──────────────────────────────────────────────────────────────────

class FacilityResult(BaseModel):
    name_of_health_facility: str
    street:                  Optional[str]
    municipality_city:       Optional[str]
    region:                  Optional[str]
    is_philhealth:           bool
    is_malasakit:            bool
    expire_date:             Optional[str]
    source:                  Optional[str]
    match_tier:              str   # "exact_city" | "fuzzy_city" | "region"


class FacilitiesResponse(BaseModel):
    query_region:    str
    query_city:      str
    is_philhealth:   bool
    is_malasakit:    bool
    match_tier_used: str   # which tier produced the results
    total_results:   int
    facilities:      list[FacilityResult]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.strip().lower() if text else ""


def _fetch_candidates(region: str, is_philhealth: bool, is_malasakit: bool) -> list[dict]:
    search_keyword = _resolve_region(region)

    query = (
        supabase.table(TABLE)
        .select(
            "name_of_health_facility, street, municipality_city, "
            "region, is_philhealth, is_malasakit, expire_date, source"
        )
        .ilike("region", f"%{search_keyword}%")
        .limit(4000)
    )

    if is_philhealth and is_malasakit:
        query = query.eq("is_philhealth", True).eq("is_malasakit", True)
    elif is_philhealth:
        query = query.eq("is_philhealth", True)
    elif is_malasakit:
        query = query.eq("is_malasakit", True)

    return query.execute().data or []


def _to_result(fac: dict, tier: str) -> FacilityResult:
    return FacilityResult(
        name_of_health_facility=fac.get("name_of_health_facility", ""),
        street=fac.get("street"),
        municipality_city=fac.get("municipality_city"),
        region=fac.get("region"),
        is_philhealth=fac.get("is_philhealth", False),
        is_malasakit=fac.get("is_malasakit", False),
        expire_date=fac.get("expire_date"),
        source=fac.get("source"),
        match_tier=tier,
    )


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.get(
    "/facilities/in-city",
    response_model=FacilitiesResponse,
    summary="All accredited health facilities in the user's city",
    tags=["Facilities"],
)
def get_facilities_in_city(
    region: str = Query(
        ...,
        description="User's region — abbreviation, full name, or numbered form. e.g. 'NCR', 'Region I', 'Davao'",
        example="NCR",
    ),
    city: str = Query(
        ...,
        description="User's city or municipality.",
        example="Taguig",
    ),
    is_philhealth: bool = Query(True,  description="Filter for PhilHealth-accredited facilities."),
    is_malasakit:  bool = Query(False, description="Filter for Malasakit-accredited facilities."),
):
    if not region.strip() or not city.strip():
        raise HTTPException(status_code=422, detail="Both 'region' and 'city' must be non-empty.")

    if not is_philhealth and not is_malasakit:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'is_philhealth' or 'is_malasakit' must be true.",
        )

    candidates = _fetch_candidates(region, is_philhealth, is_malasakit)

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No accredited facilities found for region '{region}' "
                f"(search keyword: '{_resolve_region(region)}'). "
                "Check RLS policies or re-upload the CSV."
            ),
        )

    city_norm = _normalize(city)

    # ── Tier 1: exact city match — return ALL ────────────────────────────────
    exact = [
        fac for fac in candidates
        if _normalize(fac.get("municipality_city") or "") == city_norm
    ]
    if exact:
        return FacilitiesResponse(
            query_region=region,
            query_city=city,
            is_philhealth=is_philhealth,
            is_malasakit=is_malasakit,
            match_tier_used="exact_city",
            total_results=len(exact),
            facilities=[_to_result(f, "exact_city") for f in exact],
        )

    # ── Tier 2: fuzzy city match — return ALL above threshold ────────────────
    fuzzy = [
        fac for fac in candidates
        if fuzz.token_sort_ratio(
            _normalize(fac.get("municipality_city") or ""), city_norm
        ) >= FUZZY_THRESHOLD
    ]
    if fuzzy:
        return FacilitiesResponse(
            query_region=region,
            query_city=city,
            is_philhealth=is_philhealth,
            is_malasakit=is_malasakit,
            match_tier_used="fuzzy_city",
            total_results=len(fuzzy),
            facilities=[_to_result(f, "fuzzy_city") for f in fuzzy],
        )

    # ── Tier 3: whole region fallback — return ALL in region ─────────────────
    return FacilitiesResponse(
        query_region=region,
        query_city=city,
        is_philhealth=is_philhealth,
        is_malasakit=is_malasakit,
        match_tier_used="region",
        total_results=len(candidates),
        facilities=[_to_result(f, "region") for f in candidates],
    )


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


# ── Debug (remove in production) ─────────────────────────────────────────────

@app.get("/debug/regions", tags=["Debug"])
def debug_regions():
    rows = supabase.table("health_facilities").select("region").execute().data or []
    distinct = sorted({r.get("region") or "NULL/EMPTY" for r in rows})
    return {"total_rows_checked": len(rows), "distinct_region_values": distinct, "count": len(distinct)}


@app.get("/debug/sample", tags=["Debug"])
def debug_sample(limit: int = 10):
    rows = (
        supabase.table("health_facilities")
        .select("name_of_health_facility, municipality_city, region, is_philhealth, is_malasakit")
        .limit(limit)
        .execute()
        .data or []
    )
    return {"rows_returned": len(rows), "sample": rows}