from typing import Literal

from ..schemas.chat import BenefitProfile, ChatRequest, ChatResponse, Facility


def build_emergency_reply(language: Literal["en", "fil"]) -> str:
    if language == "en":
        return (
            "EMERGENCY ITO. Call 911 now or go to the nearest Emergency Room. "
            "This chatbot is not enough for severe symptoms."
        )

    return (
        "EMERGENCY ITO. Tumawag agad sa 911 o pumunta sa pinakamalapit na Emergency Room. "
        "Hindi sapat ang chatbot para sa ganitong sintomas."
    )


def _build_mock_facility(location: str, benefits: BenefitProfile) -> Facility:
    if benefits.hasPhilcare:
        return Facility(
            id="mock-philcare-001",
            name="Mock PhilCare Partner Clinic",
            address=f"Sample area near {location}",
            distance_km=2.3,
            benefit_to_claim="PhilCare primary consult",
            what_to_say="May PhilCare card po ako, pwede po ba ako magpa-consult?",
            what_to_bring="PhilCare card, valid ID, at anumang previous reseta",
            hours="Mon-Sat 8:00 AM-6:00 PM",
            maps_url="https://maps.google.com/?q=14.5547,121.0244",
            data_source="PHILCARE_2024",
            data_year=2024,
            data_reliability="MEDIUM",
            is_emergency_capable=False,
        )

    if benefits.noBenefits:
        return Facility(
            id="mock-lgu-001",
            name="Mock LGU Health Center",
            address=f"Sample barangay clinic in {location}",
            distance_km=1.4,
            benefit_to_claim="LGU free primary care",
            what_to_say="Wala pa po akong benefit, pwede po ba sa libreng konsultasyon?",
            what_to_bring="Valid ID at proof of address kung meron",
            hours="Mon-Fri 8:00 AM-5:00 PM",
            maps_url="https://maps.google.com/?q=14.5995,120.9842",
            data_source="LGU",
            data_year=2025,
            data_reliability="MEDIUM",
            is_emergency_capable=False,
        )

    return Facility(
        id="mock-yakap-001",
        name="Mock YAKAP Primary Care Clinic",
        address=f"Sample public clinic in {location}",
        distance_km=1.1,
        benefit_to_claim="PhilHealth YAKAP",
        what_to_say="Gusto ko pong magpa-checkup at magpa-empanel sa YAKAP.",
        what_to_bring="PhilHealth ID or MDR, valid ID",
        hours="Mon-Fri 8:00 AM-5:00 PM",
        maps_url="https://maps.google.com/?q=14.6760,121.0437",
        data_source="YAKAP",
        data_year=2025,
        data_reliability="HIGH",
        is_emergency_capable=False,
    )


def build_mock_chat_response(
    payload: ChatRequest,
    response_language: Literal["en", "fil"],
) -> ChatResponse:
    if payload.location is None or not payload.location.strip():
        reply = (
            "Please share your city or barangay so I can suggest nearby facilities."
            if response_language == "en"
            else "Pakibigay ang city o barangay mo para makapagrekomenda ako ng quitmalapit na pasilidad."
        )
        return ChatResponse(
            session_id=payload.session_id,
            state="asking_location",
            reply=reply,
            facilities=[],
            is_emergency=False,
        )
    
    if payload.benefits is None:
        reply = (
            "Please select your available benefits (PhilHealth, Senior, PWD, 4Ps, PhilCare, or none)."
            if response_language == "en"
            else "Paki-select ang benefits mo (PhilHealth, Senior, PWD, 4Ps, PhilCare, o wala)."
        )
        return ChatResponse(
            session_id=payload.session_id,
            state="asking_benefits",
            reply=reply,
            facilities=[],
            is_emergency=False,
        )

    facility = _build_mock_facility(payload.location.strip(), payload.benefits)
    guide_chunks = ["(Mocked PhilHealth Context: YAKAP covers primary consultations)"]
    
    reply = (
        "I am not a doctor, but I can help you navigate where to go next. "
        f"Based on the guidelines: {guide_chunks[0]}. "
        "This is a mocked response for the initial skeleton."
        if response_language == "en"
        else "Hindi ako doktor, pero matutulungan kitang mag-navigate kung saan ka pwedeng pumunta. "
        f"Ayon sa guidelines: {guide_chunks[0]}. "
        "Mocked response ito para sa initial skeleton."
    )

    return ChatResponse(
        session_id=payload.session_id,
        state="results",
        reply=reply,
        facilities=[facility],
        is_emergency=False,
    )


# TODO: Replace mocked response generation with:
# 1) geocoding via Google Maps Geocoding API
# 2) facility search via Supabase Postgres + PostGIS
# 3) benefits retrieval via pgvector RAG
# 4) final response composition via Gemini Flash (after deterministic emergency guard)
