from dataclasses import dataclass
from typing import Protocol, Sequence

from ..schemas.chat import BenefitProfile, Facility


@dataclass(frozen=True, slots=True)
class GeoPoint:
    lat: float
    lng: float


class GeocodingService(Protocol):
    def geocode(self, location_text: str) -> GeoPoint | None:
        ...


class FacilityRepository(Protocol):
    def search_nearby(
        self,
        *,
        center: GeoPoint,
        benefits: BenefitProfile,
        needs_emergency: bool,
        limit: int = 5,
    ) -> Sequence[Facility]:
        ...


class BenefitGuideRepository(Protocol):
    def retrieve_guides(self, *, query_text: str, limit: int = 4) -> Sequence[str]:
        ...


class ResponseComposerService(Protocol):
    def compose_reply(
        self,
        *,
        user_message: str,
        language: str,
        facilities: Sequence[Facility],
        guide_chunks: Sequence[str],
        is_emergency: bool,
    ) -> str:
        ...


# TODO: Implement concrete adapters for Supabase PostGIS, pgvector retrieval, Google geocoding,
# and Gemini response composition. Keep deterministic emergency checks outside these adapters.
