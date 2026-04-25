from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FacilitySource = Literal["YAKAP", "MALASAKIT", "LGU", "OSM", "GOOGLE_MAPS", "PHILCARE_2024"]
DataReliability = Literal["HIGH", "MEDIUM", "LOW"]
ChatState = Literal[
    "idle",
    "asking_symptom",
    "asking_location",
    "asking_benefits",
    "loading",
    "results",
    "emergency",
    "error",
]


class BenefitProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hasYakap: bool = False
    hasPhilHealth: bool = False
    isSenior: bool = False
    isPwd: bool = False
    is4ps: bool = False
    hasPhilcare: bool = False
    noBenefits: bool = False


class Facility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str
    address: str
    distance_km: float | None = None
    benefit_to_claim: str | None = None
    what_to_say: str | None = None
    what_to_bring: str | None = None
    hours: str | None = None
    maps_url: str | None = None
    data_source: FacilitySource
    data_year: int | None = None
    data_reliability: DataReliability | None = None
    is_emergency_capable: bool | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)
    location: str | None = Field(default=None, max_length=120)
    benefits: BenefitProfile | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    state: ChatState
    reply: str
    facilities: list[Facility] = Field(default_factory=list)
    is_emergency: bool
