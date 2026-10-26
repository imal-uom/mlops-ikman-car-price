"""Request and response models. The request is the API's data contract:
it mirrors the Pandera schema used in training (src/data.py)."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    brand: str = Field(min_length=1, examples=["Honda"])
    model: str = Field(min_length=1, examples=["Vezel"])
    edition: str | None = Field(default=None, examples=["Z"])
    fuel_type: Literal["Petrol", "Diesel", "Hybrid", "Electric"]
    transmission: Literal["Automatic", "Manual", "Tiptronic"]
    body: str | None = Field(default=None, examples=["SUV / 4x4"])
    condition: Literal["Brand New", "Reconditioned", "Used", "Import"]
    district: str = Field(min_length=1, examples=["Colombo"])
    seller_type: Literal["dealer", "private", "member"] = "private"
    model_year: int = Field(ge=1929, le=2027)
    mileage_km: float | None = Field(default=None, ge=0, le=500_000)
    engine_cc: float | None = Field(default=None, ge=500, le=7_000)
    image_count: int = Field(default=5, ge=0, le=50)
    negotiable: bool = True
    seller_verified: bool = False
    title: str = ""
    description: str = ""
    posted_at: str | None = Field(default=None, description="ISO timestamp; defaults to now")


class TermContribution(BaseModel):
    term: str = Field(examples=["model_year"])
    contribution_log: float = Field(description="this term's contribution to log(price)")


class PredictionResponse(BaseModel):
    request_id: str
    predicted_price_lkr: float
    lower_lkr: float
    upper_lkr: float
    model_version: str
    model_support: int = Field(default=0, description="training ads of this car's model")
    top_terms: list[TermContribution] = Field(
        default_factory=list, description="the EBM's largest terms for this prediction"
    )


class FeedbackRequest(BaseModel):
    request_id: str
    actual_price_lkr: float = Field(gt=0, description="the price the car went for")
