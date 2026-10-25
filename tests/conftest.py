"""Shared test fixtures. Tests use small synthetic ads so they run without DVC or MLflow."""

import random

import pandas as pd
import pytest

BRANDS = {
    "Toyota": ["Raize", "Yaris", "Corolla", "Prado"],
    "Honda": ["Vezel", "Fit", "CRV"],
    "Suzuki": ["Wagon R", "Alto", "Wagon R FX"],
}
DISTRICTS = ["Colombo", "Gampaha", "Kandy", "Galle"]


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        brand = rnd.choice(list(BRANDS))
        year = rnd.randint(2005, 2025)
        rows.append(
            {
                "brand": brand,
                "model": rnd.choice(BRANDS[brand]),
                "edition": rnd.choice([None, "G", "Z Premium", "FX"]),
                "fuel_type": rnd.choice(["Petrol", "Hybrid", "Diesel"]),
                "transmission": rnd.choice(["Automatic", "Manual"]),
                "body": rnd.choice(["Hatchback", "SUV / 4x4", "Saloon", None]),
                "condition": rnd.choice(["Used", "Used", "Brand New", "Reconditioned"]),
                "district": rnd.choice(DISTRICTS),
                "seller_type": rnd.choice(["dealer", "private"]),
                "model_year": year,
                "mileage_km": float(rnd.randint(0, 200_000)),
                "engine_cc": float(rnd.choice([660, 1000, 1200, 1500, 2000])),
                "image_count": rnd.randint(1, 12),
                "negotiable": rnd.random() < 0.7,
                "seller_verified": rnd.random() < 0.4,
                "title": f"{brand} car {year}",
                "description": rnd.choice(["First owner", "accident free", "", "agent maintained"]),
                "posted_at": f"2026-10-{1 + i % 24:02d}T10:00:00+05:30",
                "seller_name": f"seller{rnd.randint(1, 30)}",
                "price_lkr": float(rnd.randint(2_000_000, 30_000_000)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def rows() -> pd.DataFrame:
    return make_rows(200)
