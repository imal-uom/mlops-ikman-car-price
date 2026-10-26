"""Prediction service. Every prediction and every piece of feedback is printed as one JSON
line; Azure Container Apps ships these lines to Log Analytics, where monitoring reads them."""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Send request metrics to Application Insights when running in Azure.
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor()

from fastapi import FastAPI  # noqa: E402

from app.schemas import (  # noqa: E402
    FeedbackRequest,
    PredictionRequest,
    PredictionResponse,
    TermContribution,
)
from src.features import prepare_inputs  # noqa: E402

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
TOP_TERMS = 5
STATE: dict = {}

# A dedicated logger writes one complete JSON line per event. The handler's lock stops
# lines from different request threads getting mixed together.
events = logging.getLogger("events")
events.setLevel(logging.INFO)
events.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
events.addHandler(_handler)


def log_event(event: dict) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    events.info(json.dumps(event))


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["model"] = joblib.load(MODELS_DIR / "ebm.joblib")
    STATE["spec"] = json.loads((MODELS_DIR / "spec.json").read_text())
    STATE["q_log"] = json.loads((MODELS_DIR / "conformal.json").read_text())["q_log"]
    STATE["terms"] = list(STATE["model"].term_names_)
    yield
    STATE.clear()


app = FastAPI(title="ikman.lk car price API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/version")
def version() -> dict:
    return {"model_version": MODEL_VERSION}


def top_terms(X: pd.DataFrame) -> list[TermContribution]:
    """The EBM's own decomposition of this prediction: the largest terms, exactly."""
    scores = STATE["model"].eval_terms(X)[0]
    order = np.argsort(-np.abs(scores))[:TOP_TERMS]
    return [
        TermContribution(term=STATE["terms"][i], contribution_log=round(float(scores[i]), 4))
        for i in order
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    start = time.perf_counter()
    features = req.model_dump()
    X = prepare_inputs(pd.DataFrame([features]), STATE["spec"])
    p = float(STATE["model"].predict(X)[0])
    q = STATE["q_log"]
    response = PredictionResponse(
        request_id=str(uuid.uuid4()),
        predicted_price_lkr=round(float(np.exp(p)), -3),
        lower_lkr=round(float(np.exp(p - q)), -3),
        upper_lkr=round(float(np.exp(p + q)), -3),
        model_version=MODEL_VERSION,
        model_support=int(X["model_support"].iloc[0]),
        top_terms=top_terms(X),
    )
    log_event(
        {
            "event": "prediction",
            **response.model_dump(),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "features": features,
        }
    )
    return response


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    log_event({"event": "feedback", **req.model_dump()})
    return {"status": "recorded"}
