"""
api.py
-------
FastAPI backend exposing the ML Engine as HTTP endpoints, matching the
"Backend API (FastAPI/Python)" layer in the architecture diagram.

Run with:
    uvicorn api:app --reload --port 8000

Then POST to /assess with a JSON body like the example in
risk_pipeline.py's __main__ block. Interactive docs at /docs.

NOTE on privacy: this endpoint returns an individual-level result for
the personnel member's own transparency screen. A separate,
k-anonymized aggregation endpoint (not included here) is what should
feed the Commander Dashboard -- individual risk results should never
be exposed to a commander directly, per the privacy-first design.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from risk_pipeline import RakshaMitraRiskPipeline

app = FastAPI(title="RakshaMitra ML Engine", version="0.1.0")

# The TanStack Start server function calls this API server-to-server, which
# doesn't need CORS at all. This is here for local dev convenience (e.g.
# hitting /docs or testing directly from a browser on another port) and is
# intentionally permissive for a hackathon build -- lock this down to your
# actual frontend origin before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RakshaMitraRiskPipeline()


class CheckinRequest(BaseModel):
    service_id: str = Field(..., description="Used for routing/audit only -- never passed to the model")
    age: int
    years_of_service: int
    deployment_days_last_90: int
    avg_duty_hours_per_week: float
    consecutive_duty_days: int
    leave_days_taken_last_90: int
    sleep_hours_avg: float
    family_contact_days_last_30: int
    disciplinary_flags_last_180: int
    prior_incident_count: int
    self_report_mood_score: int
    journal_text: Optional[str] = ""
    top_k: int = Field(6, ge=1, le=14, description="Number of SHAP-ranked drivers to return")


class TopFactor(BaseModel):
    feature: str
    value: float
    impact: float
    direction: str


class JournalSentiment(BaseModel):
    sentiment_compound: float
    sentiment_neg: float
    sentiment_neu: float
    sentiment_pos: float


class AssessResponse(BaseModel):
    predicted_risk_label: str
    risk_score_0_100: float
    class_probabilities: dict
    top_factors: list[TopFactor]
    journal_sentiment: JournalSentiment
    recommendation: Optional[str]
    requires_human_review: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess", response_model=AssessResponse)
def assess(req: CheckinRequest):
    try:
        payload = req.model_dump()
        payload.pop("service_id")  # deliberately excluded from the model input
        top_k = payload.pop("top_k")
        result = pipeline.assess(payload, top_k=top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
