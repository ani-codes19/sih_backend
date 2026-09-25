"""
risk_pipeline.py
------------------
The single entry point the FastAPI backend should call. Wraps the three
ML Engine components from the architecture diagram into one function:

    raw check-in (HR fields + journal text)
        -> NLP Sentiment (VADER)
        -> XGBoost score
        -> SHAP explain
        -> risk result dict

Also applies simple, transparent intervention-logic thresholds (matches
"Recommendation Engine: rotation, leave escalation, shift rebalance" and
"AI never triggers action directly - every escalation requires human
review").
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from explain_engine import RiskExplainer
from sentiment_engine import score_text

MODEL_DIR = Path(__file__).parent / "model"


class RakshaMitraRiskPipeline:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.explainer = RiskExplainer(model_dir)

    def assess(self, checkin: dict, top_k: int = 6) -> dict:
        """
        checkin keys expected (HR/self-report fields, no direct identity
        needed here -- service_id is handled separately by the caller
        for routing/auditing, never passed into the model itself):

            age, years_of_service, deployment_days_last_90,
            avg_duty_hours_per_week, consecutive_duty_days,
            leave_days_taken_last_90, sleep_hours_avg,
            family_contact_days_last_30, disciplinary_flags_last_180,
            prior_incident_count, self_report_mood_score,
            journal_text (free text from the daily conversational check-in)

        top_k: how many SHAP-ranked drivers to return (frontend renders
        these as a bar list, so 6 lines up with the original mock's
        six-driver layout).
        """
        journal_text = checkin.get("journal_text", "")
        sentiment = score_text(journal_text)

        row = {**checkin, **sentiment}
        row.pop("journal_text", None)
        X_row = pd.DataFrame([row])

        explanation = self.explainer.explain(X_row, top_k=top_k)
        explanation["journal_sentiment"] = sentiment
        explanation["recommendation"] = self._recommend(explanation)
        explanation["requires_human_review"] = explanation["predicted_risk_label"] in ("Medium", "High")
        return explanation

    @staticmethod
    def _recommend(explanation: dict) -> Optional[str]:
        """
        Rule-based recommendation layer (Python rules engine, per tech
        stack) -- deliberately simple and deliberately NOT auto-executed.
        Surfaced to a welfare officer for human decision, never applied
        automatically.
        """
        label = explanation["predicted_risk_label"]
        score = explanation["risk_score_0_100"]
        if label == "High":
            return "Flag for welfare officer review within 24h; consider duty rotation or short leave."
        if label == "Medium" and score > 55:
            return "Monitor closely; suggest workload/shift rebalance if trend continues over next check-ins."
        return None


if __name__ == "__main__":
    pipeline = RakshaMitraRiskPipeline()

    example_checkin = {
        "age": 29, "years_of_service": 6, "deployment_days_last_90": 60,
        "avg_duty_hours_per_week": 70, "consecutive_duty_days": 18,
        "leave_days_taken_last_90": 2, "sleep_hours_avg": 5.0,
        "family_contact_days_last_30": 3, "disciplinary_flags_last_180": 0,
        "prior_incident_count": 0, "self_report_mood_score": 4,
        "journal_text": "Struggling to cope, everything feels overwhelming lately.",
    }

    result = pipeline.assess(example_checkin)
    import json
    print(json.dumps(result, indent=2))
