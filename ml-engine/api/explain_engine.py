"""
explain_engine.py
-------------------
SHAP layer of the ML Engine. Turns raw SHAP values into the
human-readable "why did this score change" explanation that the pitch
deck promises personnel and commanders (Explainable Risk Scores).

Uses shap.TreeExplainer, which is fast and exact for tree-based models
like XGBoost -- no sampling/approximation needed.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# Friendly, non-clinical phrasing for each raw feature -- shown to the
# user instead of the internal column name.
FEATURE_LABELS = {
    "age": "Age",
    "years_of_service": "Years of service",
    "deployment_days_last_90": "Deployment days (last 90 days)",
    "avg_duty_hours_per_week": "Average duty hours per week",
    "consecutive_duty_days": "Consecutive duty days",
    "leave_days_taken_last_90": "Leave days taken (last 90 days)",
    "sleep_hours_avg": "Average sleep hours",
    "family_contact_days_last_30": "Days of family contact (last 30 days)",
    "disciplinary_flags_last_180": "Disciplinary flags (last 180 days)",
    "prior_incident_count": "Prior incident count",
    "self_report_mood_score": "Self-reported mood score",
    "sentiment_compound": "Overall tone of recent check-in notes",
    "sentiment_neg": "Negative language in recent check-in notes",
    "sentiment_pos": "Positive language in recent check-in notes",
}


class RiskExplainer:
    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "xgb_stress_model.joblib")
        self.encoder = joblib.load(model_dir / "label_encoder.joblib")
        self.feature_columns = joblib.load(model_dir / "feature_columns.joblib")
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X_row: pd.DataFrame, top_k: int = 4) -> dict:
        """
        X_row: single-row DataFrame with self.feature_columns.
        Returns predicted class, a 0-100 risk score, and the top_k
        features driving THIS prediction toward/away from higher risk.
        """
        X_row = X_row[self.feature_columns]

        proba = self.model.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.encoder.inverse_transform([pred_idx])[0]

        # 0-100 risk score: probability mass on Medium+High, weighted
        # so High counts more than Medium.
        classes = list(self.encoder.classes_)
        risk_weight = {"Low": 0.0, "Medium": 0.55, "High": 1.0}
        risk_score = float(sum(proba[i] * risk_weight[classes[i]] for i in range(len(classes))) * 100)

        # SHAP values: one array per class for a multiclass XGBoost model.
        shap_values = self.explainer.shap_values(X_row)
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_idx][0]
        else:
            # newer shap versions can return a single (1, n_features, n_classes) array
            class_shap = shap_values[0, :, pred_idx]

        contributions = list(zip(self.feature_columns, class_shap, X_row.iloc[0].values))
        # Sort by absolute impact on the predicted class
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        top_factors = []
        for feat, val, raw_value in contributions[:top_k]:
            direction = "increased" if val > 0 else "decreased"
            top_factors.append({
                "feature": FEATURE_LABELS.get(feat, feat),
                "value": round(float(raw_value), 2) if isinstance(raw_value, (int, float)) else raw_value,
                "impact": round(float(val), 4),
                "direction": direction,
            })

        return {
            "predicted_risk_label": pred_label,
            "risk_score_0_100": round(risk_score, 1),
            "class_probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
            "top_factors": top_factors,
        }

    @staticmethod
    def to_plain_english(explanation: dict) -> str:
        """Renders the transparency-screen sentence personnel actually see."""
        label = explanation["predicted_risk_label"]
        score = explanation["risk_score_0_100"]
        lines = [f"Current risk level: {label} (score {score}/100). Main contributing factors:"]
        for f in explanation["top_factors"]:
            lines.append(f"  - {f['feature']} ({f['value']}) {f['direction']} the risk score.")
        return "\n".join(lines)


if __name__ == "__main__":
    base = Path(__file__).parent
    explainer = RiskExplainer(base / "model")

    sample = pd.DataFrame([{
        "age": 34, "years_of_service": 12, "deployment_days_last_90": 75,
        "avg_duty_hours_per_week": 78, "consecutive_duty_days": 22,
        "leave_days_taken_last_90": 1, "sleep_hours_avg": 4.2,
        "family_contact_days_last_30": 2, "disciplinary_flags_last_180": 1,
        "prior_incident_count": 0, "self_report_mood_score": 3,
        "sentiment_compound": -0.55, "sentiment_neg": 0.35, "sentiment_pos": 0.0,
    }])

    result = explainer.explain(sample)
    print(explainer.to_plain_english(result))
    print("\nFull result:", result)
