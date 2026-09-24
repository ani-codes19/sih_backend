"""
train_model.py
----------------
Trains the XGBoost stress-risk classifier on HR features fused with
VADER sentiment features (matches "Backend API fuses HR data with
survey answers" -> "ML Engine: XGBoost score, SHAP explain, NLP
sentiment" in the architecture diagram).

Output risk is a 3-class label (Low / Medium / High) plus a continuous
0-100 risk score derived from predicted class probabilities, which is
friendlier for dashboards and trend lines than a raw label.

Run directly to train on data/personnel_checkins.csv and write
model/xgb_stress_model.joblib + model/label_encoder.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from sentiment_engine import add_sentiment_features

# Features the model actually sees. Note: no direct identity fields,
# and "latent_stress_true" is deliberately excluded -- that column only
# exists in the synthetic data to generate labels, a real deployment
# would never have it.
FEATURE_COLUMNS = [
    "age",
    "years_of_service",
    "deployment_days_last_90",
    "avg_duty_hours_per_week",
    "consecutive_duty_days",
    "leave_days_taken_last_90",
    "sleep_hours_avg",
    "family_contact_days_last_30",
    "disciplinary_flags_last_180",
    "prior_incident_count",
    "self_report_mood_score",
    "sentiment_compound",
    "sentiment_neg",
    "sentiment_pos",
]

LABEL_COLUMN = "risk_label"


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = add_sentiment_features(df, text_col="journal_text")
    return df


def train(csv_path: Path, model_dir: Path):
    model_dir.mkdir(exist_ok=True)
    df = load_and_prepare(csv_path)

    X = df[FEATURE_COLUMNS]
    y_raw = df[LABEL_COLUMN]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)  # Low/Medium/High -> 0/1/2 (alphabetical by default)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Classes:", list(encoder.classes_))
    print(classification_report(y_test, preds, target_names=encoder.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(model, model_dir / "xgb_stress_model.joblib")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    joblib.dump(FEATURE_COLUMNS, model_dir / "feature_columns.joblib")
    print(f"\nSaved model + encoder + feature list to {model_dir}/")

    return model, encoder


if __name__ == "__main__":
    base = Path(__file__).parent
    train(base / "data" / "personnel_checkins.csv", base / "model")
