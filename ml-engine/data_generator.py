"""
data_generator.py
------------------
Generates a synthetic personnel HR + daily-check-in dataset for RakshaMitra.

Why synthetic: per the pitch deck, early development shouldn't depend on
real HRMS integration. This script produces a plausible, internally
consistent dataset (feature correlations mimic real operational-stress
drivers) so the ML pipeline can be built and demoed end-to-end.

Run directly to write data/personnel_checkins.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N_PERSONNEL = 600          # distinct service members
CHECKINS_PER_PERSON = 6    # rows per person (check-ins over time)

# Journal snippet banks, roughly bucketed by underlying mood so VADER
# sentiment correlates with the "true" latent stress state we simulate.
POSITIVE_SNIPPETS = [
    "Good day today, training went well and I got some rest.",
    "Feeling steady. Caught up with family on call, that helped a lot.",
    "Solid week, workload was manageable and the team is in good spirits.",
    "Slept well last night, feeling fresh for duty.",
    "Had a great time during the sports session, morale is high.",
]
NEUTRAL_SNIPPETS = [
    "Routine day, nothing much to report.",
    "Usual duty hours, a bit tired but okay overall.",
    "Standard patrol schedule, no major issues.",
    "Been a normal week, keeping busy with regular tasks.",
    "Nothing unusual, just getting through the schedule.",
]
NEGATIVE_SNIPPETS = [
    "Extremely exhausted, haven't slept properly in days.",
    "Feeling isolated, it's been weeks since I saw my family.",
    "The back-to-back duty shifts are wearing me down badly.",
    "Struggling to cope, everything feels overwhelming lately.",
    "Really frustrated and anxious about the extended deployment.",
]


def _sample_snippet(latent_stress: float) -> str:
    """Pick a journal snippet whose sentiment loosely tracks latent stress.

    Deliberately noisy: a person's free-text mood on a given day often
    doesn't line up with their underlying HR-driven stress load (they
    vent on an easy day, or stay upbeat through a hard one). If the text
    bucket were a deterministic function of the same thresholds used for
    the label, the model could get near-perfect accuracy from sentiment
    alone and would learn to ignore the HR features -- unrealistic, and
    not what we want the model to learn.
    """
    if latent_stress < 0.35:
        true_bank = POSITIVE_SNIPPETS
    elif latent_stress < 0.65:
        true_bank = NEUTRAL_SNIPPETS
    else:
        true_bank = NEGATIVE_SNIPPETS

    roll = RNG.random()
    if roll < 0.55:
        bank = true_bank
    elif roll < 0.80:
        bank = NEUTRAL_SNIPPETS  # muted / doesn't fully vent either way
    else:
        bank = RNG.choice([POSITIVE_SNIPPETS, NEUTRAL_SNIPPETS, NEGATIVE_SNIPPETS])  # fully mismatched
    return RNG.choice(bank)


def generate_dataset(n_personnel: int = N_PERSONNEL,
                      checkins_per_person: int = CHECKINS_PER_PERSON) -> pd.DataFrame:
    rows = []
    for pid in range(1, n_personnel + 1):
        service_id = f"SVC{pid:05d}"
        age = int(RNG.integers(20, 55))
        years_of_service = int(min(age - 18, RNG.integers(1, 32)))
        base_rank_load = RNG.uniform(0.2, 0.8)  # a stable per-person load bias

        for _ in range(checkins_per_person):
            deployment_days_last_90 = int(RNG.integers(0, 91))
            avg_duty_hours_per_week = float(np.clip(RNG.normal(52, 12), 30, 96))
            consecutive_duty_days = int(RNG.integers(0, 31))
            leave_days_taken_last_90 = int(RNG.integers(0, 15))
            sleep_hours_avg = float(np.clip(RNG.normal(6.2, 1.3), 2.5, 9.5))
            family_contact_days_last_30 = int(RNG.integers(0, 20))
            disciplinary_flags_last_180 = int(RNG.poisson(0.15))
            prior_incident_count = int(RNG.poisson(0.1))
            self_report_mood_score = int(np.clip(RNG.integers(1, 11), 1, 10))  # 1=low,10=great

            # --- latent stress construction (ground truth generator, hidden from model) ---
            latent = 0.0
            latent += 0.25 * (avg_duty_hours_per_week - 30) / (96 - 30)
            latent += 0.20 * (consecutive_duty_days / 30)
            latent += 0.15 * (deployment_days_last_90 / 90)
            latent += 0.15 * (1 - sleep_hours_avg / 9.5)
            latent += 0.10 * (1 - family_contact_days_last_30 / 20)
            latent += 0.10 * (1 - leave_days_taken_last_90 / 15)
            latent += 0.05 * min(disciplinary_flags_last_180, 3) / 3
            latent += 0.15 * (1 - self_report_mood_score / 10)
            latent += base_rank_load * 0.1
            latent += RNG.normal(0, 0.06)  # noise
            latent = float(np.clip(latent, 0, 1))

            journal_text = _sample_snippet(latent)

            # risk label bucketed from latent stress (what we're trying to predict)
            # Low <0.4, Medium 0.4-0.65, High >0.65
            if latent < 0.40:
                risk_label = "Low"
            elif latent < 0.65:
                risk_label = "Medium"
            else:
                risk_label = "High"

            rows.append({
                "service_id": service_id,
                "age": age,
                "years_of_service": years_of_service,
                "deployment_days_last_90": deployment_days_last_90,
                "avg_duty_hours_per_week": round(avg_duty_hours_per_week, 1),
                "consecutive_duty_days": consecutive_duty_days,
                "leave_days_taken_last_90": leave_days_taken_last_90,
                "sleep_hours_avg": round(sleep_hours_avg, 2),
                "family_contact_days_last_30": family_contact_days_last_30,
                "disciplinary_flags_last_180": disciplinary_flags_last_180,
                "prior_incident_count": prior_incident_count,
                "self_report_mood_score": self_report_mood_score,
                "journal_text": journal_text,
                "latent_stress_true": round(latent, 3),  # kept for eval only, not a model feature
                "risk_label": risk_label,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "personnel_checkins.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["risk_label"].value_counts())
