"""
Synthetic dataset generator for HACHI demo.

Creates a simple flu detection scenario for demonstrating the HACHI framework.
"""

import random
import pandas as pd

# Flu symptoms and findings
FLU_SYMPTOMS = [
    "fever of 101.5F",
    "high temperature",
    "dry cough",
    "wet cough",
    "nasal congestion",
    "runny nose",
]

NO_FLU_COMPLAINTS = [
    "routine follow-up visit",
    "medication refill",
    "annual physical exam",
    "blood pressure check",
    "mild back pain",
    "knee pain after exercise",
    "skin rash on arm",
    "request for lab work",
]


def generate_note(has_flu: bool, patient_id: int) -> str:
    """Generate a synthetic patient note."""
    random.seed(patient_id)

    age = random.randint(25, 75)
    sex = random.choice(["male", "female"])

    if has_flu:
        symptoms = random.sample(FLU_SYMPTOMS, random.randint(3, 5))
    else:
        symptoms = random.sample(NO_FLU_COMPLAINTS, random.randint(3, 5))

    note = f"{age}yo {sex}. Reports {', '.join(symptoms)}."

    return note


def generate_demo_dataset(n_samples: int = 40, positive_rate: float = 0.4) -> pd.DataFrame:
    """Generate synthetic dataset for demo."""
    n_positive = int(n_samples * positive_rate)
    n_negative = n_samples - n_positive

    records = []

    for i in range(n_positive):
        records.append({
            "patient_id": f"FLU-{i:03d}",
            "sentence": generate_note(has_flu=True, patient_id=i),
            "y": 1
        })

    for i in range(n_negative):
        records.append({
            "patient_id": f"CTRL-{i:03d}",
            "sentence": generate_note(has_flu=False, patient_id=1000 + i),
            "y": 0
        })

    random.seed(42)

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_demo_dataset(n_samples=80, positive_rate=0.5)
    df.to_csv("demo_patients.csv", index=False)
    print(f"Generated {len(df)} records ({df['y'].sum()} flu, {len(df) - df['y'].sum()} no flu)")
    print(f"\nExample:\n{df.iloc[0]['sentence']}")
