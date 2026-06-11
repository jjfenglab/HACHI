"""
Synthetic dataset generator for HACHI demo.

Creates a simple flu detection scenario for demonstrating the HACHI framework.
"""

import random
import pandas as pd

# Flu symptoms and findings
FLU_SYMPTOMS = [
    "fever of 101.5F",
    "body aches and myalgias",
    "fatigue and malaise",
    "dry cough",
    "sore throat",
    "headache",
    "chills and sweats",
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

FLU_EXAM_FINDINGS = [
    "Temperature 101.2F",
    "Pharynx erythematous",
    "Cervical lymphadenopathy",
    "Mild tachycardia",
]

NORMAL_EXAM_FINDINGS = [
    "Temperature 98.6F",
    "Pharynx clear",
    "No lymphadenopathy",
    "Vital signs normal",
]


def generate_note(has_flu: bool, patient_id: int) -> str:
    """Generate a synthetic patient note."""
    random.seed(patient_id)

    age = random.randint(25, 75)
    sex = random.choice(["male", "female"])

    if has_flu:
        symptoms = random.sample(FLU_SYMPTOMS, random.randint(3, 5))
        exam = random.sample(FLU_EXAM_FINDINGS, random.randint(2, 3))
        cc = "flu-like symptoms"
        assessment = "Influenza likely. Recommend rest, fluids, and symptomatic treatment."
    else:
        cc = random.choice(NO_FLU_COMPLAINTS)
        symptoms = []
        exam = random.sample(NORMAL_EXAM_FINDINGS, 2)
        assessment = "No acute illness. Continue current management."

    note = f"Chief complaint: {cc}\n"
    note += f"HPI: {age}yo {sex}."
    if symptoms:
        note += f" Reports {', '.join(symptoms)}."
    note += f"\nExam: {'. '.join(exam)}."
    note += f"\nAssessment: {assessment}"

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
    random.shuffle(records)

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_demo_dataset(n_samples=40, positive_rate=0.4)
    df.to_csv("demo_patients.csv", index=False)
    print(f"Generated {len(df)} records ({df['y'].sum()} flu, {len(df) - df['y'].sum()} no flu)")
    print(f"\nExample:\n{df.iloc[0]['sentence']}")
