"""Test data generators for ensemble trainer testing."""

import numpy as np
import pandas as pd


def create_realistic_test_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create realistic synthetic medical data with learnable patterns."""
    np.random.seed(seed)

    # Medical text templates with realistic patterns
    text_templates = [
        "Patient presents with {symptom} and {secondary}. {history}",
        "History of {condition}. Patient reports {symptom}. {social}",
        "Chief complaint: {symptom}. Past medical history: {condition}. {social}",
        "Patient denies {symptom}. No history of {condition}. {social}",
        "Acute {symptom} with {secondary}. {history} {social}",
    ]

    symptoms = ["chest pain", "shortness of breath", "dizziness", "headache", "fatigue"]
    conditions = ["hypertension", "diabetes", "heart disease", "obesity", "arthritis"]
    social = [
        "Former smoker",
        "Occasional alcohol use",
        "Denies drug use",
        "Lives alone",
        "Employed",
    ]

    data = []
    for i in range(n_samples):
        # Create patterns that correlate with outcome
        has_chest_pain = np.random.choice([0, 1], p=[0.7, 0.3])
        has_hypertension = np.random.choice([0, 1], p=[0.6, 0.4])
        has_diabetes = np.random.choice([0, 1], p=[0.8, 0.2])

        # Outcome correlates with these conditions
        outcome_prob = (
            0.2 + 0.3 * has_chest_pain + 0.2 * has_hypertension + 0.1 * has_diabetes
        )
        outcome = np.random.choice([0, 1], p=[1 - outcome_prob, outcome_prob])

        # Generate text based on conditions
        template = np.random.choice(text_templates)
        symptom = "chest pain" if has_chest_pain else np.random.choice(symptoms)
        condition = []
        if has_hypertension:
            condition.append("hypertension")
        if has_diabetes:
            condition.append("diabetes")
        if not condition:
            condition = [np.random.choice(conditions)]

        text = template.format(
            symptom=symptom,
            secondary=np.random.choice(["nausea", "weakness", "palpitations"]),
            condition=", ".join(condition),
            history=f"Family history of {np.random.choice(conditions)}",
            social=np.random.choice(social),
        )

        row = {
            "sentence": text,
            "llm_output": text,
            "y": outcome,
            "sample_weight": 1.0,  # Required by concept generator
            "label_chest_pain": has_chest_pain,
            "label_hypertension": has_hypertension,
            "label_diabetes": has_diabetes,
            "label_smoking": np.random.choice([0, 1], p=[0.8, 0.2]),
            "label_alcohol": np.random.choice([0, 1], p=[0.7, 0.3]),
        }
        data.append(row)

    return pd.DataFrame(data)


def create_small_test_data(n_samples: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create small test dataset for integration tests."""
    return create_realistic_test_data(n_samples, seed)


def create_minimal_test_data(n_samples: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create minimal test dataset for unit tests."""
    return create_realistic_test_data(n_samples, seed)
