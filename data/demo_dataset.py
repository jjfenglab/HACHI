"""
Synthetic dataset generator for HACHI demo.

This creates a fictional clinical scenario for demonstrating the HACHI framework
without making real medical claims. The condition "Quantum Temporal Dissonance Syndrome"
(QTDS) is entirely fictional.
"""

import random
import pandas as pd
from typing import List, Tuple

# Fictional condition: Quantum Temporal Dissonance Syndrome (QTDS)
# A whimsical condition where patients experience temporal perception anomalies

SYMPTOMS_POSITIVE = [
    "reports experiencing time moving at different speeds",
    "describes seeing multiple versions of events simultaneously",
    "complains of 'temporal echoes' - hearing conversations before they happen",
    "notes difficulty distinguishing past memories from future predictions",
    "experiences episodes of 'chronological vertigo'",
    "reports déjà vu occurring multiple times per hour",
    "describes feeling 'unstuck in time' during stressful situations",
    "mentions seeing 'temporal shadows' of people who haven't arrived yet",
    "complains of watches and clocks appearing to run backwards",
    "reports conversations that 'loop' and repeat",
]

SYMPTOMS_NEGATIVE = [
    "denies any temporal perception issues",
    "reports normal sense of time passage",
    "no complaints of déjà vu or temporal anomalies",
    "watches and clocks appear to function normally",
    "describes linear, sequential perception of events",
    "no difficulty distinguishing past from present",
    "denies seeing temporal echoes or shadows",
    "reports stable chronological orientation",
    "no episodes of feeling 'unstuck in time'",
    "conversations proceed normally without repetition",
]

RISK_FACTORS = [
    "history of exposure to quantum computing equipment",
    "works in theoretical physics research",
    "recent travel across multiple time zones",
    "family history of temporal perception disorders",
    "uses experimental chronotherapy devices",
    "previous episode of QTDS",
    "high stress occupation",
    "irregular sleep-wake cycles",
]

PROTECTIVE_FACTORS = [
    "regular meditation practice",
    "stable daily routine",
    "no exposure to quantum equipment",
    "no family history of temporal disorders",
    "consistent sleep schedule",
    "low-stress lifestyle",
    "grounded mindfulness practice",
    "no history of QTDS episodes",
]

EXAM_FINDINGS_POSITIVE = [
    "Temporal Orientation Test: abnormal - patient predicted questions before asked",
    "Chronological Sequencing: impaired - events arranged non-linearly",
    "Clock Drawing Test: unusual - drew multiple overlapping clock faces",
    "Reaction Time: paradoxically fast, responded before stimulus",
    "Pupillary response: asymmetric temporal dilation",
]

EXAM_FINDINGS_NEGATIVE = [
    "Temporal Orientation Test: within normal limits",
    "Chronological Sequencing: intact",
    "Clock Drawing Test: normal single clock face",
    "Reaction Time: appropriate latency",
    "Pupillary response: symmetric and normal",
]

TREATMENTS = [
    "Started on temporal stabilization therapy",
    "Prescribed chronological grounding exercises",
    "Recommended temporal isolation protocol",
    "Initiated quantum decoherence treatment",
    "Advised strict routine adherence",
]


def generate_patient_note(has_qtds: bool, patient_id: int) -> Tuple[str, int]:
    """Generate a synthetic patient note for the demo."""
    random.seed(patient_id)

    # Demographics
    age = random.randint(25, 75)
    sex = random.choice(["Male", "Female"])

    # Build the note
    sections = []

    # Chief Complaint
    if has_qtds:
        cc = random.choice([
            "experiencing strange temporal sensations",
            "feeling 'out of sync' with time",
            "concerned about time perception changes",
            "episodes of temporal disorientation",
        ])
    else:
        cc = random.choice([
            "routine follow-up visit",
            "general wellness check",
            "mild headaches",
            "fatigue and low energy",
        ])
    sections.append(f"CHIEF COMPLAINT: {cc}")

    # History of Present Illness
    hpi_parts = []
    hpi_parts.append(f"{age}-year-old {sex.lower()} presenting for evaluation.")

    if has_qtds:
        # Add positive symptoms
        num_symptoms = random.randint(2, 4)
        symptoms = random.sample(SYMPTOMS_POSITIVE, num_symptoms)
        hpi_parts.extend([f"Patient {s}." for s in symptoms])

        # Add risk factors
        num_risks = random.randint(1, 3)
        risks = random.sample(RISK_FACTORS, num_risks)
        hpi_parts.append(f"Notable risk factors include: {', '.join(risks)}.")
    else:
        # Add negative symptoms
        num_symptoms = random.randint(1, 3)
        symptoms = random.sample(SYMPTOMS_NEGATIVE, num_symptoms)
        hpi_parts.extend([f"Patient {s}." for s in symptoms])

        # Add protective factors
        num_protective = random.randint(1, 2)
        protective = random.sample(PROTECTIVE_FACTORS, num_protective)
        hpi_parts.append(f"Patient reports {', '.join(protective)}.")

    sections.append("HISTORY OF PRESENT ILLNESS: " + " ".join(hpi_parts))

    # Physical Exam
    exam_parts = ["Vitals: stable. General: alert and oriented."]
    if has_qtds:
        findings = random.sample(EXAM_FINDINGS_POSITIVE, random.randint(1, 2))
    else:
        findings = random.sample(EXAM_FINDINGS_NEGATIVE, random.randint(1, 2))
    exam_parts.extend(findings)
    sections.append("PHYSICAL EXAM: " + " ".join(exam_parts))

    # Assessment/Plan
    if has_qtds:
        assessment = "Assessment: Findings consistent with Quantum Temporal Dissonance Syndrome (QTDS)."
        plan = random.choice(TREATMENTS)
    else:
        assessment = "Assessment: No evidence of temporal perception disorder."
        plan = "Continue current management. Return for follow-up as needed."
    sections.append(f"ASSESSMENT AND PLAN: {assessment} {plan}")

    return "\n\n".join(sections), 1 if has_qtds else 0


def generate_demo_dataset(n_samples: int = 80, positive_rate: float = 0.4) -> pd.DataFrame:
    """
    Generate a synthetic dataset for the HACHI demo.

    Args:
        n_samples: Number of patient notes to generate
        positive_rate: Proportion of positive (QTDS) cases

    Returns:
        DataFrame with columns: patient_id, note_text, outcome
    """
    n_positive = int(n_samples * positive_rate)
    n_negative = n_samples - n_positive

    # Generate cases
    records = []

    # Positive cases
    for i in range(n_positive):
        note, outcome = generate_patient_note(has_qtds=True, patient_id=i)
        records.append({
            "patient_id": f"QTDS-{i:03d}",
            "note_text": note,
            "outcome": outcome
        })

    # Negative cases
    for i in range(n_negative):
        note, outcome = generate_patient_note(has_qtds=False, patient_id=1000 + i)
        records.append({
            "patient_id": f"CTRL-{i:03d}",
            "note_text": note,
            "outcome": outcome
        })

    # Shuffle
    random.seed(42)
    random.shuffle(records)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Generate and save the demo dataset
    df = generate_demo_dataset(n_samples=80, positive_rate=0.4)
    df.to_csv("demo_patients.csv", index=False)
    print(f"Generated {len(df)} patient records")
    print(f"Positive cases: {df['outcome'].sum()}")
    print(f"Negative cases: {len(df) - df['outcome'].sum()}")
    print("\nSample note:")
    print(df.iloc[0]['note_text'])
