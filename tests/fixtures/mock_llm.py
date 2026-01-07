"""Mock LLM API for testing."""

import json
from typing import List, Optional

import numpy as np

from src.llm_response_types import ExtractResponse, ExtractResponseList


class RealisticMockLLMApi:
    """Sophisticated mock LLM that returns realistic responses for testing."""

    def __init__(self, seed: int = 42):
        self.call_count = 0
        self.seed = seed
        np.random.seed(seed)

        # Define realistic medical concepts
        self.medical_concepts = [
            "Does the patient have chest pain?",
            "Does the patient have hypertension?",
            "Does the patient have diabetes?",
            "Does the patient have shortness of breath?",
            "Does the patient have a smoking history?",
            "Does the patient have alcohol use?",
        ]

    def get_output(
        self, prompt: str, max_new_tokens: Optional[int] = None, response_model=None
    ):
        """Mock LLM output with realistic responses."""
        self.call_count += 1

        # Handle structured output with response_model
        if response_model is not None:
            # Handle ExtractResponseList (for feature extraction)
            if response_model == ExtractResponseList or (
                hasattr(response_model, "__name__")
                and response_model.__name__ == "ExtractResponseList"
            ):
                num_questions = prompt.count("Does the patient")
                if num_questions == 0:
                    num_questions = 3

                extractions = []
                for i in range(num_questions):
                    # Convert numpy int64 to native Python int
                    answer = float(np.random.choice([0, 1], p=[0.7, 0.3]))
                    extractions.append(
                        ExtractResponse(
                            question=i + 1,
                            answer=answer,
                            reasoning=f"Mock reasoning for question {i + 1}",
                        )
                    )

                return ExtractResponseList(
                    reasoning="Mock reasoning for extraction batch",
                    extractions=extractions,
                )

            # Handle candidate concepts generation
            elif hasattr(response_model, "to_dicts"):
                # Capture medical_concepts in local scope for the inner class
                medical_concepts = self.medical_concepts

                # Define relevant words for each medical concept
                concept_words = {
                    "Does the patient have chest pain?": [
                        "chest",
                        "pain",
                        "cardiac",
                        "heart",
                        "angina",
                    ],
                    "Does the patient have hypertension?": [
                        "blood pressure",
                        "hypertensive",
                        "bp",
                        "high pressure",
                    ],
                    "Does the patient have diabetes?": [
                        "diabetes",
                        "diabetic",
                        "glucose",
                        "insulin",
                        "blood sugar",
                    ],
                    "Does the patient have shortness of breath?": [
                        "breath",
                        "dyspnea",
                        "respiratory",
                        "breathing",
                        "sob",
                    ],
                    "Does the patient have a smoking history?": [
                        "smoking",
                        "tobacco",
                        "cigarette",
                        "nicotine",
                        "smoker",
                    ],
                    "Does the patient have alcohol use?": [
                        "alcohol",
                        "drinking",
                        "beer",
                        "wine",
                        "ethanol",
                        "etoh",
                    ],
                }

                class MockCandidateConcepts:
                    def to_dicts(self, default_prior=0.1):
                        concepts = []
                        for i, concept in enumerate(medical_concepts[:3]):
                            # Get words for this concept, or default words if not found
                            words = concept_words.get(
                                concept, ["medical", "patient", "condition"]
                            )
                            concepts.append(
                                {
                                    "concept": concept,
                                    "words": words,
                                    "reasoning": f"Candidate reasoning for {concept}",
                                    "prior": default_prior,
                                    "is_risk_factor": True,  # Required by greedy_concept_selector
                                }
                            )
                        return concepts

                return MockCandidateConcepts()

            # Default fallback for other response models
            else:
                return response_model(reasoning="Mock reasoning", extractions=[])

        # Handle concept generation (for baseline initialization) - no response_model
        if "generate" in prompt.lower() and "concepts" in prompt.lower():
            concepts = []
            for i, concept in enumerate(self.medical_concepts[:4]):  # Return 4 concepts
                concepts.append(
                    {
                        "concept": concept,
                        "reasoning": f"Clinical reasoning for {concept}",
                    }
                )

            return json.dumps({"concepts": concepts})

        # Handle feature extraction - no response_model
        if "extractions" in prompt.lower() or "answer" in prompt.lower():
            num_questions = prompt.count("Does the patient")
            if num_questions == 0:
                num_questions = 3

            extractions = []
            for i in range(num_questions):
                # Convert numpy int64 to native Python int
                answer = int(np.random.choice([0, 1], p=[0.7, 0.3]))
                extractions.append(
                    {
                        "question": i + 1,
                        "answer": answer,
                        "reasoning": f"Mock reasoning for question {i + 1}",
                    }
                )

            return json.dumps({"extractions": extractions})

        return json.dumps({"concepts": []})  # Default fallback

    async def get_outputs(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        response_model=None,
        **kwargs,
    ):
        """Mock batch LLM outputs."""
        results = []
        for prompt in prompts:
            results.append(self.get_output(prompt, max_new_tokens, response_model))
        return results
