"""Mock LLM API implementing the lab_llm ``LLMApi`` run/run_batch contract."""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

MEDICAL_CONCEPTS = [
    "Does the patient have chest pain?",
    "Does the patient have hypertension?",
    "Does the patient have diabetes?",
    "Does the patient have shortness of breath?",
    "Does the patient have a smoking history?",
    "Does the patient have alcohol use?",
]

CONCEPT_WORDS = {
    "Does the patient have chest pain?": ["chest", "pain", "cardiac", "heart", "angina"],
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


class RealisticMockLLMApi:
    """
    Mock LLM that returns realistic responses for testing.

    Mirrors ``lab_llm.LLMApi``: ``run`` takes a message string or chat list and
    returns raw content, or a validated ``response_format`` instance when one is
    given. Responses are generated as JSON strings and passed through the real
    Pydantic validation so that ``strict_response_format`` behaves as in production.
    """

    def __init__(self, seed: int = 42):
        self.call_count = 0
        self.seed = seed
        np.random.seed(seed)
        self.medical_concepts = list(MEDICAL_CONCEPTS)

    @staticmethod
    def _user_prompt(messages: Union[str, List[Any]]) -> str:
        """Pull the user turn out of whatever message shape the caller passed."""
        if isinstance(messages, str):
            return messages
        user_contents = [
            m["content"]
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_contents:
            return user_contents[-1]
        return str(messages[-1]) if messages else ""

    def _extraction_json(self, prompt: str) -> str:
        num_questions = prompt.count("Does the patient") or 3
        extractions = [
            {
                "question": i + 1,
                "answer": float(np.random.choice([0, 1], p=[0.7, 0.3])),
                "reasoning": f"Mock reasoning for question {i + 1}",
            }
            for i in range(num_questions)
        ]
        return json.dumps(
            {
                "reasoning": "Mock reasoning for extraction batch",
                "extractions": extractions,
            }
        )

    def _candidate_concepts_json(self) -> str:
        concepts = [
            {
                "concept": concept,
                "is_risk_factor": True,
                "words": CONCEPT_WORDS.get(
                    concept, ["medical", "patient", "condition"]
                ),
            }
            for concept in self.medical_concepts[:3]
        ]
        return json.dumps(
            {"reasoning": "Mock reasoning for candidate concepts", "concepts": concepts}
        )

    def _keyphrases_json(self) -> str:
        return json.dumps(
            {
                "reasoning": "Mock reasoning for keyphrase extraction",
                "keyphrases": ["chest pain", "hypertension", "diabetes"],
            }
        )

    def _content_for(self, prompt: str, response_format: Optional[type]) -> str:
        """Realistic raw model content, dispatched on the requested schema."""
        if response_format is None:
            if "generate" in prompt.lower() and "concepts" in prompt.lower():
                return json.dumps(
                    {
                        "concepts": [
                            {
                                "concept": concept,
                                "reasoning": f"Clinical reasoning for {concept}",
                            }
                            for concept in self.medical_concepts[:4]
                        ]
                    }
                )
            if "extractions" in prompt.lower() or "answer" in prompt.lower():
                return self._extraction_json(prompt)
            return json.dumps({"concepts": []})

        fields = set(getattr(response_format, "model_fields", {}))
        if "extractions" in fields:
            return self._extraction_json(prompt)
        if "concepts" in fields:
            return self._candidate_concepts_json()
        if "keyphrases" in fields:
            return self._keyphrases_json()
        return json.dumps({"reasoning": "Mock reasoning"})

    def run(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        max_tool_calls: Optional[int] = None,
        model: Optional[str] = None,
        strict_response_format: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        prompt = self._user_prompt(messages)
        response_format = kwargs.get("response_format")
        content = self._content_for(prompt, response_format)

        if response_format is None:
            return content
        try:
            return response_format.model_validate_json(content)
        except Exception:
            if strict_response_format:
                raise
            return content

    async def run_batch(
        self,
        messages_list: List[Union[str, List[Any]]],
        tools: Optional[List[Any]] = None,
        max_tool_calls: Optional[int] = None,
        max_parallel_jobs: Optional[int] = None,
        model: Optional[str] = None,
        return_exceptions: bool = False,
        strict_response_format: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        results: List[Any] = []
        for messages in messages_list:
            try:
                results.append(
                    self.run(
                        messages,
                        tools=tools,
                        max_tool_calls=max_tool_calls,
                        model=model,
                        strict_response_format=strict_response_format,
                        **kwargs,
                    )
                )
            except Exception as exc:
                if not return_exceptions:
                    raise
                results.append(exc)
        return results


class FailingMockLLMApi(RealisticMockLLMApi):
    """Mock whose calls fail, for exercising per-item error handling."""

    def __init__(self, seed: int = 42, error: Optional[Exception] = None):
        super().__init__(seed=seed)
        self.error = error or RuntimeError("Mock LLM failure")

    def run(self, messages, **kwargs):
        self.call_count += 1
        raise self.error
