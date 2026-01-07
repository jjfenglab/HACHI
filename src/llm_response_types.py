from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

class TBIExclusionResult(BaseModel):
    exclude: bool = Field(
            description="Whether this patient should be excluded from the TBI prediction study"
            )
    reasoning: Optional[str] = Field(
            default=None,
            description="Brief explanation of why this case should be excluded, if applicable"
            )
    evidence: Optional[str] = Field(
            default=None,
            description="Specific text snippet from the note that supports exclusion, if applicable"
            )

class ProtoConceptExtract(BaseModel):
    reasoning: Optional[str] = Field(
        description="string describing reasoning for which keyphrases to include"
    )
    keyphrases: List[str] = Field(
        description="string of keyphrases that are highly related to each other"
    )


class ProtoConceptExtractGrouped(BaseModel):
    reasoning: Optional[str] = Field(
        description="string describing reasoning for which keyphrases to include"
    )
    keyphrases: List[str] = Field(description="string of keyphrases for each bird")


class PriorCandidate(BaseModel):
    candidate_id: int = Field(description="Candidate concept ID")
    reasoning: Optional[str] = Field(
        description="Reasoning for the prior assigned to this candidate"
    )
    prior: float = Field(description="Prior assigned to this candidate concept")


class PriorResponse(BaseModel):
    candidate_priors: List[PriorCandidate] = Field(
        description="List of candidate concepts and their priors"
    )

    def fill_candidate_concept_dicts(self, candidate_concept_dicts):
        prior_dict = {}
        for prior_candidate in self.candidate_priors:
            prior_dict[prior_candidate.candidate_id] = prior_candidate.prior

        for idx, concept_dict in enumerate(candidate_concept_dicts):
            if idx in prior_dict:
                concept_dict["prior"] = prior_dict[idx]
        return candidate_concept_dicts


class CandidateConcept(BaseModel):
    concept: str = Field(description="Concept defined as a yes/no question")
    is_risk_factor: bool = Field(description="whether the coef should be positive")
    words: List[str] = Field(description="Words that are synonyms or antonyms")


class CandidateConcepts(BaseModel):
    reasoning: Optional[str] = Field(
        description="Reasoning through each one of the attributes"
    )
    concepts: List[CandidateConcept] = Field(description="List of candidate concepts")

    def to_dicts(self, default_prior: float = 1):
        all_concept_dicts = []
        for concept in self.concepts:
            all_concept_dicts.append(
                {
                    "concept": concept.concept,
                    "words": concept.words,
                    "is_risk_factor": concept.is_risk_factor,
                }
            )
        return all_concept_dicts


class ExtractResponse(BaseModel):
    question: int = Field(..., description="question number")
    reasoning: Optional[str] = Field(description="reasoning")
    answer: float = Field(
        ..., description="binary answer, 1=yes, 0=no, probability if unsure"
    )

    @field_validator("question", mode="before")
    def ensure_question(cls, value):
        return int(value)


class ExtractResponseList(BaseModel):
    reasoning: Optional[str] = Field(description="reasoning")
    extractions: List[ExtractResponse] = Field(..., description="List of extractions")


class GroupedExtractResponses(BaseModel):
    all_extractions: List[ExtractResponseList] = Field(
        ..., description="List of extractions"
    )


# --- Concept Card Models ---
ValueType = Literal["binary", "categorical", "ordinal", "count", "duration_days"]


class EvidenceSpan(BaseModel):
    span: str = Field(description="Quoted text used by the model")
    date: Optional[str] = Field(default=None, description="YYYY-MM-DD or 'HD#'")
    loc: Optional[str] = Field(default=None, description="Note type/section if present")


class ConceptRubric(BaseModel):
    positive: List[str] = Field(description="Rules that indicate the concept")
    negative: List[str] = Field(description="Common confounders that should NOT count")


class ConceptCard(BaseModel):
    slug: str = Field(description="≤6 words, e.g., 'ID consult >48h'")
    clinical_intent: str = Field(description="1–2 sentences on LOS relevance")
    value_type: ValueType
    extraction_rubric: ConceptRubric
    evidence: List[EvidenceSpan]
    derived: Optional[Dict[str, float]] = Field(
        default=None, description="Computed metrics (e.g., delay_days)"
    )


class ConceptCardList(BaseModel):
    reasoning: Optional[str] = None
    concepts: List[ConceptCard]

    def to_rows(
        self,
        default_prior: float = 1.0,
        row_idx: Optional[int] = None,
        patient_id: Optional[str] = None,
    ):
        import json

        rows = []
        for c in self.concepts:
            rows.append(
                {
                    "row_idx": row_idx,
                    "patient_id": patient_id,
                    "slug": c.slug,
                    "clinical_intent": c.clinical_intent,
                    "value_type": c.value_type,
                    "extraction_positive": "; ".join(c.extraction_rubric.positive),
                    "extraction_negative": "; ".join(c.extraction_rubric.negative),
                    "evidence_json": json.dumps([e.model_dump() for e in c.evidence]),
                    "derived_json": json.dumps(c.derived or {}),
                    "prior": default_prior,
                }
            )
        return rows


# Evidence-aware extraction models for evidence-span enhancement
class ConceptWithEvidence(BaseModel):
    """Single concept with its supporting evidence from the clinical summary."""

    concept: str = Field(description="Concept phrase with synonyms/alternatives")
    evidence: str = Field(description="Exact text from summary supporting this concept")

    @field_validator("concept")
    @classmethod
    def validate_concept_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Concept cannot be empty")
        return v.strip()

    @field_validator("evidence")
    @classmethod
    def validate_evidence_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Evidence cannot be empty")
        return v.strip()


class ProtoConceptExtractWithEvidence(BaseModel):
    """Response model using list format for OpenAI API compatibility."""

    concepts: List[ConceptWithEvidence] = Field(
        description="List of concepts with their supporting evidence from the clinical summary"
    )

    @field_validator("concepts")
    @classmethod
    def validate_concepts_not_empty(
        cls, v: List[ConceptWithEvidence]
    ) -> List[ConceptWithEvidence]:
        """Ensure we have at least one concept."""
        if not v:
            raise ValueError("Must provide at least one concept")
        return v

    def to_keyphrases_string(self) -> str:
        """Convert to comma-separated string for backward compatibility."""
        keyphrases = [item.concept for item in self.concepts]
        return ",".join(keyphrases)

    def to_evidence_dict(self) -> Dict[str, str]:
        """Create mapping from concept phrases to evidence."""
        evidence_map = {}
        for item in self.concepts:
            # Map each phrase variant to the evidence
            # Split on commas to handle multiple phrasings
            for phrase in item.concept.split(","):
                phrase = phrase.strip()
                if phrase:
                    evidence_map[phrase] = item.evidence
        return evidence_map

    def to_keyphrases(self) -> List[str]:
        """Get just the concepts as list (legacy support)."""
        return [item.concept for item in self.concepts]
