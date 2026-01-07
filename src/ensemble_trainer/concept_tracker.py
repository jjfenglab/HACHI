"""
Concept tracking for ensemble training.

This module contains classes for tracking concept ownership across multiple
initializations.
"""

from typing import Dict, List


class ConceptTracker:
    """Tracks concept ownership across multiple initializations."""

    def __init__(self, init_seeds: List[int]):
        self.init_seeds = init_seeds
        self.concept_to_init: Dict[str, int] = {}  # concept -> init_seed
        self.init_to_concepts: Dict[int, List[str]] = {seed: [] for seed in init_seeds}

    def add_concept(self, init_seed: int, concept: str):
        """Add a concept to the tracker."""
        if concept not in self.concept_to_init:
            self.concept_to_init[concept] = init_seed
            self.init_to_concepts[init_seed].append(concept)

    def get_concepts_for_init(self, init_seed: int) -> List[str]:
        """Get all concepts for a specific initialization."""
        return self.init_to_concepts[init_seed]

    def get_all_concepts(self) -> List[str]:
        """Get all concepts across all initializations."""
        return list(self.concept_to_init.keys())

    def get_init_for_concept(self, concept: str) -> int:
        """Get the initialization that owns a concept."""
        return self.concept_to_init.get(concept)

    def get_state(self) -> Dict[str, any]:
        """Get serializable state for checkpointing."""
        return {
            "init_seeds": self.init_seeds,
            "concept_to_init": self.concept_to_init,
            "init_to_concepts": self.init_to_concepts,
        }

    def restore_state(self, state: Dict[str, any]):
        """Restore from serialized state."""
        self.init_seeds = state["init_seeds"]
        self.concept_to_init = state["concept_to_init"]
        self.init_to_concepts = state["init_to_concepts"]
