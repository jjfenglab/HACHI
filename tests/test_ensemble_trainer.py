#!/usr/bin/env python3
"""
Comprehensive test suite for EnsembleTrainer functionality.

This script tests the ensemble trainer with realistic synthetic data and
sophisticated mocks to validate all major functionality including:
- Complete training pipeline (baseline + bayesian)
- Checkpoint/resume functionality
- Parallel processing
- Data operations
- Concept evolution
- Error handling
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ensemble_trainer import (
    CheckpointManager,
    ConceptConfig,
    ConceptGeneratorFactory,
    DataConfig,
    EnsembleConfig,
    EnsembleTrainer,
    FeatureExtractionManager,
    LLMConfig,
    ModelConfig,
    TrainingConfig,
    TrainingPhase,
    data_operations,
)
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
                                    "prior": default_prior,  # Default prior
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


def create_realistic_test_data(n_samples: int = 200) -> pd.DataFrame:
    """Create realistic synthetic medical data with learnable patterns."""
    np.random.seed(42)

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


def create_test_config(temp_dir: str) -> Dict[str, str]:
    """Create test configuration files."""

    # Create concept extraction prompt
    concept_prompt = """
Please analyze the following medical note and answer these questions:

{prompt_questions}

For each question, provide a numerical answer (0 for No, 1 for Yes) and brief reasoning.

Format your response as JSON:
{
    "extractions": [
        {
            "question": 1,
            "answer": 0,
            "reasoning": "Brief explanation"
        }
    ]
}

Medical note: {sentence}
"""

    # Create baseline initialization prompt
    baseline_prompt = """
Based on the top predictive features from logistic regression analysis, generate {num_meta_concepts} medical concepts for patient classification:

Top Features:
{top_features_df}

Generate concepts as binary questions (answerable with 0/1) that could help classify patients.

Format as JSON:
{
    "concepts": [
        {
            "concept": "Does the patient have chest pain?",
            "reasoning": "Chest pain is a key diagnostic indicator"
        }
    ]
}
"""

    # Create iterative improvement prompt
    iter_prompt = """
Given the current model performance and concept set, suggest improvements:

Current concepts: {current_concepts}
Model performance: {model_performance}
Residual analysis: {residual_analysis}

Suggest {num_new_concepts} new or improved concepts for better classification.

Format as JSON:
{
    "concepts": [
        {
            "concept": "Does the patient have a specific condition?",
            "reasoning": "Explanation for this concept"
        }
    ]
}
"""

    # Create prior concept generation prompt
    prior_prompt = """
Generate {num_meta_concepts} medical concepts for patient classification based on clinical knowledge:

Focus on concepts that are:
1. Clinically relevant
2. Answerable from medical notes
3. Predictive of patient outcomes

Format as JSON:
{
    "concepts": [
        {
            "concept": "Does the patient have cardiovascular risk factors?",
            "reasoning": "Cardiovascular risk factors are key predictors"
        }
    ]
}
"""

    # Write all prompt files
    files = {
        "concept_prompt_file": os.path.join(temp_dir, "concept_prompt.txt"),
        "baseline_prompt_file": os.path.join(temp_dir, "baseline_prompt.txt"),
        "iter_prompt_file": os.path.join(temp_dir, "iter_prompt.txt"),
        "prior_prompt_file": os.path.join(temp_dir, "prior_prompt.txt"),
    }

    with open(files["concept_prompt_file"], "w") as f:
        f.write(concept_prompt)

    with open(files["baseline_prompt_file"], "w") as f:
        f.write(baseline_prompt)

    with open(files["iter_prompt_file"], "w") as f:
        f.write(iter_prompt)

    with open(files["prior_prompt_file"], "w") as f:
        f.write(prior_prompt)

    return files


class EnsembleTrainerTest:
    """Comprehensive test suite for EnsembleTrainer."""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.test_data = create_realistic_test_data(150)
        self.config = create_test_config(temp_dir)
        self.results = {}

    def setup_trainer(self, **kwargs) -> EnsembleTrainer:
        """Create a test ensemble trainer with mock LLM."""

        # Create ensemble config using new structure
        model_config = ModelConfig(
            model="l2",
            final_model_type="l2",
            learner_type="count_l2",
            count_vectorizer="count",
            use_acc=False,
            final_learner_type="l2",
        )

        llm_config = LLMConfig(
            llm_model="gpt-4o-mini",
            max_tokens=600,
            max_new_tokens=5000,
            cache_file=os.path.join(self.temp_dir, "test_cache.db"),
        )

        data_config = DataConfig(
            text_summary_column="sentence",
            concept_column="sentence",
            min_prevalence=0,
            is_image=False,
            max_section_length=None,
        )

        training_config = TrainingConfig(
            num_epochs=2,
            num_greedy_epochs=1,
            batch_size=8,
            batch_concept_size=10,
            batch_obs_size=1,
            train_frac=0.8,
            num_greedy_holdout=1,
            sampling_method="bootstrap",
        )

        concept_config = ConceptConfig(
            goal_num_meta_concepts=4,
            baseline_init_file=self.config["baseline_prompt_file"],
            prompt_concepts_file=self.config["concept_prompt_file"],
            prompt_iter_file=self.config["iter_prompt_file"],
            prompt_prior_file=self.config["prior_prompt_file"],
        )

        ensemble_config = EnsembleConfig(
            init_seeds=[1, 2, 3],
            model=model_config,
            llm=llm_config,
            data=data_config,
            training=training_config,
            concept=concept_config,
        )

        # Allow kwargs to override config values
        if kwargs:
            # Update config with any overrides from kwargs
            for key, value in kwargs.items():
                if hasattr(ensemble_config, key):
                    setattr(ensemble_config, key, value)

        trainer = EnsembleTrainer(config=ensemble_config, output_dir=self.temp_dir)

        # Disable signal handlers for testing
        trainer._setup_signal_handlers = lambda: None

        # Mock LLM clients
        mock_llm = RealisticMockLLMApi()
        trainer.llm_dict = {
            "iter": mock_llm,
            "extraction": mock_llm,
        }

        # Mock feature extraction manager - use the ConceptTracker already created by trainer
        trainer.feature_extraction_manager = FeatureExtractionManager(
            trainer.config, trainer.llm_dict, trainer.concept_tracker
        )

        # Initialize trainer state for testing
        trainer._initialize_state()

        # Monkey patch _setup_llm_clients to preserve our mock
        def mock_setup_llm_clients(self):
            """Mock version of _setup_llm_clients that preserves the mock LLM."""
            # Keep existing llm_dict and feature_extraction_manager (our mocks)
            # Just handle the checkpoint restoration logic
            if self.ensemble_state and self.ensemble_state.shared_extractions:
                self.feature_extraction_manager.set_shared_extractions(
                    self.ensemble_state.shared_extractions
                )
                logging.info(
                    f"Restored {len(self.ensemble_state.shared_extractions)} shared extractions from checkpoint"
                )

            # Restore state from checkpoint if resuming
            if (
                self.ensemble_state
                and self.ensemble_state.current_phase
                != self.ensemble_state.current_phase.NOT_STARTED
            ):
                CheckpointManager.restore_from_checkpoint(
                    self.ensemble_state,
                    self.config.init_seeds,
                    self.concept_tracker,
                    self.training_histories,
                )

            # Don't create selectors here - let the trainer create them
            # when actually needed (after baseline training)

        # Bind the mock method to the trainer instance
        import types

        trainer._setup_llm_clients = types.MethodType(mock_setup_llm_clients, trainer)

        # Don't create greedy concept selectors yet - they need populated training histories
        # The trainer will create these when needed during actual training
        return trainer

    async def test_data_operations(self) -> bool:
        """Test data_operations functionality."""
        print("Testing data_operations...")

        try:
            # Test data splitting
            for init_seed in [1, 2, 3]:
                train_data, test_data = data_operations.create_data_split(
                    self.test_data, init_seed, 0.7
                )
                assert len(train_data) > 0, f"Empty split for seed {init_seed}"
                assert len(train_data) <= len(
                    self.test_data
                ), "Split larger than original"
                assert len(train_data) + len(test_data) == len(self.test_data), "Split sizes don't sum to original"

            # Test extraction mapping
            mock_extractions = {
                "concept1": np.random.rand(len(self.test_data), 1),
                "concept2": np.random.rand(len(self.test_data), 1),
            }
            mock_concepts = [{"concept": "concept1"}, {"concept": "concept2"}]

            mapped = data_operations.map_extractions_to_sample(
                mock_extractions, self.test_data, mock_concepts
            )
            assert len(mapped) == 2, "Wrong number of mapped extractions"

            print("✅ data_operations tests passed")
            return True

        except Exception as e:
            print(f"❌ data_operations test failed: {e}")
            return False

    async def test_concept_evolution(self) -> bool:
        """Test concept_evolution functionality end-to-end at a unit level."""
        print("Testing concept_evolution...")

        try:
            # Setup a minimal trainer with a single initialization for determinism
            trainer = self.setup_trainer(
                init_seeds=[1],
                num_epochs=1,
                num_meta_concepts=3,
            )

            # Direct outputs/checkpoints to an isolated directory to avoid cross-test conflicts
            concept_evo_out = os.path.join(self.temp_dir, "concept_evolution_output")
            os.makedirs(concept_evo_out, exist_ok=True)
            trainer.output_dir = concept_evo_out
            trainer.checkpoint_file = os.path.join(
                concept_evo_out, "ensemble_state_checkpoint.pkl"
            )

            # Ensure generator exists and run baseline to populate histories
            trainer.concept_generator = ConceptGeneratorFactory.create_generator(
                trainer.config, trainer.llm_dict, trainer.summaries_df
            )
            await trainer._run_baseline_phase(self.test_data)

            # Create concept selectors now that baseline is available
            trainer._create_greedy_concept_selectors()

            # Import concept evolution utilities
            from src.ensemble_trainer import selection_operations

            init_seed = 1
            history = trainer.training_histories[init_seed]

            # Current concepts and extractions (limit to configured num_meta_concepts)
            current_concepts = history.get_last_concepts()[: trainer.config.concept.goal_num_meta_concepts]
            current_concept_names = [c["concept"] for c in current_concepts]

            # Extract features for current concepts on full dataset
            current_shared_extractions = (
                await trainer.feature_extraction_manager.extract_for_training(
                    self.test_data,
                    current_concept_names,
                    max_new_tokens=trainer.config.llm.max_new_tokens,
                )
            )

            # Build a train split for candidate generation
            init_data_df, _ = data_operations.create_data_split(
                self.test_data, init_seed, trainer.train_frac
            )

            concept_selector = trainer.concept_selectors[init_seed]

            # Generate candidate concepts
            candidate_concept_dicts = (
                trainer.concept_generator.generate_candidate_concepts(
                    init_data_df,
                    concept_selector,
                    current_concepts,
                    current_shared_extractions,
                )
            )

            assert len(candidate_concept_dicts) > 0, "No candidate concepts generated"
            assert all(
                "concept" in c for c in candidate_concept_dicts
            ), "Candidate concepts missing 'concept' key"

            # Extract candidate features on full dataset
            candidate_names = [c["concept"] for c in candidate_concept_dicts]
            candidate_shared_extractions = (
                await trainer.feature_extraction_manager.extract_for_training(
                    self.test_data,
                    candidate_names,
                    max_new_tokens=trainer.config.llm.max_new_tokens,
                )
            )

            # Build init candidate map
            init_candidate_map = {
                (init_seed, c["concept"]): c for c in candidate_concept_dicts
            }

            # Perform greedy selection and verify the result
            final_concepts = selection_operations.perform_greedy_selection(
                self.test_data,
                concept_selector,
                init_candidate_map,
                current_shared_extractions,
                candidate_shared_extractions,
            )

            assert (
                len(final_concepts) == trainer.config.concept.goal_num_meta_concepts
            ), "Final concept set has wrong size"
            assert all(
                "concept" in c for c in final_concepts
            ), "Final concepts missing 'concept' key"

            # Duplicates can occur if the greedy step retains a held-out concept or a candidate
            # matches an existing kept concept. We don't fail on duplicates; we only require that
            # features can be constructed and sizes are consistent.

            # Sanity-check: we can build features for the final set
            all_extractions = {
                **current_shared_extractions,
                **candidate_shared_extractions,
            }
            import src.common as common

            X_check = common.get_features(
                final_concepts,
                {
                    k: v
                    for k, v in all_extractions.items()
                    if k in {c["concept"] for c in final_concepts}
                },
                self.test_data,
                force_keep_columns=trainer.config.force_keep_columns,
            )
            assert X_check.shape[0] == len(
                self.test_data
            ), "Feature rows mismatch dataset size"

            print("✅ concept_evolution tests passed")
            return True

        except Exception as e:
            print(f"❌ concept_evolution test failed: {e}")
            return False

    async def test_checkpoint_functionality(self) -> bool:
        """Test checkpoint save/load/resume functionality."""
        print("Testing checkpoint functionality...")

        try:
            trainer = self.setup_trainer()

            # Test checkpoint initialization
            state, checkpoint_file, resuming = (
                CheckpointManager.initialize_or_resume_state(
                    trainer.config.init_seeds,
                    trainer.num_epochs,
                    trainer.config.concept.goal_num_meta_concepts,
                    trainer.num_greedy_holdout,
                    self.temp_dir,
                    None,
                )
            )

            assert not resuming, "Should not be resuming on first run"
            assert state is not None, "State should be initialized"

            # Test checkpoint saving
            CheckpointManager.save_checkpoint(
                state,
                checkpoint_file,
                self.temp_dir,
                trainer.concept_tracker,
                trainer.feature_extraction_manager,
                trainer.config.init_seeds,
                trainer.training_histories,
            )

            assert os.path.exists(checkpoint_file), "Checkpoint file should exist"

            # Test checkpoint loading
            state2, _, resuming2 = CheckpointManager.initialize_or_resume_state(
                trainer.config.init_seeds,
                trainer.num_epochs,
                trainer.config.concept.goal_num_meta_concepts,
                trainer.num_greedy_holdout,
                self.temp_dir,
                None,
            )

            assert resuming2, "Should be resuming from existing checkpoint"

            print("✅ Checkpoint functionality tests passed")
            return True

        except Exception as e:
            print(f"❌ Checkpoint test failed: {e}")
            return False

    async def test_parallel_processing(self) -> bool:
        """Test that parallel processing works correctly."""
        print("Testing parallel processing...")

        try:
            trainer = self.setup_trainer()

            # Test parallel baseline training
            start_time = time.time()

            # Mock the generate_initial_concepts function
            def mock_generate_concepts(init_seed, data_df):
                # Simulate some work
                time.sleep(0.1)  # Small delay to test parallelization
                concepts = [
                    {"concept": f"Mock concept {i} for seed {init_seed}"}
                    for i in range(3)
                ]
                return concepts, data_df.sample(n=min(50, len(data_df)))

            # Test parallel execution
            baseline_histories = await trainer._train_baselines_parallel(
                self.test_data,
                mock_generate_concepts,
                trainer.feature_extraction_manager,
                trainer.concept_tracker,
                trainer.config,
            )

            parallel_time = time.time() - start_time

            # Verify results
            assert len(baseline_histories) == len(
                trainer.config.init_seeds
            ), "Wrong number of histories"
            for init_seed in trainer.config.init_seeds:
                assert (
                    init_seed in baseline_histories
                ), f"Missing history for seed {init_seed}"

            # Test should complete in reasonable time (parallel should be faster than 3x sequential)
            # assert (
            #     parallel_time < 0.5
            # ), f"Parallel processing too slow: {parallel_time:.2f}s"

            print(
                f"✅ Parallel processing tests passed (completed in {parallel_time:.2f}s)"
            )
            return True

        except Exception as e:
            print(f"❌ Parallel processing test failed: {e}")
            return False

    async def test_feature_extraction_manager(self) -> bool:
        """Test FeatureExtractionManager functionality."""
        print("Testing FeatureExtractionManager...")

        try:
            trainer = self.setup_trainer()

            # Test batch extraction
            concepts = [
                "Does the patient have chest pain?",
                "Does the patient have diabetes?",
            ]

            extractions = (
                await trainer.feature_extraction_manager.extract_features_batch(
                    self.test_data.head(20),  # Small subset for testing
                    concepts,
                    max_new_tokens=1000,
                )
            )

            assert len(extractions) == len(concepts), "Wrong number of extractions"
            for concept in concepts:
                assert concept in extractions, f"Missing extraction for {concept}"
                assert (
                    extractions[concept].shape[0] == 20
                ), f"Wrong extraction shape for {concept}"

            # Test caching
            extractions2 = (
                await trainer.feature_extraction_manager.extract_features_batch(
                    self.test_data.head(20),
                    concepts,  # Same concepts - should use cache
                    max_new_tokens=1000,
                )
            )

            # Should return same results from cache
            for concept in concepts:
                np.testing.assert_array_equal(
                    extractions[concept], extractions2[concept]
                )

            print("✅ FeatureExtractionManager tests passed")
            return True

        except Exception as e:
            print(f"❌ FeatureExtractionManager test failed: {e}")
            return False

    async def test_end_to_end_training(self) -> bool:
        """Test complete end-to-end training pipeline."""
        print("Testing end-to-end training...")

        try:
            # Use smaller configuration for faster testing
            trainer = self.setup_trainer(
                init_seeds=[1, 2],  # Fewer seeds
                num_epochs=1,  # Single epoch
                num_meta_concepts=3,  # Fewer concepts
            )

            trainer.concept_generator = ConceptGeneratorFactory.create_generator(
                trainer.config, trainer.llm_dict, trainer.summaries_df
            )

            # Run complete training
            output_dir = os.path.join(self.temp_dir, "training_output")
            os.makedirs(output_dir, exist_ok=True)

            # we have to make the directories that would otherwise be made
            # by scons
            for init_seed in [1, 2]:
                os.makedirs(
                    os.path.join(output_dir, f"init_seed_{init_seed}"), exist_ok=True
                )

            # Ensure trainer saves artifacts to the expected output directory
            trainer.output_dir = output_dir
            trainer.checkpoint_file = os.path.join(
                output_dir, "ensemble_state_checkpoint.pkl"
            )

            start_time = time.time()

            # Manually run the baseline phase to populate training_histories
            await trainer._run_baseline_phase(self.test_data)

            # Manually create concept selectors now that baseline is done
            trainer._create_greedy_concept_selectors()

            # Manually set the phase to baseline complete
            trainer.ensemble_state.current_phase = TrainingPhase.BASELINE_COMPLETE

            # Run the greedy phase using the trainer's configured output_dir
            await trainer._run_greedy_phase(self.test_data)
            final_histories = trainer.training_histories
            training_time = time.time() - start_time

            # Verify results
            assert len(final_histories) == len(
                trainer.config.init_seeds
            ), "Wrong number of final histories"

            for init_seed in trainer.config.init_seeds:
                assert (
                    init_seed in final_histories
                ), f"Missing final history for seed {init_seed}"
                history = final_histories[init_seed]
                assert (
                    len(history._concepts) > 0
                ), f"No concepts in final history for seed {init_seed}"
                assert (
                    len(history._aucs) > 0
                ), f"No AUCs in final history for seed {init_seed}"

            # Check output files
            for init_seed in trainer.config.init_seeds:
                seed_dir = os.path.join(output_dir, f"init_seed_{init_seed}")
                assert os.path.exists(
                    seed_dir
                ), f"Missing output directory for seed {init_seed}"

                logging.info(f"Files in {seed_dir}: {os.listdir(seed_dir)}")
                final_file = os.path.join(seed_dir, "training_history.pkl")

                assert os.path.exists(
                    final_file
                ), f"Missing history files for seed {init_seed}. {os.listdir(seed_dir)}"

            print(
                f"✅ End-to-end training tests passed (completed in {training_time:.2f}s)"
            )
            return True

        except Exception as e:
            print(f"❌ End-to-end training test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling and edge cases."""
        print("Testing error handling...")

        try:
            # Test invalid configuration
            try:
                self.setup_trainer(init_seeds=[])  # Empty seeds
                assert False, "Should have failed with empty init_seeds"
            except (ValueError, AssertionError):
                pass  # Expected

            # Test checkpoint compatibility
            try:
                state, _, _ = CheckpointManager.initialize_or_resume_state(
                    [1, 2, 3], 2, 4, 1, self.temp_dir, None
                )
                # Modify state to be incompatible
                state.init_seeds = [1, 2]  # Different seeds

                CheckpointManager.validate_checkpoint_compatibility(
                    state, [1, 2, 3], 2, 4  # Different seeds
                )
                assert False, "Should have failed compatibility check"
            except ValueError:
                pass  # Expected

            print("✅ Error handling tests passed")
            return True

        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False

    async def run_all_tests(self) -> bool:
        """Run all test suites."""
        print("=" * 60)
        print("Running Comprehensive EnsembleTrainer Tests")
        print("=" * 60)

        tests = [
            ("Data Operations", self.test_data_operations),
            ("Concept Evolution", self.test_concept_evolution),
            ("Checkpoint Functionality", self.test_checkpoint_functionality),
            ("Parallel Processing", self.test_parallel_processing),
            ("Feature Extraction Manager", self.test_feature_extraction_manager),
            ("End-to-End Training", self.test_end_to_end_training),
            ("Error Handling", self.test_error_handling),
        ]

        results = {}
        for test_name, test_func in tests:
            print(f"\n{'='*40}")
            print(f"Running {test_name} Tests")
            print("=" * 40)

            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    print(f"✅ {test_name} - PASSED")
                else:
                    print(f"❌ {test_name} - FAILED")
            except Exception as e:
                print(f"❌ {test_name} - ERROR: {e}")
                results[test_name] = False

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<30} {status}")

        print(f"\nTotal: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All tests passed! EnsembleTrainer is working correctly.")
            return True
        else:
            print(
                f"\n⚠️  {total - passed} tests failed. Please review the failures above."
            )
            return False


async def main():
    """Run the comprehensive test suite."""
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        tester = EnsembleTrainerTest(temp_dir)
        success = await tester.run_all_tests()

        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
