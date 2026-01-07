"""Unit tests for trainer state management."""

import pytest

from src.ensemble_trainer import TrainerState, TrainingPhase


@pytest.mark.unit
def test_trainer_state_initialization():
    """Test TrainerState initialization with defaults."""
    state = TrainerState(
        init_seeds=[1, 2, 3],
        num_epochs=2,
        max_meta_concepts=6,
        num_greedy_holdout=1,
    )

    assert state.init_seeds == [1, 2, 3]
    assert state.num_epochs == 2
    assert state.max_meta_concepts == 6
    assert state.num_greedy_holdout == 1

    # Check initial phase
    assert state.current_phase == TrainingPhase.NOT_STARTED

    # Check initial epoch tracking
    assert state.working_epoch == -1
    assert state.completed_epoch == -1
    assert state.completed_concept_iteration == -1

    # Check completion tracking
    assert len(state.baseline_complete) == 3
    assert len(state.greedy_complete) == 3
    assert all(not completed for completed in state.baseline_complete.values())
    assert all(not completed for completed in state.greedy_complete.values())


@pytest.mark.unit
def test_phase_transitions():
    """Test training phase transitions."""
    state = TrainerState(
        init_seeds=[1, 2],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initial state
    assert state.current_phase == TrainingPhase.NOT_STARTED

    # Transition to baseline running
    state.current_phase = TrainingPhase.BASELINE_RUNNING
    assert state.current_phase == TrainingPhase.BASELINE_RUNNING

    # Transition to baseline complete
    state.current_phase = TrainingPhase.BASELINE_COMPLETE
    assert state.current_phase == TrainingPhase.BASELINE_COMPLETE

    # Transition to greedy running
    state.current_phase = TrainingPhase.GREEDY_RUNNING
    assert state.current_phase == TrainingPhase.GREEDY_RUNNING

    # Transition to complete
    state.current_phase = TrainingPhase.COMPLETE
    assert state.current_phase == TrainingPhase.COMPLETE


@pytest.mark.unit
def test_epoch_tracking():
    """Test epoch tracking functionality."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=3,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Start working on epoch 0
    state.working_epoch = 0
    assert state.working_epoch == 0
    assert state.completed_epoch == -1

    # Complete epoch 0
    state.completed_epoch = 0
    assert state.working_epoch == 0
    assert state.completed_epoch == 0

    # Start working on epoch 1
    state.working_epoch = 1
    assert state.working_epoch == 1
    assert state.completed_epoch == 0

    # Complete epoch 1
    state.completed_epoch = 1
    assert state.completed_epoch == 1


@pytest.mark.unit
def test_concept_iteration_tracking():
    """Test concept iteration tracking."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=6,
        num_greedy_holdout=1,
    )

    # Initial state
    assert state.completed_concept_iteration == -1

    # Complete iterations
    state.completed_concept_iteration = 0
    assert state.completed_concept_iteration == 0

    state.completed_concept_iteration = 1
    assert state.completed_concept_iteration == 1

    # Reset for new epoch
    state.completed_concept_iteration = -1
    assert state.completed_concept_iteration == -1


@pytest.mark.unit
def test_baseline_completion_tracking():
    """Test baseline completion tracking per initialization."""
    state = TrainerState(
        init_seeds=[1, 2, 3],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially all False
    assert not state.baseline_complete[1]
    assert not state.baseline_complete[2]
    assert not state.baseline_complete[3]

    # Mark some as complete
    state.baseline_complete[1] = True
    assert state.baseline_complete[1]
    assert not state.baseline_complete[2]

    state.baseline_complete[2] = True
    state.baseline_complete[3] = True

    # All complete
    assert all(state.baseline_complete.values())


@pytest.mark.unit
def test_greedy_completion_tracking():
    """Test greedy completion tracking per initialization."""
    state = TrainerState(
        init_seeds=[1, 2],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially all False
    assert not state.greedy_complete[1]
    assert not state.greedy_complete[2]

    # Mark as complete
    state.greedy_complete[1] = True
    assert state.greedy_complete[1]
    assert not state.greedy_complete[2]


@pytest.mark.unit
def test_is_baseline_complete():
    """Test is_baseline_complete method."""
    state = TrainerState(
        init_seeds=[1, 2, 3],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially not complete
    assert not state.is_baseline_complete()

    # Mark some as complete - still not all complete
    state.baseline_complete[1] = True
    state.baseline_complete[2] = True
    assert not state.is_baseline_complete()

    # Mark all as complete
    state.baseline_complete[3] = True
    assert state.is_baseline_complete()


@pytest.mark.unit
def test_is_greedy_complete():
    """Test is_greedy_complete method."""
    state = TrainerState(
        init_seeds=[1, 2],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially not complete
    assert not state.is_greedy_complete()

    # Mark one as complete - still not all complete
    state.greedy_complete[1] = True
    assert not state.is_greedy_complete()

    # Mark all as complete
    state.greedy_complete[2] = True
    assert state.is_greedy_complete()


@pytest.mark.unit
def test_shared_extractions_tracking():
    """Test shared extractions dictionary."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially empty
    assert len(state.shared_extractions) == 0

    # Add extractions
    import numpy as np

    state.shared_extractions["concept1"] = np.array([[1, 2, 3]])
    state.shared_extractions["concept2"] = np.array([[4, 5, 6]])

    assert len(state.shared_extractions) == 2
    assert "concept1" in state.shared_extractions
    assert "concept2" in state.shared_extractions


@pytest.mark.unit
def test_training_histories_files_tracking():
    """Test training history file paths tracking."""
    state = TrainerState(
        init_seeds=[1, 2],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially empty
    assert len(state.training_histories_files) == 0

    # Add file paths
    state.training_histories_files[1] = "/path/to/history_1.pkl"
    state.training_histories_files[2] = "/path/to/history_2.pkl"

    assert len(state.training_histories_files) == 2
    assert state.training_histories_files[1] == "/path/to/history_1.pkl"
    assert state.training_histories_files[2] == "/path/to/history_2.pkl"


@pytest.mark.unit
def test_state_validation_valid():
    """Test state validation with valid state."""
    state = TrainerState(
        init_seeds=[1, 2],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Valid initial state
    state.validate_state()  # Should not raise


@pytest.mark.unit
def test_state_validation_working_epoch_bounds():
    """Test state validation catches invalid working_epoch."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Set invalid working epoch (beyond num_epochs)
    state.working_epoch = 3  # num_epochs is 2, so max valid is 1

    with pytest.raises(ValueError, match="working_epoch"):
        state.validate_state()


@pytest.mark.unit
def test_state_validation_completed_epoch_bounds():
    """Test state validation catches invalid completed_epoch."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Set invalid completed epoch
    state.completed_epoch = 2  # num_epochs is 2, so max valid is 1

    with pytest.raises(ValueError, match="completed_epoch"):
        state.validate_state()


@pytest.mark.unit
def test_state_validation_epoch_relationship():
    """Test state validation catches invalid epoch relationships."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Set completed_epoch > working_epoch (invalid)
    state.working_epoch = 0
    state.completed_epoch = 1

    with pytest.raises(ValueError, match="completed_epoch.*>.*working_epoch"):
        state.validate_state()


@pytest.mark.unit
def test_interruption_count():
    """Test interruption count tracking."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initial count
    assert state.interruption_count == 0

    # Increment on interruptions
    state.interruption_count += 1
    assert state.interruption_count == 1

    state.interruption_count += 1
    assert state.interruption_count == 2


@pytest.mark.unit
def test_phase_start_times_tracking():
    """Test phase start times dictionary."""
    state = TrainerState(
        init_seeds=[1],
        num_epochs=2,
        max_meta_concepts=4,
        num_greedy_holdout=1,
    )

    # Initially empty
    assert len(state.phase_start_times) == 0

    # Add phase start times
    import time

    state.phase_start_times["baseline"] = time.time()
    state.phase_start_times["greedy"] = time.time()

    assert len(state.phase_start_times) == 2
    assert "baseline" in state.phase_start_times
    assert "greedy" in state.phase_start_times
