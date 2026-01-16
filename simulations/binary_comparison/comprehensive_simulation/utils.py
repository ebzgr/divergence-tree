"""
Utility functions for comprehensive simulation.

Helper functions for folder management, naming conventions, and data saving.
"""

import os
import pickle
from typing import Dict, Any, Optional


def create_scenario_name(
    mode: str,
    aspect: Optional[str] = None,
    aspect_value: Optional[Any] = None,
    scenario_id: Optional[int] = None,
    run_number: Optional[int] = None,
) -> str:
    """
    Create consistent naming for scenarios and runs.

    Parameters
    ----------
    mode : str
        Either "one_at_a_time" or "grid_search".
    aspect : str, optional
        Aspect name for one_at_a_time mode (e.g., "complexity", "noise").
    aspect_value : Any, optional
        Value of the aspect (e.g., 2, 0.5).
    scenario_id : int, optional
        Scenario ID for grid_search mode.
    run_number : int, optional
        Run number (1-indexed).

    Returns
    -------
    str
        Scenario/run name.
    """
    if mode == "one_at_a_time" or mode == "one_at_a_time_constrained":
        if aspect is None or aspect_value is None:
            raise ValueError(
                "aspect and aspect_value required for one_at_a_time mode"
            )
        # Format aspect value appropriately
        if isinstance(aspect_value, float):
            value_str = f"{aspect_value:.2f}".replace(".", "p")
        else:
            value_str = str(aspect_value)
        name = f"{aspect}_{value_str}"
    elif mode == "grid_search":
        if scenario_id is None:
            raise ValueError("scenario_id required for grid_search mode")
        name = f"scenario_{scenario_id:03d}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if run_number is not None:
        name += f"_run{run_number:03d}"

    return name


def get_data_dir(base_dir: str, mode: str, scenario_name: str) -> str:
    """
    Get data directory path for a scenario.

    Parameters
    ----------
    base_dir : str
        Base directory for simulation.
    mode : str
        Either "one_at_a_time" or "grid_search".
    scenario_name : str
        Scenario name.

    Returns
    -------
    str
        Data directory path.
    """
    return os.path.join(base_dir, "data", mode, scenario_name)


def safe_makedirs(dirpath: str) -> None:
    """
    Safely create directory, handling race conditions in parallel execution.
    
    Multiple workers may try to create the same directory simultaneously.
    This function catches and ignores errors that occur when the directory
    is created by another process between the check and creation attempt.

    Parameters
    ----------
    dirpath : str
        Directory path to create.
    """
    try:
        os.makedirs(dirpath, exist_ok=True)
    except (FileExistsError, OSError) as e:
        # Directory was created by another process between check and creation
        # On Windows, OSError with errno 183 (ERROR_ALREADY_EXISTS) can occur
        # Verify it exists now (it should)
        if not os.path.exists(dirpath):
            # If it still doesn't exist, re-raise the error
            raise
        # If it exists, silently ignore the error (race condition resolved)


def get_results_dir(base_dir: str, mode: str, scenario_name: str) -> str:
    """
    Get results directory path for a scenario.

    Parameters
    ----------
    base_dir : str
        Base directory for simulation.
    mode : str
        Either "one_at_a_time" or "grid_search".
    scenario_name : str
        Scenario name.

    Returns
    -------
    str
        Results directory path.
    """
    return os.path.join(base_dir, "results", mode, scenario_name)


def get_aggregated_dir(base_dir: str, mode: str) -> str:
    """
    Get aggregated results directory path.

    Parameters
    ----------
    base_dir : str
        Base directory for simulation.
    mode : str
        Either "one_at_a_time" or "grid_search".

    Returns
    -------
    str
        Aggregated results directory path.
    """
    return os.path.join(base_dir, "aggregated", mode)


def save_data(
    data: Dict[str, Any], filepath: str
) -> None:
    """
    Save data dictionary to pickle file.

    Parameters
    ----------
    data : dict
        Data dictionary to save.
    filepath : str
        Path to save file.
    """
    dirname = os.path.dirname(filepath)
    if dirname:  # Only create directory if path has a directory component
        safe_makedirs(dirname)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_data(filepath: str) -> Dict[str, Any]:
    """
    Load data dictionary from pickle file.

    Parameters
    ----------
    filepath : str
        Path to load file from.

    Returns
    -------
    dict
        Loaded data dictionary.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_results(
    results: Dict[str, Any], filepath: str
) -> None:
    """
    Save results dictionary to pickle file.

    Parameters
    ----------
    results : dict
        Results dictionary to save.
    filepath : str
        Path to save file.
    """
    dirname = os.path.dirname(filepath)
    if dirname:  # Only create directory if path has a directory component
        safe_makedirs(dirname)
    with open(filepath, "wb") as f:
        pickle.dump(results, f)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results dictionary from pickle file.

    Parameters
    ----------
    filepath : str
        Path to load file from.

    Returns
    -------
    dict
        Loaded results dictionary.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

