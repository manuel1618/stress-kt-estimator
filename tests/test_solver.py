from __future__ import annotations

import pandas as pd

from kt_optimizer.models import (
    FORCE_COLUMNS,
    ObjectiveMode,
    SignMode,
    SolverSettings,
    TABLE_COLUMNS,
)
from kt_optimizer.solver import find_minimal_unlink, suggest_unlink_from_data, solve


def _sample_df():
    data = [
        ["LC1", 100, 0, 0, 0, 0, 0, 100],
        ["LC2", 0, 100, 0, 0, 0, 0, 100],
        ["LC3", 40, 40, 0, 0, 0, 0, 90],
        ["LC4", -100, 0, 0, 0, 0, 0, 80],
    ]
    return pd.DataFrame(data, columns=TABLE_COLUMNS)


def test_minmax_mode_solution():
    df = _sample_df()
    settings = SolverSettings(
        use_separate_sign=True, objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION
    )
    result = solve(df, settings)
    assert result.success
    assert result.min_error >= -1e-6
    assert len(result.kt_values) == 12


def test_flexible_column_mapping_works():
    df = pd.DataFrame(
        {
            "LC": [3, 5, 36, 38, 59],
            "Side": [0, 0, 13, 0, 0],
            "Drag": [9.588, 22.533, 0, 0, -23.838],
            "Vertical": [38.35, 28.895, 18.249, 10.787, 3.182],
            "-Mz": [0, 0, 0, -573.81, 0],
            "Stress": [124.63, 285.51, 58.979, 31.947, 286.43],
        }
    )

    result = solve(df, SolverSettings(use_separate_sign=True))
    assert result.success
    assert len(result.per_case) == 5
    assert min(c.predicted - c.actual for c in result.per_case) >= -1e-6


def test_bad_safety_factor_fails():
    df = _sample_df()
    settings = SolverSettings(safety_factor=0)
    result = solve(df, settings)
    assert not result.success


def test_suggest_unlink_from_data_fat_lower_bracket():
    """With Fy both + and - and different stress/force behaviour, Fy is suggested."""
    df = _sample_df()
    # Add a case with negative Fy and different implied Kt
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [["LC5", 0, -100, 0, 0, 0, 0, 60]],
                columns=TABLE_COLUMNS,
            ),
        ],
        ignore_index=True,
    )
    suggested = suggest_unlink_from_data(df, ratio_threshold=2.0)
    assert "Fy" in suggested


def test_find_minimal_unlink_returns_valid_config():
    """find_minimal_unlink returns a valid sign_mode list and result."""
    df = pd.DataFrame(
        {
            "Case Name": ["3", "5", "59"],
            "Fx": [0, 0, 0],
            "Fy": [10, 22, -24],
            "Fz": [38, 29, 3],
            "Mx": [0, 0, 0],
            "My": [0, 0, 0],
            "Mz": [0, 0, 0],
            "Stress": [125, 286, 286],
        }
    )
    base = SolverSettings(
        use_separate_sign=True,
        objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION,
        safety_factor=1.0,
        enforce_nonnegative_kt=True,
    )
    modes, result = find_minimal_unlink(
        df, base, max_underprediction_tol=-0.01, max_rms_error=1.0
    )
    unlinked = [
        FORCE_COLUMNS[i] for i in range(len(modes)) if modes[i] == SignMode.INDIVIDUAL
    ]
    assert len(modes) == 6
    assert all(m in (SignMode.LINKED, SignMode.INDIVIDUAL) for m in modes)
    assert result is not None
    assert result.success
    assert len(unlinked) <= 6
