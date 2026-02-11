from __future__ import annotations

import pandas as pd

from kt_optimizer.models import ObjectiveMode, SolverSettings, TABLE_COLUMNS
from kt_optimizer.solver import solve


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
