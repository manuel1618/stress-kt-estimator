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


def test_min_sum_conservative_solution():
    df = _sample_df()
    settings = SolverSettings(use_separate_sign=True, objective_mode=ObjectiveMode.MIN_SUM_KT)
    result = solve(df, settings)
    assert result.success
    assert result.min_error >= -1e-6


def test_minmax_mode_solution():
    df = _sample_df()
    settings = SolverSettings(use_separate_sign=True, objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION)
    result = solve(df, settings)
    assert result.success
    assert result.min_error >= -1e-6


def test_bad_safety_factor_fails():
    df = _sample_df()
    settings = SolverSettings(safety_factor=0)
    result = solve(df, settings)
    assert not result.success
