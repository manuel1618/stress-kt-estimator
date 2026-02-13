from __future__ import annotations

import pandas as pd

from kt_optimizer.models import (
    CANONICAL_KT_ORDER,
    FORCE_COLUMNS,
    TABLE_COLUMNS,
    ObjectiveMode,
    SignMode,
    SolverSettings,
)
from kt_optimizer.solver import (
    find_minimal_unlink,
    recalculate_with_kt,
    solve,
    suggest_unlink_from_data,
)


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


def test_negative_kt_values_propagate_when_unconstrained():
    """Canonical Kt slots allow stress reconstruction via Kt_slot × |F|.

    With negative Fx and positive stress the LINKED solver finds k = -1.0.
    Canonical conversion: Fx+ = k = -1.0, Fx- = -k = 1.0.
    Reconstruction: Fx+ × |F| gives negative stress (correct for positive F),
    Fx- × |F| gives positive stress (correct for negative F).
    """
    df = pd.DataFrame(
        [
            ["LC1", -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ["LC2", -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=False,
        objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION,
        safety_factor=1.0,
    )
    result = solve(df, settings)
    assert result.success
    assert "Fx+" in result.kt_names and "Fx-" in result.kt_names
    fx_plus_idx = result.kt_names.index("Fx+")
    fx_minus_idx = result.kt_names.index("Fx-")
    fx_plus_val = result.kt_values[fx_plus_idx]
    fx_minus_val = result.kt_values[fx_minus_idx]
    # LINKED k = -1: Fx+ = -1 (negative), Fx- = +1 (positive).
    # Reconstruction: for F < 0 (the only cases here), use Fx- × |F|:
    #   LC1: 1.0 × 1.0 = 1.0 ✓,  LC2: 1.0 × 2.0 = 2.0 ✓
    assert fx_plus_val < 0.0  # k is negative
    assert fx_minus_val > 0.0  # -k is positive
    assert abs(fx_plus_val - (-1.0)) < 1e-6
    assert abs(fx_minus_val - 1.0) < 1e-6


def test_constraint_status_strongly_under_constrained():
    """Very few load cases relative to variables should be flagged as strongly under-constrained."""
    # Default settings: 6 Kt variables (Fx..Mz), but only 2 load cases here.
    df = pd.DataFrame(
        [
            ["LC1", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ["LC2", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=False,
        objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION,
        safety_factor=1.0,
    )
    result = solve(df, settings)
    assert result.diagnostics.get("constraint_status") == "strongly_under_constrained"


def test_set_mode_excludes_design_variables():
    """SET components should not count as design variables."""
    df = _sample_df()
    # All linked: 6 design vars
    settings_linked = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=[SignMode.LINKED] * 6,
    )
    result_linked = solve(df, settings_linked)
    assert result_linked.diagnostics["n_kt_variables"] == 6

    # Set Fx and Fy, rest linked: 4 design vars
    settings_set = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=[
            SignMode.SET, SignMode.SET,
            SignMode.LINKED, SignMode.LINKED, SignMode.LINKED, SignMode.LINKED,
        ],
        fixed_kt_values=[(1.0, 0.8), (1.0, 0.6), (0, 0), (0, 0), (0, 0), (0, 0)],
    )
    result_set = solve(df, settings_set)
    assert result_set.diagnostics["n_kt_variables"] == 4


def test_set_mode_fixed_values_contribute_to_prediction():
    """SET Kt values should contribute to the predicted stress."""
    # Two cases: only Fx active. Set Fx with Kt+=2.0, Kt-=1.5
    df = pd.DataFrame(
        [
            ["LC1", 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0],
            ["LC2", -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0],
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=[SignMode.SET] * 6,
        fixed_kt_values=[(2.0, 1.5), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    )
    result = solve(df, settings)
    assert result.success
    # Predicted: LC1 = 10*2.0 = 20, LC2 = 10*1.5 = 15
    assert abs(result.sigma_pred[0] - 20.0) < 1e-6
    assert abs(result.sigma_pred[1] - 15.0) < 1e-6


def test_set_mode_zero_values_deactivate():
    """SET with both values 0 should deactivate the component (no contribution)."""
    df = pd.DataFrame(
        [
            ["LC1", 100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            ["LC2", 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        ],
        columns=TABLE_COLUMNS,
    )
    # SET Fx to (0,0) -> Fx contributes nothing; only Fy is a design variable.
    settings = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=[
            SignMode.SET, SignMode.LINKED,
            SignMode.LINKED, SignMode.LINKED, SignMode.LINKED, SignMode.LINKED,
        ],
        fixed_kt_values=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    )
    result = solve(df, settings)
    assert result.success
    assert result.diagnostics["n_kt_variables"] == 5


def test_set_mode_constraint_status_reflects_fewer_variables():
    """With SET reducing variable count, constraint status should improve."""
    # 2 load cases with all 6 components linked -> strongly under-constrained.
    # Setting 4 components to SET -> 2 vars, 2 cases -> well_determined.
    df = pd.DataFrame(
        [
            ["LC1", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ["LC2", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=[
            SignMode.LINKED, SignMode.LINKED,
            SignMode.SET, SignMode.SET, SignMode.SET, SignMode.SET,
        ],
        fixed_kt_values=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    )
    result = solve(df, settings)
    assert result.diagnostics["n_kt_variables"] == 2
    assert result.diagnostics["constraint_status"] == "well_determined"


def test_recalculate_with_kt_conservative():
    """Test recalculate_with_kt with conservative Kt values."""
    df = _sample_df()
    settings = SolverSettings(use_separate_sign=True)

    # First solve to get optimized Kt values
    original_result = solve(df, settings)
    assert original_result.success

    # Recalculate with the same Kt values - should be conservative
    recalc_result = recalculate_with_kt(
        df,
        original_result.kt_names,
        original_result.kt_values,
        settings
    )
    assert recalc_result.success
    assert recalc_result.max_underprediction >= -1e-6

    # Predictions should match
    for i in range(len(original_result.sigma_pred)):
        assert abs(original_result.sigma_pred[i] - recalc_result.sigma_pred[i]) < 1e-3


def test_recalculate_with_kt_non_conservative():
    """Test recalculate_with_kt with non-conservative Kt values."""
    df = _sample_df()
    settings = SolverSettings(use_separate_sign=True)

    # Use artificially low Kt values that will underpredict
    low_kt_values = [0.1] * 12
    kt_names = list(CANONICAL_KT_ORDER)

    recalc_result = recalculate_with_kt(df, kt_names, low_kt_values, settings)

    # Should detect non-conservative predictions
    assert not recalc_result.success
    assert recalc_result.max_underprediction < -1e-6
    assert "NON-CONSERVATIVE" in recalc_result.message
