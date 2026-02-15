from __future__ import annotations

import pandas as pd

from kt_optimizer.models import (
    FORCE_COLUMNS,
    TABLE_COLUMNS,
    ObjectiveMode,
    SignMode,
    SolverSettings,
)
from kt_optimizer.solver import find_minimal_unlink, solve, suggest_unlink_from_data


def _sample_df():
    data = [
        ["LC1", 100, 0, 0, 0, 0, 0, 100],
        ["LC2", 0, 100, 0, 0, 0, 0, 100],
        ["LC3", 40, 40, 0, 0, 0, 0, 90],
        ["LC4", -100, 0, 0, 0, 0, 0, -80],  # Fixed: negative force → negative stress
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


def test_exact_column_names_required():
    """Verify that exact column names are required (no aliasing)."""
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

    # Should raise ValueError due to missing required columns
    try:
        result = solve(df, SolverSettings(use_separate_sign=True))
        assert False, "Expected ValueError for missing columns"
    except ValueError as e:
        assert "Missing required columns" in str(e)


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


def test_linked_mode_same_kt_for_both_signs():
    """LINKED mode: Kt_Fx+ should equal Kt_Fx-, both non-negative.

    Physics: σ = Kt × F where Kt ≥ 0 is a geometric property.
    Stress sign comes from force sign, not from Kt sign.
    """
    df = pd.DataFrame(
        [
            ["LC1", 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0],  # Tension: Kt ≈ 2
            ["LC2", -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, -200.0],  # Compression: Kt ≈ 2
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=False,
        objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION,
        safety_factor=1.0,
    )
    result = solve(df, settings)
    assert result.success, f"Solver failed: {result.message}"

    # Find Fx+ and Fx- in canonical results
    assert "Fx+" in result.kt_names and "Fx-" in result.kt_names
    fx_plus_idx = result.kt_names.index("Fx+")
    fx_minus_idx = result.kt_names.index("Fx-")
    fx_plus_val = result.kt_values[fx_plus_idx]
    fx_minus_val = result.kt_values[fx_minus_idx]

    # LINKED mode: same Kt for both directions
    assert abs(fx_plus_val - fx_minus_val) < 1e-6, (
        f"LINKED mode should have same Kt: Fx+={fx_plus_val}, Fx-={fx_minus_val}"
    )

    # Both should be non-negative
    assert fx_plus_val >= -1e-6, f"Kt should be non-negative: Fx+={fx_plus_val}"
    assert fx_minus_val >= -1e-6, f"Kt should be non-negative: Fx-={fx_minus_val}"

    # Expected value around 2.0 (200 / 100)
    assert abs(fx_plus_val - 2.0) < 0.1, f"Expected Kt ≈ 2.0, got {fx_plus_val}"

    # Verify stress predictions preserve signs
    assert result.sigma_pred[0] > 0, "Positive force should give positive stress"
    assert result.sigma_pred[1] < 0, "Negative force should give negative stress"

    # Conservative constraint satisfied
    assert result.min_error >= -1e-6, "Conservative constraint violated"


def test_individual_mode_asymmetric_kt():
    """INDIVIDUAL mode: Kt+ and Kt- can differ, both non-negative.

    Tests asymmetric behavior; negative side uses signed value so Kt- × F gives
    negative stress (e.g. Kt- ≈ 3 when F=-100 and stress=-300).
    """
    df = pd.DataFrame(
        [
            ["LC1", 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0],  # Tension: Kt+ ≈ 2
            [
                "LC2",
                -100.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -300.0,
            ],  # Compression: F=-100, stress=-300 → Kt- ≈ 3
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=[SignMode.INDIVIDUAL] + [SignMode.LINKED] * 5,
        objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION,
        safety_factor=1.0,
    )
    result = solve(df, settings)
    assert result.success, f"Solver failed: {result.message}"

    # Find Fx+ and Fx- in results
    assert "Fx+" in result.kt_names and "Fx-" in result.kt_names
    fx_plus_idx = result.kt_names.index("Fx+")
    fx_minus_idx = result.kt_names.index("Fx-")
    fx_plus_val = result.kt_values[fx_plus_idx]
    fx_minus_val = result.kt_values[fx_minus_idx]

    # INDIVIDUAL mode: can have different values
    assert abs(fx_plus_val - fx_minus_val) > 0.5, (
        "Expected asymmetric Kt values for different load directions"
    )

    # Both should be non-negative
    assert fx_plus_val >= -1e-6, f"Kt+ should be non-negative: {fx_plus_val}"
    assert fx_minus_val >= -1e-6, f"Kt- should be non-negative: {fx_minus_val}"

    # Expected values: Kt+ ≈ 2.0, Kt- ≈ 3.0
    assert abs(fx_plus_val - 2.0) < 0.1, f"Expected Kt+ ≈ 2.0, got {fx_plus_val}"
    assert abs(fx_minus_val - 3.0) < 0.1, f"Expected Kt- ≈ 3.0, got {fx_minus_val}"

    # Predicted: LC1 = 2*100 = 200, LC2 = 3*(-100) = -300 (signed convention)
    assert result.sigma_pred[0] > 0, "Positive force should give positive stress"
    assert result.sigma_pred[1] < 0, "Negative force should give negative stress"

    # Conservative constraint satisfied
    assert result.min_error >= -1e-6, "Conservative constraint violated"


def test_kt_nonnegativity_enforced():
    """Verify solver enforces Kt ≥ 0 for all modes.

    Cases where force and stress have opposite signs should be rejected as
    physically inconsistent (would require negative Kt).
    """
    # Pathological case: negative force, positive stress (impossible with Kt ≥ 0)
    df = pd.DataFrame(
        [
            [
                "LC1",
                -100.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                200.0,
            ],  # σ = Kt × F → 200 = Kt × (-100) → Kt = -2 ✗
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings(
        use_separate_sign=False,
        objective_mode=ObjectiveMode.MINIMIZE_MAX_DEVIATION,
        safety_factor=1.0,
    )
    result = solve(df, settings)

    # Solver should either:
    # 1. Fail to find solution (infeasible), OR
    # 2. Find solution with Kt ≥ 0 but violate conservative constraint
    if result.success:
        # If it succeeded, all Kt values must be non-negative
        for kt_val in result.kt_values:
            assert kt_val >= -1e-6, f"Kt should be non-negative, got {kt_val}"
        # But it should violate the conservative constraint (min_error < 0)
        assert result.min_error < -1e-3, (
            "Should not satisfy conservative constraint with Kt ≥ 0 for inconsistent data"
        )
    else:
        # Expected: infeasible
        assert (
            "infeasible" in result.message.lower() or "failed" in result.message.lower()
        )


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
            SignMode.SET,
            SignMode.SET,
            SignMode.LINKED,
            SignMode.LINKED,
            SignMode.LINKED,
            SignMode.LINKED,
        ],
        fixed_kt_values=[(1.0, 0.8), (1.0, 0.6), (0, 0), (0, 0), (0, 0), (0, 0)],
    )
    result_set = solve(df, settings_set)
    assert result_set.diagnostics["n_kt_variables"] == 4


def test_set_mode_fixed_values_contribute_to_prediction():
    """SET Kt values should contribute to the predicted stress.

    Signed convention: σ = Kt+×max(F,0) + Kt-×min(F,0).
    """
    df = pd.DataFrame(
        [
            ["LC1", 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0],  # Kt+×10 = 20 → 2.0
            ["LC2", -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, -15.0],  # Kt-×(-10) = -15 → 1.5
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
    # Predicted: LC1 = 2.0×10 = 20, LC2 = 1.5×(−10) = -15
    assert abs(result.sigma_pred[0] - 20.0) < 1e-6
    assert abs(result.sigma_pred[1] - (-15.0)) < 1e-6


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
            SignMode.SET,
            SignMode.LINKED,
            SignMode.LINKED,
            SignMode.LINKED,
            SignMode.LINKED,
            SignMode.LINKED,
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
            SignMode.LINKED,
            SignMode.LINKED,
            SignMode.SET,
            SignMode.SET,
            SignMode.SET,
            SignMode.SET,
        ],
        fixed_kt_values=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    )
    result = solve(df, settings)
    assert result.diagnostics["n_kt_variables"] == 2
    assert result.diagnostics["constraint_status"] == "well_determined"
