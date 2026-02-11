from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from kt_optimizer.models import (
    FORCE_COLUMNS,
    SignMode,
    SolverResult,
    SolverSettings,
    TABLE_COLUMNS,
    ValidationCase,
    expand_kt_to_canonical,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept common variants from spreadsheet exports and map to canonical columns."""
    work = df.copy()
    renamed: dict[str, str] = {}

    for col in work.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in {"lc", "loadcase", "casename", "case", "id"}:
            renamed[col] = "Case Name"
        elif key in {"fx", "drag"}:
            renamed[col] = "Fx"
        elif key in {"fy", "side"}:
            renamed[col] = "Fy"
        elif key in {"fz", "vertical"}:
            renamed[col] = "Fz"
        elif key in {"mx", "roll", "-mx"}:
            renamed[col] = "Mx"
        elif key in {"my", "pitch", "-my"}:
            renamed[col] = "My"
        elif key in {"mz", "yaw", "-mz"}:
            renamed[col] = "Mz"
        elif key in {"stress", "sigma", "σ"}:
            renamed[col] = "Stress"

    work = work.rename(columns=renamed)

    if "Case Name" not in work.columns:
        work["Case Name"] = [f"LC{i + 1}" for i in range(len(work))]

    for col in FORCE_COLUMNS + ["Stress"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["Case Name"] = work["Case Name"].astype(str)
    work = work.dropna(subset=["Stress"]).reset_index(drop=True)
    return work[TABLE_COLUMNS]


def _build_force_matrix(df: pd.DataFrame, settings: SolverSettings):
    f = df[FORCE_COLUMNS].to_numpy(dtype=float)
    use_separate = settings.use_separate_sign
    modes = settings.sign_mode_per_component

    if not use_separate:
        return f.copy(), list(FORCE_COLUMNS)

    # When no per-component modes given, default to all individual (backward compat)
    if modes is None or len(modes) != 6:
        modes = [SignMode.INDIVIDUAL] * 6

    # Per-component: linked => one signed column; individual => two columns (+, -)
    cols_list: list[np.ndarray] = []
    names: list[str] = []
    for i, comp in enumerate(FORCE_COLUMNS):
        mode = modes[i] if i < len(modes) else SignMode.LINKED
        if mode == SignMode.LINKED:
            cols_list.append(f[:, i : i + 1])
            names.append(comp)
        else:
            f_pos = np.maximum(f[:, i], 0.0).reshape(-1, 1)
            f_neg = np.abs(np.minimum(f[:, i], 0.0)).reshape(-1, 1)
            cols_list.append(f_pos)
            cols_list.append(f_neg)
            names.append(f"{comp}+")
            names.append(f"{comp}-")
    return np.hstack(cols_list), names


def _build_bounds(n_vars: int, enforce_nonnegative: bool):
    if enforce_nonnegative:
        return [(0.0, None)] * n_vars
    return [(None, None)] * n_vars


def _solve_min_max_deviation(f_mat, sigma, settings: SolverSettings):
    n = f_mat.shape[1]
    c = np.concatenate([np.zeros(n), [1.0]])

    # Conservative constraint: f*k >= sigma -> -f*k <= -sigma
    A1 = np.hstack([-f_mat, np.zeros((f_mat.shape[0], 1))])
    b1 = -sigma

    # Upper deviation: f*k - sigma <= t
    A2 = np.hstack([f_mat, -np.ones((f_mat.shape[0], 1))])
    b2 = sigma

    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([b1, b2])
    bounds = _build_bounds(n, settings.enforce_nonnegative_kt) + [(0.0, None)]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return False, res.message, np.zeros(n)
    return True, str(res.message), res.x[:-1]


def suggest_unlink_from_data(
    df: pd.DataFrame,
    *,
    ratio_threshold: float = 2.0,
    dominant_frac: float = 0.5,
) -> list[str]:
    """Suggest which components to unlink based on data alone (no solve).

    For each component, if both positive and negative forces appear, we check whether
    the implied Kt (stress/force when this component dominates) is consistent between
    + and -. If the ratio of effective Kt differs beyond ratio_threshold, we suggest
    unlinking that direction.

    Returns list of component names (e.g. ['Fy']) that are good candidates to set
    to Individual (+/- separate).
    """
    normalized = _normalize_columns(df)
    if len(normalized) < 2:
        return []

    f = normalized[FORCE_COLUMNS].to_numpy(dtype=float)
    stress = normalized["Stress"].to_numpy(dtype=float)
    # Per row: which component has largest magnitude?
    abs_f = np.abs(f)
    max_per_row = np.max(abs_f, axis=1, keepdims=True)
    suggested: list[str] = []

    for i, comp in enumerate(FORCE_COLUMNS):
        col = f[:, i]
        # Use rows where this component is dominant (at least dominant_frac of max force)
        dominant = (abs_f[:, i] >= dominant_frac * max_per_row.ravel()) & (
            abs_f[:, i] > 1e-10
        )
        pos_mask = dominant & (col > 1e-10)
        neg_mask = dominant & (col < -1e-10)

        if not np.any(pos_mask) or not np.any(neg_mask):
            continue

        # Linked model: stress = Kt * F => Kt = stress / F (signed)
        k_pos = np.mean(stress[pos_mask] / col[pos_mask])
        k_neg = np.mean(stress[neg_mask] / col[neg_mask])

        # Same sign and similar magnitude => linking is plausible
        denom = min(abs(k_pos), abs(k_neg))
        if denom < 1e-10:
            # One side near zero => different effective stiffness
            suggested.append(comp)
            continue
        ratio = max(abs(k_pos), abs(k_neg)) / denom
        if ratio > ratio_threshold:
            suggested.append(comp)
        elif (k_pos > 0) != (k_neg > 0):
            suggested.append(comp)
    return suggested


def find_minimal_unlink(
    df: pd.DataFrame,
    base_settings: SolverSettings,
    *,
    max_underprediction_tol: float = -0.01,
    max_rms_error: float | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[SignMode], SolverResult | None]:
    """Find a sign_mode_per_component with fewest Individual directions that still
    gives a successful solve and acceptable fit.

    Uses brute force: try 0 unlinks (all linked), then all 6 single-unlink, then
    all 15 pairs, etc. Stops at the first configuration that succeeds, has
    max_underprediction >= max_underprediction_tol, and (if max_rms_error is set)
    rms_error <= max_rms_error.

    If max_rms_error is None, it is set to 1% of mean stress in the data so that
    "all linked" with a poor fit (e.g. huge RMS) is rejected.

    Returns (sign_mode_per_component, result). If no good config found, returns
    (all INDIVIDUAL, last result) so the UI can still show something.
    """
    logger = logger or logging.getLogger("kt_optimizer")
    normalized = _normalize_columns(df)
    if normalized.empty:
        return ([SignMode.INDIVIDUAL] * len(FORCE_COLUMNS), None)
    mean_stress = float(normalized["Stress"].mean())
    rms_threshold = max_rms_error
    if rms_threshold is None:
        rms_threshold = max(0.01 * mean_stress, 1e-6)

    settings = SolverSettings(
        use_separate_sign=True,
        sign_mode_per_component=None,
        objective_mode=base_settings.objective_mode,
        safety_factor=base_settings.safety_factor,
        enforce_nonnegative_kt=base_settings.enforce_nonnegative_kt,
    )
    n = len(FORCE_COLUMNS)
    best_result: SolverResult | None = None
    best_modes: list[SignMode] | None = None

    def _acceptable(r: SolverResult) -> bool:
        if not r.success:
            return False
        if r.max_underprediction < max_underprediction_tol:
            return False
        if r.rms_error > rms_threshold:
            return False
        return True

    for k in range(n + 1):
        for indices in itertools.combinations(range(n), k):
            modes = [
                SignMode.INDIVIDUAL if i in indices else SignMode.LINKED
                for i in range(n)
            ]
            settings.sign_mode_per_component = modes
            result = solve(df, settings=settings, logger=logger)
            if _acceptable(result):
                best_modes = modes
                best_result = result
                logger.info(
                    "Minimal unlink: %d direction(s) unlinked -> %s",
                    k,
                    [FORCE_COLUMNS[i] for i in indices],
                )
                return (best_modes, best_result)
            # Fallback: keep successful result with fewest unlinks
            if result.success and (
                best_result is None
                or k < sum(1 for m in (best_modes or []) if m == SignMode.INDIVIDUAL)
            ):
                best_result = result
                best_modes = modes

    return (best_modes or [SignMode.INDIVIDUAL] * n, best_result)


def solve(
    df: pd.DataFrame, settings: SolverSettings, logger: logging.Logger | None = None
) -> SolverResult:
    logger = logger or logging.getLogger("kt_optimizer")
    normalized = _normalize_columns(df)

    logger.info("Parsed %d load cases", len(normalized))
    if normalized.empty:
        return SolverResult(
            success=False, message="No valid load cases with Stress provided"
        )

    f_mat, kt_names = _build_force_matrix(normalized, settings)
    sigma = normalized["Stress"].to_numpy(dtype=float).copy()
    logger.info("Building force matrix (%dx%d)", f_mat.shape[0], f_mat.shape[1])

    if settings.safety_factor <= 0:
        return SolverResult(success=False, message="Safety factor must be > 0")
    sigma *= float(settings.safety_factor)

    if settings.use_separate_sign and settings.sign_mode_per_component:
        individual = [
            FORCE_COLUMNS[i]
            for i, m in enumerate(settings.sign_mode_per_component)
            if i < len(settings.sign_mode_per_component) and m == SignMode.INDIVIDUAL
        ]
        if individual:
            logger.info("Individual + / - for: %s", ", ".join(individual))
        else:
            logger.info("Linked + / - (signed) for all directions")

    logger.info(
        "Solving min-max deviation LP (%d constraints, %d variables)",
        f_mat.shape[0],
        f_mat.shape[1] + 1,
    )
    success, message, k = _solve_min_max_deviation(f_mat, sigma, settings)

    sigma_pred = f_mat @ k
    error = sigma_pred - sigma
    min_error = float(np.min(error))
    max_error = float(np.max(error))
    rms_error = float(np.sqrt(np.mean(np.square(error))))

    with np.errstate(divide="ignore", invalid="ignore"):
        margin_pct = np.where(
            sigma != 0, (sigma_pred - sigma) / np.abs(sigma) * 100.0, np.nan
        )

    per_case: list[ValidationCase] = []
    for row_idx, row in normalized.iterrows():
        mc = float(margin_pct[row_idx]) if not np.isnan(margin_pct[row_idx]) else 0.0
        per_case.append(
            ValidationCase(
                case_name=str(row["Case Name"]),
                actual=float(sigma[row_idx]),
                predicted=float(sigma_pred[row_idx]),
                margin_pct=mc,
            )
        )
        logger.info(
            "Case %s | Actual: %.3f | Predicted: %.3f | Margin: %+.2f%%",
            row["Case Name"],
            float(sigma[row_idx]),
            float(sigma_pred[row_idx]),
            mc,
        )

    cond_number = float(np.linalg.cond(f_mat)) if f_mat.size else 0.0
    if cond_number > 1e6:
        logger.warning("Force matrix condition number is high: %.3e", cond_number)

    sensitivity_violations = 0
    for j in range(len(k)):
        k2 = k.copy()
        k2[j] *= 0.99
        if np.any((f_mat @ k2 - sigma) < 0):
            sensitivity_violations += 1

    if success and min_error >= -1e-6:
        logger.info("All load cases satisfied conservatively")
    elif success:
        logger.warning("Optimization succeeded but conservative check failed")
    else:
        logger.error("Optimization failed: %s", message)

    kt_names_canonical, kt_values_canonical = expand_kt_to_canonical(
        kt_names, k.tolist()
    )
    return SolverResult(
        success=bool(success),
        message=str(message),
        kt_names=kt_names_canonical,
        kt_values=kt_values_canonical,
        sigma_target=sigma.tolist(),
        sigma_pred=sigma_pred.tolist(),
        min_error=min_error,
        max_error=max_error,
        rms_error=rms_error,
        worst_case_margin=float(np.nanmin(margin_pct)) if len(margin_pct) else 0.0,
        max_overprediction=float(np.max(error)),
        max_underprediction=float(np.min(error)),
        condition_number=cond_number,
        sensitivity_violations=sensitivity_violations,
        diagnostics={
            "objective": getattr(
                settings.objective_mode, "value", settings.objective_mode
            )
        },
        per_case=per_case,
    )
