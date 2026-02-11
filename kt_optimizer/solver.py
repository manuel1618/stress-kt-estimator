from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.optimize import linprog, minimize

from kt_optimizer.models import FORCE_COLUMNS, ObjectiveMode, SolverResult, SolverSettings, ValidationCase


def _build_force_matrix(df, use_separate_sign: bool):
    f = df[FORCE_COLUMNS].to_numpy(dtype=float)
    if not use_separate_sign:
        names = FORCE_COLUMNS.copy()
        return f, names
    f_pos = np.maximum(f, 0.0)
    f_neg = np.abs(np.minimum(f, 0.0))
    names = [f"{c}+" for c in FORCE_COLUMNS] + [f"{c}-" for c in FORCE_COLUMNS]
    return np.hstack([f_pos, f_neg]), names


def _build_bounds(n_vars: int, enforce_nonnegative: bool):
    if enforce_nonnegative:
        return [(0.0, None)] * n_vars
    return [(None, None)] * n_vars


def _solve_lp(f_mat, sigma, settings: SolverSettings):
    n = f_mat.shape[1]
    c = np.ones(n)
    res = linprog(
        c=c,
        A_ub=-f_mat,
        b_ub=-sigma,
        bounds=_build_bounds(n, settings.enforce_nonnegative_kt),
        method="highs",
    )
    return res.success, res.message, res.x if res.success else np.zeros(n)


def _solve_least_squares(f_mat, sigma, settings: SolverSettings):
    n = f_mat.shape[1]

    def objective(k):
        r = f_mat @ k - sigma
        penalty = np.where(r < 0, 1e5 * np.square(r), 0.0)
        return float(np.mean(np.square(r)) + np.mean(penalty))

    bounds = _build_bounds(n, settings.enforce_nonnegative_kt)
    x0 = np.ones(n)
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    feasible = np.all((f_mat @ res.x - sigma) >= -1e-6)
    success = bool(res.success and feasible)
    msg = res.message if success else "Least-squares mode could not find conservative solution"
    return success, str(msg), res.x


def _solve_min_max_deviation(f_mat, sigma, settings: SolverSettings):
    n = f_mat.shape[1]
    # Variables: [k..., t]
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
    return True, res.message, res.x[:-1]


def solve(df, settings: SolverSettings, logger: logging.Logger | None = None) -> SolverResult:
    logger = logger or logging.getLogger("kt_optimizer")
    logger.info("Parsed %d load cases", len(df))
    if df.empty:
        return SolverResult(success=False, message="No load cases provided")

    f_mat, kt_names = _build_force_matrix(df, settings.use_separate_sign)
    sigma = df["Stress"].to_numpy(dtype=float).copy()
    logger.info("Building force matrix (%dx%d)", f_mat.shape[0], f_mat.shape[1])

    if settings.safety_factor <= 0:
        return SolverResult(success=False, message="Safety factor must be > 0")
    sigma *= float(settings.safety_factor)

    if settings.use_separate_sign:
        logger.info("Using separate + / - formulation")

    if settings.objective_mode == ObjectiveMode.MIN_SUM_KT:
        logger.info("Solving LP (%d constraints, %d variables)", f_mat.shape[0], f_mat.shape[1])
        success, message, k = _solve_lp(f_mat, sigma, settings)
    elif settings.objective_mode == ObjectiveMode.LEAST_SQUARES_CONSERVATIVE:
        logger.info("Solving least-squares conservative optimization")
        success, message, k = _solve_least_squares(f_mat, sigma, settings)
    else:
        logger.info("Solving min-max deviation optimization")
        success, message, k = _solve_min_max_deviation(f_mat, sigma, settings)

    sigma_pred = f_mat @ k
    error = sigma_pred - sigma
    min_error = float(np.min(error))
    max_error = float(np.max(error))
    rms_error = float(np.sqrt(np.mean(np.square(error))))

    actual = sigma
    with np.errstate(divide="ignore", invalid="ignore"):
        margin_pct = np.where(actual != 0, (sigma_pred - actual) / np.abs(actual) * 100.0, np.nan)

    per_case = []
    for i, row in df.iterrows():
        mc = float(margin_pct[i]) if not np.isnan(margin_pct[i]) else 0.0
        per_case.append(
            ValidationCase(
                case_name=str(row["Case Name"]),
                actual=float(actual[i]),
                predicted=float(sigma_pred[i]),
                margin_pct=mc,
            )
        )
        logger.info(
            "Case %s | Actual: %.3f | Predicted: %.3f | Margin: %+.2f%%",
            row["Case Name"],
            float(actual[i]),
            float(sigma_pred[i]),
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

    return SolverResult(
        success=bool(success),
        message=str(message),
        kt_names=kt_names,
        kt_values=k.tolist(),
        sigma_target=sigma.tolist(),
        sigma_pred=sigma_pred.tolist(),
        min_error=min_error,
        max_error=max_error,
        rms_error=rms_error,
        worst_case_margin=float(np.min(margin_pct)) if len(margin_pct) else 0.0,
        max_overprediction=float(np.max(error)),
        max_underprediction=float(np.min(error)),
        condition_number=cond_number,
        sensitivity_violations=sensitivity_violations,
        diagnostics={"objective": settings.objective_mode.value},
        per_case=per_case,
    )
