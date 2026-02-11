"""Export solver inputs, Kt results, and validation to a single Excel file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kt_optimizer.models import (
    FORCE_COLUMNS,
    SignMode,
    SolverResult,
    SolverSettings,
)


def _pad_row(row: list, width: int) -> list:
    return list(row) + [""] * (width - len(row))


def export_to_excel(
    load_cases_df: pd.DataFrame,
    result: SolverResult,
    settings: SolverSettings,
    out_path: str | Path,
) -> Path:
    """Write all tables to one Excel sheet (Summary, Load cases, Kt values, Validation)."""
    out_path = Path(out_path)
    out_path = out_path.with_suffix(".xlsx")

    objective = getattr(settings.objective_mode, "value", settings.objective_mode)
    rows: list[list] = [
        ["Setting", "Value"],
        ["Success", str(result.success)],
        ["Message", result.message],
        ["Objective", str(objective)],
        ["Safety factor", str(settings.safety_factor)],
        ["Enforce Kt ≥ 0", str(settings.enforce_nonnegative_kt)],
        ["Separate +/− for directions", str(settings.use_separate_sign)],
        [],
        ["Worst-case margin (%)", f"{result.worst_case_margin:.4f}"],
        ["Max overprediction", f"{result.max_overprediction:.6f}"],
        ["Max underprediction", f"{result.max_underprediction:.6f}"],
        ["RMS error", f"{result.rms_error:.6f}"],
        ["Condition number", f"{result.condition_number:.3e}"],
        ["Sensitivity violations", str(result.sensitivity_violations)],
    ]
    if settings.use_separate_sign and settings.sign_mode_per_component:
        rows.append([])
        rows.append(["Component", "Sign mode"])
        for i, comp in enumerate(FORCE_COLUMNS):
            if i < len(settings.sign_mode_per_component):
                m = settings.sign_mode_per_component[i]
                label = "Linked" if m == SignMode.LINKED else "Individual"
                rows.append([comp, label])

    rows.append([])
    rows.append([])
    rows.append(list(load_cases_df.columns))
    for _, r in load_cases_df.iterrows():
        rows.append(list(r))
    rows.append([])
    rows.append([])
    if result.kt_names and result.kt_values:
        rows.append(list(result.kt_names))
        rows.append([f"{v:.6f}" for v in result.kt_values])
        rows.append([])
        rows.append([])
    rows.append(["Case", "Actual", "Predicted", "Margin %"])
    for c in result.per_case:
        rows.append([c.case_name, c.actual, c.predicted, c.margin_pct])

    max_cols = max(len(r) for r in rows)
    padded = [_pad_row(r, max_cols) for r in rows]
    pd.DataFrame(padded).to_excel(
        out_path, sheet_name="Kt Export", index=False, header=False
    )
    return out_path
