from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from kt_optimizer.models import ObjectiveMode, SignMode


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LoadCaseIn(BaseModel):
    case_name: str
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float
    stress: float


class SolverSettingsIn(BaseModel):
    use_separate_sign: bool = False
    sign_mode_per_component: list[SignMode] | None = None
    fixed_kt_values: list[tuple[float, float]] | None = None
    objective_mode: ObjectiveMode = ObjectiveMode.MINIMIZE_MAX_DEVIATION
    safety_factor: float = Field(default=1.0, gt=0)


class SolveRequest(BaseModel):
    load_cases: list[LoadCaseIn] = Field(..., min_length=1)
    settings: SolverSettingsIn = Field(default_factory=SolverSettingsIn)


class RecalcRequest(BaseModel):
    load_cases: list[LoadCaseIn] = Field(..., min_length=1)
    settings: SolverSettingsIn = Field(default_factory=SolverSettingsIn)
    kt_values: list[float] = Field(..., min_length=12, max_length=12)


class SuggestUnlinkRequest(BaseModel):
    load_cases: list[LoadCaseIn] = Field(..., min_length=2)


class FindMinimalUnlinkRequest(BaseModel):
    load_cases: list[LoadCaseIn] = Field(..., min_length=1)
    settings: SolverSettingsIn = Field(default_factory=SolverSettingsIn)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ValidationCaseOut(BaseModel):
    case_name: str
    actual: float
    predicted: float
    margin_pct: float


class SolverResultOut(BaseModel):
    success: bool
    message: str
    kt_names: list[str] = []
    kt_values: list[float] = []
    sigma_target: list[float] = []
    sigma_pred: list[float] = []
    min_error: float = 0.0
    max_error: float = 0.0
    rms_error: float = 0.0
    worst_case_margin: float = 0.0
    max_overprediction: float = 0.0
    max_underprediction: float = 0.0
    condition_number: float = 0.0
    sensitivity_violations: int = 0
    diagnostics: dict[str, Any] = {}
    per_case: list[ValidationCaseOut] = []


class SuggestUnlinkResponse(BaseModel):
    suggested_components: list[str]


class FindMinimalUnlinkResponse(BaseModel):
    sign_modes: list[SignMode]
    result: SolverResultOut | None


class PlotRequest(BaseModel):
    kt_names: list[str]
    kt_values: list[float]
    sigma_target: list[float] = []
    sigma_pred: list[float] = []
    per_case: list[ValidationCaseOut] = []


class PlotResponse(BaseModel):
    bar_chart: str
    scatter_chart: str


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def load_cases_to_dataframe(cases: list[LoadCaseIn]) -> pd.DataFrame:
    """Convert a list of LoadCaseIn Pydantic models into the DataFrame
    format expected by ``kt_optimizer.solver``."""
    rows = [
        {
            "Case Name": c.case_name,
            "Fx": c.fx,
            "Fy": c.fy,
            "Fz": c.fz,
            "Mx": c.mx,
            "My": c.my,
            "Mz": c.mz,
            "Stress": c.stress,
        }
        for c in cases
    ]
    return pd.DataFrame(rows)


def settings_in_to_dataclass(s: SolverSettingsIn):
    """Convert a Pydantic SolverSettingsIn to the dataclass used by the solver."""
    from kt_optimizer.models import SolverSettings

    return SolverSettings(
        use_separate_sign=s.use_separate_sign,
        sign_mode_per_component=s.sign_mode_per_component,
        fixed_kt_values=s.fixed_kt_values,
        objective_mode=s.objective_mode,
        safety_factor=s.safety_factor,
    )


def solver_result_to_out(r) -> SolverResultOut:
    """Convert a ``kt_optimizer.models.SolverResult`` dataclass to a Pydantic response."""
    return SolverResultOut(
        success=r.success,
        message=r.message,
        kt_names=r.kt_names,
        kt_values=r.kt_values,
        sigma_target=r.sigma_target,
        sigma_pred=r.sigma_pred,
        min_error=r.min_error,
        max_error=r.max_error,
        rms_error=r.rms_error,
        worst_case_margin=r.worst_case_margin,
        max_overprediction=r.max_overprediction,
        max_underprediction=r.max_underprediction,
        condition_number=r.condition_number,
        sensitivity_violations=r.sensitivity_violations,
        diagnostics=r.diagnostics,
        per_case=[
            ValidationCaseOut(
                case_name=vc.case_name,
                actual=vc.actual,
                predicted=vc.predicted,
                margin_pct=vc.margin_pct,
            )
            for vc in r.per_case
        ],
    )
