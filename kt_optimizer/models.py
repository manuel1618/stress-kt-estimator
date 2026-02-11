from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class ObjectiveMode(str, Enum):
    MINIMIZE_MAX_DEVIATION = "minimize_max_deviation"


@dataclass(slots=True)
class LoadCase:
    case_name: str
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float
    stress: float


@dataclass(slots=True)
class SolverSettings:
    use_separate_sign: bool = False
    objective_mode: ObjectiveMode = ObjectiveMode.MINIMIZE_MAX_DEVIATION
    safety_factor: float = 1.0
    enforce_nonnegative_kt: bool = True


@dataclass(slots=True)
class ValidationCase:
    case_name: str
    actual: float
    predicted: float
    margin_pct: float


@dataclass(slots=True)
class SolverResult:
    success: bool
    message: str
    kt_names: list[str] = field(default_factory=list)
    kt_values: list[float] = field(default_factory=list)
    sigma_target: list[float] = field(default_factory=list)
    sigma_pred: list[float] = field(default_factory=list)
    min_error: float = 0.0
    max_error: float = 0.0
    rms_error: float = 0.0
    worst_case_margin: float = 0.0
    max_overprediction: float = 0.0
    max_underprediction: float = 0.0
    condition_number: float = 0.0
    sensitivity_violations: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    per_case: list[ValidationCase] = field(default_factory=list)

    def to_kt_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"Kt": self.kt_names, "Value": self.kt_values})


FORCE_COLUMNS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
TABLE_COLUMNS = ["Case Name", *FORCE_COLUMNS, "Stress"]
