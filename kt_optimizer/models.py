from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class ObjectiveMode(str, Enum):
    MINIMIZE_MAX_DEVIATION = "minimize_max_deviation"


class SignMode(str, Enum):
    """Per-direction +/- handling when separate + / - is enabled."""

    LINKED = (
        "linked"  # one Kt: + and - same magnitude, opposite sign (signed force column)
    )
    INDIVIDUAL = "individual"  # separate Kt for + and - (two design vars per direction)


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
    """When True, per-direction dropdowns apply: linked (signed) or individual (+/- separate)."""
    sign_mode_per_component: list[SignMode] | None = None
    """Length 6, order Fx,Fy,Fz,Mx,My,Mz. Used only when use_separate_sign is True."""
    objective_mode: ObjectiveMode = ObjectiveMode.MINIMIZE_MAX_DEVIATION
    safety_factor: float = 1.0


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

# Display order: + then - for Fx, Fy, Fz, then same for Mx, My, Mz
CANONICAL_KT_ORDER = [
    "Fx+",
    "Fx-",
    "Fy+",
    "Fy-",
    "Fz+",
    "Fz-",
    "Mx+",
    "Mx-",
    "My+",
    "My-",
    "Mz+",
    "Mz-",
]


def expand_kt_to_canonical(
    kt_names: list[str], kt_values: list[float]
) -> tuple[list[str], list[float]]:
    """Return Kt names and values in canonical order, always 12 entries.

    For display/export, we treat Kt as a magnitude per direction and encode the sign
    convention into the +/- slots:

    - ``Fx+`` / ``Fy+`` / ... are always non-negative (``abs`` of the underlying Kt).
    - ``Fx-`` / ``Fy-`` / ... are always non-positive (``-abs`` of the underlying Kt).

    The underlying solver variables may be signed or unsigned depending on the model,
    but this helper ensures a consistent visual convention where negative directions
    always show negative Kt values.
    """
    name_to_val = dict(zip(kt_names, kt_values))
    values_out: list[float] = []
    for name in CANONICAL_KT_ORDER:
        if name in name_to_val:
            raw = name_to_val[name]
        else:
            base = name[:-1]  # "Fx+" -> "Fx", "Fx-" -> "Fx"
            raw = name_to_val[base]

        if name.endswith("+"):
            values_out.append(abs(raw))
        elif name.endswith("-"):
            values_out.append(-abs(raw))
        else:
            values_out.append(raw)
    return list(CANONICAL_KT_ORDER), values_out
