from __future__ import annotations

from pathlib import Path

import pandas as pd

from kt_optimizer.models import SolverSettings, TABLE_COLUMNS
from kt_optimizer.solver import solve
from kt_optimizer.export_excel import export_to_excel


def test_excel_export_contains_canonical_kt_headers(tmp_path: Path):
    """Excel export should always contain canonical Fx+/Fx-/... headers when a solution exists."""
    # Simple data set that produces a valid solution.
    df = pd.DataFrame(
        [
            ["LC1", 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            ["LC2", -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        ],
        columns=TABLE_COLUMNS,
    )
    settings = SolverSettings()
    result = solve(df, settings)
    out_path = export_to_excel(df, result, settings, tmp_path / "kt_export.xlsx")

    assert out_path.exists()

    # Read back the sheet and verify the Kt header row contains Fx+.
    exported = pd.read_excel(out_path, sheet_name="Kt Export", header=None)
    # The Kt header row is after the load case table and two blank rows; easiest is
    # to just search for a row that contains "Fx+".
    assert (exported == "Fx+").any().any()

