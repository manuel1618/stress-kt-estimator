# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conservative Kt (stress concentration factor) optimizer prototype. Fits Kt factors from mixed load cases using min-max linear programming so that predictions are always conservative (never underpredict stress) while minimizing worst-case overprediction. PySide6 desktop GUI, scipy linprog (HiGHS) solver backend.

## Commands

All commands use `uv` as the package manager. Taskfile.yml wraps common workflows:

```bash
uv sync                        # Install/sync dependencies
uv run python -m kt_optimizer.main  # Launch GUI
uv run pytest tests/ -v        # Run all tests
uv run pytest tests/test_solver.py::test_name -v  # Run single test
uv run ruff check .            # Lint
uv run ruff format .           # Format
uv run python -m PyInstaller kt_optimizer.spec --distpath dist --workpath build  # Build .exe
```

Or via task runner: `task gui`, `task test`, `task lint`, `task format`, `task build`.

## Architecture

```
kt_optimizer/
├── main.py           # Entry point: creates QApplication + MainWindow
├── models.py         # Dataclasses (LoadCase, SolverSettings, SolverResult, ValidationCase)
│                     #   Enums: ObjectiveMode, SignMode (LINKED vs INDIVIDUAL)
│                     #   CANONICAL_KT_ORDER: 12-entry display order (Fx+, Fx-, ... Mz-)
├── solver.py         # Core optimization: builds LP, calls scipy linprog, returns diagnostics
├── logger.py         # Python logging with optional GUI callback (GuiLogHandler)
├── export_excel.py   # Single-sheet XLSX export of settings, Kt values, validation
└── ui/
    ├── main_window.py   # Main UI: splitter layout, CSV I/O, settings panel, action buttons
    ├── result_panel.py  # Matplotlib charts (Kt bar chart, actual vs predicted scatter)
    ├── table_model.py   # QAbstractTableModel for editable load case table
    └── style.qss        # Qt stylesheet
```

**Data flow:** CSV input -> LoadCaseTableModel (editable table) -> SolverSettings configured by user -> `solver.solve()` (validate columns, build force matrix, linprog) -> SolverResult (Kt values, margins, diagnostics) -> ResultPanel (charts) / export_excel (XLSX).

**Key solver concepts:**
- Sign modes: LINKED (single Kt per component) vs INDIVIDUAL (separate +/- Kt per component)
- `suggest_unlink_from_data()`: heuristic to recommend which components need +/- split
- `find_minimal_unlink()`: brute-force search for minimal unlinking with acceptable fit
- Conservative constraint: predicted >= actual for all load cases
- CSV columns: Exact column names required (Case Name, Fx, Fy, Fz, Mx, My, Mz, Stress)

## Sign Conventions

**Stress Concentration Factors (Kt):**
- Always non-negative: Kt ≥ 0 (enforced by LP bounds)
- Represents geometric stress amplification factor
- Physical property independent of load direction

**Forces:**
- Signed values: F ∈ ℝ
- Positive = tension/+direction, Negative = compression/-direction

**Stresses:**
- Signed values: σ ∈ ℝ
- Sign matches physical behavior (tension +, compression -)

**LINKED Mode (Symmetric Behavior):**
- Single Kt per component applies to both tension and compression
- Relationship: σ = Kt × F (where both σ and F retain their signs)
- Display shows same Kt for Fx+ and Fx- (not negated)
- Example: Kt_Fx = 2.0 means F = +100 → σ = +200, F = -100 → σ = -200

**INDIVIDUAL Mode (Asymmetric Behavior):**
- Separate Kt+ and Kt- per component (both non-negative)
- Force matrix uses magnitudes for the negative side: f_pos = max(F, 0), f_neg = |min(F, 0)|
- Relationship: σ = Kt+ × max(F, 0) + Kt- × |min(F, 0)| (so negative F can contribute positive σ via Kt-)
- Allows asymmetric behavior and fits data where stress magnitude is reported (e.g. critical stress)
- Example: Kt_Fx+ = 2.0, Kt_Fx- = 3.0 means:
  - F = +100 → σ = 2.0 × 100 = 200
  - F = -100 → σ = 3.0 × 100 = 300 (same magnitude of force, different amplification)

**SET Mode:**
- User-specified fixed Kt values (not optimized)
- Same physics as INDIVIDUAL: σ = Kt+ × max(F, 0) + Kt- × |min(F, 0)|

## Conventions

- Python 3.10+, `from __future__ import annotations` in all modules
- Type hints throughout, dataclasses with `slots=True`
- Ruff for linting and formatting
- Tests use pytest with DataFrame fixtures and `tmp_path`
