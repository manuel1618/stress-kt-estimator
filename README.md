# Stress Kt Estimator

Prototype internal tool to derive conservative Kt factors from mixed load cases using optimization.

## Dependency management (uv)

```bash
uv sync
```

## Run

```bash
uv run python -m kt_optimizer.main
```

## Test

```bash
uv run pytest
```

## CSV input notes

The app accepts either canonical columns:

`Case Name, Fx, Fy, Fz, Mx, My, Mz, Stress`

or common aliases from spreadsheets, including:

- `LC` / `Case` / `LoadCase` → `Case Name`
- `Drag` → `Fx`
- `Side` → `Fy`
- `Vertical` → `Fz`
- `-Mz` → `Mz`

Only the min-max conservative solver mode is exposed in the GUI.
