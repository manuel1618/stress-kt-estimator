# Stress Kt Estimator

Prototype internal tool to derive conservative Kt factors from mixed load cases using optimization.

## Min-max optimization approach

The solver uses a **min-max (minimize maximum deviation)** linear program to fit Kt factors so that predictions are conservative and the worst-case overprediction is minimized.

- **Model**: Stress at the hot spot is approximated as a linear combination of load components:  
  \( \sigma \approx F_x k_{Fx} + F_y k_{Fy} + F_z k_{Fz} + M_x k_{Mx} + M_y k_{My} + M_z k_{Mz} \)  
  (with optional split into separate +/− coefficients per component when sign matters).

- **Conservative constraint**: For every load case, predicted stress must be **≥** actual stress:  
  \( F_i \cdot k \geq \sigma_i \).  
  So the fitted Kt set never underpredicts; designs based on it remain on the safe side.

- **Objective**: Among all such conservative \(k\), minimize the **maximum overprediction** over load cases.  
  That is, minimize \(t\) subject to \(F_i \cdot k - \sigma_i \leq t\) for all \(i\).  
  So the solution is as tight as possible while staying conservative.

- **Implementation**: Formulated as a linear program (variables: Kt coefficients and the scalar \(t\)), solved with `scipy.optimize.linprog` (HiGHS). Optional constraints can enforce non-negative Kt and a safety factor on the target stress.

Only this min-max conservative solver mode is exposed in the GUI.

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
