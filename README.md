# Stress Kt Estimator

Conservative Kt optimizer – derives conservative stress concentration factors from mixed load cases using min-max linear programming optimization. Available as a **FastAPI web service** (primary) and an optional PySide6 desktop GUI.

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

## Dependency management (uv)

```bash
uv sync
```

## Run the web service

```bash
uv run python -m app.main
```

The server starts on `http://0.0.0.0:8000`. Interactive API docs are at `/docs` (Swagger) and `/redoc`.

### Environment variables

| Variable    | Default   | Description            |
|-------------|-----------|------------------------|
| `HOST`      | `0.0.0.0` | Bind address           |
| `PORT`      | `8000`    | Bind port              |
| `LOG_LEVEL` | `info`    | Uvicorn log level      |

## API endpoints

### `GET /health`

Health check. Returns `{"status": "ok"}`.

### `POST /api/solve`

Run the min-max optimization to derive Kt factors.

```json
{
  "load_cases": [
    {"case_name": "LC1", "fx": 100, "fy": 0, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": 100},
    {"case_name": "LC2", "fx": 0, "fy": 100, "fz": 0, "mx": 0, "my": 0, "mz": 0, "stress": 100}
  ],
  "settings": {
    "use_separate_sign": true,
    "sign_mode_per_component": null,
    "safety_factor": 1.0
  }
}
```

### `POST /api/recalc`

Recalculate predicted stresses with user-provided Kt values (12 canonical values in order Fx+, Fx-, Fy+, Fy-, Fz+, Fz-, Mx+, Mx-, My+, My-, Mz+, Mz-).

### `POST /api/suggest-unlink`

Data-driven suggestion of which force/moment components to set to Individual (separate +/-) mode. Requires at least 2 load cases.

### `POST /api/find-minimal-unlink`

Brute-force search for the sign-mode configuration with the fewest Individual directions that still produces an acceptable fit.

## Docker deployment

Build and run with Docker Compose:

```bash
cd docker
docker compose up --build -d
```

The service will be available on port 80 (mapped to container port 8000).

## Run the desktop GUI (optional)

Install the GUI extras first:

```bash
uv sync --extra gui
uv run python -m kt_optimizer.main
```

## Test

```bash
uv run pytest
```

## CSV input requirements

The CSV file (for the GUI) must contain exactly these column names (case-sensitive):

`Case Name, Fx, Fy, Fz, Mx, My, Mz, Stress`

All column names must match exactly. No aliases or alternative names are accepted.
