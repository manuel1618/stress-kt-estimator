from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kt_optimizer.models import FORCE_COLUMNS, SignMode, SolverResult, SolverSettings


def _format_sign_modes(settings: SolverSettings) -> str:
    if not settings.use_separate_sign or not settings.sign_mode_per_component:
        return "N/A (single signed per direction)"
    parts = []
    for i, comp in enumerate(FORCE_COLUMNS):
        if i < len(settings.sign_mode_per_component):
            m = settings.sign_mode_per_component[i]
            parts.append(f"{comp}={m.value}")
    return ", ".join(parts) if parts else "N/A"


def _render_html(
    result: SolverResult, settings: SolverSettings, load_cases_html: str, kt_html: str
) -> str:
    rows = "\n".join(
        f"<tr><td>{c.case_name}</td><td>{c.actual:.3f}</td><td>{c.predicted:.3f}</td><td>{c.margin_pct:+.2f}%</td></tr>"
        for c in result.per_case
    )
    return f"""
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1, h2 {{ color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>Kt Optimizer Report</h1>
<p class="small">Generated: {datetime.utcnow().isoformat()}Z</p>
<h2>Summary</h2>
<ul>
<li>Success: {result.success}</li>
<li>Message: {result.message}</li>
<li>Objective: {getattr(settings.objective_mode, 'value', settings.objective_mode)}</li>
<li>Safety factor: {settings.safety_factor}</li>
<li>Separate +/- mode: {settings.use_separate_sign}</li>
<li>Per-direction: {_format_sign_modes(settings)}</li>
</ul>
<h2>Load Cases</h2>
{load_cases_html}
<h2>Derived Kt Values</h2>
{kt_html}
<h2>Validation Results</h2>
<table>
<tr><th>Case</th><th>Actual</th><th>Predicted</th><th>Margin %</th></tr>
{rows}
</table>
<h2>Optimization Diagnostics</h2>
<ul>
<li>Worst-case margin (%): {result.worst_case_margin:.3f}</li>
<li>Max overprediction: {result.max_overprediction:.6f}</li>
<li>Max underprediction: {result.max_underprediction:.6f}</li>
<li>RMS error: {result.rms_error:.6f}</li>
<li>Condition number: {result.condition_number:.3e}</li>
<li>Sensitivity violations: {result.sensitivity_violations}</li>
</ul>
</body>
</html>
"""


def generate_report(
    load_cases_df, result: SolverResult, settings: SolverSettings, out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    load_cases_html = load_cases_df.to_html(index=False)
    kt_html = result.to_kt_dataframe().to_html(index=False)
    html = _render_html(result, settings, load_cases_html, kt_html)

    if out_path.suffix.lower() == ".html":
        out_path.write_text(html, encoding="utf-8")
        return out_path

    if out_path.suffix.lower() == ".pdf":
        from xhtml2pdf import pisa

        with open(out_path, "wb") as dest:
            pisa_status = pisa.CreatePDF(
                html.encode("utf-8"), dest=dest, encoding="utf-8"
            )
        if pisa_status.err:
            raise RuntimeError("PDF generation failed (xhtml2pdf reported errors)")
        return out_path

    raise ValueError("Output path must end with .html or .pdf")
