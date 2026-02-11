from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kt_optimizer.models import FORCE_COLUMNS, SignMode, SolverResult, SolverSettings


def _sign_modes_table_html(settings: SolverSettings) -> str:
    """Per-direction sign mode as a table when separate +/- is enabled."""
    if not settings.use_separate_sign or not settings.sign_mode_per_component:
        return "<p class=\"muted\">Single signed Kt per direction (no +/- split).</p>"
    rows = []
    for i, comp in enumerate(FORCE_COLUMNS):
        if i < len(settings.sign_mode_per_component):
            m = settings.sign_mode_per_component[i]
            label = "Linked (+/− same magnitude, opposite sign)" if m == SignMode.LINKED else "Individual (separate + and − Kt)"
            rows.append(f"<tr><td>{comp}</td><td>{label}</td></tr>")
    return "<table class=\"report-table\"><thead><tr><th>Component</th><th>Sign mode</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def _settings_section_html(result: SolverResult, settings: SolverSettings) -> str:
    """Summary and settings as a structured section with optional per-direction table."""
    objective = getattr(settings.objective_mode, "value", settings.objective_mode)
    return f"""
<div class="settings-grid">
  <table class="report-table settings-table">
    <thead><tr><th>Setting</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Success</td><td>{result.success}</td></tr>
      <tr><td>Message</td><td>{result.message}</td></tr>
      <tr><td>Objective</td><td>{objective}</td></tr>
      <tr><td>Safety factor</td><td>{settings.safety_factor}</td></tr>
      <tr><td>Enforce Kt ≥ 0</td><td>{settings.enforce_nonnegative_kt}</td></tr>
      <tr><td>Separate +/− for directions</td><td>{settings.use_separate_sign}</td></tr>
    </tbody>
  </table>
</div>
<h3>Per-direction sign mode</h3>
{_sign_modes_table_html(settings)}
"""


def _kt_table_html(result: SolverResult) -> str:
    """Kt values as a styled table with component grouping and clear formatting."""
    if not result.kt_names or not result.kt_values:
        return "<p class=\"muted\">No Kt values.</p>"
    rows = []
    for name, value in zip(result.kt_names, result.kt_values):
        # Highlight +/- pairs for readability
        cell_class = ""
        if name.endswith("+") or name.endswith("-"):
            cell_class = ' class="kt-signed"'
        rows.append(f"<tr><td{cell_class}>{name}</td><td class=\"num\">{value:.6f}</td></tr>")
    return (
        "<table class=\"report-table kt-table\">"
        "<thead><tr><th>Kt</th><th>Value</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def _render_html(
    result: SolverResult, settings: SolverSettings, load_cases_html: str
) -> str:
    rows = "\n".join(
        f"<tr><td>{c.case_name}</td><td>{c.actual:.3f}</td><td>{c.predicted:.3f}</td><td>{c.margin_pct:+.2f}%</td></tr>"
        for c in result.per_case
    )
    kt_html = _kt_table_html(result)
    settings_html = _settings_section_html(result, settings)
    return f"""
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ color: #1a1a1a; border-bottom: 2px solid #333; padding-bottom: 6px; }}
h2 {{ color: #333; margin-top: 24px; }}
h3 {{ color: #444; font-size: 1.05em; margin-top: 16px; }}
.report-table {{ border-collapse: collapse; width: 100%; max-width: 640px; margin-bottom: 18px; }}
.report-table th, .report-table td {{ border: 1px solid #bbb; padding: 8px 12px; text-align: left; }}
.report-table th {{ background: #374151; color: #fff; font-weight: 600; }}
.report-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.report-table tbody tr:nth-child(even) {{ background: #f5f5f5; }}
.report-table tbody tr:hover {{ background: #eee; }}
.kt-table {{ max-width: 360px; }}
.kt-table .kt-signed {{ color: #055; }}
.settings-table {{ max-width: 480px; }}
.settings-grid {{ margin-bottom: 8px; }}
.muted {{ color: #666; font-size: 14px; }}
.load-cases-section table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
.load-cases-section th, .load-cases-section td {{ border: 1px solid #bbb; padding: 6px 10px; text-align: right; }}
.load-cases-section th {{ background: #374151; color: #fff; }}
.load-cases-section th:first-child, .load-cases-section td:first-child {{ text-align: left; }}
.load-cases-section tbody tr:nth-child(even) {{ background: #f9f9f9; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>Kt Optimizer Report</h1>
<p class="small">Generated: {datetime.utcnow().isoformat()}Z</p>
<h2>Summary &amp; settings</h2>
{settings_html}
<h2>Load cases</h2>
<div class="load-cases-section">{load_cases_html}</div>
<h2>Derived Kt values</h2>
{kt_html}
<h2>Validation results</h2>
<table class="report-table">
<tr><th>Case</th><th>Actual</th><th>Predicted</th><th>Margin %</th></tr>
{rows}
</table>
<h2>Optimization diagnostics</h2>
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
    html = _render_html(result, settings, load_cases_html)

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
