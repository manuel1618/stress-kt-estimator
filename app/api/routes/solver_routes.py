from __future__ import annotations

import asyncio
import base64
import io
import logging
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from app.models.api_models import (
    ExportExcelRequest,
    FindMinimalUnlinkRequest,
    FindMinimalUnlinkResponse,
    PlotRequest,
    PlotResponse,
    RecalcRequest,
    SolveRequest,
    SolverResultOut,
    SuggestUnlinkRequest,
    SuggestUnlinkResponse,
    load_cases_to_dataframe,
    result_out_to_dataclass,
    settings_in_to_dataclass,
    solver_result_to_out,
)
from kt_optimizer.export_excel import export_to_excel
from kt_optimizer.solver import (
    find_minimal_unlink,
    recalc_with_fixed_kt,
    solve,
    suggest_unlink_from_data,
)

logger = logging.getLogger("kt_optimizer")


def mk_solver_routes(app: FastAPI) -> None:

    @app.post("/api/solve", response_model=SolverResultOut, tags=["solver"])
    async def api_solve(body: SolveRequest) -> SolverResultOut:
        df = load_cases_to_dataframe(body.load_cases)
        settings = settings_in_to_dataclass(body.settings)
        try:
            result = await asyncio.to_thread(solve, df, settings, logger)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return solver_result_to_out(result)

    @app.post("/api/recalc", response_model=SolverResultOut, tags=["solver"])
    async def api_recalc(body: RecalcRequest) -> SolverResultOut:
        df = load_cases_to_dataframe(body.load_cases)
        settings = settings_in_to_dataclass(body.settings)
        try:
            result = await asyncio.to_thread(
                recalc_with_fixed_kt, df, settings, body.kt_values, logger,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return solver_result_to_out(result)

    @app.post(
        "/api/suggest-unlink",
        response_model=SuggestUnlinkResponse,
        tags=["solver"],
    )
    async def api_suggest_unlink(body: SuggestUnlinkRequest) -> SuggestUnlinkResponse:
        df = load_cases_to_dataframe(body.load_cases)
        try:
            suggested = await asyncio.to_thread(suggest_unlink_from_data, df)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return SuggestUnlinkResponse(suggested_components=suggested)

    @app.post(
        "/api/find-minimal-unlink",
        response_model=FindMinimalUnlinkResponse,
        tags=["solver"],
    )
    async def api_find_minimal_unlink(
        body: FindMinimalUnlinkRequest,
    ) -> FindMinimalUnlinkResponse:
        df = load_cases_to_dataframe(body.load_cases)
        settings = settings_in_to_dataclass(body.settings)
        try:
            modes, result = await asyncio.to_thread(
                find_minimal_unlink, df, settings, logger=logger,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return FindMinimalUnlinkResponse(
            sign_modes=modes,
            result=solver_result_to_out(result) if result else None,
        )

    @app.post("/api/plot", response_model=PlotResponse, tags=["solver"])
    async def api_plot(body: PlotRequest) -> PlotResponse:
        bar_b64, scatter_b64 = await asyncio.to_thread(_render_plots, body)
        return PlotResponse(bar_chart=bar_b64, scatter_chart=scatter_b64)

    @app.post("/api/export-excel", tags=["solver"])
    async def api_export_excel(body: ExportExcelRequest):
        """Export load cases, settings, and solver result to an Excel file."""
        df = load_cases_to_dataframe(body.load_cases)
        settings = settings_in_to_dataclass(body.settings)
        result = result_out_to_dataclass(body.result)
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".xlsx", delete=False
            ) as tmp:
                path = tmp.name
            await asyncio.to_thread(
                export_to_excel, df, result, settings, path
            )
            content = pathlib.Path(path).read_bytes()
            try:
                pathlib.Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=kt_export.xlsx"
            },
        )


def _render_plots(body: PlotRequest) -> tuple[str, str]:
    """Render Kt bar chart and predicted-vs-actual scatter, return base64 PNGs."""
    bar_b64 = _render_bar_chart(body.kt_names, body.kt_values)
    scatter_b64 = _render_scatter_chart(body.sigma_target, body.sigma_pred, body.per_case)
    return bar_b64, scatter_b64


def _render_bar_chart(names: list[str], values: list[float]) -> str:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    x = np.arange(len(names))
    colors = ["#3b82f6" if v >= 0 else "#ef4444" for v in values]
    ax.bar(x, values, color=colors, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Kt value")
    ax.set_title("Kt values", fontsize=10)
    ax.axhline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _render_scatter_chart(sigma_target, sigma_pred, per_case) -> str:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    if sigma_target and sigma_pred:
        t = np.array(sigma_target)
        p = np.array(sigma_pred)
        ax.scatter(t, p, s=20, alpha=0.7, color="#3b82f6", edgecolors="#1d4ed8", linewidths=0.5)
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        margin = (hi - lo) * 0.05 or 1
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", linewidth=1, label="y = x")
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.legend(fontsize=7)
    ax.set_xlabel("Actual stress")
    ax.set_ylabel("Predicted stress")
    ax.set_title("Predicted vs Actual", fontsize=10)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
