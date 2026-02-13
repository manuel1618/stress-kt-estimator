from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from kt_optimizer.models import SolverResult


class ResultPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.summary = QLabel("No results yet")
        # One row, one column per Kt direction (widescreen: fx | -fx | fy | ...)
        self.kt_table = QTableWidget(1, 0)
        self.kt_table.verticalHeader().setVisible(False)
        self.kt_table.setMaximumHeight(90)
        self.kt_table.setEditTriggers(QAbstractItemView.AllEditTriggers)

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)

        layout.addWidget(self.summary)
        layout.addWidget(self.kt_table)
        layout.addWidget(self.canvas)

    def update_result(self, result: SolverResult) -> None:
        summary_base = (
            f"Worst margin: {result.worst_case_margin:+.3f}% | "
            f"Max over: {result.max_overprediction:.4f} | "
            f"Max under: {result.max_underprediction:.4f}"
        )

        warnings: list[str] = []
        status = (
            result.diagnostics.get("constraint_status") if result.diagnostics else None
        )
        if status in {
            "under_constrained",
            "strongly_under_constrained",
            "over_constrained_or_infeasible",
        }:
            if status == "strongly_under_constrained":
                txt = "system strongly under-constrained"
            elif status == "under_constrained":
                txt = "system under-constrained"
            else:
                txt = "system over-constrained / possibly infeasible"
            warnings.append(txt)

        if result.diagnostics and result.diagnostics.get("signed_kt_inconsistent"):
            warnings.append(
                "signed-Kt interpretation may produce negative stress for some cases"
            )

        if warnings:
            warning_html = " | ".join(
                f'<span style="color:red;">WARNING: {w}</span>' for w in warnings
            )
            summary = f"{summary_base} | {warning_html}"
        else:
            summary = summary_base

        self.summary.setText(summary)
        self.kt_table.setColumnCount(len(result.kt_names))
        self.kt_table.setHorizontalHeaderLabels(result.kt_names)
        for col, value in enumerate(result.kt_values):
            self.kt_table.setItem(0, col, QTableWidgetItem(f"{value:.6f}"))

        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        x = np.arange(len(result.kt_names))
        ax1.bar(x, result.kt_values)
        ax1.set_xticks(x)
        ax1.set_xticklabels(result.kt_names, rotation=60, fontsize=7)
        ax1.set_title("Kt values")

        ax2.scatter(result.sigma_target, result.sigma_pred, s=20)
        if result.sigma_target:
            lo = min(result.sigma_target)
            hi = max(result.sigma_target)
            ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_title("Predicted vs actual")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def current_kt_values(self) -> list[float]:
        """Return current Kt values from the table (row 0, ordered by columns)."""
        values: list[float] = []
        n_cols = self.kt_table.columnCount()
        for col in range(n_cols):
            item = self.kt_table.item(0, col)
            text = item.text().strip() if item is not None and item.text() is not None else ""
            if not text:
                values.append(0.0)
            else:
                try:
                    values.append(float(text))
                except ValueError:
                    # Keep raw value as 0.0; caller should validate if needed.
                    values.append(0.0)
        return values

