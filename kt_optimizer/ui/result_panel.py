from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kt_optimizer.models import SolverResult


class ResultPanel(QWidget):
    recalc_requested = Signal(list, list)  # kt_names, kt_values

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.summary = QLabel("No results yet")

        # Kt table header with recalc button
        kt_header = QHBoxLayout()
        kt_label = QLabel("Kt Values (editable):")
        self.recalc_btn = QPushButton("Recalculate Deviations")
        self.recalc_btn.setEnabled(False)
        kt_header.addWidget(kt_label)
        kt_header.addWidget(self.recalc_btn)
        kt_header.addStretch()

        # One row, one column per Kt direction (widescreen: fx | -fx | fy | ...)
        self.kt_table = QTableWidget(1, 0)
        self.kt_table.verticalHeader().setVisible(False)
        self.kt_table.setMaximumHeight(90)

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)

        layout.addWidget(self.summary)
        layout.addLayout(kt_header)
        layout.addWidget(self.kt_table)
        layout.addWidget(self.canvas)

        self.recalc_btn.clicked.connect(self._on_recalc_clicked)
        self.current_result = None

    def _on_recalc_clicked(self) -> None:
        """Emit signal to recalculate with user-modified Kt values."""
        kt_names = []
        kt_values = []
        for col in range(self.kt_table.columnCount()):
            kt_names.append(self.kt_table.horizontalHeaderItem(col).text())
            try:
                value = float(self.kt_table.item(0, col).text())
                kt_values.append(value)
            except (ValueError, AttributeError):
                kt_values.append(0.0)
        self.recalc_requested.emit(kt_names, kt_values)

    def update_result(self, result: SolverResult) -> None:
        self.current_result = result
        self.recalc_btn.setEnabled(True)

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

        # Check for non-conservative predictions
        if result.max_underprediction < -1e-6:
            warnings.append(
                f"NON-CONSERVATIVE: underpredicting by {abs(result.max_underprediction):.4f}"
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
            item = QTableWidgetItem(f"{value:.6f}")
            self.kt_table.setItem(0, col, item)

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
