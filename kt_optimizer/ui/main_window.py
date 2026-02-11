from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from kt_optimizer.logger import attach_gui_handler, build_logger
from kt_optimizer.models import ObjectiveMode, SolverSettings
from kt_optimizer.report import generate_report
from kt_optimizer.solver import solve
from kt_optimizer.ui.result_panel import ResultPanel
from kt_optimizer.ui.table_model import LoadCaseTableModel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kt Optimizer Prototype")
        self.logger = build_logger()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.table_model = LoadCaseTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)

        add_btn = QPushButton("Add Row")
        del_btn = QPushButton("Delete Row")
        load_btn = QPushButton("Import CSV")
        save_btn = QPushButton("Export CSV")

        top_buttons = QHBoxLayout()
        for b in [add_btn, del_btn, load_btn, save_btn]:
            top_buttons.addWidget(b)

        self.settings_panel = self._build_settings_panel()
        self.results_panel = ResultPanel()
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        attach_gui_handler(self.logger, self.log_panel.appendPlainText)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self.results_panel)
        right_split.addWidget(self.log_panel)

        mid_split = QSplitter(Qt.Horizontal)
        mid_split.addWidget(self.settings_panel)
        mid_split.addWidget(right_split)
        mid_split.setSizes([300, 700])

        actions = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        self.report_btn = QPushButton("Generate Report")
        actions.addWidget(self.report_btn)
        actions.addWidget(self.solve_btn)

        root_layout.addLayout(top_buttons)
        root_layout.addWidget(self.table)
        root_layout.addWidget(mid_split)
        root_layout.addLayout(actions)

        add_btn.clicked.connect(self.table_model.add_row)
        del_btn.clicked.connect(self._delete_selected)
        load_btn.clicked.connect(self._load_csv)
        save_btn.clicked.connect(self._save_csv)
        self.solve_btn.clicked.connect(self._solve)
        self.report_btn.clicked.connect(self._report)

        self.last_result = None

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.use_sign = QCheckBox("Use separate Kt for + / - direction")
        self.objective = QComboBox()
        self.objective.addItem("Minimize sum(Kt)", ObjectiveMode.MIN_SUM_KT)
        self.objective.addItem("Least squares conservative", ObjectiveMode.LEAST_SQUARES_CONSERVATIVE)
        self.objective.addItem("Minimize max deviation", ObjectiveMode.MINIMIZE_MAX_DEVIATION)

        self.safety_factor = QLineEdit("1.0")
        self.nonnegative = QCheckBox("Enforce Kt >= 0")
        self.nonnegative.setChecked(True)

        layout.addWidget(self.use_sign)
        layout.addWidget(QLabel("Objective mode"))
        layout.addWidget(self.objective)
        layout.addWidget(QLabel("Safety factor"))
        layout.addWidget(self.safety_factor)
        layout.addWidget(self.nonnegative)
        layout.addStretch(1)

        return panel

    def _settings(self) -> SolverSettings:
        return SolverSettings(
            use_separate_sign=self.use_sign.isChecked(),
            objective_mode=self.objective.currentData(),
            safety_factor=float(self.safety_factor.text()),
            enforce_nonnegative_kt=self.nonnegative.isChecked(),
        )

    def _delete_selected(self) -> None:
        idx = self.table.currentIndex()
        self.table_model.remove_row(idx.row())

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", str(Path.cwd()), "CSV (*.csv)")
        if path:
            self.table_model.load_csv(path)

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", str(Path.cwd() / "load_cases.csv"), "CSV (*.csv)")
        if path:
            self.table_model.save_csv(path)

    def _solve(self) -> None:
        try:
            settings = self._settings()
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Safety factor must be numeric")
            return

        result = solve(self.table_model.df, settings=settings, logger=self.logger)
        self.last_result = result
        self.results_panel.update_result(result)
        if not result.success:
            QMessageBox.warning(self, "Solve failed", result.message)

    def _report(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "No results", "Solve first before generating report")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Generate report", str(Path.cwd() / "kt_report.pdf"), "PDF (*.pdf);;HTML (*.html)")
        if not path:
            return
        out = generate_report(self.table_model.df, self.last_result, self._settings(), path)
        QMessageBox.information(self, "Report generated", f"Wrote report to {out}")
