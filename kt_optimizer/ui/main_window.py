from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from kt_optimizer.logger import attach_gui_handler, build_logger
from kt_optimizer.models import FORCE_COLUMNS, ObjectiveMode, SignMode, SolverSettings
from kt_optimizer.report import generate_report
from kt_optimizer.solver import find_minimal_unlink, suggest_unlink_from_data, solve
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
        self.table.setMinimumHeight(100)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        add_btn = QPushButton("Add Row")
        del_btn = QPushButton("Delete Row")
        load_btn = QPushButton("Import CSV")
        save_btn = QPushButton("Export CSV")

        top_buttons = QHBoxLayout()
        for b in [add_btn, del_btn, load_btn, save_btn]:
            top_buttons.addWidget(b)

        # Input matrix section: toolbar + table, in a frame so it’s one logical block
        input_section = QWidget()
        input_layout = QVBoxLayout(input_section)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addLayout(top_buttons)
        input_layout.addWidget(self.table)

        self.settings_panel = self._build_settings_panel()
        self.results_panel = ResultPanel()
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(180)
        attach_gui_handler(self.logger, self.log_panel.appendPlainText)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self.results_panel)
        right_split.addWidget(self.log_panel)
        right_split.setSizes([340, 180])

        mid_split = QSplitter(Qt.Horizontal)
        mid_split.addWidget(self.settings_panel)
        mid_split.addWidget(right_split)
        mid_split.setSizes([260, 500])

        actions = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        self.suggest_unlink_btn = QPushButton("Suggest unlink")
        self.report_btn = QPushButton("Generate Report")
        actions.addWidget(self.suggest_unlink_btn)
        actions.addWidget(self.report_btn)
        actions.addWidget(self.solve_btn)

        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(mid_split)
        bottom_layout.addLayout(actions)
        bottom_section.setMinimumHeight(260)

        # Main vertical splitter: input matrix gets more space and can be resized
        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(input_section)
        main_split.addWidget(bottom_section)
        main_split.setSizes([420, 380])  # table area larger by default
        main_split.setStretchFactor(
            0, 2
        )  # input section grows 2x when window is resized
        main_split.setStretchFactor(1, 1)
        main_split.setChildrenCollapsible(False)

        root_layout.addWidget(main_split)

        add_btn.clicked.connect(self.table_model.add_row)
        del_btn.clicked.connect(self._delete_selected)
        load_btn.clicked.connect(self._load_csv)
        save_btn.clicked.connect(self._save_csv)
        self.solve_btn.clicked.connect(self._solve)
        self.suggest_unlink_btn.clicked.connect(self._suggest_unlink)
        self.report_btn.clicked.connect(self._report)

        self.last_result = None

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.use_sign = QCheckBox("Use separate + / - for directions")
        self.use_sign.setChecked(True)
        self.use_sign.toggled.connect(self._on_use_sign_toggled)

        # Container for per-direction dropdowns; visible only when use_sign is checked
        self.sign_mode_widget = QWidget()
        sign_form = QFormLayout(self.sign_mode_widget)
        sign_form.setContentsMargins(0, 0, 0, 0)
        self.sign_mode_combo: dict[str, QComboBox] = {}
        for comp in FORCE_COLUMNS:
            combo = QComboBox()
            combo.addItem("Linked (+/− same magnitude, opposite sign)", SignMode.LINKED)
            combo.addItem("Individual (separate + and − Kt)", SignMode.INDIVIDUAL)
            combo.setCurrentIndex(0)
            self.sign_mode_combo[comp] = combo
            sign_form.addRow(f"{comp}:", combo)
        self.sign_mode_widget.setVisible(True)
        layout.addWidget(self.use_sign)
        layout.addWidget(self.sign_mode_widget)

        self.objective = QComboBox()
        self.objective.addItem(
            "Minimize max deviation (recommended)", ObjectiveMode.MINIMIZE_MAX_DEVIATION
        )
        self.objective.setEnabled(False)

        self.safety_factor = QLineEdit("1.0")
        self.nonnegative = QCheckBox("Enforce Kt >= 0")
        self.nonnegative.setChecked(True)

        layout.addWidget(QLabel("Objective mode"))
        layout.addWidget(self.objective)
        layout.addWidget(QLabel("Safety factor"))
        layout.addWidget(self.safety_factor)
        layout.addWidget(self.nonnegative)
        layout.addStretch(1)

        return panel

    def _on_use_sign_toggled(self, checked: bool) -> None:
        self.sign_mode_widget.setVisible(checked)

    def _settings(self) -> SolverSettings:
        data = self.objective.currentData()
        objective_mode = (
            data if isinstance(data, ObjectiveMode) else ObjectiveMode(data)
        )
        use_separate = self.use_sign.isChecked()
        sign_modes: list[SignMode] | None = None
        if use_separate:
            sign_modes = [
                self.sign_mode_combo[comp].currentData() or SignMode.LINKED
                for comp in FORCE_COLUMNS
            ]
        return SolverSettings(
            use_separate_sign=use_separate,
            sign_mode_per_component=sign_modes,
            objective_mode=objective_mode,
            safety_factor=float(self.safety_factor.text()),
            enforce_nonnegative_kt=self.nonnegative.isChecked(),
        )

    def _delete_selected(self) -> None:
        idx = self.table.currentIndex()
        self.table_model.remove_row(idx.row())

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", str(Path.cwd()), "CSV (*.csv)"
        )
        if path:
            self.table_model.load_csv(path)

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(Path.cwd() / "load_cases.csv"), "CSV (*.csv)"
        )
        if path:
            self.table_model.save_csv(path)

    def _suggest_unlink(self) -> None:
        try:
            base = self._settings()
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Safety factor must be numeric")
            return
        if self.table_model.df.empty:
            QMessageBox.warning(self, "No data", "Load or add load cases first.")
            return

        # Data-based hint (no solve)
        data_suggested = suggest_unlink_from_data(self.table_model.df)
        if data_suggested:
            self.logger.info(
                "Data suggests unlinking (different +/− behaviour): %s",
                ", ".join(data_suggested),
            )

        # Find minimal unlink set and apply
        modes, result = find_minimal_unlink(
            self.table_model.df,
            base,
            max_underprediction_tol=-0.01,
            logger=self.logger,
        )
        self.use_sign.setChecked(True)
        for i, comp in enumerate(FORCE_COLUMNS):
            combo = self.sign_mode_combo[comp]
            combo.setCurrentIndex(
                1 if (i < len(modes) and modes[i] == SignMode.INDIVIDUAL) else 0
            )

        unlinked = [
            FORCE_COLUMNS[i]
            for i in range(len(modes))
            if modes[i] == SignMode.INDIVIDUAL
        ]
        if result and result.success:
            self.last_result = result
            self.results_panel.update_result(result)
            QMessageBox.information(
                self,
                "Suggest unlink",
                f"Unlinked {len(unlinked)} direction(s): {', '.join(unlinked) or 'none'}.\n"
                f"Solve succeeded (max underprediction: {result.max_underprediction:.4f})."
                + (
                    f"\nData also suggested: {', '.join(data_suggested)}."
                    if data_suggested
                    else ""
                ),
            )
        else:
            msg = result.message if result else "No result"
            self.logger.warning("Minimal unlink could not meet tolerance: %s", msg)
            if result:
                self.last_result = result
                self.results_panel.update_result(result)
            QMessageBox.warning(
                self,
                "Suggest unlink",
                f"Applied unlink for: {', '.join(unlinked) or 'none'}.\n"
                f"Solve did not meet tolerance: {msg}",
            )

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
            QMessageBox.information(
                self, "No results", "Solve first before generating report"
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Generate report",
            str(Path.cwd() / "kt_report.pdf"),
            "PDF (*.pdf);;HTML (*.html)",
        )
        if not path:
            return
        try:
            out = generate_report(
                self.table_model.df, self.last_result, self._settings(), path
            )
        except Exception as exc:
            QMessageBox.critical(self, "Report generation failed", str(exc))
            return
        QMessageBox.information(self, "Report generated", f"Wrote report to {out}")
