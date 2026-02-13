from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from kt_optimizer.export_excel import export_to_excel
from kt_optimizer.logger import attach_gui_handler, build_logger
from kt_optimizer.models import FORCE_COLUMNS, ObjectiveMode, SignMode, SolverSettings
from kt_optimizer.solver import (
    find_minimal_unlink,
    recalc_with_fixed_kt,
    solve,
    suggest_unlink_from_data,
)
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

        # Connect table model signals to update constraint status
        self.table_model.rowsInserted.connect(self._update_constraint_status)
        self.table_model.rowsRemoved.connect(self._update_constraint_status)
        self.table_model.modelReset.connect(self._update_constraint_status)

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
        self.recalc_btn = QPushButton("Recalc (manual Kt)")
        self.suggest_unlink_btn = QPushButton("Suggest unlink")
        self.export_excel_btn = QPushButton("Export Excel")
        actions.addWidget(self.suggest_unlink_btn)
        actions.addWidget(self.export_excel_btn)
        actions.addWidget(self.recalc_btn)
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
        self.recalc_btn.clicked.connect(self._recalc_fixed_kt)
        self.suggest_unlink_btn.clicked.connect(self._suggest_unlink)
        self.export_excel_btn.clicked.connect(self._export_excel)

        self.last_result = None

        # Initialize constraint status
        self._update_constraint_status()

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Constraint status indicator at the top
        self.constraint_status_widget = self._build_constraint_status_widget()
        layout.addWidget(self.constraint_status_widget)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        self.use_sign = QCheckBox("Use separate + / - for directions")
        self.use_sign.setChecked(True)
        self.use_sign.toggled.connect(self._on_use_sign_toggled)
        self.use_sign.toggled.connect(self._update_constraint_status)

        # Container for per-direction dropdowns; visible only when use_sign is checked
        self.sign_mode_widget = QWidget()
        sign_form = QFormLayout(self.sign_mode_widget)
        sign_form.setContentsMargins(0, 0, 0, 0)
        self.sign_mode_combo: dict[str, QComboBox] = {}
        self.fixed_kt_edits: dict[str, tuple[QLineEdit, QLineEdit]] = {}
        for comp in FORCE_COLUMNS:
            combo = QComboBox()
            combo.addItem("Linked (+/− same magnitude, opposite sign)", SignMode.LINKED)
            combo.addItem("Individual (separate + and − Kt)", SignMode.INDIVIDUAL)
            combo.addItem("Set (manual Kt values)", SignMode.SET)
            combo.setCurrentIndex(0)

            kt_plus_edit = QLineEdit("0")
            kt_minus_edit = QLineEdit("0")
            kt_plus_edit.setFixedWidth(50)
            kt_minus_edit.setFixedWidth(50)

            set_container = QWidget()
            set_layout = QHBoxLayout(set_container)
            set_layout.setContentsMargins(0, 0, 0, 0)
            set_layout.addWidget(QLabel("+:"))
            set_layout.addWidget(kt_plus_edit)
            set_layout.addWidget(QLabel("\u2212:"))
            set_layout.addWidget(kt_minus_edit)
            set_container.setVisible(False)

            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            row_layout.addWidget(combo)
            row_layout.addWidget(set_container)

            combo.currentIndexChanged.connect(
                lambda _idx, sc=set_container, cb=combo: sc.setVisible(
                    cb.currentData() == SignMode.SET
                )
            )
            combo.currentIndexChanged.connect(self._update_constraint_status)

            self.sign_mode_combo[comp] = combo
            self.fixed_kt_edits[comp] = (kt_plus_edit, kt_minus_edit)
            sign_form.addRow(f"{comp}:", row_widget)
        self.sign_mode_widget.setVisible(True)
        layout.addWidget(self.use_sign)
        layout.addWidget(self.sign_mode_widget)

        self.objective = QComboBox()
        self.objective.addItem(
            "Minimize max deviation (recommended)", ObjectiveMode.MINIMIZE_MAX_DEVIATION
        )
        self.objective.setEnabled(False)

        self.safety_factor = QLineEdit("1.0")

        layout.addWidget(QLabel("Objective mode"))
        layout.addWidget(self.objective)
        layout.addWidget(QLabel("Safety factor"))
        layout.addWidget(self.safety_factor)
        layout.addStretch(1)

        return panel

    def _on_use_sign_toggled(self, checked: bool) -> None:
        self.sign_mode_widget.setVisible(checked)

    def _build_constraint_status_widget(self) -> QWidget:
        """Build the live constraint status indicator."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Constraint Status")
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Status label with colored background
        self.constraint_status_label = QLabel("Unknown")
        self.constraint_status_label.setAlignment(Qt.AlignCenter)
        self.constraint_status_label.setStyleSheet(
            "padding: 6px; border-radius: 3px; background-color: #cccccc;"
        )
        layout.addWidget(self.constraint_status_label)

        # Details label
        self.constraint_details_label = QLabel("No load cases")
        self.constraint_details_label.setWordWrap(True)
        self.constraint_details_label.setStyleSheet("color: #666666; font-size: 9pt;")
        layout.addWidget(self.constraint_details_label)

        return widget

    def _count_design_variables(self) -> int:
        """Count the number of design variables based on current sign mode settings."""
        if not self.use_sign.isChecked():
            # All components use signed (linked) mode
            return len(FORCE_COLUMNS)

        count = 0
        for comp in FORCE_COLUMNS:
            mode = self.sign_mode_combo[comp].currentData()
            if mode == SignMode.LINKED:
                count += 1
            elif mode == SignMode.INDIVIDUAL:
                count += 2
            # SET adds 0 variables (user-specified)
        return count

    def _update_constraint_status(self) -> None:
        """Update the constraint status indicator based on current load cases and settings."""
        n_cases = len(self.table_model.df)
        n_vars = self._count_design_variables()

        # Determine constraint status
        if n_cases == 0:
            status_text = "No load cases"
            details = "Add or import load cases to begin"
            bg_color = "#cccccc"  # Gray
        elif n_cases < n_vars:
            status_text = "UNDERCONSTRAINED"
            details = f"{n_cases} equation(s), {n_vars} variable(s)\nNeed at least {n_vars} load cases"
            bg_color = "#ff4444"  # Red
        elif n_cases == n_vars:
            status_text = "Just determined"
            details = f"{n_cases} equation(s), {n_vars} variable(s)\nSystem is exactly determined"
            bg_color = "#ffaa00"  # Orange
        else:
            status_text = "Well constrained"
            details = f"{n_cases} equation(s), {n_vars} variable(s)\nSystem is overdetermined (good)"
            bg_color = "#44cc44"  # Green

        # Update UI
        self.constraint_status_label.setText(status_text)
        self.constraint_status_label.setStyleSheet(
            f"padding: 6px; border-radius: 3px; background-color: {bg_color}; "
            f"color: white; font-weight: bold;"
        )
        self.constraint_details_label.setText(details)

    def _settings(self) -> SolverSettings:
        data = self.objective.currentData()
        objective_mode = (
            data if isinstance(data, ObjectiveMode) else ObjectiveMode(data)
        )
        use_separate = self.use_sign.isChecked()
        sign_modes: list[SignMode] | None = None
        fixed_kt: list[tuple[float, float]] | None = None
        if use_separate:
            sign_modes = [
                self.sign_mode_combo[comp].currentData() or SignMode.LINKED
                for comp in FORCE_COLUMNS
            ]
            fixed_kt = []
            for comp in FORCE_COLUMNS:
                mode = self.sign_mode_combo[comp].currentData()
                if mode == SignMode.SET:
                    plus_edit, minus_edit = self.fixed_kt_edits[comp]
                    fixed_kt.append(
                        (float(plus_edit.text() or "0"), float(minus_edit.text() or "0"))
                    )
                else:
                    fixed_kt.append((0.0, 0.0))
        safety_text = (self.safety_factor.text() or "").strip()
        safety_factor = float(safety_text) if safety_text else 1.0
        return SolverSettings(
            use_separate_sign=use_separate,
            sign_mode_per_component=sign_modes,
            fixed_kt_values=fixed_kt,
            objective_mode=objective_mode,
            safety_factor=safety_factor,
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

    def _show_constraint_warning(self, result) -> None:
        """Show a prominent warning dialog when the solver reports constraint issues."""
        if not result or not getattr(result, "diagnostics", None):
            return

        status = result.diagnostics.get("constraint_status")
        if status not in {
            "under_constrained",
            "strongly_under_constrained",
            "over_constrained_or_infeasible",
        }:
            return

        if status == "strongly_under_constrained":
            title = "Strongly under-constrained system"
            body = (
                "The load case setup is strongly under-constrained:\n"
                "there are fewer independent load cases than Kt variables.\n\n"
                "Results may not be unique or reliable."
            )
        elif status == "under_constrained":
            title = "Under-constrained system"
            body = (
                "The load case setup is under-constrained:\n"
                "the force matrix rank is lower than the number of Kt variables.\n\n"
                "Results may not be unique or reliable."
            )
        else:
            title = "Over-constrained / infeasible system"
            body = (
                "The optimization indicates an over-constrained or possibly infeasible setup.\n\n"
                "Check your load cases and constraints; a consistent solution may not exist."
            )

        note = result.diagnostics.get("constraint_status_note")
        if note:
            body += f"\n\nDetails: {note}"

        QMessageBox.warning(self, title, body)

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
            # Preserve SET components — don't overwrite user's manual values
            if combo.currentData() == SignMode.SET:
                continue
            if i < len(modes) and modes[i] == SignMode.INDIVIDUAL:
                combo.setCurrentIndex(1)
            elif i < len(modes) and modes[i] == SignMode.SET:
                combo.setCurrentIndex(2)
            else:
                combo.setCurrentIndex(0)

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
            # If the underlying solve is under/over-constrained, surface that as well.
            self._show_constraint_warning(result)
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
            # Even when tolerance is not met, still show detailed constraint diagnostics.
            self._show_constraint_warning(result)

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
            return

        # Surface constraint issues very prominently so they are hard to miss.
        self._show_constraint_warning(result)

    def _recalc_fixed_kt(self) -> None:
        """Recalculate predicted stresses using the currently displayed Kt values.

        This does NOT run the optimizer; it only evaluates the model with the
        user-edited Kt table.
        """
        if self.table_model.df.empty:
            QMessageBox.warning(self, "No data", "Load or add load cases first.")
            return

        n_cols = self.results_panel.kt_table.columnCount()
        if n_cols == 0:
            QMessageBox.information(
                self,
                "No Kt values",
                "Solve once or enter Kt values in the Kt table before recalculating.",
            )
            return

        try:
            settings = self._settings()
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Safety factor must be numeric")
            return

        # Read Kt values from the first row of the Kt table.
        kt_values: list[float] = []
        for col in range(n_cols):
            item = self.results_panel.kt_table.item(0, col)
            text = item.text().strip() if item is not None and item.text() is not None else ""
            if not text:
                kt_values.append(0.0)
                continue
            try:
                kt_values.append(float(text))
            except ValueError:
                header_item = self.results_panel.kt_table.horizontalHeaderItem(col)
                name = header_item.text() if header_item is not None else f"col {col}"
                QMessageBox.critical(
                    self,
                    "Input Error",
                    f"Invalid Kt value for {name}: '{text}'",
                )
                return

        try:
            result = recalc_with_fixed_kt(
                self.table_model.df, settings=settings, kt_values_canonical=kt_values, logger=self.logger
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Recalc failed", str(exc))
            return

        self.last_result = result
        self.results_panel.update_result(result)

    def _export_excel(self) -> None:
        if self.last_result is None:
            QMessageBox.information(
                self, "No results", "Solve first before exporting Excel"
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Excel",
            str(Path.cwd() / "kt_export.xlsx"),
            "Excel (*.xlsx)",
        )
        if not path:
            return
        try:
            out = export_to_excel(
                self.table_model.df, self.last_result, self._settings(), path
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export done", f"Wrote {out}")
