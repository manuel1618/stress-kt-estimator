from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from kt_optimizer.models import FORCE_COLUMNS, TABLE_COLUMNS


def _normalize_loaded_csv(data: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for col in data.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in {"lc", "loadcase", "casename", "case", "id"}:
            renamed[col] = "Case Name"
        elif key in {"fx", "drag"}:
            renamed[col] = "Fx"
        elif key in {"fy", "side"}:
            renamed[col] = "Fy"
        elif key in {"fz", "vertical"}:
            renamed[col] = "Fz"
        elif key in {"mx", "roll", "-mx"}:
            renamed[col] = "Mx"
        elif key in {"my", "pitch", "-my"}:
            renamed[col] = "My"
        elif key in {"mz", "yaw", "-mz"}:
            renamed[col] = "Mz"
        elif key in {"stress", "sigma", "σ"}:
            renamed[col] = "Stress"

    normalized = data.rename(columns=renamed).copy()

    if "Case Name" not in normalized.columns:
        normalized["Case Name"] = [f"LC{i + 1}" for i in range(len(normalized))]

    for col in FORCE_COLUMNS + ["Stress"]:
        if col not in normalized.columns:
            normalized[col] = 0.0
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0)

    normalized["Case Name"] = normalized["Case Name"].astype(str)
    return normalized[TABLE_COLUMNS]


class LoadCaseTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self.df = pd.DataFrame(columns=TABLE_COLUMNS)

    def rowCount(self, parent=QModelIndex()) -> int:  # type: ignore[override]
        return len(self.df)

    def columnCount(self, parent=QModelIndex()) -> int:  # type: ignore[override]
        return len(self.df.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        value = self.df.iat[index.row(), index.column()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            return "" if pd.isna(value) else str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # type: ignore[override]
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.df.columns[section]
        return str(section + 1)

    def flags(self, index: QModelIndex):  # type: ignore[override]
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index: QModelIndex, value, role=Qt.EditRole):  # type: ignore[override]
        if role != Qt.EditRole or not index.isValid():
            return False
        col = self.df.columns[index.column()]
        if col != "Case Name":
            try:
                value = float(value)
            except ValueError:
                return False
        self.df.iat[index.row(), index.column()] = value
        self.dataChanged.emit(index, index, [Qt.DisplayRole])
        return True

    def add_row(self) -> None:
        self.beginInsertRows(QModelIndex(), len(self.df), len(self.df))
        self.df.loc[len(self.df)] = ["", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.endInsertRows()

    def remove_row(self, row: int) -> None:
        if row < 0 or row >= len(self.df):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        self.df = self.df.drop(self.df.index[row]).reset_index(drop=True)
        self.endRemoveRows()

    def load_csv(self, path: str) -> None:
        data = pd.read_csv(path, encoding="utf-8-sig")
        self.beginResetModel()
        self.df = _normalize_loaded_csv(data)
        self.endResetModel()

    def save_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)
