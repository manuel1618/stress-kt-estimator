from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from kt_optimizer.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1300, 800)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
