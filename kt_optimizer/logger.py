from __future__ import annotations

import logging
from typing import Callable


class GuiLogHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        self.callback(self.format(record))


def build_logger(name: str = "kt_optimizer") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(stream)
    return logger


def attach_gui_handler(
    logger: logging.Logger, callback: Callable[[str], None]
) -> GuiLogHandler:
    handler = GuiLogHandler(callback)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return handler
