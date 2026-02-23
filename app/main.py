from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.__main__ import mk_routes
from app.utils.settings import Settings

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()

_APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Stress Kt Estimator starting up")
    yield
    logger.info("Stress Kt Estimator shutting down")


app = FastAPI(
    title="Stress Kt Estimator",
    version="0.1.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(_APP_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(_APP_DIR / "static")), name="static")

mk_routes(app, templates)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
