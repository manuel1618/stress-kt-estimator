from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from app.api.routes.index_routes import mk_index_routes
from app.api.routes.solver_routes import mk_solver_routes


def mk_routes(app: FastAPI, templates: Jinja2Templates) -> None:
    mk_index_routes(app, templates)
    mk_solver_routes(app)
