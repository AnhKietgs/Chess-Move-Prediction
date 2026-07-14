"""CORS configuration, isolated so `main.py` stays a thin composition root."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings


def add_cors_middleware(app: FastAPI) -> None:
    """Attach CORS middleware to the app using origins from settings.

    Args:
        app: The FastAPI application instance to configure.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
