

from fastapi import FastAPI

from src.middleware.cors import add_cors_middleware
from src.routes import play

app = FastAPI(
    title="Fischer-Style Chess AI API",
    description="Backend for the Behavioral Cloning chess move prediction system.",
    version="0.1.0",
)

add_cors_middleware(app)

app.include_router(play.router)


@app.get("/api/health")
def health_check() -> dict:
    """Lightweight endpoint for Railway health checks and frontend connectivity tests."""
    return {"status": "ok"}
