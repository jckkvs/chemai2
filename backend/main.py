"""
Application Entrypoint - chemai2/backend/main.py
Production-ready FastAPI application with lifecycle management
"""
import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Note: opentelemetry may need to be installed or mocked
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.core.security import setup_security
from backend.routers import health, data, ml, chem, export, ws
from backend.utils.logger import setup_logger

# Initialize structured logging
setup_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle manager"""
    logger.info("🚀 ChemAI ML Studio Backend starting...")
    
    # Initialize database connections
    await init_db()
    
    # Initialize task queue connections
    from backend.core.celery_app import celery_app
    celery_app.conf.update(broker_url=settings.REDIS_URL, result_backend=settings.CELERY_RESULT_BACKEND)
    
    # Load chemical plugins
    from backend.chem.plugins import DescriptorPluginRegistry
    DescriptorPluginRegistry()  # Auto-discovery
    
    logger.info("✅ All systems initialized. Ready for requests.")
    yield
    
    logger.info("🛑 Shutting down gracefully...")
    await close_db()
    logger.info("🔌 Database connections closed.")


# Create FastAPI app
app = FastAPI(
    title="ChemAI ML Studio API",
    description="Integrated Chemoinformatics & Machine Learning Platform",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Instrumentation (OpenTelemetry)
if HAS_OTEL and settings.DEBUG:
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Task-ID"],
)

# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(id(request))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Ensure request_id is available even if middleware failed or was skipped
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception [{request_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request_id}
    )

# Register Routers
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(data.router, prefix="/api/v1/data", tags=["Data Management"])
app.include_router(chem.router, prefix="/api/v1/chem", tags=["Chemical Descriptors"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(export.router, prefix="/api/v1/export", tags=["Export"])
app.include_router(ws.router, prefix="/api/v1/ws", tags=["WebSocket"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "ChemAI ML Studio",
        "version": "2.0.0",
        "docs": "/api/docs",
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run expects settings to have HOST and PORT
    host = getattr(settings, "HOST", "0.0.0.0")
    port = getattr(settings, "PORT", 8000)
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        access_log=not settings.DEBUG,
        use_colors=True
    )
