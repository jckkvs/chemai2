# backend/main.py

"""
ChemAI ML Studio - Main FastAPI Application

Integrated platform for chemoinformatics, machine learning, and LLM assistance.
"""
from __future__ import annotations

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.utils.logger import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Import lifecycle manager
from backend.llm.lifecycle import lifecycle


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager with LLM and DB integration"""
    logger.info("🚀 ChemAI ML Studio starting...")
    
    # 1. Initialize database connections
    await init_db()
    
    # 2. Initialize task queue connections
    from backend.core.celery_app import celery_app
    celery_app.conf.update(
        broker_url=settings.REDIS_URL, 
        result_backend=settings.CELERY_RESULT_BACKEND
    )
    
    # 3. Load chemical plugins
    from backend.chem.plugins import DescriptorPluginRegistry
    DescriptorPluginRegistry()  # Auto-discovery
    
    # 4. Initialize LLM lifecycle
    async with lifecycle.lifespan():
        logger.info("✅ All systems operational.")
        yield
    
    # Shutdown
    logger.info("🛑 ChemAI ML Studio shutting down...")
    await close_db()
    logger.info("🔌 Database connections closed.")


# Create FastAPI application
app = FastAPI(
    title="ChemAI ML Studio",
    description="Integrated Chemoinformatics & Machine Learning Platform with LLM Assistant",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=app_lifespan,
)

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
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception [{request_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if settings.DEBUG else None
        }
    )

# ========== Register Routers ==========

# LLM Assistant Router (Prefix is in router: /api/v1/llm)
from backend.routers import llm as llm_router
app.include_router(llm_router.router)

# Existing routers
try:
    from backend.routers import health, data, ml, chem, export, ws
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    app.include_router(data.router, prefix="/api/v1/data", tags=["Data Management"])
    app.include_router(ml.router, prefix="/api/v1/ml", tags=["Machine Learning"])
    app.include_router(chem.router, prefix="/api/v1/chem", tags=["Chemical Descriptors"])
    app.include_router(export.router, prefix="/api/v1/export", tags=["Export"])
    app.include_router(ws.router, prefix="/api/v1/ws", tags=["WebSocket"])
except ImportError as e:
    logger.warning(f"Some routers not available: {e}")

# ========== Root Endpoint ==========

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "ChemAI ML Studio",
        "version": "2.0.0",
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc",
            "openapi": "/api/openapi.json"
        },
        "api_endpoints": {
            "health": "/api/v1/health",
            "llm_status": "/api/v1/llm/status",
            "llm_initialize": "/api/v1/llm/initialize",
            "data_upload": "/api/v1/data/upload",
            "ml_automl": "/api/v1/ml/automl"
        },
        "status": "operational"
    }


@app.get("/api/health", tags=["Health"])
async def api_health():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "chemai-api"}


# ========== Main Entry Point ==========

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ChemAI ML Studio backend...")
    
    host = getattr(settings, "HOST", "0.0.0.0")
    port = getattr(settings, "PORT", 8000)
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
        use_colors=True,
    )
