"""Dependency injection utilities"""
from fastapi import Depends, HTTPException, status
from typing import Optional
from .main import get_session_backend, SessionBackend

async def require_session(
    session_id: str,
    backend: SessionBackend = Depends(get_session_backend)
) -> dict:
    """Require valid session"""
    session = backend.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

async def require_data_loaded(
    session: dict = Depends(require_session)
) -> dict:
    """Require data to be loaded in session"""
    if session.get("df") is None:
        raise HTTPException(status_code=400, detail="No data loaded in session")
    return session
