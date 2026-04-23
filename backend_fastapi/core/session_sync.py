"""
backend_fastapi/core/session_sync.py
JWT認証 & Redis ベースの NiceGUI/Next.js 間状態同期
"""
import os
import redis
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)
# Redis connection
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)

security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET", "chemai-dev-secret-change-in-prod")
ALGORITHM = "HS256"

class SessionManager:
    @staticmethod
    def create_token(user_id: str, session_id: Optional[str] = None) -> str:
        payload = {
            "sub": user_id,
            "session_id": session_id or os.urandom(16).hex(),
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        try:
            return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    @staticmethod
    async def sync_state(source: str, session_id: str, state_delta: Dict[str, Any]) -> Dict[str, Any]:
        """source: 'nicegui' | 'nextjs'"""
        key = f"session:{session_id}"
        current = redis_client.get(key)
        state = eval(current) if current else {}
        state.update({k: {"value": v, "_updated_at": datetime.utcnow().isoformat(), "_source": source} for k, v in state_delta.items()})
        redis_client.setex(key, 86400, str(state))
        return {"status": "synced", "keys": list(state_delta.keys())}

session_manager = SessionManager()
