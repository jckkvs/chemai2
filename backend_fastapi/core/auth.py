"""backend_fastapi/core/auth.py
JWT発行・検証 & Redisセッション同期基盤
"""
import os, jwt, redis, logging
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
security = HTTPBearer()
SECRET = os.getenv("JWT_SECRET", "chemai-dev-secret")
ALGO = "HS256"

def create_token(user_id: str, session_id: Optional[str] = None) -> str:
    return jwt.encode({
        "sub": user_id, 
        "sid": session_id or os.urandom(8).hex(), 
        "exp": datetime.utcnow() + timedelta(hours=24)
    }, SECRET, algorithm=ALGO)

def verify_token(cred: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    try:
        return jwt.decode(cred.credentials, SECRET, algorithms=[ALGO])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def sync_state(source: str, session_id: str, delta: Dict[str, Any]) -> Dict[str, Any]:
    key = f"session:{session_id}"
    # Use hset to store multiple fields in a hash
    # Note: Values must be converted to strings for Redis if they aren't already
    data_to_store = {
        k: f"{v}|{datetime.utcnow().isoformat()}|{source}" 
        for k, v in delta.items()
    }
    if data_to_store:
        redis_client.hset(key, mapping=data_to_store)
        redis_client.expire(key, 86400)
    return {"status": "synced", "keys": list(delta.keys())}
