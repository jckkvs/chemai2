import os
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import List, Optional
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("JWT_SECRET", "chemai-dev-secret")
ALGORITHM = "HS256"
security = HTTPBearer()

class TenantUser:
    def __init__(self, user_id: str, org_id: str, roles: List[str]):
        self.user_id = user_id
        self.org_id = org_id
        self.roles = roles

def create_tenant_token(user_id: str, org_id: str, roles: List[str]) -> str:
    return jwt.encode(
        {"sub": user_id, "org_id": org_id, "roles": roles, "exp": datetime.utcnow() + timedelta(hours=24)},
        SECRET_KEY, algorithm=ALGORITHM
    )

async def get_current_tenant_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TenantUser:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return TenantUser(
            user_id=payload["sub"],
            org_id=payload.get("org_id", "default"),
            roles=payload.get("roles", [])
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(*allowed_roles: str):
    async def role_checker(user: TenantUser = Depends(get_current_tenant_user)):
        if not any(role in user.roles for role in allowed_roles):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker
