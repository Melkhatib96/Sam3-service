"""
API Key authentication with per-user origin enforcement.
Mirrors the pattern used by all services in this stack.
"""
import logging
from typing import Optional
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.core.cache import api_key_cache
from app.core.config import settings
from app.db.database import get_db
from app.db.models import User

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user_from_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    db: Session = Depends(get_db),
) -> User:
    """
    Validate the X-API-Key header against the shared PostgreSQL users table.

    Origin enforcement:
    - If the user has custom ``allowed_origins`` set → origin must match that list.
    - Otherwise → origin must be in the static ``CORS_ORIGINS`` env var.
    - Requests with no Origin/Referer header (curl, server-to-server) are always allowed.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Cache lookup ───────────────────────────────────────────────────────────
    cached = api_key_cache.get(api_key)
    if cached:
        user = User()
        user.id = cached["id"]
        user.username = cached["username"]
        user.api_key = cached["api_key"]
        user.is_admin = cached["is_admin"]
        user.allowed_origins = cached["allowed_origins"]
    else:
        user = db.query(User).filter(User.api_key == api_key).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        api_key_cache.set(
            api_key,
            {
                "id": user.id,
                "username": user.username,
                "api_key": user.api_key,
                "is_admin": user.is_admin,
                "allowed_origins": user.allowed_origins,
            },
        )

    # ── Origin enforcement ─────────────────────────────────────────────────────
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin and origin.startswith("http"):
        parsed = urlparse(origin)
        origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin:
        if user.allowed_origins:
            allowed = [o.strip() for o in user.allowed_origins.split(",") if o.strip()]
            if origin not in allowed:
                logger.warning(
                    "User %s blocked from origin %s (custom list)", user.username, origin
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Origin '{origin}' is not allowed for this API key.",
                )
        else:
            if origin not in settings.cors_origins_list:
                logger.warning(
                    "User %s blocked from origin %s (static list)", user.username, origin
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Origin '{origin}' is not allowed.",
                )

    return user
