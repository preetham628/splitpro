"""
Google OAuth + JWT helpers for SplitPro.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request
from jose import JWTError, jwt

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_INFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

_jwt_secret: Optional[str] = None
_jwt_algorithm: str = "HS256"
_jwt_expire_minutes: int = 10080  # 7 days
_google_client_id: str = ""
_google_client_secret: str = ""
_google_redirect_uri: str = "http://localhost:8000/auth/google/callback"


def configure(
    jwt_secret: str,
    jwt_algorithm: str,
    jwt_expire_minutes: int,
    google_client_id: str,
    google_client_secret: str,
    google_redirect_uri: str,
) -> None:
    """Called once at startup from server.py with values from AppConfig + env vars."""
    global _jwt_secret, _jwt_algorithm, _jwt_expire_minutes
    global _google_client_id, _google_client_secret, _google_redirect_uri

    if not jwt_secret:
        raise ValueError("JWT_SECRET env var must be set to a secure random value.")
    if not google_client_id:
        raise ValueError("GOOGLE_CLIENT_ID env var is required for OAuth.")
    if not google_client_secret:
        raise ValueError("GOOGLE_CLIENT_SECRET env var is required for OAuth.")

    _jwt_secret = jwt_secret
    _jwt_algorithm = jwt_algorithm
    _jwt_expire_minutes = jwt_expire_minutes
    _google_client_id = google_client_id
    _google_client_secret = google_client_secret
    _google_redirect_uri = google_redirect_uri


# ---------- Google OAuth ----------

def build_google_auth_url(state: str) -> str:
    params = {
        "client_id": _google_client_id,
        "redirect_uri": _google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    """Exchange OAuth code for access token. Returns token response dict."""
    response = httpx.post(GOOGLE_TOKEN_URL, data={
        "code": code,
        "client_id": _google_client_id,
        "client_secret": _google_client_secret,
        "redirect_uri": _google_redirect_uri,
        "grant_type": "authorization_code",
    })
    response.raise_for_status()
    return response.json()


def get_google_user_info(access_token: str) -> dict:
    """Fetch user profile from Google. Returns dict with id, email, name, picture."""
    response = httpx.get(
        GOOGLE_INFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    response.raise_for_status()
    return response.json()


# ---------- JWT ----------

def create_jwt(user_id: int) -> str:
    if not _jwt_secret:
        raise RuntimeError("Auth not configured — call configure() at startup.")
    expire = datetime.now(timezone.utc) + timedelta(minutes=_jwt_expire_minutes)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, _jwt_secret, algorithm=_jwt_algorithm)


def verify_jwt(token: str) -> Optional[dict]:
    if not _jwt_secret:
        return None
    try:
        return jwt.decode(token, _jwt_secret, algorithms=[_jwt_algorithm])
    except JWTError:
        return None


# ---------- FastAPI dependency ----------

def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency. Reads the 'access_token' httpOnly cookie,
    verifies the JWT, and returns {'id': int, ...} or raises 401.
    """
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"id": int(payload["sub"])}
