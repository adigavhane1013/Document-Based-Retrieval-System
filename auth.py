"""
auth.py

Authentication via Firebase (email/password on the frontend,
ID token verification on the backend).

Design:
  - The browser (docmind_ui.html) signs users up/in directly against
    Firebase Auth using the Firebase JS SDK — the backend never sees
    or stores passwords.
  - Every request from the frontend includes the Firebase ID token
    (JWT) in the Authorization header. The backend verifies that
    token using firebase-admin + the service account key — it does
    NOT mint its own tokens.
  - get_current_user is the FastAPI dependency other routes plug into
    to require auth and scope data (sessions, uploads) by user_id
    (Firebase's uid).
"""

from pathlib import Path

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials
from pydantic import BaseModel

from configs.settings import settings
from observability.logger import get_logger

logger = get_logger("auth")

bearer_scheme = HTTPBearer()

_SERVICE_ACCOUNT_PATH = settings.BASE_DIR / "firebase-service-account.json"


def init_auth_db() -> None:
    """
    Initialize the Firebase Admin SDK. Call once at startup.
    (Function name kept as init_auth_db for drop-in compatibility with
    the previous SQLite-based auth module.)
    """
    if firebase_admin._apps:
        logger.info("Firebase Admin SDK already initialized")
        return

    if not _SERVICE_ACCOUNT_PATH.exists():
        raise RuntimeError(
            f"Firebase service account key not found at {_SERVICE_ACCOUNT_PATH}. "
            "Download it from Firebase Console → Project Settings → Service Accounts "
            "→ Generate new private key, and save it as 'firebase-service-account.json' "
            "in the project root."
        )

    cred = credentials.Certificate(str(_SERVICE_ACCOUNT_PATH))
    firebase_admin.initialize_app(cred)
    logger.info("Firebase Admin SDK initialized")


# ── FastAPI dependency ───────────────────────────────────────────────────

class CurrentUser(BaseModel):
    user_id: str
    email: str


def get_current_user(
    credentials_: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> CurrentUser:
    """
    FastAPI dependency — plug into any route with:
        current_user: CurrentUser = Depends(get_current_user)

    Verifies the Firebase ID token from the Authorization header.
    Raises 401 if missing/invalid/expired.
    """
    token = credentials_.credentials
    try:
        decoded = firebase_auth.verify_id_token(token)
        user_id = decoded["uid"]
        email = decoded.get("email", "")
        return CurrentUser(user_id=user_id, email=email)
    except Exception as e:
        logger.warning(f"Firebase token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )