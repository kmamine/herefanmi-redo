"""Password hashing (bcrypt) and JWT issuance/verification.

Uses bcrypt directly (passlib 1.7.x is incompatible with bcrypt 5.x).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
from hrf_shared.config import Settings
from jose import JWTError, jwt


class AuthError(Exception):
    """Raised when a token is missing, malformed, or fails verification."""


def hash_password(password: str) -> str:
    # bcrypt has a hard 72-byte input limit; truncate deterministically.
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(uid: str, email: str, settings: Settings) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": uid, "email": email, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise AuthError(str(exc)) from exc
