"""TDD for app.auth — password hashing and JWT."""

import pytest
from app.auth import (
    AuthError,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    h = hash_password("s3cret-pw")
    assert h != "s3cret-pw"
    assert verify_password("s3cret-pw", h) is True
    assert verify_password("wrong", h) is False


def test_verify_handles_long_password():
    long_pw = "x" * 200
    h = hash_password(long_pw)
    assert verify_password(long_pw, h) is True


def test_token_roundtrip(settings):
    token = create_access_token("uid-123", "a@b.com", settings)
    claims = decode_token(token, settings)
    assert claims["sub"] == "uid-123"
    assert claims["email"] == "a@b.com"


def test_decode_invalid_token_raises(settings):
    with pytest.raises(AuthError):
        decode_token("not-a-jwt", settings)


def test_decode_wrong_secret_raises(settings):
    token = create_access_token("uid-123", "a@b.com", settings)
    other = settings.model_copy(update={"jwt_secret": "different-secret"})
    with pytest.raises(AuthError):
        decode_token(token, other)
