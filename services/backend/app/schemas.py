"""Request/response models for the Backend API."""

from __future__ import annotations

from pydantic import BaseModel


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    uid: str
    points: int
    is_admin: bool = False


class PointsBody(BaseModel):
    points: int


class MedicalTalkRequest(BaseModel):
    data: str
    opinion: str = "0"
    backend: str = ""


class SaveRequest(BaseModel):
    reference: str
    rating: str


class PointSaveRequest(BaseModel):
    points: int


class PointCheckRequest(BaseModel):
    user: str | None = None
