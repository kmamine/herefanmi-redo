"""Backend orchestrator FastAPI app.

Auth:   POST /auth/signup, POST /auth/login
Core:   POST /medicalTalk (JWT) -> classified verdict
Extras: POST /save, POST /pointcheck, POST /pointsave (JWT)
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from hrf_shared.config import Settings, get_settings
from hrf_shared.contracts import SourceCreate, SourceUpdate
from pydantic import BaseModel

from app.auth import (
    AuthError,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.clients import LLMClient, RAGClient
from app.db import Repository, User, init_engine
from app.gateway import AdminGateway
from app.orchestrator import Orchestrator, OutOfPointsError
from app.schemas import (
    LoginRequest,
    MedicalTalkRequest,
    PointCheckRequest,
    PointSaveRequest,
    PointsBody,
    SaveRequest,
    SignupRequest,
    TokenResponse,
)
from app.validation import validate_sources

app = FastAPI(title="HeReFaNMi Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None


def get_settings_dep() -> Settings:
    return get_settings()


def get_repo() -> Repository:
    global _engine
    settings = get_settings()
    if _engine is None:
        _engine = init_engine(f"sqlite:///{settings.sqlite_path}")
    return Repository(_engine)


def get_rag_client() -> RAGClient:
    return RAGClient(get_settings())


def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())


def get_source_validator():
    return validate_sources


def get_admin_gateway(settings: Settings = Depends(get_settings_dep)) -> AdminGateway:
    return AdminGateway(settings)


def get_orchestrator(
    repo: Repository = Depends(get_repo),
    rag=Depends(get_rag_client),
    llm=Depends(get_llm_client),
    validator=Depends(get_source_validator),
    settings: Settings = Depends(get_settings_dep),
) -> Orchestrator:
    return Orchestrator(
        rag_client=rag,
        llm_client=llm,
        repo=repo,
        source_validator=validator,
        settings=settings,
    )


def get_current_user(
    authorization: str | None = Header(default=None),
    repo: Repository = Depends(get_repo),
    settings: Settings = Depends(get_settings_dep),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        claims = decode_token(token, settings)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    user = repo.get_user_by_uid(claims.get("sub", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user")
    return user


def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/signup", response_model=TokenResponse)
def signup(
    body: SignupRequest,
    repo: Repository = Depends(get_repo),
    settings: Settings = Depends(get_settings_dep),
) -> TokenResponse:
    is_admin = body.email.lower() in settings.admin_email_set()
    user = repo.create_user(
        body.email, hash_password(body.password), points=settings.signup_points, is_admin=is_admin
    )
    if user is None:
        raise HTTPException(status_code=400, detail="Email already registered")
    token = create_access_token(user.uid, user.email, settings)
    return TokenResponse(
        access_token=token, uid=user.uid, points=user.points, is_admin=user.is_admin
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    repo: Repository = Depends(get_repo),
    settings: Settings = Depends(get_settings_dep),
) -> TokenResponse:
    user = repo.get_user_by_email(body.email)
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Refresh admin status from the allowlist (env may have changed).
    should_be_admin = user.email.lower() in settings.admin_email_set()
    if should_be_admin != user.is_admin:
        repo.set_admin(user.uid, should_be_admin)
        user = repo.get_user_by_uid(user.uid)
    token = create_access_token(user.uid, user.email, settings)
    return TokenResponse(
        access_token=token, uid=user.uid, points=user.points, is_admin=user.is_admin
    )


@app.post("/medicalTalk")
async def medical_talk(
    body: MedicalTalkRequest,
    user: User = Depends(get_current_user),
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> dict:
    try:
        return await orchestrator.handle(
            data=body.data, opinion=body.opinion, uid=user.uid, backend=body.backend
        )
    except OutOfPointsError as exc:
        raise HTTPException(status_code=403, detail="Out of query points") from exc


@app.post("/save")
def save_rating(
    body: SaveRequest,
    user: User = Depends(get_current_user),
    repo: Repository = Depends(get_repo),
) -> dict:
    if not repo.set_rating(body.reference, body.rating):
        raise HTTPException(status_code=404, detail="Unknown reference")
    return {"status": "SUCCESS"}


@app.post("/pointcheck")
def point_check(
    body: PointCheckRequest,
    user: User = Depends(get_current_user),
) -> dict:
    return {"points": user.points}


@app.post("/pointsave")
def point_save(
    body: PointSaveRequest,
    user: User = Depends(get_current_user),
    repo: Repository = Depends(get_repo),
) -> dict:
    repo.set_points(user.uid, body.points)
    return {"status": "SUCCESS"}


# ---- Admin panel (admin-only) ----------------------------------------------


class ScrapeBody(BaseModel):
    sources: list[str] | None = None


@app.get("/admin/stats")
async def admin_stats(
    _: User = Depends(get_current_admin),
    repo: Repository = Depends(get_repo),
    gateway: AdminGateway = Depends(get_admin_gateway),
) -> dict:
    kb = await gateway.kb_stats()
    try:
        sources = await gateway.list_sources()
    except Exception:  # noqa: BLE001 - degrade gracefully if scraper is down
        sources = []
    return {
        "users": repo.count_users(),
        "queries": repo.count_queries(),
        "chunks": kb.get("chunks", 0),
        "per_source_chunks": kb.get("sources", {}),
        "sources": len(sources),
    }


@app.get("/admin/sources")
async def admin_list_sources(
    _: User = Depends(get_current_admin),
    gateway: AdminGateway = Depends(get_admin_gateway),
) -> dict:
    return {"sources": await gateway.list_sources()}


@app.post("/admin/sources")
async def admin_create_source(
    body: SourceCreate,
    _: User = Depends(get_current_admin),
    gateway: AdminGateway = Depends(get_admin_gateway),
) -> dict:
    return await gateway.create_source(body.model_dump())


@app.patch("/admin/sources/{name}")
async def admin_update_source(
    name: str,
    body: SourceUpdate,
    _: User = Depends(get_current_admin),
    gateway: AdminGateway = Depends(get_admin_gateway),
) -> dict:
    return await gateway.update_source(name, body.model_dump(exclude_unset=True))


@app.delete("/admin/sources/{name}")
async def admin_delete_source(
    name: str,
    _: User = Depends(get_current_admin),
    gateway: AdminGateway = Depends(get_admin_gateway),
) -> dict:
    return await gateway.delete_source(name)


@app.post("/admin/scrape")
async def admin_scrape(
    body: ScrapeBody,
    _: User = Depends(get_current_admin),
    gateway: AdminGateway = Depends(get_admin_gateway),
) -> dict:
    return await gateway.run_scrape(body.sources)


@app.get("/admin/users")
def admin_users(
    _: User = Depends(get_current_admin),
    repo: Repository = Depends(get_repo),
) -> dict:
    users = [
        {"uid": u.uid, "email": u.email, "points": u.points, "is_admin": u.is_admin}
        for u in repo.list_users()
    ]
    return {"users": users}


@app.post("/admin/users/{uid}/points")
def admin_set_points(
    uid: str,
    body: PointsBody,
    _: User = Depends(get_current_admin),
    repo: Repository = Depends(get_repo),
) -> dict:
    if repo.get_user_by_uid(uid) is None:
        raise HTTPException(status_code=404, detail="Unknown user")
    repo.set_points(uid, body.points)
    return {"status": "SUCCESS", "points": body.points}


@app.get("/admin/queries")
def admin_queries(
    limit: int = 50,
    _: User = Depends(get_current_admin),
    repo: Repository = Depends(get_repo),
) -> dict:
    queries = [
        {
            "id": q.id,
            "uid": q.uid,
            "question": q.question,
            "label": q.label,
            "medical": q.medical,
            "rating": q.rating,
        }
        for q in repo.recent_queries(limit)
    ]
    return {"queries": jsonable_encoder(queries)}
