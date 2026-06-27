"""TDD for the admin panel backend — is_admin gating + admin endpoints."""

import pytest
from app.db import Repository, init_engine
from app.main import (
    app,
    get_admin_gateway,
    get_repo,
    get_settings_dep,
)
from fastapi.testclient import TestClient


class FakeGateway:
    def __init__(self):
        self.sources = [{"name": "cdc", "enabled": False, "interval_minutes": 1440}]
        self.calls: list = []

    async def list_sources(self):
        return self.sources

    async def create_source(self, body):
        self.calls.append(("create", body))
        return {**body, "last_run_at": None, "last_status": None}

    async def update_source(self, name, body):
        self.calls.append(("update", name, body))
        return {"name": name, **body}

    async def delete_source(self, name):
        self.calls.append(("delete", name))
        return {"status": "deleted"}

    async def run_scrape(self, sources):
        self.calls.append(("scrape", sources))
        return {"stored": 1, "scraped": 1, "sources": {"cdc": {"scraped": 1, "new": 1}}}

    async def kb_stats(self):
        return {"chunks": 12, "sources": {"cdc": 12}}


@pytest.fixture
def tc(tmp_path, settings):
    repo = Repository(init_engine(f"sqlite:///{tmp_path / 'admin.sqlite'}"))
    gateway = FakeGateway()
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_admin_gateway] = lambda: gateway
    client = TestClient(app)
    client.repo = repo  # type: ignore[attr-defined]
    client.gateway = gateway  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()


def _signup(tc, email, password="pw123456"):
    return tc.post("/auth/signup", json={"email": email, "password": password}).json()


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def test_signup_admin_email_sets_is_admin(tc):
    data = _signup(tc, "admin@e.com")
    assert data["is_admin"] is True


def test_signup_normal_email_not_admin(tc):
    data = _signup(tc, "user@e.com")
    assert data["is_admin"] is False


def test_login_refreshes_admin_flag(tc):
    _signup(tc, "admin@e.com")
    login = tc.post("/auth/login", json={"email": "admin@e.com", "password": "pw123456"})
    assert login.json()["is_admin"] is True


def test_non_admin_forbidden(tc):
    token = _signup(tc, "user@e.com")["access_token"]
    assert tc.get("/admin/stats", headers=_h(token)).status_code == 403


def test_admin_stats(tc):
    token = _signup(tc, "admin@e.com")["access_token"]
    r = tc.get("/admin/stats", headers=_h(token))
    assert r.status_code == 200
    body = r.json()
    assert body["users"] >= 1
    assert body["chunks"] == 12
    assert body["sources"] == 1


def test_admin_source_crud_proxied(tc):
    token = _signup(tc, "admin@e.com")["access_token"]
    assert tc.get("/admin/sources", headers=_h(token)).status_code == 200
    tc.post(
        "/admin/sources",
        headers=_h(token),
        json={"name": "nhs", "base_url": "https://nhs.uk", "listing_url": "https://nhs.uk/n"},
    )
    tc.patch("/admin/sources/nhs", headers=_h(token), json={"enabled": True})
    tc.delete("/admin/sources/nhs", headers=_h(token))
    kinds = [c[0] for c in tc.gateway.calls]
    assert kinds == ["create", "update", "delete"]


def test_admin_scrape_proxied(tc):
    token = _signup(tc, "admin@e.com")["access_token"]
    r = tc.post("/admin/scrape", headers=_h(token), json={"sources": ["cdc"]})
    assert r.status_code == 200
    assert ("scrape", ["cdc"]) in tc.gateway.calls


def test_admin_users_and_points(tc):
    token = _signup(tc, "admin@e.com")["access_token"]
    other = _signup(tc, "user@e.com")
    users = tc.get("/admin/users", headers=_h(token)).json()["users"]
    assert any(u["email"] == "user@e.com" for u in users)
    r = tc.post(f"/admin/users/{other['uid']}/points", headers=_h(token), json={"points": 99})
    assert r.status_code == 200
    assert tc.repo.get_user_by_uid(other["uid"]).points == 99


def test_admin_queries(tc):
    token = _signup(tc, "admin@e.com")["access_token"]
    r = tc.get("/admin/queries", headers=_h(token))
    assert r.status_code == 200
    assert "queries" in r.json()
