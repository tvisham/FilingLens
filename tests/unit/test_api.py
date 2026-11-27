"""The web API over the mocked SEC and an in-memory risk store (no database)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_evals import _facts
from test_risk_pipeline import _ingest, routes

from filinglens.api.app import Backend, create_app
from filinglens.config import Universe
from filinglens.mcp_server import tools
from filinglens.risk.classify import Classifier
from filinglens.risk.embed import HashingEmbedder
from filinglens.risk.store import InMemoryRiskStore
from helpers import mock_sec_client


@pytest.fixture()
def client(tmp_path: Path, classifier: Classifier) -> TestClient:
    store = InMemoryRiskStore()
    _ingest(tmp_path / "ingest", store, classifier)
    sec = mock_sec_client(
        tmp_path / "api",
        routes() | {"https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json": _facts()},
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    report = {
        "suite": "smoke",
        "started_at": "2026-11-26T18:44:30+00:00",
        "passed": False,
        "cost_usd": 0.0,
        "git_sha": None,
        "metrics": [
            {
                "metric": "financial_xbrl_match",
                "value": 0.5,
                "passed": False,
                "threshold": {"min": 1.0},
                "n_cases": 2,
            }
        ],
        "cases": [
            {
                "case_id": "EXMP-revenue-FY2025Q4",
                "metric": "financial_xbrl_match",
                "score": 0.0,
                "passed": False,
                "details": {"check": "derived value could not be recomputed"},
            },
            {"case_id": "ok", "metric": "financial_xbrl_match", "score": 1.0, "passed": True, "details": {}},
        ],
    }
    (reports / "smoke-20261126T184446Z.json").write_text(json.dumps(report), encoding="utf-8")

    def ctx() -> tools.ToolContext:
        return tools.ToolContext(
            client=sec,
            universe=Universe.model_validate({"sectors": {"tech": ["EXMP"]}}),
            today=lambda: date(2026, 11, 24),
            store=store,
            embedder_factory=HashingEmbedder,
        )

    return TestClient(create_app(Backend(ctx, None, api_token="s3cret", reports_dir=reports)))


@pytest.fixture()
def classifier() -> Classifier:
    from types import SimpleNamespace

    from test_risk_pipeline import AnsweringMessages

    return Classifier(
        SimpleNamespace(beta=SimpleNamespace(messages=AnsweringMessages())), model="claude-opus-5-5"
    )


def test_health_and_companies(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"ok": True, "database": False, "risk_store": True}
    assert client.get("/api/companies").json() == [{"ticker": "EXMP", "sector": "tech"}]


def test_company_detail(client: TestClient) -> None:
    body = client.get("/api/companies/exmp").json()
    assert body["company"]["ticker"] == "EXMP" and body["company"]["sector"] == "tech"
    quarters = body["financials"]["revenue"]["quarters"]
    assert [q["value_usd"] for q in quarters][-2:] == ["120", "130"]  # Q3, derived Q4 (460 - 330)
    assert quarters[-1]["derived"] is True
    assert body["risks"]["status"] == "available" and len(body["risks"]["factors"]) == 8
    assert body["changes"]["counts"]["new"] == 1
    assert {f["form"] for f in body["filings"]["filings"]} == {"10-K"}


def test_company_errors(client: TestClient) -> None:
    assert client.get("/api/companies/PRIVATECO").status_code == 404
    assert client.get("/api/companies/drop;table").status_code == 422  # path pattern


def test_chunks(client: TestClient) -> None:
    chunk_id = client.get("/api/companies/EXMP").json()["risks"]["factors"][0]["cited_chunk_ids"][0]
    body = client.get(f"/api/chunks/{chunk_id}").json()
    assert body["text"].startswith("<filing_text>") and body["filing_url"].endswith("k26.htm")
    assert client.get("/api/chunks/999999").status_code == 404


def test_evals_from_report_files(client: TestClient) -> None:
    (run,) = client.get("/api/evals").json()
    assert run["id"] == "file:smoke-20261126T184446Z" and run["passed"] is False
    detail = client.get(f"/api/evals/{run['id']}").json()
    assert [c["case_id"] for c in detail["failed_cases"]] == ["EXMP-revenue-FY2025Q4"]
    assert client.get("/api/evals/file:nope").status_code == 404


def test_review_needs_db_and_token(client: TestClient) -> None:
    assert client.get("/api/review").status_code == 503
    body = {"decision": "approve", "reviewer": "t", "reason": "ok"}
    assert client.post("/api/review/1/decision", json=body).status_code == 401
    assert (
        client.post("/api/review/1/decision", json=body, headers={"X-API-Token": "wrong"}).status_code == 401
    )
    assert (
        client.post("/api/review/1/decision", json=body, headers={"X-API-Token": "s3cret"}).status_code == 503
    )
    bad = {"decision": "delete", "reviewer": "t", "reason": "x"}
    assert (
        client.post("/api/review/1/decision", json=bad, headers={"X-API-Token": "s3cret"}).status_code == 422
    )


def test_cors_allows_only_the_web_origin(client: TestClient) -> None:
    ok = client.options(
        "/api/companies", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    other = client.options(
        "/api/companies", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert "access-control-allow-origin" not in other.headers


def test_web_client_schema_is_current() -> None:
    """web/openapi.json drives the generated TypeScript client; it must match the API.
    Fix: `uv run filinglens api openapi && (cd web && npm run gen:api)`."""
    spec = create_app(Backend(lambda: None, None)).openapi()  # type: ignore[arg-type, return-value]
    committed = json.loads((Path(__file__).parents[2] / "web" / "openapi.json").read_text(encoding="utf-8"))
    assert committed == spec
