# CLAUDE.md — FilingLens

## Hard rules
- **Never run `git commit` or `git push`.** Tvisha commits herself. `git status` / `git diff` are fine.
- Never put secrets in code; they live in `.env` (gitignored). `.env.example` documents them.
- Work one phase at a time (see PLAN.md). At the end of each phase: run tests, then stop and summarize what to test and what to commit.

## What this is
Portfolio project: an agentic system that analyzes SEC filings for large US public companies.
Numbers (cadence, financials) are computed deterministically from SEC data; the LLM only reads
and classifies risk-factor text and must cite chunk IDs. Full roadmap, schema and risks: `PLAN.md`.

## Current status
- Phase 0 + Phase 1 written: EDGAR client (User-Agent check, 8 req/s token bucket, retry/backoff,
  disk cache), ticker->CIK, cadence (`cadence.py`), quarterly revenue/net income (`financials.py`),
  SQLAlchemy models + Alembic migration 0001, idempotent upserts, `filinglens` CLI.
- Phase 1 verified on this machine (2026-11-24): `uv sync` (Python 3.12 pinned in `.python-version`),
  ruff, mypy --strict, 59 offline tests, 5 live SEC tests (AAPL, JPM, COST, WMT, XOM). Figures printed
  by the live tests are pending Tvisha's hand-check against the filings.
- Successor registrants: `predecessors:` in `config/companies.yaml` (XOM -> old CIK 0000034088).
- NOT yet verified: DB tests (`TEST_DATABASE_URL`) and `alembic upgrade head` on Neon.
- uv on this machine is a pip `--user` install, not on PATH:
  `C:\Users\tvish\AppData\Roaming\Python\Python313\Scripts`.
- Phase 2 written (2026-11-24), pending review: MCP server on `mcp` 2.3.0 (`MCPServer`; spec 2026-07-28),
  5 read-only tools over stdio, Claude Agent SDK 0.2.x runner for scenario (a), `agent_runs` (migration 0002),
  `filinglens agent profile TICKER`. Verified: 83 offline tests, live stdio test. NOT yet run: a real agent
  call (costs money) and migration 0002 against a DB.
- `search_filings` searches filing-index metadata only. EDGAR full-text search (efts.sec.gov) is not a
  documented API, so it is not used; open question for Tvisha.
- Phase 3 written (2026-11-26), pending review: `risk/` (html, extract, chunk, embed, classify, diff,
  store, documents, ingest), migration 0003 (pgvector), `SqlRiskStore`, 5 new MCP tools (9 total),
  scenarios (b) sector_comparison and (c) what_changed, `filinglens risk ingest|show|changes`.
  Verified: 130 offline tests; live extraction on all 25 latest 10-Ks (24 at confidence 1.0, WFC 0.6 =
  annual report beside a wrapper 10-K, flagged for review); live bge-small embedding test.
  NOT yet run: Claude classification (costs money), agent scenarios, migration 0003 / DB tests.
- Embeddings need `uv sync --extra embeddings` (torch); unit tests use `HashingEmbedder`.
- Phase 4 written (2026-11-26), pending review: `src/filinglens/evals/` (metrics, golden, deterministic,
  judge, llm_evals, runner, cli), `evals/thresholds.yaml`, `evals/golden/` (20+20 UNVERIFIED snapshot rows),
  migration 0004 (eval_runs/eval_results), `.github/workflows/{ci,nightly-eval}.yml`, `question` scenario.
  Verified: 143 offline tests; `eval run --suite smoke` passes (XBRL 20/20, extraction 5/5, guardrails 6/6).
  NOT yet run: full suite (costs money), CI workflows (need a GitHub push + SEC_USER_AGENT / ANTHROPIC_API_KEY
  secrets), DB tests. Golden metrics stay skipped until Tvisha runs `eval label ...`.
- Phases 5-7 written back to back at Tvisha's request (2026-11-26), pending review:
  - 5: `guardrails/` (numbers vs XBRL, free-text amounts, citations, advice policy, disclaimer), one schema
    repair retry in the same agent session, migration 0005 (review_queue, append-only audit_log,
    agent_runs.guardrails/repairs), `filinglens review ...`, scenario (d) `agent brief`, `<filing_text>` wrapping.
  - 6: FastAPI `src/filinglens/api/app.py` (`filinglens api serve|openapi`), Next.js 16 app in `web/`
    (typed client from `web/openapi.json`; a test fails if it drifts).
  - 7: MCP Streamable HTTP (`filinglens-mcp --http`), `edgar/bulk.py` + `bulk download` / `universe build`
    (public-float ranking) / `ingest --bulk-dir`, `risk ingest --sample --max-cost`, Dockerfiles,
    `docs/deploy-azure.md` (not deployed), README rewritten as a presentation.
  - Verified: 186 offline tests, 14 live tests (incl. MCP over HTTP), smoke gate passes, web typecheck + build,
    pages rendered and screenshotted. NOT run: agent/classification/full eval (cost), DB tests, Docker builds,
    bulk download (several GB), CI.
- Every tool here (Bash heredocs, Write, Edit) unescapes `\u`, `\n` and `\b` in content: write Python without
  backslash escapes (use chr(), print(), triple quotes) and grep the result.
- Local database (2026-11-26): `DATABASE_URL="local"` = embedded open-source PostgreSQL 16 + pgvector 0.6
  (`pgserver`, `uv sync --extra localdb`, data in `data/pgdata`); `TEST_DATABASE_URL="local:filinglens_test"`
  for DB tests. Migrations 0001-0005 applied; all 25 companies ingested; risk factors for AAPL, XOM, JPM
  (unclassified). Storing data found and fixed two bugs: MSFT "Q5" (restated quarter with a shifted start)
  and >65,535-parameter inserts for heavy filers (JPM/BAC/GS).
- Tvisha: no paid API use. The full eval workflow is manual-only; nothing calls Claude unless she asks.
- Free local classification (2026-11-26): `risk classify --local` (bge zero-shot categories, extractive
  summaries, severity left empty, model tagged `local-zero-shot`). All 25 companies ingested + classified
  (~1,380 factors, $0); `risk report --all` -> `reports/risk-summaries.md`. Smoke evals over all 25 pass.
  Fixes found on real data: bold running "Part I" headers split JPM factors; run-in "." fragments; YoY diff
  now matches near-identical text (cosine >= 0.92) even when the heading was rewritten.
- FilingLens never gives buy/sell recommendations or price targets (policy guardrail; Tvisha asked once).
- Stock prices (2026-11-27): Alpha Vantage free key (`ALPHAVANTAGE_API_KEY`, 25 requests/day, compact = last
  100 days); Stooq is behind bot protection and Yahoo scraping breaks its terms, so neither is used.
  `prices fetch|show`, migration 0006 `prices`, MCP tool `get_price_history` (10 tools), price chart and
  approximate market cap (close x SEC shares outstanding) on the company page. Policy guardrail extended
  ("strong buy", buy/sell at a $ level, target price / fair value with a $ figure).
- Silver labels (2026-11-27): `evals/silver.py` scores stored classifications (latest 10-K) against the
  filers' own section headings ("Legal and Regulatory Risks" -> regulatory|litigation; generic headings
  skipped). Report-only metric `silver_category_agreement` in every smoke run with a DB: local classifier
  0.56 primary / 0.67 any-category over 393 of 680 factors. Human `eval label risk` stays the gold standard.
- All seven phases are written; everything free is verified. Remaining needs Tvisha: labels, a GitHub push
  (+ SEC_USER_AGENT secret), and any paid run she chooses.

## Decisions
- Windows laptop, Python 3.12 via `uv`; ruff + mypy --strict + pytest.
- DB: Neon Postgres + pgvector (dev). CI uses a `pgvector/pgvector` service container.
  DB tests reset `TEST_DATABASE_URL` — never point it at Neon.
- Embeddings (Phase 3): local `sentence-transformers` `BAAI/bge-small-en-v1.5` (384 dims) behind an `Embedder` interface.
- Universe: `config/companies.yaml`, 25 tickers across 5 sectors.
- Revenue concept order: RevenuesNetOfInterestExpense → Revenues → RevenueFromContractWithCustomerExcludingAssessedTax
  → ...IncludingAssessedTax → SalesRevenueNet; first concept covering the latest quarter wins.
- Quarters classified by dates/duration, never by XBRL `fy`/`fp` (those describe the filing).
- Only SEC's documented endpoints; SEC_USER_AGENT must include name + email.

## Commands
```
uv sync
uv run ruff format . ; uv run ruff check --fix . ; uv run mypy
uv run pytest
$env:RUN_LIVE=1; uv run pytest tests/live -s
uv run alembic upgrade head
uv run filinglens report financials AAPL
uv run filinglens ingest --dry-run
```
