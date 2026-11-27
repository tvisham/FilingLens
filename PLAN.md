# FilingLens — Plan

An agentic system that analyzes SEC filings for large US public companies: filing cadence, recent financials, and LLM-assisted risk analysis with citations, evaluations, guardrails and human review.

**Status:** draft for approval. Nothing is built until this is approved.
**Rule:** Claude never runs `git commit` or `git push`. Tvisha commits.

---

## 1. Decisions (from clarifying questions)

| Topic | Decision |
|---|---|
| Where code runs | Tvisha's Windows laptop (folder connected via the Claude desktop app). Claude writes files; Tvisha runs commands and tests. |
| Database | Hosted Postgres with pgvector: **Neon** free tier (default; Supabase works the same). CI uses a `pgvector/pgvector` service container so tests never touch the hosted DB. |
| Embeddings | Local open-source: `sentence-transformers` with **`BAAI/bge-small-en-v1.5`** (384 dims, fast on a laptop CPU). Wrapped behind an `Embedder` interface so Voyage/Azure OpenAI can be swapped in later for comparison. |
| LLM | Claude via the Claude Agent SDK (agent) and Anthropic API (classification, judge). Model names live in config. |
| Python tooling | Python 3.12, `uv` (env + lockfile), `ruff` (lint/format), `mypy --strict` on `src/`, `pytest`. |
| Starting companies | 25 tickers below, chosen to include non-December fiscal years and awkward revenue tags. |

### Starting universe (`config/companies.yaml`)

| Sector | Tickers | Edge cases exercised |
|---|---|---|
| Technology | AAPL, MSFT, NVDA, GOOGL, ORCL | FY ends Sep (AAPL), Jun (MSFT), Jan (NVDA), May (ORCL); 52/53-week years |
| Financials | JPM, BAC, WFC, GS, V | Banks don't use a plain `Revenues` tag; V has a Sep FY |
| Healthcare | UNH, JNJ, PFE, CVS, ABBV | Large 8-K volume; acquisitions |
| Energy | XOM, CVX, COP, PSX, MPC | Revenue tag variety (`RevenueFromContractWithCustomer…` vs `Revenues`) |
| Consumer | WMT, AMZN, COST, HD, PG | FY ends Jan (WMT, HD), Jun (PG); COST has a 12/12/12/16-week fiscal year |

Ticker → CIK mapping comes from `https://www.sec.gov/files/company_tickers.json` (cached), never hand-typed.

---

## 2. Architecture

```
          ┌──────────────────────────────── Next.js (TypeScript) ────────────────────────────────┐
          │ Company report  │  Eval dashboard  │  Review queue (approve / edit / reject)           │
          └──────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │ REST (JSON)
                                    ┌────────────▼────────────┐
                                    │   FastAPI (Python)      │
                                    └───┬───────────────┬─────┘
                                        │               │
                          ┌─────────────▼───┐    ┌──────▼─────────────────────┐
                          │ Postgres +      │◄───┤ Agent (Claude Agent SDK)   │
                          │ pgvector (Neon) │    │  scenarios a–d, guardrails │
                          └─────────▲───────┘    └──────┬─────────────────────┘
                                    │                   │ MCP (stdio → Streamable HTTP)
                          ┌─────────┴───────┐    ┌──────▼─────────────────────┐
                          │ Ingestion jobs  │    │ FilingLens MCP server      │
                          │ (cadence, XBRL, │    │ read-only tools            │
                          │  Item 1A, embed)│    └────────────────────────────┘
                          └─────────▲───────┘
                                    │ rate-limited, cached HTTPS
                          ┌─────────┴───────┐
                          │ SEC EDGAR        │  data.sec.gov/submissions, /api/xbrl/companyfacts,
                          │ (documented only)│  sec.gov/Archives (filing docs), efts full-text search
                          └──────────────────┘
```

**Core principle:** numbers and dates are computed deterministically from SEC data. The LLM is used only for reading and classifying risk-factor text, and every claim it makes must cite a stored chunk.

---

## 3. Phases

Each phase ends with: green tests, a short "what to test / what to commit" summary from Claude, and a stop for review.

### Phase 0 — Scaffolding (small, done with Phase 1)
- `pyproject.toml` (uv), ruff, mypy, pytest config, `.env.example`, `.gitignore`, README skeleton.
- `config/companies.yaml`, settings via `pydantic-settings`.
- Alembic initialized against `DATABASE_URL`.

### Phase 1 — EDGAR client, schema, cadence, financials
- **`edgar/client.py`**: `httpx` client; `User-Agent` from `SEC_USER_AGENT` (startup fails if missing or lacks an email); token-bucket limiter at **8 req/s** (headroom under SEC's 10/s); retries with exponential backoff + jitter on 429/5xx (small in-house `edgar/retry.py`, no `tenacity`); gzip; on-disk cache keyed by URL with per-endpoint TTL (submissions 12h, companyfacts 24h, tickers 7d).
- **`edgar/tickers.py`**: ticker → 10-digit zero-padded CIK.
- **`cadence.py`**: from `submissions/CIK##########.json`, merge `filings.recent` **plus the paged `filings.files[]`** (heavy 8-K filers overflow `recent` within 3 years). Count and date 10-K, 10-Q, 8-K, DEF 14A (amendments like `10-K/A` tracked separately) over a 3-year window. Report the company's `fiscalYearEnd` and the expected vs actual cadence (1 10-K + 3 10-Qs per fiscal year).
- **`financials.py`**: from `companyfacts`:
  - Revenue concept fallback list (the first concept that covers the most recent quarter wins, so stale tags are skipped; the chosen concept is stored): `RevenuesNetOfInterestExpense` (banks' headline net revenue; non-banks don't use it) → `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `RevenueFromContractWithCustomerIncludingAssessedTax` → `SalesRevenueNet`. *Changed during Phase 1:* banks often tag only fee income as `RevenueFromContractWithCustomer…`, so putting that first would understate their revenue. Net income: `NetIncomeLoss` → `ProfitLoss` → `NetIncomeLossAvailableToCommonStockholdersBasic`.
  - Identify quarters by **period start/end dates**, not by the fact's `fy`/`fp` fields (those describe the *filing*, so a 10-K's prior-year comparatives carry the new year's `fy`).
  - Quarter = duration of ~84–119 days (covers 12-week, 13-week, 14-week and COST's 16-week Q4). Annual = ~350–380 days. 9-month YTD = ~250–280 days.
  - Q4 not reported directly → derive `Q4 = FY − 9M YTD` (fallback: `FY − (Q1+Q2+Q3)`), mark `derived=true` with a `derivation` note, and keep both source accessions.
  - Duplicates/restatements: for the same concept + period, keep the most recently filed value and store its accession number and form.
  - Output: last two completed fiscal quarters of revenue and net income, USD.
- **Successor registrants** *(added during Phase 1)*: after a holding-company reorganization SEC's ticker file maps only the new CIK (XOM → ExxonMobil Holdings Corp, 0002115436, 8-K12B on 2026-07-01). `config/companies.yaml` has a `predecessors:` map (the only hand-typed CIKs); cadence merges the predecessors' filings, financials use the current CIK, and each filing's Archive URL uses the CIK that listed it.
- **CLI**: `filinglens ingest --companies config/companies.yaml` and `filinglens report cadence|financials TICKER`.
- **Tests** (all offline by default): trimmed, recorded JSON fixtures for AAPL, JPM, COST, WMT and one synthetic case; mocked HTTP via `httpx.MockTransport` (no `respx`); unit tests for limiter, cache, retry, concept fallback, period classification, Q4 derivation, restatement de-dupe, cadence window/amendments. `@pytest.mark.live` tests hit SEC and are skipped unless `RUN_LIVE=1`.

### Phase 2 — MCP server + first agent scenario
- Confirm the **official Python MCP SDK** version that implements spec **2026-07-28** (stateless; per-request `_meta`, `server/discover`). Pin it in `pyproject.toml` and record the spec revision in the README. If the stable SDK doesn't support it yet, pin the latest stable, note the revision it targets, and keep the server stateless in design so the upgrade is trivial.
- *Confirmed during Phase 2:* `mcp` 2.x implements 2026-07-28 (stable since v2.0.0, 28 Jul 2026); pinned `mcp>=2.3,<3`, locked at 2.3.0. `FastMCP` is now `MCPServer`.
- Read-only tools with Pydantic input/output schemas: `lookup_company`, `get_filing_index`, `get_financials`, `get_risk_factors` (stub until Phase 3), `search_filings`. *Changed during Phase 2:* `search_filings` searches the documented submissions data (form, 8-K item codes, dates, document description), not EDGAR full-text search, because `efts.sec.gov` is not a documented API. Full-text search over filings arrives with Phase 3's chunk index.
- Transport: **stdio** first; test with MCP Inspector.
- Agent (Claude Agent SDK) as MCP client, tool allowlist = the five tools above. Scenario **(a) single-company risk profile** end to end (financials + cadence now, risk section after Phase 3).
- Every run is logged to `agent_runs` (model, tokens in/out, cost, latency, tool calls).

### Phase 3 — Risk factors: extraction, RAG, classification, diff
- Fetch the latest and prior-year 10-K primary documents from `sec.gov/Archives` (documented, directory-browsable).
- **Item 1A extraction**: HTML → text, find the *body* "Item 1A. Risk Factors" heading (skipping the table of contents) and end at "Item 1B"/"Item 1C"/"Item 2". Store extraction confidence; low confidence → review queue.
- Chunk by risk-factor heading (most 10-Ks bold each risk) with a token-window fallback (~400 tokens, 50 overlap). Embed with bge-small; store in `chunks.embedding vector(384)` with an HNSW index.
- **Classification**: Claude classifies each risk factor into the fixed taxonomy (macro, regulatory, cyber, supply chain, competition, litigation, AI/technology, liquidity, ESG/climate, other) with a severity score, a one-line summary and **required `chunk_ids`**. Structured output validated by Pydantic.
- **YoY diff**: match risk factors across years by embedding similarity + heading similarity. Classes: `new`, `removed`, `changed` (matched but textual change above threshold), `unchanged`.
- Scenarios **(b) sector comparison** and **(c) what changed**.
- *Changed during Phase 3:*
  - Factor headings are found by **style signature** (bold/italic/caps, block vs run-in), the most frequent signature being the factor level, because filers differ (AAPL bold sentences, AMZN bold-italic titles, XOM/CVX run-in leads, PFE caps topics).
  - When the primary document has no real Item 1A (WFC files the annual report beside a cross-reference 10-K), other documents in the filing folder (`index.json`) are searched; such sections are capped at confidence 0.6 and flagged for review.
  - Fiscal year comes from the 10-K's XBRL `fy` (HD's year ending 2026-02-01 is FY2025; WMT's ending 2026-01-31 is FY2026).
  - Taxonomy slugs: `macro, regulatory, cyber, supply_chain, competition, litigation, ai_technology, liquidity, esg_climate, other`.
  - Classification: Claude Opus 5.5, effort `low`, structured output, batches of 8, server-side refusal fallback (`fallbacks="default"`); cited chunk ids must be a subset of the factor's own chunks. Each run is logged in `agent_runs`. The in-memory store backs tests and `--no-db`.
  - Agent runs fail as `unsupported_citation` if they cite a chunk id no tool returned in that run (early Phase 5 guardrail).
  - Diff thresholds (match 0.80, unchanged >= 0.97 text similarity) are untuned; Phase 4 should add a "minor edit" band.

### Phase 4 — Evals + CI gate
- `/evals` package + `filinglens eval run --suite {smoke,full}`.
- **Golden set**: `evals/golden/*.jsonl` (format below) + `evals/label.py` CLI for hand-labeling risk categories and coherence scores.
- Metrics:
  - *Financial exact-match* vs XBRL (deterministic, must be 100%).
  - *Cadence correctness* vs hand-checked counts.
  - *Risk category precision/recall/F1* (micro + per-category).
  - *Faithfulness*: each claim's cited chunk entails it (LLM judge) + a lexical-overlap sanity check; uncited claims count as failures.
  - *Coherence*: LLM-as-judge 1–5 rubric; **agreement with Tvisha's labels** (Cohen's κ / Spearman) reported next to the score.
  - *Performance*: latency p50/p95, tool calls per run.
  - *Cost*: tokens and USD per report.
  - *Adversarial*: prompt injection planted in a filing chunk, a company that doesn't file with the SEC (e.g. a large private company), and buy/sell-advice requests. Pass = refuses/ignores correctly.
- Results → `eval_runs` / `eval_results`. Thresholds in `evals/thresholds.yaml`.
- GitHub Actions: lint + typecheck + unit tests on every PR; smoke eval on PR (fails below threshold); full eval nightly (scheduled), uploads a summary artifact.
- *Changed during Phase 4:*
  - Code lives in `src/filinglens/evals/` (typed, mypy strict); data stays in `evals/` (golden, thresholds, reports).
  - **smoke** = no LLM: financial exact match re-derived from raw XBRL (incl. derived Q4), golden matches, extraction confidence (5 companies), and protocol-level tool guardrails (non-SEC company, injection-shaped and oversized arguments, all tools read-only). **full** = smoke over 25 companies + category P/R/F1 and kappa, faithfulness (judge + lexical floor), agent success/latency/tool calls/cost, coherence, judge-human Spearman, adversarial LLM cases.
  - Golden financial/cadence rows start as unverified snapshots (`eval snapshot`) and count only after `eval label`. Risk labels are keyed by accession + heading hash.
  - Skipped metrics (no labels yet) are reported and do not fail the gate.
  - Adversarial advice / non-SEC / prompt-extraction cases use a new free-form `question` agent scenario.
  - Judge: `claude-opus-5-5` by default (`JUDGE_MODEL`); a cheaper judge can be compared once human coherence labels exist.

### Phase 5 — Guardrails, review queue, audit log
- Tool allowlist and read-only enforcement in the agent and the MCP server.
- Pydantic schemas on every agent output; invalid output → one repair retry → review queue.
- **Numeric cross-check**: any number in agent output is parsed and matched against `financial_facts`; mismatch blocks the report and queues it.
- Policy layer: refuses investment recommendations; standard "not investment advice" footer.
- Treat filing text as data: chunks are wrapped in delimiters with an instruction that their content is never a command (tested by adversarial evals).
- Confidence threshold → `review_queue` (approve / edit / reject with reason). Every agent run and reviewer action → `audit_log`.
- Scenario **(d) client-meeting prep brief**.

- *Changed during Phase 5:* the guardrail verdict is pass / review / block from concrete checks (numbers vs XBRL, free-text dollar amounts vs known figures or quoted tool text, citations from the run's tool results, advice patterns with negation handling, disclaimer, low-confidence extractions); confidence is derived from the issues. The audit log is append-only by a database trigger. Guardrails are also an eval metric (`guardrail_checks`, smoke suite).

### Phase 6 — Frontend (Next.js + TypeScript)
- `/companies/[ticker]`: cadence timeline, revenue/net income charts (last 8 quarters, latest two highlighted), risk table (category, severity, summary, clickable citations that open the chunk and link to the filing on EDGAR), YoY diff view.
- `/evals`: scores over time per metric, cost and latency trends, drill-down into failures.
- `/review`: queue with approve/edit/reject.
- Typed API client generated from FastAPI's OpenAPI schema.

- *Changed during Phase 6:* the API reuses the MCP tool functions for company data (one source of truth). Evals are read from the database or, without one, from `evals/reports/*.json`. Review decisions need `X-API-Token` when `API_TOKEN` is set. Charts are hand-written SVG (one measure per chart, latest quarters highlighted, hover + table view). Next.js 16 (15 pulled a vulnerable PostCSS).

### Phase 7 — Scale + polish
- Streamable HTTP transport for the MCP server.
- Scale deterministic ingestion to ~500 companies (bulk nightly ZIPs from SEC instead of per-company calls); LLM analysis on a sampled subset with cost reporting.
- Optional: Azure deployment (Container Apps + Azure Database for PostgreSQL) and Azure OpenAI as a second provider for a cost/quality comparison in the eval dashboard.
- README as a presentation: problem, architecture, eval results, demo GIF.

---

- *Changed during Phase 7:* the ~500-company universe is ranked by each company's latest `dei:EntityPublicFloat` from the bulk companyfacts archive (no scraped index list); sectors are SEC SIC divisions. The MCP server runs stateless over Streamable HTTP. Azure deployment is a runbook (`docs/deploy-azure.md`), not automated. Azure OpenAI comparison not implemented.

---

## 4. Database schema

Phase 1 tables are **bold**; the rest are added by later Alembic migrations.

| Table | Key columns |
|---|---|
| **companies** | `cik` PK (char 10), `ticker`, `name`, `sector`, `sic`, `fiscal_year_end` (MMDD), `updated_at` |
| **filings** | `accession` PK, `cik` FK, `form`, `is_amendment`, `filing_date`, `report_date`, `primary_document`, `url` |
| **financial_facts** | `id` PK, `cik` FK, `metric` (revenue / net_income), `concept`, `period_start`, `period_end`, `duration_days`, `period_type` (quarter / ytd9 / annual), `fiscal_year`, `fiscal_quarter`, `value` numeric, `unit`, `accession`, `form`, `filed`, `derived` bool, `derivation` text, `source_accessions` text[]; unique (`cik`,`metric`,`period_start`,`period_end`) |
| **cadence_snapshots** | `cik`, `window_start`, `window_end`, `form`, `count`, `dates` date[], `computed_at` |
| risk_sections | `id`, `accession` FK, `fiscal_year`, `text`, `extraction_confidence` |
| chunks | `id`, `risk_section_id` FK, `ordinal`, `heading`, `text`, `token_count`, `embedding vector(384)` (HNSW) |
| risk_factors | `id`, `cik`, `fiscal_year`, `category`, `severity`, `summary`, `chunk_ids` int[], `model`, `agent_run_id` |
| risk_diffs | `id`, `cik`, `from_year`, `to_year`, `status` (new/removed/changed/unchanged), `from_factor_id`, `to_factor_id`, `similarity` |
| agent_runs | `id`, `scenario`, `input`, `output` jsonb, `model`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`, `tool_calls`, `status`, `created_at` |
| eval_runs | `id`, `suite`, `git_sha`, `model`, `started_at`, `finished_at`, `passed` |
| eval_results | `id`, `eval_run_id` FK, `case_id`, `metric`, `score`, `threshold`, `passed`, `details` jsonb |
| review_queue | `id`, `agent_run_id` FK, `reason`, `status` (pending/approved/edited/rejected), `reviewer`, `edited_output` jsonb, `decided_at` |
| audit_log | `id`, `actor` (agent/user), `action`, `entity`, `entity_id`, `payload` jsonb, `created_at` |

### Golden-set format (`evals/golden/risk_labels.jsonl`)
```json
{"case_id": "AAPL-FY2026-rf-07", "cik": "0000320193", "accession": "…", "chunk_id": 1234,
 "categories": ["supply chain", "macro"], "labeler": "tvisha", "labeled_at": "2026-12-01"}
```

---

## 5. File tree (target)

```
filinglens/
├── PLAN.md
├── README.md
├── pyproject.toml            # uv-managed
├── uv.lock
├── .env.example              # SEC_USER_AGENT, DATABASE_URL, ANTHROPIC_API_KEY, CACHE_DIR
├── .gitignore
├── alembic.ini
├── config/
│   └── companies.yaml
├── src/filinglens/
│   ├── settings.py
│   ├── cli.py
│   ├── db/
│   │   ├── models.py
│   │   ├── session.py
│   │   └── migrations/       # alembic
│   ├── edgar/
│   │   ├── client.py         # httpx + limiter + retry + cache
│   │   ├── ratelimit.py
│   │   ├── cache.py
│   │   └── tickers.py
│   ├── cadence.py
│   ├── financials.py
│   ├── risk/                 # phase 3: extract.py, chunk.py, embed.py, classify.py, diff.py
│   ├── mcp_server/           # phase 2: server.py, tools.py, schemas.py
│   ├── agent/                # phase 2+: runner.py, scenarios/, guardrails.py
│   └── api/                  # FastAPI app: main.py, routes/
├── evals/                    # phase 4: runner.py, metrics/, golden/, label.py, thresholds.yaml
├── tests/
│   ├── fixtures/             # trimmed recorded SEC JSON
│   ├── unit/
│   └── live/                 # @pytest.mark.live
├── web/                      # phase 6: Next.js + TypeScript
└── .github/workflows/
    ├── ci.yml                # lint, mypy, tests, smoke eval on PR
    └── nightly-eval.yml
```

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| XBRL `fy`/`fp` describe the filing, not the fact → wrong quarter labels | Classify by period dates and duration; unit tests on comparative-period facts |
| 52/53-week years and COST's 16-week Q4 | Wide duration windows (84–119 days); explicit COST fixture test |
| Revenue tag varies (banks especially) | Ordered fallback list; store chosen concept; flag companies where none match |
| Q4 derivation errors (restatements between 10-Q and 10-K) | Use YTD and FY from the same 10-K where possible; record both accessions; eval exact-match |
| `filings.recent` doesn't cover 3 years for heavy filers | Always merge paged `filings.files[]` |
| SEC 403/429 | Mandatory User-Agent check, 8 req/s limiter, backoff, cache |
| Item 1A extraction from varied HTML | TOC-skipping heuristics, confidence score, review queue, extraction eval on 25 companies |
| Prompt injection in filing text | Chunks treated as data, tools read-only, adversarial eval cases |
| MCP SDK lags the 2026-07-28 spec | Verify in Phase 2, pin, document targeted revision, stateless design |
| Hosted DB free-tier limits / network flakiness | CI uses a local pgvector container; Neon only for dev/demo |
| Local embedding model download/CPU speed on laptop | Small model (bge-small, ~130 MB), batch embedding, cache embeddings by chunk hash |
| LLM cost at 500 companies | Deterministic parts for all; LLM on a sample; prompt caching; cost tracked per run |
| Copyrighted/licensed company lists | Universe is self-chosen tickers; no scraping of the Fortune list |

---

## 7. What Tvisha needs before Phase 1 runs
1. An empty folder (suggested `C:\Users\<you>\filinglens`) connected to this chat via **Add folder** in the Claude desktop app.
2. `uv` installed (`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`), then `uv python install 3.12`.
3. A Neon project with pgvector enabled (`CREATE EXTENSION vector;`) and its connection string for `.env`.
4. `SEC_USER_AGENT` value, e.g. `FilingLens Tvisha Mishra <your-illinois-email>`.
5. (Phase 2+) `ANTHROPIC_API_KEY`.
