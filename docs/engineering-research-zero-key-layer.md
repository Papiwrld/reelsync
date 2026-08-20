# Zero-Key Research Layer — Engineering Report

**Status:** implemented, tested (all offline — every test mocks the network)
**Scope:** supplemental research sources for the agentic content planner, without any API key
**Activation:** `[research] zero_key_enabled = true` in `config.toml`

## 1. Goal

The agentic planner's research stage (`app/services/research.py`) previously
relied on user notes and LLM model knowledge (`provenance: model_knowledge`),
optionally augmented by one generic JSON search endpoint. This layer adds a
second, fully offline-friendly option: real, attributable sources fetched
from public APIs that require **no account and no API key**.

The layer is opt-in, budgeted, cached, deduplicated and rate-limit aware so
it stays polite even when every feature is enabled.

## 2. Providers

Nine public APIs, selected after verifying each one's 2026 usage policy:

| Provider | What it covers | Policy notes | Tier mapping |
|---|---|---|---|
| Wikipedia (MediaWiki Action API) | general knowledge, article extracts | UA required (403 otherwise); ~200 req/min, ≤3 concurrent | secondary |
| Wikidata (WB API + optional SPARQL) | structured facts, statistics, citations via sitelinks | same Wikimedia pool; SPARQL opt-in only | secondary |
| OpenAlex | academic works | **key-gated since Feb 2026**; keyless demo tier (~100 credits/day) still works; optional `openalex_api_key` raises limits | academic |
| Crossref | bibliographic metadata | public pool 5 req/s; `mailto` → polite pool 10 req/s | academic |
| arXiv | preprints | 1 req/3 s, single connection | academic |
| World Bank | country GDP / population / life expectancy indicators | no key, no auth, caching encouraged; multi-indicator `;` batching | government |
| NASA Open APIs | APOD + NEO feed | `DEMO_KEY` (no registration) 30 req/h; optional free key | government |
| Open-Meteo | geocoding + current weather | keyless 10k req/day; forecasts cached with a short TTL | secondary |
| Nominatim | place search, geodata | strict: 1 req/s, single thread, real UA required, results **must** be cached, attribution (ODbL) | secondary |

Rejected: **REST Countries** — since 2026 it requires a bearer token/API key
with a 500 req/month non-commercial free tier, so it no longer belongs in a
zero-key layer.

## 3. Architecture

```
app/services/research_layer/
├── schema.py      ResearchResult / ResearchPackage / ResearchClaim / SourceReference
├── metrics.py     thread-safe counters; requests_avoided = cache_hits + coalesced
│                  + duplicate_queries_prevented + batched_items_saved + budget_blocked
├── budget.py      per-run caps: max_external_requests (20), max_requests_per_provider (5)
├── dedupe.py      query normalization + Jaccard ≥ 0.75 dedup (job-scoped)
├── cache.py       persistent cache storage/research_cache/ (zk- prefix, sha256 keys,
│                  256 shard locks, coalescing, atomic writes, TTL per source)
├── throttle.py    per-provider min-interval + max-concurrency + 429 cooldowns
├── http.py        single retrying GET client: UA, (10,30) timeouts, exponential
│                  backoff + jitter, Retry-After (≤30 s), 5xx retry, never raises
├── quality.py     score = 0.2 relevance + 0.35 authority + 0.15 freshness
│                  + 0.2 corroboration + 0.1 completeness
├── router.py      keyword classification → minimal provider set per query
├── providers/     base.py + 9 thin providers (search/fetch/health_check)
└── __init__.py    ResearchClient — the single entry point
```

The **router** never selects all providers: e.g. `population of France` →
`[worldbank, wikidata, wikipedia]`; `weather in Berlin` → `[openmeteo,
wikipedia]`; `nasa asteroid news` → `[nasa, wikipedia, openalex]`;
everything else → `[wikipedia, wikidata]`.

## 4. Integration with the agentic graph

`research.py` gained a `ZeroKeyResearchProvider` bridge, registered in
`_resolve_providers()` between the web-search provider and the model-knowledge
provider. It:

- activates only when `zero_key_enabled = true` (otherwise `discover()` raises
  `ResearchProviderError`, caught by the orchestrator — zero impact by default);
- lazily imports `ResearchClient` (no import-time cost, no circular imports);
- builds one client per research run (per-run budget; process-level metrics);
- maps `ResearchResult`s into the existing provider dict contract
  (`title/url/tier/is_primary/note/provenance="zero_key"`), so the existing
  dedupe, tiering and confidence logic in `run_research` applies unchanged;
- exposes `zero_key_metrics()` / `zero_key_last_error()` for the WebUI.

Any failure inside the layer (network down, malformed responses, budget
exhaustion) degrades to fewer/zero zero-key sources — it can never break a
video run.

## 5. Reliability & etiquette

- **Cache first:** identical queries within a source's TTL cost zero external
  requests. TTLs: wikipedia/wikidata 30 d, crossref/openalex 14 d,
  arxiv/worldbank 7 d, nasa 1 d, openmeteo 1 h (configurable), nominatim 30 d.
  Empty results are cached for ≤ 1 h. The cache is atomic (temp + `os.replace`)
  and survives restarts (`storage/research_cache/`).
- **Coalescing:** concurrent identical requests share one external call
  (256 sharded locks + double-check read, same pattern as `material_cache.py`).
- **Budget:** a research run never exceeds 20 external requests (5 per
  provider); once exhausted, cached results still serve, network calls stop.
- **Throttles mirror provider policy:** arXiv 3 s, Nominatim 1 s, NASA 2 s,
  Wikimedia ≤ 3 concurrent, etc. A 429 pushes a cooldown (interval × 4,
  capped by Retry-After ≤ 30 s) and is counted as a rate-limit event.
- **Batching:** Wikipedia/OpenAlex/Crossref/Wikidata fetch many items in one
  request where supported (e.g. OpenAlex `filter=ids.openalex:W1|W2`).
- **Attribution is preserved end-to-end** in `raw_metadata` and surfaced in
  the source `note` (CC BY-SA / CC0 / CC BY 4.0 / ODbL / © OSM contributors).
- **Safety:** the SPARQL endpoint is off by default and only answers exact
  `population of X` / `gdp of X` / `capital of X` patterns (regex-validated,
  non-injectable); Nominatim can be disabled; OpenAlex keyless requests
  degrade gracefully; malformed responses from any source are dropped, never
  crash the run.

## 6. Configuration (`config.toml` → `[research]`)

```toml
zero_key_enabled = false          # opt-in
cache_enabled = true              # persistent response cache
deduplication_enabled = true      # near-duplicate queries skipped per job
batching_enabled = true           # multi-item fetches in one request
max_external_requests = 20        # per research run
max_requests_per_provider = 5     # per provider, per run
user_agent = ""                   # some providers require a real UA (Nominatim)
contact_email = ""                # Crossref polite pool + UA suffix
openalex_api_key = ""             # optional: ~10x daily budget
nasa_api_key = ""                 # optional; "DEMO_KEY" default
openmeteo_ttl_minutes = 60        # forecast freshness
nominatim_enabled = true
enable_sparql = false             # Wikidata SPARQL, strict patterns only
```

Fix included in this work: `save_config()` previously dropped the `research`
section when persisting the WebUI settings — it now writes it.

## 7. Observability

The WebUI "Content Intelligence" section gained a **Zero-Key Research
Metrics** expander showing, for the last research run:

- **External requests** made
- **Requests avoided** — cache hits + coalesced + deduplicated + batched +
  budget-blocked (the headline figure)
- **Cache hit rate**
- Requests per provider

`zero_key_metrics()` also feeds any future instrumentation; metrics are
process-level and reset with the app.

## 8. Files changed

- `app/services/research_layer/` (new) — schema, metrics, budget, dedupe,
  cache, throttle, http, quality, router, providers (base + 9), client.
- `app/services/research.py` — `ZeroKeyResearchProvider` bridge,
  `zero_key_metrics()`, provider registration.
- `app/config/config.py` — new `[research]` keys; `save_config()` now persists
  the `research` section.
- `config.example.toml` — documented `[research]` zero-key block.
- `webui/Main.py` — Zero-Key Research Metrics expander.
- `webui/i18n/*.json` — 7 new keys in all 9 locales.
- `test/services/test_research_layer.py` (new) — 64 tests, all offline.
- Fix: `webui/Main.py` — subtitle background-opacity slider previously
  referenced `subtitle_background_enabled` before its assignment (NameError on
  first render); now reads the checkbox session state.

## 9. Test coverage

`test_research_layer.py` (64 tests, network fully mocked via
`patch("...http.requests.get")`):

- router classification/selection, dedup (exact + fuzzy + provider scoping),
  budget caps, metrics/requests-avoided math
- cache round-trip/expiry/empty-TTL cap/corrupt files/coalescing
  (5 concurrent threads → 1 loader), cleanup
- throttles (interval, concurrency, 429 cooldown), HTTP (429 Retry-After,
  5xx retry, 4xx no-retry, exception retry, exhausted retries → None)
- per-provider parsing incl. malformed payloads, batching, attribution,
  config toggles (nominatim off, openalex/nasa keys, SPARQL guard)
- client pipeline (dedup prevention, provider failure isolation, quality
  ordering, batch_fetch) and the research.py bridge (disabled → raises,
  mapping/tiering, metrics exposure)

All pre-existing suites stay green: `test_research.py` + `test_agentic.py` +
`test_config.py` (102 passed), subtitle/video/schema (108 passed), WebUI i18n
(10 passed, 7 217 subtests). The only failing suite in the repo
(`test_webui_task.py`, `test_webui_startup.py` startup test) fails on the
unrelated external `MoneyPrinter Custom` checkout on this machine and is not
touched by this work.

## 10. Example flow

`topic = "What is the GDP of Japan?"` →

1. normalize → `gdp japan`; not a duplicate this job.
2. router → `country_statistics` → providers `[worldbank, wikidata, wikipedia]`.
3. World Bank: country-code list (cached 30 d) → one `;`-batched indicator
   call (GDP, GDP/capita, population, ...) → 4 sources with government tier.
4. Wikidata: `wbsearchentities` + `wbgetentities` → population/GDP facts.
5. Wikipedia: title search + batched extract → secondary-tier article source.
6. Results scored (relevance + authority + freshness + corroboration +
   completeness), corroborated claims assembled into the `ResearchPacket`,
   sources flow into the existing script agent with `provenance: zero_key`.

Budget for this run: ≤ 5 external requests (budget default 20); on a repeat
query: 0 (cache + job dedup).