# SERAPH — Tech Stack & Coding Context

**This is the canonical rules file.** AGENTS.md is read natively by most coding
agents (Cursor, Windsurf/Devin, Copilot, Codex, Gemini CLI, and others). Claude
Code is the one exception — it reads `CLAUDE.md`, which imports this file as
its first line. Don't fork this content into another file; edit it here and
everything else points back.

---

## 0. How context is tiered in this repo — read this first

- **Tier 0 (this file):** always loaded, every session, every tool. Hard rules,
  stack choices, invariants. Keep it true, keep it short — every line here
  costs context budget on every single request.
- **Tier 1 (`docs/`):** `SPEC.md`, `ARCHITECTURE.md`, `DATA-ACQUISITION.md`,
  `SERAPH-BUILD-ROADMAP.md`. Load on demand for whichever component you're
  working — not by default on every prompt.
- **Tier 2 (`docs/archive/`):** `SPEC-DELTA-01.md`. **Superseded — do not build
  from it, do not cite it as current spec.** It's excluded from this repo's
  AI indexing (see `.cursorindexingignore` / `.devinignore` / `.claude/settings.json`)
  on purpose, because it reads like current spec but two of its sections are
  actively wrong (FR52 curation approach, pre-acquisition data-volume
  estimates — see its own banner). It's kept only for report §5.1.2 and the
  viva. If a task is specifically about that, say so explicitly and read it —
  otherwise treat it as if it isn't in the repo.

**If you're picking up a session cold:** read this file, then `docs/SPEC.md`
§2 (objectives) and `docs/ARCHITECTURE.md` §1–2 (components + interfaces)
before touching any component you haven't worked on yet.

---

## 1. Source of truth — read order, and who wins on conflict

1. `docs/SPEC.md` v3.1 — requirements (FR/NFR), objectives, acceptance criteria. **Authoritative.**
2. `docs/ARCHITECTURE.md` v2 — component boundaries, interface contracts, the D1–D5 decisions, contract tests CT-1…CT-10. Derived from SPEC, adds nothing new.
3. `docs/DATA-ACQUISITION.md` — data sources, costs, week-1 checklist.
4. `docs/archive/SPEC-DELTA-01.md` — **superseded, historical only.** Do not build from it.

If code and docs disagree, **the docs win.** If you (the AI assistant) think a doc is wrong, say so and ask — don't silently deviate and don't silently invent behavior for anything marked `[GAP]` in SPEC.

---

## 2. Stack — exact choices

| Concern | Use | Why |
|---|---|---|
| Runtime | **Python 3.12** | Newer than SPEC NFR36's ">=3.10" floor for free interpreter speed, fully compatible |
| Ingestion/ETL (C2, per-symbol C4) | **Polars** (`polars>=0.20`) | Multi-threaded columnar engine — matters at ~19M intraday rows. Convert to NumPy at the pillar-engine boundary; pillars work on small dense arrays, not big tables |
| Pillar math (C5/C6/C7/C8/C9) | **NumPy** (`numpy>=1.26`, built against **OpenBLAS or MKL** via conda-forge) + **SciPy** (`scipy>=1.11`) | `numpy.linalg.eigh` on the daily 500×500 correlation matrix is the one place BLAS backend visibly matters |
| Schema / contracts | **Pydantic v2** (`pydantic>=2.5`) | Mirrors every TS shape in ARCHITECTURE §0/§2 exactly — `BaseModel` + `Literal` discriminators for the `Ok`/`Err` union, `frozen=True` config wherever ARCHITECTURE marks `readonly` |
| DB | **PostgreSQL + TimescaleDB** | Per NFR34 (hypertables, partitioned on timestamp axis) |
| DB driver | **`asyncpg`** (or `psycopg[binary]>=3.1`) direct — **no ORM** | 14 known entities, fixed shapes — SQLAlchemy's abstraction buys nothing here and adds a real latency/complexity tax |
| Bulk historical load | **Postgres `COPY`**, not row `INSERT` | Real difference at millions of rows for the EOD2/Kite backfill |
| Hawkes MLE (C5) | **`tick`** (X-DataInitiative) as baseline; `scipy.optimize.minimize` with **analytic gradients** for the `S_t`-conditional likelihood `tick` doesn't cover | Runs at bar frequency (FR10) — the one genuinely latency-sensitive numerical component, per NFR1/NFR4 |
| Hamilton filter (C7) | **`statsmodels.tsa.regime_switching.MarkovSwitching`** with `exog_tvtp` | Native TVTP support — don't hand-roll EM unless this genuinely can't express the state-dependent structural-break spec |
| Kalman (C8) | **Hand-rolled NumPy**, not a framework | Fixed 3-dim state — trivial to write correctly, lets you inline D4's `R^(p)(Δ)` noise model directly |
| Fusion (C9) | Pure NumPy, zero dependencies beyond it | Must stay a pure function — see §5 |
| Parallelism | `concurrent.futures.ProcessPoolExecutor` — one process per pillar minimum (NFR5), pool across symbols/sectors inside Hawkes MLE | Targets the 8-core floor in NFR30 |
| Scheduling | **Cron + idempotent scripts** | Single-node (NFR35), daily/bar-frequency cadence — a workflow engine is pure overhead here |
| Dashboard | **Streamlit** (`streamlit>=1.30`) + `st.cache_data` on the fetch layer | Per SPEC §7 out-of-scope item 7 — Plotly Dash was considered and explicitly not chosen |
| Containers | Docker Compose — one service (Postgres+TimescaleDB), single-node per NFR35 | No k8s — nothing here needs it |
| Testing | **pytest** + **Hypothesis** (`hypothesis>=6.0`) | Hypothesis specifically for the recurring invariant checks: weights sum to 1, `P` stays PSD, mask renormalization sums correctly |

---

## 3. Explicitly rejected — don't suggest these even though they appear in older notes

| Rejected | Was considered where | Use instead |
|---|---|---|
| `rpy2` + R + `MSwM` | NFR36 lists as "optional secondary" | `statsmodels` TVTP already covers it. Cross-language marshaling is a pure tax with no offsetting benefit here. |
| `filterpy` / `pykalman` | NFR36 lists as an option | Hand-rolled NumPy — see §2 |
| SQLAlchemy ORM | — | `asyncpg`/psycopg3 direct |
| Airflow / Prefect | — | Cron + idempotent scripts |
| Plotly Dash | SPEC §7 out-of-scope item 7 | Streamlit (already decided) |
| GPU / CUDA anything | v1 had this for the EEMD–LSTM overlay | Cut with the overlay (NFR32). Nothing in the current scope needs a GPU. |
| True LOB/tick-level microstructure work | SPEC §7 out-of-scope item 12 | Estimators only (Abdi–Ranaldo, tick-test) — never claim queue-depth or quote-level behavior in code comments, docs, or the eventual report |
| pandas as the *primary* engine for large intraday tables | — | Polars for ingestion/feature layers; pandas is fine for small, already-aggregated frames if convenient |

---

## 4. Repo structure

```
seraph/
├── pyproject.toml
├── docker-compose.yml
├── seraph/
│   ├── shared_types/        # pydantic mirrors of ARCHITECTURE §0/§2 — EVERY component imports from here
│   ├── store/                # C1 — schema.sql, writer.py, reader.py
│   ├── ingestion/              # C2 — eod2_client.py, bhavcopy_reader.py, kite_client.py, macro_client.py, flow_client.py, tariff_events.py
│   ├── universe/                 # C3 — constructor.py
│   ├── features/                   # C4 — jump_extraction.py, spread_estimators.py, tick_test.py, cross_sectional.py, tariff_covariate.py
│   ├── pillars/
│   │   ├── rmt/        engine.py     # C6
│   │   ├── hamilton/   engine.py     # C7
│   │   └── hawkes/     engine.py     # C5
│   ├── reconciliation/                 # C8 — kalman.py, noise_model.py
│   ├── fusion/                           # C9 — score.py (PURE), shapley.py, standardize.py
│   ├── validation/                         # C10 — harness.py, ablation.py, preregistration.py
│   ├── decision_support/                     # C11
│   └── dashboard/                              # C12
├── fixtures/
│   └── mock_generator.py    # T0
└── tests/
    └── contract/              # test_ct1_*.py … test_ct10_*.py, written before the component they test
```

---

## 5. Non-negotiable invariants

These encode the subtle parts of ARCHITECTURE.md — the mistakes an assistant coding one component in isolation is most likely to make. Check against this list before finalizing any component's code. (`scripts/check_invariants.py` mechanically checks the ones that can be checked mechanically — run it, don't just re-read this list.)

- **`PILLAR_ORDER = ["hawkes", "rmt", "hamilton"]` is global and fixed.** Every `Vec3`, `Mat3`, and `AvailabilityMask` uses this order everywhere. Getting this wrong silently corrupts the CSRS — no error is raised.
- **C9 (`score()`) is a pure function.** No DB calls, no hidden state, no import of C10. Identical inputs → byte-identical output, 100/100 times (CT-6).
- **`unavailable` pillar emissions are `Ok`, never `Err`.** A structurally-missing Hawkes score pre-2015 is expected, not a failure — modeling it as an error breaks the `R → ∞` mechanism O6 depends on.
- **`structural` vs `transient` absence controls the D2 mask.** Structural → pillar excluded from the mask (`false`). Transient → stays in the mask, D4 age-inflation handles it.
- **`R^(p)(Δ)` must be bounded.** Use D4's saturating exponential: `R₀ + (R_max − R₀)(1 − e^{−Δ/h_p})`. Never a linear or unbounded form in production code — unbounded noise sends `P → ∞` and makes the FR29 confidence interval infinite.
- **FR52: filter before rank, always.** ≥1000 listing days first, *then* rank by 6-month median `TOTTRDVAL`, *then* take top 500. Reversing this order is a named bug trap in the spec — it silently produces fewer than 500 names and breaks `Q = T/N = 2`.
- **`Ω_t` is computed only on common support** — `members(t) ∩ members(t−Δ)`, both eigenvectors renormalized before the inner product (D3). Computing it on full 500-dim vectors across a rebalance date is mathematically undefined, not just noisy.
- **`AR_t` lives in C4, never in C6.** Putting it in the RMT engine makes Hamilton secretly depend on RMT and breaks FR35 (pillars must be independently evaluable). Duplicating the covariance computation is the correct price.
- **Weight artefact flow is one-directional: C10 → `FusionWeights` (immutable, content-hashed file) → C9.** C9 never imports C10 or holds epoch labels. Reversing this reinstates a dependency cycle (D1).
- **Every `CsrsPoint` carries `mask` and `weightsArtefactId`.** This is the audit trail that makes ablation rows falsifiable (FR35/36) — never simplify these fields away.
- **`writeBatch` idempotency is mandatory**, not an optimization. Upsert on `(entity, natural key)` — e.g. `(symbol, ts)` or `(symbol, date)`. Re-running an ingestion batch must produce `inserted: 0` the second time (CT-1).
- **All timestamps are IST and explicit**: `"YYYY-MM-DDTHH:mm:ss+05:30"`. Never naive or UTC-implicit.
- **`Result<T> = Ok<T> | Err` throughout.** Model as a pydantic discriminated union on `status`/`kind`. Don't raise exceptions across component boundaries — return an `Err` variant instead.
- **`seraph/shared_types/` is treated as frozen once Phase 1 (L0) lands.** Every one of the 12 components imports from it — ask before editing it, don't fold it into an unrelated refactor.

---

## 6. Coding conventions

- Every module's docstring cites the component ID and FR range it implements, e.g. `# C6 — RMT Engine (FR16–FR19)`. This is what keeps ARCHITECTURE §12's traceability table true as code evolves.
- No component reaches into another component's owned tables or state directly — always go through the owning component's typed reader/writer (the S1–S10 interfaces in ARCHITECTURE §2).
- Contract test file (`tests/contract/test_ctN_*.py`) is written **before** the component it tests, wherever the component's inputs already exist (many are runnable against T0 mocks from day one — see `docs/SERAPH-BUILD-ROADMAP.md` §3).
- Anything ARCHITECTURE marks `readonly` → pydantic `frozen=True` config, not just a type hint.
- Warning codes and error kinds are typed `Literal` unions matching ARCHITECTURE §0 exactly (`WarningCode`, `SeraphError`) — don't invent ad hoc string codes.

---

## 7. Environment & commands

```bash
# Python
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# DB
docker-compose up -d          # Postgres + TimescaleDB, single node

# Tests
pytest tests/contract -v      # run CT-1..CT-10
pytest tests/ -v              # everything

# Invariant checks (mechanical subset of §5)
python scripts/check_invariants.py

# Mock stream (T0) — no real data needed
python -m fixtures.mock_generator
```

`pyproject.toml` core deps (verify current stable versions before pinning — floors shown, not exact pins):
```
numpy>=1.26
scipy>=1.11
polars>=0.20
pandas            # only where genuinely convenient, not the primary ingestion engine
statsmodels>=0.14
tick
pydantic>=2.5
asyncpg>=0.29
streamlit>=1.30
pytest>=8.0
hypothesis>=6.0
```

---

## 8. Component ownership — quick reference

| ID | Component | FRs | Owner (per SPEC §10) |
|---|---|---|---|
| C1 | Store & Schema | 8, 38 | A |
| C2 | Ingestion | 1, 2, 3, 4, 7, 51 | A/B/D |
| C3 | Universe Constructor | 52, 54 | A |
| C4 | Feature Deriver | 5, 6, 22, 50, 53 | A/B — split: per-symbol/intraday (jump extraction, `S_t`) is B's; cross-sectional (`AR_t`, `BAS_t`, `G_t`) is A's |
| C5 | Hawkes Engine | 10–15 | B |
| C6 | RMT Engine | 16–19 | A |
| C7 | Hamilton Engine | 20, 21, 23, 24 | C |
| C8 | Reconciliation | 25–27 | C |
| C9 | Fusion Engine | 28, 29, 31 | D |
| C10 | Validation Harness | 30, 32–37 | D |
| C11 | Decision Support | 43–49 | D |
| C12 | Dashboard | 39–42 | D |

One AI session per component, matching this table. Don't let a single session's diff span two owners' directories — it's the single biggest lever for keeping review tractable and hallucination radius small.

---

## 9. When something's undefined in the spec

`docs/SPEC.md` §8 has real open questions (`[GAP]` / OQ items) — jump-test significance level, epoch-6 break date, undefined constants like kernel order `K`. **Don't silently invent a value.** Check `docs/SERAPH-BUILD-ROADMAP.md` §4 for the agreed default first; if it's not there, flag it and ask rather than guessing — several of these (e.g. FR52's filter/rank order) are exactly the kind of thing that looks like a reasonable guess and is actually a documented bug trap.
