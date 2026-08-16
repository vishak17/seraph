# SERAPH — Build Roadmap
**From spec to running system.** Synthesizes `SPEC.md` v3.1 + `ARCHITECTURE.md` v2 + `DATA-ACQUISITION.md` into a coding sequence. `SPEC-DELTA-01.md` is history only — not used here.

---

## 0. The one thing to internalize before writing code

**O1→O10 is not your build order.** It's the paper's narrative order. ARCHITECTURE.md §3–4 already solved dependency ordering for you: **12 components (C1–C12), 9 layers (L0–L8)**. Build in layer order. Objectives get satisfied as a *side effect* of components existing — not the other way round.

| Objective | Satisfied when… | Layer |
|---|---|---|
| O3 RMT | C6 exists | **L4 — build this pillar first** |
| O4 Hamilton | C7 exists | L4 |
| O8 Tariff covariate | C4 (`G_t`) + C7 exist | L3/L4 |
| O2 Hawkes | C5 exists | L4 — **build last of the three pillars, highest risk** |
| O6 Kalman reconciliation | C8 exists | L5 |
| O1 Unified CSRS | C9 + ≥1 pillar | L6 |
| O5 Regime-conditional weighting | C9 + C10 calibration | L6/L7 |
| O7 Ablation | C10 | L7 |
| O10 Decision support | C11 | L7 |
| O9 Cross-market | C10 + `yfinance` data | L7 — stretch goal, descope-safe |

The second thing to internalize: **T0, the mock-data generator, is the highest-leverage single artefact in this project.** Build it in the first two days. It emits synthetic `PillarEmission` / `ReconciledState` / `CsrsPoint` streams conforming to ARCHITECTURE §2's shapes. With it, C8→C9→C10→C11→C12 can all be built and tested **before a single real pillar exists.** Skipping this is the single most common way student projects end up with four people blocked on each other in week 6.

---

## 1. Repo skeleton (write this before anything else)

```
seraph/
├── pyproject.toml
├── docker-compose.yml                 # postgres+timescaledb, one command up
├── seraph/
│   ├── shared_types/                  # pydantic mirrors of ARCHITECTURE §0/§2 — Result, Warning,
│   │                                   # SeraphError, PillarEmission, ReconciledState, CsrsPoint, FusionWeights…
│   │                                   # EVERY component imports from here. Nothing redefines a shape locally.
│   ├── store/                          # C1
│   │   ├── schema.sql                  # DDL, hypertables on `ts`/`date`
│   │   ├── writer.py                   # StoreWriter.writeBatch — idempotent on (entity, natural key)
│   │   └── reader.py                   # StoreReader.* query methods
│   ├── ingestion/                       # C2
│   │   ├── eod2_client.py               # daily OHLCV, wraps BennyThadikaran/eod2
│   │   ├── bhavcopy_reader.py            # legacy + UDiFF schema, fallback path
│   │   ├── kite_client.py                 # intraday bars, checkpointed/resumable
│   │   ├── macro_client.py                 # RBI DBIE + FRED
│   │   ├── flow_client.py                   # FII/DII (E14)
│   │   └── tariff_events.py                  # E5, hand-curated table
│   ├── universe/                          # C3
│   │   └── constructor.py                  # FR52 liquidity-rank rule + D3 buffer rule
│   ├── features/                            # C4
│   │   ├── jump_extraction.py                # FR50, Lee–Mykland
│   │   ├── spread_estimators.py               # FR6, Abdi–Ranaldo / Corwin–Schultz / Roll
│   │   ├── tick_test.py                        # FR5, bar-level signed volume
│   │   ├── cross_sectional.py                   # AR_t (PCA), BAS_t — computed here, NOT in C6
│   │   └── tariff_covariate.py                    # FR22, G_t
│   ├── pillars/
│   │   ├── rmt/            engine.py           # C6 — build first
│   │   ├── hamilton/       engine.py           # C7 — build second
│   │   └── hawkes/         engine.py           # C5 — build last
│   ├── reconciliation/                            # C8
│   │   ├── kalman.py                                # predict/update
│   │   └── noise_model.py                            # D4 R^(p)(Δ), D2 mask
│   ├── fusion/                                        # C9 — PURE, no I/O
│   │   ├── score.py
│   │   ├── shapley.py                                  # 3 pillars ⇒ exact, 8 subsets, no approximation needed
│   │   └── standardize.py                              # rolling 5y empirical-CDF
│   ├── validation/                                      # C10
│   │   ├── harness.py                                    # FR30 calibration, D5 LOEO
│   │   ├── ablation.py                                    # FR35 Table A/B runner
│   │   └── preregistration.py                              # D5(d) hash-gated config
│   ├── decision_support/                                    # C11
│   └── dashboard/                                            # C12 — Streamlit
├── fixtures/
│   └── mock_generator.py            # T0 — build in the first 2 days, before C1 is even finished
└── tests/
    └── contract/                     # CT-1 … CT-10, one file each — write the test before the component
```

---

## 2. Phase-by-phase plan

Each phase lists: what to code, which FRs it satisfies, how you know it's done, and rough effort against a ~16-week window. Adjust the weeks to your actual deadline — the **order** is what matters, not the pacing.

### Phase 0 — Setup (Week 1)

- [ ] Repo skeleton above, `docker-compose up` gives a working Postgres+TimescaleDB
- [ ] Zerodha account + Kite Connect paid tier confirmed at signup (₹500/mo per DATA-ACQUISITION §1)
- [ ] Clone `BennyThadikaran/eod2`, run one full sync, spot-check 5 known split events adjust correctly
- [ ] **Probe the earliest real 5-min candle for 3 symbols** (Nifty 50 name / mid-cap / a solar name) — this is the ten-minute test that pins your actual Pillar-1 start date. Don't assume ~2015, verify it.
- [ ] Pull RBI repo rate + credit growth (DBIE), Brent + INR (FRED), confirm India VIX starts Nov 2007
- [ ] Test-pull FII/DII bulk history, note how far back it actually goes (OQ8b — unresolved in your docs)
- [ ] **Resolve the open questions that block early phases** — see §5 below, don't skip this

**Definition of done:** you can run one script that pulls one day of real data from each of the 4 external sources and lands it in a dataframe. Nothing persisted yet.

---

### Phase 1 (L0) — C1 Store & Schema + T0 Mock Generator

Two people can do these in parallel; one person can do them sequentially in ~4–5 days.

**C1 — Store & Schema**
- [ ] `shared_types/`: pydantic models for every field in SPEC §6 (E1–E14) — `DailyBar`, `IntradayBar`, `MacroRow`, `MicroState`, etc., exactly matching ARCHITECTURE §2's TS shapes
- [ ] `schema.sql`: hypertables partitioned on the timestamp axis (FR8), one table per entity
- [ ] `StoreWriter.writeBatch()` — **idempotent on (entity, natural key).** This is a hard requirement, not a nice-to-have — NSE sources fail often and you'll re-run ingestion constantly. Use `INSERT … ON CONFLICT DO UPDATE` keyed on `(symbol, ts)` or `(symbol, date)`.
- [ ] `getWatermark()` per `(source, entity)` — tracks how far each ingestion source has progressed
- [ ] `StoreReader`: `dailyBars()`, `intradayBars()`, `macroCovariates()`, `microstructureState()`

**T0 — Mock generator** (build this alongside C1, don't wait)
- [ ] Generate synthetic `PillarEmission` streams for all three pillars, including deliberately-`unavailable` periods (both `structural` and `transient` absence)
- [ ] Generate synthetic `ReconciledState` and `CsrsPoint` conforming exactly to ARCHITECTURE §2 shapes
- [ ] This one script is what lets C8/C9/C10/C11/C12 start immediately, in parallel with C2–C7

**Definition of done — CT-1** (ingestion idempotency): `writeBatch()` called twice on identical data → second call reports `inserted: 0`; watermark advances exactly once. Write this test file before `writer.py` is finished.

---

### Phase 2 (L1) — C2 Ingestion

- [ ] `eod2_client.py` — wraps the EOD2 sync, handles both bhavcopy schemas (pre/post July-2024 UDiFF transition per DATA-ACQUISITION §2)
- [ ] `kite_client.py` — **checkpointed and resumable**, respecting the per-interval request caps (5-min bars: 100 days/request, ~3,700 requests total, budget 1–2h with backoff). Don't write a script that dies at request 2,000 and starts over.
- [ ] `macro_client.py`, `flow_client.py`, `tariff_events.py` (hand-curated — this one's manual data entry, not code, budget real hours per DATA-ACQUISITION §3)

**Definition of done:** full historical sync completes and is idempotent; NFR8 volumes roughly match (~2.6M daily rows, ~19M intraday rows at 5-min).

---

### Phase 3 (L2) — C3 Universe Constructor + C4 (per-symbol features)

**C3 — Universe Constructor (FR52)**
- [ ] Filter to symbols with ≥1000 trading days history **as of each semi-annual rebalance date**
- [ ] Rank by 6-month median `TOTTRDVAL`, take top 500 — **filter before rank**, this order is explicitly called out as a bug trap in FR52
- [ ] D3 buffer rule: incumbent retained while rank ≤ 550, entrant admitted only at rank ≤ 450 (reduces rebalance churn)
- [ ] `members.length === 500` is a hard invariant — underfilled universe raises `UNIVERSE_UNDERFILLED`, never silently ships 480 names

**C4 — per-symbol features**
- [ ] `jump_extraction.py` (FR50) — Lee–Mykland bipower-variation test on intraday log-returns → `is_jump`, `jump_sign`. **You still need to pick the significance level and local window length** — this is OQ16, undefined in your own spec. Pick a defensible default (Lee–Mykland's own paper's recommended significance, e.g. α=0.01 with a local window of ~20–40 observations) and document the choice; don't leave it open indefinitely.
- [ ] `tick_test.py` (FR5) — sign bar volume by close-to-close return sign
- [ ] `spread_estimators.py` (FR6) — Abdi–Ranaldo (primary), Corwin–Schultz (fallback), Roll (sanity check); discretize spread × imbalance into `S_t` (25-value joint quintile)

**Definition of done — run these two contract tests here, before C5/C6 exist:**
- **CT-2** (universe cardinality/churn) — synthetic 600-symbol panel with engineered traps; assert exactly 500 members, short-history names excluded, churn lower with the buffer rule than without
- **CT-3** (jump sufficiency, **run this first, it's your highest-probability failure mode — R2 in the risk register**) — one real symbol-month of 5-min bars, no estimation, pure counting: negative jumps/symbol-day above floor, events-per-`S_t`-state above floor across ≥2 states. **If this fails, you need to know in week 3, not week 10** when you're debugging why the Hawkes MLE won't converge.

---

### Phase 4 (L3) — C4 (cross-sectional)

- [ ] `cross_sectional.py`: `AR_t` from top-10 principal components of the Nifty-500-analogous universe's covariance matrix; `BAS_t` as cross-sectional median Abdi–Ranaldo spread
- [ ] Deliberately computed **here in C4, not inside C6** — ARCHITECTURE explicitly calls this out: putting `AR_t` in the RMT engine would make Pillar 3 secretly depend on Pillar 2, which breaks FR35 (you couldn't evaluate Hamilton standalone). Duplicating a covariance computation is the correct price to pay.

---

### Phase 5 (L4) — The three pillar engines

Build in this order regardless of team size: **RMT → Hamilton → Hawkes.** RMT is lowest-risk and gets you a working end-to-end path fastest; Hawkes carries the real technical uncertainty (R2) and should not be the thing blocking your first full run.

**C6 — RMT Engine (O3) — build first**
- [ ] Rolling `C_t = (1/T) R_t R_tᵀ`, T=1000 days, N=500 (`Q=T/N=2`)
- [ ] `numpy.linalg.eigh` for the eigendecomposition; MP-exceedance fraction `F_t`, eigenvalue collapse velocity `V_t^eig`, leading-eigenvector rotation speed `Ω_t`
- [ ] **`Ω_t` on common support only** (D3a) — restrict to `members(t) ∩ members(t−Δ)`, renormalize both eigenvectors before the inner product. This is described in ARCHITECTURE as a *latent correctness bug*, not cosmetic — implement it correctly the first time.
- [ ] 1,000-replication parametric bootstrap → p-value alongside `F_t`
- [ ] `SCA_t` — calibrated-weight aggregate of the three spectral signals

**Definition of done — CT-10**: synthetic panel, forced 15% membership change at a known date, correlation structure held constant → `Ω_t` shouldn't spike at the rebalance; `commonSupportFraction ≥ 0.85`.

**C7 — Hamilton Engine (O4)**
- [ ] EM estimation of a 3-state TVTP Hamilton filter on `y_t = (RV_t, AR_t, BAS_t)`. **`statsmodels.tsa.regime_switching.markov_switching`** supports TVTP directly via its `exog_tvtp` parameter — check it against your spec before hand-rolling the EM loop.
- [ ] `RV_t` via Yang–Zhang OHLC estimator (full corpus) / true 5-min RV where available; multinomial logit on `z_t` for transition probabilities
- [ ] Emit `ξ_{j,t}`, `LSD_t = ξ_2 + 2ξ_3`, `τ_{1/2} = ln(2)/(1−p̂_jj)`
- [ ] `tariff_covariate.py` (FR22, O8) feeds into `z_t` here: `G_t = Σ |Δr_k| e^{−(t−τ_k)/η}` over the dated tariff-event table

**C5 — Hawkes Engine (O2) — build last, start research earliest**
Because this is your highest-risk component, it's worth starting the *literature/API* work in Phase 0–2 downtime even though the estimation code comes last.
- [ ] MLE per rolling one-day window on negative jump events, conditional on `S_t` — spectral-radius regularization enforcing `n(S_t) < 1` on every window (FR11)
- [ ] Asymmetric kernels (inhibitory positive→negative), `MTS_t = n(S_t)/(1−n(S_t))` gated by asymmetry threshold
- [ ] Per-sector re-estimation → amber micro-flags; contagion network from steady-state `Φ(S_t)`
- [ ] **The `tick` package** (X-DataInitiative) is in your NFR36 stack list and handles standard multivariate Hawkes MLE, but likely doesn't support `S_t`-conditional excitation kernels out of the box — plan to use it as a baseline/sanity-check, with custom `scipy.optimize` for the state-dependent likelihood.
- [ ] **If CT-3 (Phase 3) showed sparse jump counts**, apply the pre-agreed mitigation ladder in order: drop to 1-min bars → coarsen `S_t` to 9 states (3×3 terciles) → pool across symbols within sector → lengthen the estimation window.

---

### Phase 6 (L5) — C8 Reconciliation

- [ ] Kalman predict step every tick (random-walk transition, `Q_proc`)
- [ ] Kalman update step on each pillar emission, using `R^(p)(Δ)` — the **D4 bounded saturating exponential**: `R^(p)(Δ) = R₀ + (R_max−R₀)(1−e^{−Δ/h_p})`, `h_p` initialized to each pillar's native cadence, `R_max = 100·R₀`
- [ ] D2 availability mask: `structural` absence → pillar excluded from mask; `transient` absence → stays in mask, noise inflates
- [ ] Forward-fill retained only as the ablation baseline (FR27)

**You can build and test this entire component against T0 mocks before a single real pillar exists** — this is the whole point of T0. Don't wait for Phase 5 to finish.

**Definition of done — CT-4** (this test *is* Objective 6, per ARCHITECTURE): mock stream where Hawkes returns `structural` absence for 200 ticks while RMT/Hamilton observe normally → `xHat[0]` doesn't collapse to 0, `trace(P)` grows then saturates (not unbounded), `mask[0]===false`; on the first real Hawkes emission, `P[0][0]` drops and `mask[0]` flips true.

---

### Phase 7 (L6) — C9 Fusion

- [ ] `score()` — **pure function, no I/O.** `CSRS_t = Σ_j ξ_{j,t} w_jᵀ x̂_t`, each sub-score standardized via rolling 5-year empirical-CDF first
- [ ] D2 mask renormalization: `w_j^(m) = (m ⊙ w_j)/Σ(m ⊙ w_j)`
- [ ] Shapley decomposition — **with only 3 pillars there are 2³=8 coalition subsets, so compute exact Shapley by brute-force enumeration.** No need for an approximation library here.
- [ ] `Var(CSRS_t) = Σ_j ξ²_j w_jᵀ P_t w_j` + confidence interval

Also buildable against T0 mocks before C8 is finished, per the same logic as Phase 6.

**Definition of done:**
- **CT-6** (purity/leakage): `score()` called 100× on identical inputs → 100 byte-identical outputs
- **CT-9** (mask renormalization, D2): scoring the same `ReconciledState` with mask `[T,T,T]` vs `[F,T,T]` — masked result must be **exactly identical** to the corresponding Table B ablation row. This equality is a design invariant, not a coincidence — if it doesn't hold, something's wrong in the renormalization.

---

### Phase 8 (L7) — C10 Validation Harness + C11 Decision Support

**C10 — Validation Harness**
- [ ] `harness.py`: FR30 joint weight optimization, `L = κ_miss Σ_{t∈C}(1−CSRS_t) + κ_fa Σ_{t∉C} CSRS_t`
- [ ] **D5 split policy — leave-one-epoch-out.** Table A (4 Hawkes-covered epochs, 4 folds), Table B (all 6, 6 folds). Report mean AUC **with per-fold spread**, never mean alone — at n=6 that's the honest number.
- [ ] `preregistration.py`: hash the split policy + κ ratios + amber-threshold rule + full 14-config list **before computing the first AUC**. Every result row carries this hash; a mismatched hash makes the result unreportable.
- [ ] `ablation.py`: the 14-configuration runner (7 subsets × 2 reconciliation modes on Table A epochs, 3 subsets × 2 modes on Table B). This doubles as your system integration test.
- [ ] **`leakageCheck`** — machine check, intersect `weights.calibratedOn.epochs` with `testEpochs`; non-clean rows are not reportable

**C11 — Decision Support (7 artefacts, FR43–49)**
- [ ] Portfolio overlay (FR43), template-based plain-language explanation (FR44), market-level flow diagnostic (FR45, downgraded per SPEC — market-level only, no per-instrument), historical playbook (FR46), shock simulator (FR47), policy-path simulator (FR48), regulatory action log (FR49)

**Definition of done — CT-7**: `simulateShock` with magnitude 0 reproduces the unperturbed CSRS series exactly — if it doesn't, the simulator has forked the scoring logic instead of calling `score()` directly.

**Resolve before this phase's numbers are final:** which of the candidate 2025–26 tariff dates is Epoch 6's labelled break (OQ: 50% peak / Feb-2026 interim framework / solar CVD single-day drop), and where exactly your corpus's 2026 end date falls. Both need fixing now for reproducibility — pick one, write it down, move on.

---

### Phase 9 (L8) — C12 Dashboard

- [ ] Streamlit (per NFR36 and the "pick one now" instruction in SPEC §7 — Plotly Dash is explicitly out of scope)
- [ ] RBAC enforced **at the gateway, not the view layer** — retail/institutional/regulator payloads differ at the transport level
- [ ] CSRS gauge + CI, 3 pillar sub-scores with sector micro-flags, regime-probability stack, MP-density plot, contagion network, Shapley decomposition
- [ ] **Compliance note (DATA-ACQUISITION §1):** Kite-derived data may not be publicly redistributed. Keep this dashboard local/demo-only unless you clear otherwise — raise it with your guide before anyone builds a hosted version.

**Definition of done — CT-8**: retail-role payload contains no institution-level fields *at the transport layer* — absent, not merely hidden in the UI.

---

### Phase 10 — Backtest, report, stretch goals

- [ ] Run the full six-epoch validation, produce Tables A and B
- [ ] O9 cross-market transfer (stretch, AMBER feasibility) — RMT+Hamilton subset only, via `yfinance` on Ibovespa/JSE/IDX, weights bit-identical to NSE run (no retraining)
- [ ] Report amendments per SPEC §11 — abstract terminology, §5.1.2 observation-channel subsection, bibliography additions [17]–[22]

---

## 3. Contract-test run order (not build order — run these the moment their inputs exist)

| Test | Proves | Earliest runnable |
|---|---|---|
| **CT-3** | Jump stream dense enough to identify a Hawkes kernel | L2, before C5 exists — **run first, highest-risk item** |
| **CT-4** | Structural absence → `R→∞`, never zero | L0, against T0 mocks |
| **CT-6** | C9 is pure; train/test leakage caught mechanically | L0, against T0 mocks |
| **CT-9** | Degraded-mask scoring ≡ corresponding ablation subset | L0, against T0 mocks |
| **CT-2** | Universe always 500, filter-before-rank enforced | L2 |
| **CT-10** | `Ω_t` doesn't spike at rebalances | L2, before real data |
| **CT-5** | `Var(CSRS)` tracks the real covariance `P` | L0 |
| **CT-1** | Re-running ingestion doesn't duplicate rows | L0 |
| **CT-7** | Shock simulator doesn't fork scoring logic | L7 |
| **CT-8** | RBAC enforced at transport, not render | L8 |

Four of these (CT-4, CT-6, CT-9, CT-5) are runnable **on day 2**, against T0, before you've written a single pillar. Do that.

---

## 4. Open questions that will block you if left unresolved

Your own SPEC §8 flags these as open — resolve the ones marked ⚠ before the phase listed, or you'll stall mid-phase:

| Open question | Blocks | Suggested move |
|---|---|---|
| ⚠ Jump-test significance level + window (OQ16) | Phase 3 | Pick Lee–Mykland's standard default, document it, revisit only if CT-3 fails |
| ⚠ Hawkes symbol universe — Nifty 50 only vs + tariff-sector reps | Phase 3/5 | Go with Nifty 50 + tariff-exposed reps now — you need it for the §5.1.1 micro-flags anyway |
| Solar sector basket — no clean NSE index exists | Phase 3 | Hand-pick, document the selection rule (small task, don't block on it) |
| Epoch 6 break date (3 candidates) | Phase 8 | Pick one now, e.g. the 50% peak date — consistency matters more than which one |
| Corpus end date | Phase 2/8 | Fix a literal date now for reproducibility |
| Rebalance dates + lag (FR52) | Phase 3 | Two fixed calendar dates/year (e.g. Jan 1 / Jul 1), no lag, unless you have a reason otherwise |
| OQ8b — FII/DII retrievable depth | Phase 2/8 | Test-pull it in Phase 0; if it doesn't reach early epochs, FR45 quietly degrades — know this early, not in week 14 |

Undefined constants (kernel order K, amber/red CSRS thresholds, `κ_miss/κ_fa`, sector micro-flag threshold) don't need to be resolved up front — they're the things FR34's sensitivity analysis exists to sweep. Pick reasonable starting values and let the ablation runner (Phase 8) tell you if they matter.

---

## 5. If you're building this with a team, not solo

Your own SPEC §10 / ARCHITECTURE §4 already define a 4-person split that maps cleanly onto the phases above:

| Owner | Scope | Maps to |
|---|---|---|
| **A** | Data platform + RMT | Phases 1–4, then C6 in Phase 5 |
| **B** | Hawkes | C2(Kite) in Phase 2, then C5 in Phase 5 — hardest track, full-time from day 1 |
| **C** | Hamilton + Kalman reconciliation | C7 (Phase 5) then C8 (Phase 6) — same predict/update theory applied twice |
| **D** | Fusion + validation + dashboard + decision support | T0 (Phase 1) then C9→C10→C11→C12 (Phases 7–9), built against mocks from day 1 |

The mechanism that makes this actually parallel rather than sequential: **everyone codes against the shared-types contract with T0 mocks from day one.** C and D never wait on A or B. If you're solo, the phase order above already reflects the right sequence — you're just doing all four tracks yourself, RMT-first to get a working two-pillar system as early as possible.

---

*Each phase here can be expanded into actual code whenever you're ready to start it — ask for a deep-dive on any specific component (e.g. "let's write C1's schema" or "let's build the T0 mock generator") and we can go file by file.*
