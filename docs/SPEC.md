# SPEC.md — SERAPH — v3.1

**Source:** `draft_new.pdf` — *SERAPH: Systemic Early-Risk Alert for Portfolio Health*, Major Project Report, Dept. of CSE, BGSCET, VTU 2022 Scheme, 2026–27.
**Status of source:** Design-stage report. Implementation, calibration, back-testing and dashboard deployment are stated as planned future work; every quantitative figure in the paper is a design target, not a measured result.

### Changelog v1 → v2
1. **Tick data cut.** NSE tick-level order-flow data is unobtainable at project budget (≈₹20 lakh). Pillar 1 is re-specified as a **jump-event** Hawkes process on intraday bars.
2. **Corpus extended to 2026.** Six structural-break epochs, not five. Resolves the O8 conflict flagged in v1.
3. **LOB reconstruction cut.** State variable `S_t` is now built from spread and imbalance *estimators* computable from OHLCV.
4. **Hardware requirements collapse.** No tick ingestion → laptop-class deployment.
5. **Build order and team split added** (§9, §10).

### Changelog v2 → v3 — data acquisition merge
1. **Point-in-time index membership abandoned.** NSE's official inclusion/exclusion file stopped updating ~July 2020 and no reliable free source exists. FR52 is rewritten as a **liquidity-rank universe reconstruction**, which is survivorship-bias-free and reproducible. This removes the largest unbounded task in the project. Two places in v2 still instructed the team to curate membership by hand — both corrected.
2. **Data sources named.** Kite Connect (₹500/month, confirmed) for intraday; EOD2 for daily; nsepython or MrChartist/fii-dii-data for FII/DII.
3. **Sector taxonomy resolved** — NSE sectoral indices, not GICS (licensed).
4. **Redistribution constraint added to Out of Scope** — Kite-derived data may not be publicly redistributed.
5. **Six open questions closed** (1, 2, 4, 8, 11, and the OQ3 supersession confirmed).

### Annotation legend
- *(unmarked)* — stated explicitly in the paper.
- `[INFERRED]` — not in the paper; derived by me.
- `[GAP]` — the paper raises it and leaves it undefined. No value invented.
- `[v2]` — changed by the data feasibility amendment.
- `[v3]` — changed by the data acquisition merge.
- `[v3.1]` — open question closed by an ARCHITECTURE.md decision. **No requirement text was changed in v3.1** — only §8 open questions and the two notes below.

### Changelog v3 → v3.1 — architecture decision merge
**Requirement content is unchanged and remains frozen.** v3.1 records only that two open questions are now closed, and adds two clarifying notes where ARCHITECTURE.md resolves an ambiguity in existing requirements. No FR, NFR, objective or entity was modified.
1. **OQ10 closed** by decision D4 (`R^(p)(·)` form).
2. **OQ12 closed** by decision D5 (train/test split).
3. **FR28 clarification note** added — what the formula means when a pillar is structurally absent (decision D2).
4. **FR17 clarification note** added — `Ω_t` on common support (decision D3).

---

## 1. PROBLEM STATEMENT

Systemic stress in equity markets concentrates along three structural fault lines — self-exciting sell-side pressure, breakdown of the inter-asset correlation eigenstructure, and non-linear liquidity deterioration during regime transitions — which the existing literature models in isolation, at mismatched temporal resolutions, and reports in academic-paper form rather than as decision-grade output. No published framework fuses Hawkes-process contagion modelling, Random Matrix Theory spectral monitoring, and Markov regime-switching into a single composite score; no reviewed composite framework propagates estimation uncertainty into its output or represents dated trade-policy events as a regime covariate; and only two of fifteen surveyed studies use Indian market data. The consequence is that participants on India's NSE have no real-time, integrated, uncertainty-aware, interpretable early-warning instrument operating at a resolution sufficient to support preventive action.

---

## 2. OBJECTIVES

Objective text is verbatim from Chapter 6. All acceptance criteria are `[INFERRED]` — the paper states none.

### O1 — Unified composite framework
> "To design and implement SERAPH, a unified quantitative early-warning framework for systemic financial-market instability on the Indian NSE, integrating order-flow toxicity, correlation eigenstructure monitoring, and volatility-regime detection into a single jointly calibrated composite score (CSRS)."

**FUZZY.** "Unified" and "integrating" have no failure condition; satisfied by any pipeline emitting one number.

**Reformulation `[INFERRED]`:** One process shall emit, for every timestamp in the 2005–2026 validation set, a scalar `CSRS_t ∈ [0,1]` such that (a) perturbing any one pillar input while holding the others fixed changes `CSRS_t`, and (b) all nine weight parameters `{w_j}` come from a single optimisation run under one loss function.

**ACCEPTANCE:** Sensitivity test passes for all three pillars; calibration log shows exactly one optimisation objective producing all nine weights.

**Feasibility: GREEN.**

---

### O2 — Sell-side Hawkes contagion modelling
> "To model self-exciting and cross-exciting order-flow dynamics using a sell-side conditional multivariate Hawkes process with state-dependent excitation kernels, extracting the branching ratio as a real-time indicator of market endogeneity and toxic trading behaviour that adversely affects price fairness and liquidity provision."

`[v2]` **Observation channel changed.** Order-flow events are unobservable at budget. The pillar models **negative price-jump events** extracted from intraday bars. The mathematics of Equations 5.1–5.4 is unchanged; only the definition of an event changes. See §3 (FR50) and §11.

**ACCEPTANCE:** MLE converges on ≥95% of rolling one-day windows; `n(S_t) < 1` on 100% of windows post-regularisation; fitted `α, β` differ significantly across at least two distinct `S_t` states (state-dependence non-degenerate); `MTS_t` emitted at bar frequency with no gaps during NSE trading hours; negative/positive jump asymmetry is computable and non-constant.

**Partially untestable:** "adversely affects price fairness and liquidity provision" is a claim about the world, not the system. Drop from acceptance, or reformulate as a stated-significance correlation between `n(S_t)` and contemporaneous spread-estimator widening. `[GAP]` — no target in the paper.

**Feasibility: AMBER.** Highest technical risk of the three pillars. Requires the §11 report rewrite and three new citations.

---

### O3 — RMT spectral monitoring of Nifty 500
> "To monitor the spectral structure of the Nifty 500 cross-asset correlation matrix using Random Matrix Theory, detecting systemic coupling and diversification failure through the fraction of eigenvalues exceeding the Marchenko–Pastur upper bound, the eigenvalue collapse velocity, and the eigenvector rotation speed as forward-looking precursors of correlation breakdown affecting household savings and pension funds."

**ACCEPTANCE:** `F_t`, `V_t^eig`, `Ω_t` computed daily across 2005–2026 with `Q = T/N = 2`; a bootstrap p-value accompanies every `F_t`; `V_t^eig` crosses its calibrated threshold 2–5 trading days ahead of each labelled break (NFR15).

**Feasibility: GREEN.** `[v2]` Entirely unaffected by the data downgrade — this pillar was never tick-dependent. **Build first.**

---

### O4 — Three-state Hamilton filter regime detection
> "To detect transitions among normal, stressed, and crisis volatility regimes using a multivariate three-state Hamilton filter under a TVTP specification conditioned on realised volatility, the Absorption Ratio, and bid–ask spread dynamics, and to estimate liquidity recovery half-life as a forward-looking diagnostic of market resilience following shock events."

`[v2]` `RV_t` and `BAS_t` are computed from estimators rather than intraday/quote data. `AR_t` is unchanged.

**ACCEPTANCE:** EM converges to a non-degenerate three-regime solution with ordered means; `max_j ξ_{j,t} > 0.90` for the correct state within one trading day of each labelled transition (NFR16); crisis-regime RV mean ≥3σ below tranquil-regime mean (NFR17); `τ_{1/2}` finite and emitted daily for both stressed and crisis regimes.

**Feasibility: GREEN.**

---

### O5 — Regime-conditional dynamic fusion
> "To develop a novel regime-conditional dynamic weighting mechanism that adapts signal aggregation to the detected market state, calibrating the three component signals jointly and asymmetrically toward left-tail precursor detection, with empirical validation across five structural-break epochs (the 2008 GFC, the 2013 taper tantrum, the 2018 IL&FS crisis, the 2020 COVID-19 crash, and the 2022 tightening cycle) on NSE data spanning 2005–2023."

`[v2]` **Six** epochs, corpus **2005–2026**. Epoch 6 is the 2025–26 US–India tariff sequence.

**ACCEPTANCE:** Three distinct weight vectors `w_1 ≠ w_2 ≠ w_3`, each satisfying `1ᵀw_j = 1, w_j ⪰ 0`; held-out AUC > 0.85 averaged over epochs (NFR11); mean lead time of first amber crossing in 5–15 trading days (NFR12).

**Note:** The paper's anticipated weight ordering (spectral dominant in tranquil, contagion in stressed, regime in crisis) is stated as economic intuition, **not** a requirement. A different ordering is not a failure.

**Feasibility: GREEN.**

---

### O6 — Uncertainty-aware multi-timescale reconciliation
> "To design and validate a state-space (Kalman-filter) reconciliation layer that replaces naive forward-fill propagation across the tick-to-monthly frequency spectrum of the three pillars, producing a reconciled pillar vector x̂ₜ and a closed-form CSRS confidence interval, and to quantify the improvement of this layer over forward-fill on identical validation folds."

`[v2]` "tick-to-monthly" → **bar-to-monthly**.

`[v2]` **Strengthened by the downgrade.** Pillar 1 does not exist before ~2015 (§4, coverage table). Equation 5.18 already models each pillar as an intermittently-arriving observation with age-inflating noise `R^(p)(t − τ^(p))`; a permanently-absent pillar is the limiting case `R → ∞`. Pre-2015 epochs are handled by the existing machinery with **zero new mathematics**, and forward-fill has no defensible answer to a pillar that never existed. This is the strongest available empirical argument for O6.

**ACCEPTANCE:** Kalman and forward-fill variants run on byte-identical folds; false-alarm rate at fixed recall strictly lower for Kalman on days 1–2 of each labelled transition; `Var(CSRS_t)` monotonically non-decreasing between pillar updates and strictly decreasing at each arrival; empirical CI coverage within a stated tolerance of nominal. `[GAP]` — no required improvement magnitude, no coverage tolerance.

**Feasibility: GREEN.**

---

### O7 — Empirical ablation of the three-way fusion claim
> "To empirically test, rather than assert, the central premise of MG-1 by evaluating each pillar standalone, each pairwise combination, and the full three-way fusion on identical structural-break epochs, reporting AUC and mean lead time for every subset and establishing whether – and under what conditions – three-way fusion dominates."

`[v2]` **Two tables**, not one, because of the Hawkes coverage gap:
- **Table A** — 7 subsets × 2 reconciliation modes = 14 configurations, on the 4 Hawkes-covered epochs (2018, 2020, 2022, 2025–26).
- **Table B** — 3 subsets (RMT, Hamilton, RMT+Hamilton) × 2 modes = 6 configurations, on the 2 pre-2015 epochs (2008, 2013).

This is a natural experiment handed to you by the data constraint, not a deficiency. Report it as such.

**ACCEPTANCE:** Both tables produced on identical epochs, splits and evaluation windows, each row reporting AUC and mean lead time. The objective is met by *producing* the tables — dominance of the triple is a hypothesis, not an acceptance condition.

**Feasibility: GREEN.** Cheap once the harness exists; it is re-running, not new code.

---

### O8 — Event-based trade-policy covariate
> "To construct and incorporate a Tariff and Geopolitical Shock Index into the TVTP covariate vector zₜ, calibrated against the dated 2025–2026 US–India tariff sequence, and to test whether its inclusion improves regime-detection lead time relative to the standard macro covariate set alone."

`[v2]` **v1 CONFLICT RESOLVED.** The corpus now runs to 2026, so the tariff events fall inside the validation window. Epoch 6 is the targeted sub-analysis window and is the only fully-instrumented recent epoch (all three pillars available).

**ACCEPTANCE:** `G_t` computed from the dated event table; Pillar 3 estimated twice (with and without `G_t` in `z_t`); lead-time difference reported with a significance test on the epoch-6 window.

**Feasibility: GREEN.**

---

### O9 — Cross-market transferability
> "To calibrate SERAPH's fusion weights on NSE data and evaluate the same fixed weights, without retraining, on at least one additional emerging-market equity index, reporting out-of-sample AUC as a test of structural generalisability versus India-specific overfitting."

`[v2]` **Scoped down.** Daily data for Ibovespa / JSE / IDX is free via `yfinance`, so the **RMT + Hamilton subset** transfer test is fully deliverable. Full three-pillar transfer requires a second market's intraday history and is a stretch goal. Report the scoping explicitly rather than silently omitting it.

**ACCEPTANCE:** Weight vectors bit-identical between NSE and second-market runs (verified by parameter-file hash); out-of-sample AUC reported on ≥1 of {IDX, Ibovespa, JSE}; execution log contains no refit step.

**Feasibility: AMBER.** `[GAP]` — no minimum acceptable degradation bound; the paper only anticipates "moderate."

---

### O10 — End-user decision-support layer
> "To design a decision-support layer that translates CSRS and pillar-level outputs into role-specific, actionable content for retail investors (portfolio overlay, plain-language explanation, herd-share indicator), institutional risk managers (historical playbook, scenario simulator), and regulators (counterfactual policy simulation, feedback logging), so that SERAPH's output functions as a recommendation rather than only a reading."

**FUZZY.** The verb is "to design"; "functions as a recommendation" has no pass/fail condition.

**Reformulation `[INFERRED]`:** Seven named artefacts shall be demonstrable end-to-end against a live or replayed CSRS stream: (1) portfolio overlay reporting exposure as % of portfolio value; (2) single-sentence template explanation naming the dominant pillar; (3) herd-share indicator; (4) per-epoch historical playbook returning a ranked defensive action and its realised-drawdown reduction; (5) single-pillar shock scenario simulator; (6) counterfactual `z_t`-path policy simulator; (7) regulatory action log keyed to alert decomposition.

**ACCEPTANCE:** Each produces correct output for ≥1 scripted end-to-end test. `[v2]` Artefact (3) is **downgraded** to a market-level daily diagnostic (FR45) — per-instrument counterparty tags do not exist in bar data. Artefact (7) is logging only; using the log to recalibrate `κ` remains out of scope per the paper.

**Feasibility: AMBER.** 5 of 7 artefacts at full spec, 1 downgraded, 1 already long-term in the paper.

---

## 3. FUNCTIONAL REQUIREMENTS

IDs are stable across versions. Cut requirements are retained as tombstones so traceability does not silently break.

### Data acquisition & pre-processing
| ID | Requirement | Obj |
|---|---|---|
| FR1 `[v3]` | The system shall ingest daily OHLCV for all NSE equities covering 2005–2026 via the EOD2 pipeline (or direct bhavcopy, handling both pre- and post-July-2024 UDiFF schemas), and 5-minute OHLCV bars for a curated ~100-symbol subset via Kite Connect for the available lookback period. | O1, O2, O3 |
| FR2 | The system shall ingest daily adjusted closing prices for Nifty 500 constituents for correlation-matrix construction. | O3 |
| FR3 | The system shall ingest the macro covariate set — RBI repo rate, YoY bank credit growth, India VIX, trade-weighted INR, Brent crude — at each series' native publication frequency. | O4 |
| FR4 | The system shall ingest a hand-curated, dated event table of USTR / Section 301 / Section 122 announcements and Indian government responses, cross-checked against contemporaneous news coverage for announcement-date accuracy. | O8 |
| FR5 `[v2]` | The system shall sign each intraday bar's volume by the sign of its close-to-close return (bar-level tick test). *(Replaces Lee–Ready, which requires quotes.)* | O2 |
| FR6 `[v2]` | The system shall compute the Abdi–Ranaldo close-high-low spread estimator and a rolling signed-volume imbalance, and discretise the pair into the joint-quintile state variable `S_t`. *(Replaces LOB reconstruction.)* | O2 |
| FR7 | The system shall correct all price series for survivorship bias and corporate actions before use. | O1, O3 |
| FR8 | The system shall persist all series in PostgreSQL + TimescaleDB, partitioned on the timestamp axis. | O1 |
| FR9 | ~~Preserve retail/institutional counterparty tags~~ — **CUT `[v2]`.** Not present in bar data. Partially rescued by FR51. | — |
| FR50 `[v2]` **NEW** | The system shall extract negative and positive jump events per symbol from intraday log-returns using the Lee–Mykland bipower-variation test. | O2 |
| FR51 `[v2]` **NEW** | The system shall ingest daily NSE FII/DII net cash-market activity and total turnover, deriving a residual (non-institutional) flow share. | O10 |
| FR52 `[v3]` **REWRITTEN** | The system shall reconstruct a point-in-time universe at each semi-annual rebalance date by: (1) filtering to NSE equities with ≥1000 trading days of history as at that date; (2) ranking the eligible set by six-month median daily traded value (`TOTTRDVAL`); (3) selecting the top 500. **Filter before ranking** — ranking first and filtering after yields fewer than 500 names and breaks `Q = T/N = 2`. | O3 |
| FR54 `[v3]` **NEW** | The system shall report the correlation between the reconstructed universe's equal-weighted return and the official Nifty 500 index level, as evidence the reconstruction is representative. | O3 |

### Pillar 1 — Hawkes (jump-event)
| ID | Requirement | Obj |
|---|---|---|
| FR10 `[v2]` | The system shall estimate the conditional multivariate Hawkes process by maximum likelihood on rolling one-trading-day windows of **negative jump events**, conditional on `S_t`. | O2 |
| FR11 | The system shall apply spectral-radius regularisation so that `n(S_t) < 1` holds on every window. | O2 |
| FR12 `[v2]` | The system shall use asymmetric (inhibitory positive→negative) kernels so that positive-jump excitation does not contaminate the downside measure. | O2 |
| FR13 `[v2]` | The system shall emit `MTS_t = n(S_t)/(1 − n(S_t)) · 1{negative/positive jump asymmetry exceeds threshold}` at bar frequency, zeroed outside high-toxicity regimes. | O2 |
| FR14 | The system shall repeat the estimation per sector and raise an amber micro-flag when a sector-level branching ratio crosses its threshold. | O2, O10 |
| FR15 | The system shall construct a directed contagion network from the steady-state branching matrix `Φ(S_t)` for sectoral attribution. | O2, O10 |

### Pillar 2 — RMT *(unchanged from v1)*
| ID | Requirement | Obj |
|---|---|---|
| FR16 | The system shall compute, on every trading day, `C_t = (1/T)R_t R_tᵀ` over a rolling window of T = 1000 trading days across N = 500 constituents (Q = 2). | O3 |
| FR17 | The system shall eigen-decompose `C_t` and emit the MP-exceedance fraction `F_t`, the eigenvalue collapse velocity `V_t^eig`, and the leading-eigenvector rotation speed `Ω_t`. | O3 |
| | `[v3.1]` **Clarification (ARCHITECTURE D3), not a change:** `Ω_t` is an inner product of two leading eigenvectors. When universe membership changes at a rebalance, those vectors span different coordinate spaces and the product is undefined, not merely noisy. It is therefore computed on the common support `members(t) ∩ members(t−Δ)` with both vectors renormalised. This specifies *how* FR17 is computed; *what* it measures is unchanged. | |
| FR18 | The system shall aggregate the three spectral signals into `SCA_t` using calibrated weights `w^p`. | O3 |
| FR19 | The system shall run a parametric bootstrap with 1,000 replications at every daily update and publish a p-value for the MP-exceedance null alongside `SCA_t`. | O3 |

### Pillar 3 — Hamilton / TVTP
| ID | Requirement | Obj |
|---|---|---|
| FR20 `[v2]` | The system shall estimate a multivariate three-state TVTP Hamilton filter by EM on `y_t = (RV_t, AR_t, BAS_t)ᵀ`, where `RV_t` uses the Yang–Zhang OHLC estimator across the full corpus (and true 5-minute RV where intraday data exists), `AR_t` uses the top k = 10 principal components of the Nifty 500 covariance matrix, and `BAS_t` is the cross-sectional median Abdi–Ranaldo spread estimate. | O4 |
| FR21 | The system shall model transition probabilities via multinomial logit on `z_t`, estimating TVTP coefficients jointly with regime-conditional moments under a state-dependent structural-break process. | O4 |
| FR22 | The system shall compute `G_t = Σ_{k: τ_k ≤ t} |Δr_k| · e^{−(t−τ_k)/η}` and append it to `z_t`, plus a sector-weighted variant restricted to steel, aluminium, auto components, solar and pharmaceuticals. | O8 |
| FR23 | The system shall emit `{ξ_{j,t}}` and `LSD_t = ξ_{2,t} + 2ξ_{3,t}` daily, with per-update estimation uncertainty. | O4, O6 |
| FR24 | The system shall compute `τ_{1/2} = ln(2)/(1 − p̂_jj)` for the stressed and crisis regimes at the current `z_t`, published alongside `LSD_t`. | O4 |
| FR53 `[v2]` **NEW** | The system shall report the correlation between Yang–Zhang OHLC volatility and true 5-minute realised volatility over the overlapping period, as a validation check on the estimator substitution. | O4 |

### Reconciliation & fusion *(unchanged from v1)*
| ID | Requirement | Obj |
|---|---|---|
| FR25 | The system shall run a Kalman predict step at every update tick, advancing `x̂_t` and `P_t` under a random-walk transition with covariance `Q_proc`. | O6 |
| FR26 | The system shall run a Kalman update step whenever a pillar emits a new value, using observation-noise variance `R^(p)(t − τ^(p))` that grows monotonically with time since that pillar's previous update. | O6 |
| FR27 | The system shall pass only `x̂_t` and `P_t` to the fusion layer in production, retaining forward-fill solely as a selectable ablation baseline. | O6, O7 |
| FR28 | The system shall compute `CSRS_t = Σ_j ξ_{j,t} w_jᵀ x̂_t` subject to `1ᵀw_j = 1, w_j ⪰ 0`, after each sub-score is standardised to [0,1] by empirical-CDF transform on a rolling five-year window. | O1, O5 |
| | `[v3.1]` **Clarification (ARCHITECTURE D2), not a change:** FR28 does not define the case where a pillar is structurally absent (e.g. Hawkes pre-2015, SPEC §4). Weights are renormalised over an availability mask, `w_j^(m) = (m ⊙ w_j)/Σ(m ⊙ w_j)`, which still satisfies `1ᵀw_j = 1, w_j ⪰ 0` — it selects a point inside FR28's existing constraint set rather than altering it. A CSRS computed under a degraded mask must be labelled as such wherever it appears. | |
| FR29 | The system shall compute `Var(CSRS_t) = Σ_j ξ²_{j,t} w_jᵀ P_t w_j` and store a calibrated confidence interval with every CSRS value. | O6 |
| FR30 | The system shall jointly optimise `{w_j}` on labelled crisis windows under `L = κ_miss Σ_{t∈C}(1 − CSRS_t) + κ_fa Σ_{t∉C} CSRS_t`, with `κ_miss ≫ κ_fa`. | O5 |
| FR31 | The system shall compute pillar-level Shapley contributions at every update step. | O1, O5, O10 |

### Validation
| ID | Requirement | Obj |
|---|---|---|
| FR32 `[v2]` | The system shall evaluate against **six** labelled structural-break epochs (2008 GFC, 2013 taper tantrum, 2018 IL&FS, 2020 COVID-19, 2022 tightening, 2025–26 tariff sequence), each crisis window being the 30 trading days following the labelled break, with a 5–15 trading-day early-warning lead window. | O5, O8 |
| FR33 | The system shall report AUC against binary crisis labels, mean lead time to first amber crossing, and lead-window-averaged pillar Shapley contributions. | O5, O7 |
| FR34 | The system shall run sensitivity analyses varying rolling-window length, Hawkes kernel order K, Absorption-Ratio component count, and `κ_miss/κ_fa`. | O5 |
| FR35 `[v2]` | The system shall produce **Table A** (7 subsets × 2 reconciliation modes, on the 4 Hawkes-covered epochs) and **Table B** (3 subsets × 2 modes, on the 2 pre-2015 epochs), each reporting AUC and mean lead time. | O7 |
| FR36 | The system shall run each ablation protocol twice — forward-fill and Kalman — to isolate the reconciliation layer's marginal contribution. | O6, O7 |
| FR37 `[v2]` | The system shall apply NSE-calibrated fusion weights, without retraining, to ≥1 additional emerging-market index and report out-of-sample AUC, **scoped to the RMT + Hamilton subset** unless intraday data for that market proves free and available. | O9 |
| FR38 | The system shall archive all sub-scores, regime probabilities, reconciliation covariances, CSRS values, ablation results and cross-market results for retrospective analysis. | O1, O7 |

### Dashboard & decision support
| ID | Requirement | Obj |
|---|---|---|
| FR39 | The system shall render in real time: the CSRS gauge with confidence interval, the three pillar sub-scores including sector micro-flags, the regime-probability stack chart, the MP-density plot with shaded outlier zone, the directed contagion network, and the pillar Shapley decomposition. | O1, O10 |
| FR40 | The system shall trigger colour-coded amber and red alerts when CSRS crosses calibrated thresholds. | O5, O10 |
| FR41 | The system shall overlay historical CSRS on benchmark Nifty 500 returns to support visual back-testing. | O5 |
| FR42 | The system shall enforce role-based access control: regulators see institution-level decompositions, institutional users see portfolio-level views, retail users see a simplified traffic-light interface. | O10 |
| FR43 | The system shall accept an optional user-supplied holdings list and map sector-level micro-flags onto it, reporting exposure as a percentage of portfolio value. | O10 |
| FR44 | The system shall generate a template-based, single-sentence plain-language explanation of every alert from the Shapley decomposition. | O10 |
| FR45 `[v2]` | The system shall report a **market-level daily** institutional-versus-residual flow diagnostic derived from FII/DII net activity and total turnover. *(Downgraded from per-instrument counterparty decomposition, which bar data cannot support.)* | O10 |
| FR46 | The system shall expose, per labelled epoch, which defensive action (sector rotation, hedge-ratio adjustment, index-put overlay) would have most reduced realised drawdown and by how much, conditioned on the pillar decomposition active at the time. | O10 |
| FR47 | The system shall expose a self-serve scenario simulator injecting a hypothetical shock into any one pillar while holding the others at baseline, returning the resulting CSRS trajectory. | O10 |
| FR48 | The system shall expose a counterfactual policy simulator varying `z_t` along a hypothetical macro or policy path. | O10 |
| FR49 | The system shall record the regulator's action per alert (no action / soft warning / circuit-breaker) against that alert's pillar decomposition. | O10 |

---

## 4. COVERAGE MATRIX `[v2]`

Pillar 1 exists only for the intraday-data lookback period (~2015/16 onwards, pending broker API confirmation).

| Epoch | RMT | Hamilton | Hawkes |
|---|---|---|---|
| 2008 GFC | ✔ | ✔ ¹ | ✘ |
| 2013 taper tantrum | ✔ | ✔ | ✘ |
| 2018 IL&FS | ✔ | ✔ | ✔ |
| 2020 COVID-19 | ✔ | ✔ | ✔ |
| 2022 tightening | ✔ | ✔ | ✔ |
| **2025–26 tariff (NEW)** | ✔ | ✔ | ✔ |

¹ India VIX is unavailable before Nov 2007. Either drop the VIX covariate for that window or substitute Nifty index realised volatility. **Decide and document** — see OQ5.

**Handled by existing machinery.** Equation 5.18 models pillars as intermittently-arriving observations with age-inflating noise; an absent pillar is `R → ∞`. No new mathematics is required, and this is the cleanest empirical justification for O6.

---

## 5. NON-FUNCTIONAL REQUIREMENTS

### Latency & throughput
| ID | Requirement | Value |
|---|---|---|
| NFR1 `[v2]` | Bar-close → dashboard propagation. | < 5 s *(was < 1 s from tick)* |
| NFR2 `[v2]` | CSRS refresh cadence. | Bar frequency (1-min or 5-min). Sub-second is no longer meaningful and must not be claimed. |
| NFR3 | Kalman step overhead relative to pillar computation. | `[GAP]` — "negligible" unquantified; no ms budget |
| NFR4 `[v2]` | Pillar cadences. | Hawkes: bar frequency. RMT: daily. Hamilton: daily. |
| NFR5 | Pillars shall compute in parallel on dedicated worker processes. | — |

### Scale
| ID | Requirement | Value |
|---|---|---|
| NFR6 `[v2]` | Security count. | 500 for RMT (daily); ~100 for Hawkes (intraday) |
| NFR7 | Concurrent dashboard users. | ≥ 100 |
| NFR8 `[v3]` | Data volume. | ≈2.6M daily rows + **≈19M intraday rows at 5-min** (≈94M if 1-min). Fetch cost ≈3,700 Kite requests, ~1–2 h with backoff. Storage well under 50 GB. |

### Availability
| ID | Requirement | Value |
|---|---|---|
| NFR9 `[v2]` | Uptime. | 99.9% during NSE hours — **retained as a design target for a hypothetical production deployment, not a project deliverable.** State it as such in the report. |
| NFR10 `[v2]` | Failover. | **Downgraded to single-node with a documented restart procedure.** A four-person student project should not claim a failover cluster it will not build. |

### Accuracy targets *(design targets; none measured)*
| ID | Requirement | Value |
|---|---|---|
| NFR11 | Out-of-sample AUC averaged across epochs. | > 0.85 |
| NFR12 | Mean lead time of first amber crossing. | 5–15 trading days |
| NFR13 | Precision at fixed recall vs best single-pillar baseline. | ≥ +20% |
| NFR14 | Hawkes criticality crossing ahead of each major sell-off. | 3–7 trading days |
| NFR15 | RMT collapse-velocity lead on correlation breakdown. | 2–5 trading days |
| NFR16 | Hamilton posterior for the correct state within one day of a labelled transition. | > 0.90 |
| NFR17 | Crisis-vs-tranquil separation in the RV component. | ≥ 3 σ |
| NFR18 | Kalman shall reduce false-alarm rate at fixed recall vs forward-fill on days 1–2 of a transition. | Direction only; `[GAP]` on magnitude |
| NFR19 | CSRS CI shall widen before a Hamilton update and narrow after. | `[GAP]` — "measurably" unquantified |
| NFR20 | Cross-market AUC degradation. | `[GAP]` — "moderate"; no bound |

### Interpretability
| ID | Requirement |
|---|---|
| NFR21 | Every CSRS reading shall carry a Shapley decomposition, a confidence interval, sectoral attribution where applicable, and a plain-language explanation. |
| NFR22 | The retail view shall be comprehensible without a quantitative-finance background. `[GAP]` — no usability test defined. |

### Security
| ID | Requirement |
|---|---|
| NFR23 | TLS for all data in transit. |
| NFR24 | AES-256 at rest. |
| NFR25 | Role-based access control via OAuth2. |
| NFR26 | Immutable audit logs of every alert issued, every dashboard view rendered, and every logged regulatory action. |
| NFR27 | Log retention period. `[GAP]` |

### Maintainability
| ID | Requirement |
|---|---|
| NFR28 | Modular package boundaries separating the three pillars, the reconciliation layer, the fusion layer and the decision-support layer. |
| NFR29 | Automated test suite covering unit, integration and back-test regression. `[GAP]` on coverage %. |

### Platform `[v2]` — collapses to laptop class
| ID | Requirement | v1 | v2 |
|---|---|---|---|
| NFR30 | CPU | ≥ 16 physical cores | **≥ 8 cores** |
| NFR31 | RAM | ≥ 64 GB | **16 GB minimum, 32 GB comfortable** — the 500×500 correlation matrix is ~2 MB; the 64 GB figure existed only to hold tick data |
| NFR32 | GPU | 16 GB VRAM recommended | **CUT** — only ever served the out-of-scope EEMD–LSTM overlay |
| NFR33 | Network | 10 Gbps NIC | **CUT** — no live tick ingestion; ordinary broadband |
| NFR34 | Storage | 4 TB redundant SSD | **250 GB** |
| NFR35 | Deployment | 2-node replication | **Single node** |
| NFR36 | Stack | — | Python ≥ 3.10 (NumPy, SciPy, pandas, statsmodels, `tick`), `filterpy` or custom NumPy Kalman, R + `MSwM` via `rpy2` (optional secondary), PostgreSQL + TimescaleDB, Streamlit + Plotly, Docker, Ubuntu 22.04 LTS or equivalent |

---

## 6. DATA CONTRACT

All field-level types are `[INFERRED]` — the paper specifies no schema.

### E1 `[v2]` — `intraday_bars` *(replaces `tick_order_flow`)*
`symbol` string · `ts` timestamp · `open/high/low/close` decimal · `volume` int · `signed_volume` int (FR5) · `log_return` float · `is_jump` bool (FR50) · `jump_sign` enum{−1,+1}
**Source:** broker historical API. **Frequency:** 1-min or 5-min. **Coverage:** ~2015–2026, ~100 symbols.

### E2 `[v2]` — `microstructure_state` *(replaces `lob_state`)*
`symbol` · `ts` · `spread_ar` float (Abdi–Ranaldo) · `spread_cs` float (Corwin–Schultz cross-check) · `amihud_illiq` float · `signed_vol_imbalance` float · `spread_quintile` int 1–5 · `imbalance_quintile` int 1–5 · `S_t` int 1–25
**Source:** derived, not ingested.

### E3 `[v2]` — `daily_prices`
`symbol` · `date` · `open/high/low/close/volume` · `tottrdval` decimal · `adj_close` decimal · `log_return_std` float · `yz_volatility` float · `universe_member_flag` bool (FR52) · `listing_days_to_date` int
**Source:** EOD2 / NSE bhavcopy. **Coverage:** 2005–2026, all NSE equities (universe narrows to 500 per rebalance). **Frequency:** daily.

`[v3]` **Note:** `TOTTRDVAL` is traded *value* and is therefore split-invariant — a 1:10 split multiplies quantity by ten and divides price by ten. EOD2's documented caveat that pre-2005 data may not be fully split-adjusted therefore **does not affect universe selection at all**; it affects only the return series feeding `C_t`. State this in report §7.1 — it materially narrows the blast radius of that caveat.

### E4 `[v2]` — `macro_covariates` (`z_t`)
`date` · `rbi_repo_rate` (RBI DBIE) · `bank_credit_growth_yoy` (RBI DBIE) · `india_vix` (NSE, **Nov 2007+**) · `vix_available` bool · `inr_twi` (RBI) · `brent_price` (FRED/EIA) · `G_t` (derived from E5)
`[GAP]` — no alignment rule for mixing publication frequencies inside `z_t`.

### E5 — `tariff_events`
`event_id` · `tau_k` date · `delta_r_k` float · `severity_score` float *(scale undefined `[GAP]`)* · `sectors_affected` array\<string\> · `source_ref` string
**Source:** hand-curated. **Frequency:** event-driven.
**Known sequence:** ~25% mid-2025 → 50% peak (Russian-oil secondary sanctions) → 18% (Feb 2026 interim framework) → 10% Section 122 baseline post-Supreme-Court; sector duties 50% steel/aluminium, 25% certain auto components, solar CVD action (Feb 2026).

### E6 `[v2]` — `hawkes_output`
`ts` · `scope` enum{market, sector, instrument} · `branching_ratio_n` float (<1 enforced) · `MTS_t` float ≥0 · `jump_count_neg` int · `jump_count_pos` int · `asymmetry_ratio` float · `sector_microflag` enum{none, amber} · `branching_matrix_Phi` float[D][D] · `alpha, beta` per (i,j,k,S)
*(`retail_toxicity_share` removed — moved to E14.)*

### E7 — `rmt_output`
`date` · `eigenvalues` float[500] · `v1` float[500] · `F_t`, `V_eig_t`, `Omega_t`, `SCA_t` float · `mp_pvalue` float · `lambda_minus`, `lambda_plus` float. **Frequency:** daily.

### E8 `[v2]` — `hamilton_output`
`date` · `RV_t_yz`, `RV_t_5min` (nullable), `AR_t`, `BAS_t` float · `xi_1/2/3` float (sums to 1) · `LSD_t` float ∈[0,3] · `p_hat_22`, `p_hat_33`, `tau_half_stressed`, `tau_half_crisis` float · `estimation_uncertainty` float · `gamma_ij` float[3][3][dim z]. **Frequency:** daily.

### E9 — `reconciled_state`
`ts` · `x_hat` float[3] · `P_t` float[3][3] · `tau_last_update` timestamp[3] · `mode` enum{kalman, forward_fill}

### E10 — `csrs`
`ts` · `CSRS_t` float ∈[0,1] · `var_CSRS_t`, `ci_lower`, `ci_upper` float · `shapley_hawkes/rmt/hamilton` float · `alert_level` enum{green, amber, red} · `regime_weights_used` float[3][3]

### E11 — `user_portfolio`
`user_id` · `symbol` · `quantity` · `value` · `sector` · `exposure_pct` float. **Frequency:** user-supplied.

### E12 — `regulatory_action_log`
`alert_id` · `ts` · `action_taken` enum{none, soft_warning, circuit_breaker} · `pillar_decomposition_snapshot` float[3]

### E13 `[v2]` — `validation_results`
`pillar_subset` enum (7 values) · `reconciliation_mode` enum{kalman, forward_fill} · `epoch` enum (6 values) · `table` enum{A, B} · `auc`, `mean_lead_time_days`, `precision_at_recall` float

### E14 `[v2]` **NEW** — `flow_aggregates`
`date` · `fii_net` decimal · `dii_net` decimal · `total_turnover` decimal · `residual_flow_share` float
**Source:** NSE daily reports. **Frequency:** daily. **Powers:** FR45.

---

## 7. OUT OF SCOPE

Mentioned in the paper but not committed to:

1. **EEMD–LSTM forecasting overlay** — optional extension; the GPU requirement existed only for it. `[v2]` GPU cut with it.
2. **Volatility-jump contagion channel** (Nyawa et al.) — longer-horizon extension. *(Note: the double-Hawkes price/volatility split is now closer to reach than in v1, since the pillar already operates on jumps. Still out of scope for this build.)*
3. **Four-state Hamilton specification** (adding a recovery regime).
4. **Higher-order spectral statistics** — eigenvalue gap, participation ratio of the leading eigenvector.
5. **Using the regulatory feedback log to recalibrate `κ`** — the paper itself flags the data-accumulation horizon as too long. Logging (FR49) is in scope; recalibration is not.
6. **Production deployment of the cross-market variant** — only the one-off evaluation (FR37) is in scope.
7. **Plotly Dash** — an alternative, not a requirement. Pick one.
8. **R / `MSwM` via `rpy2`** — secondary environment; optional if the Python path suffices.
9. **Graph-theoretic risk measures** (centrality, PageRank, clustering coefficient) — surveyed, never required.
10. **NSE history back to 1994** — the committed corpus is 2005–2026.
11. **Competing aggregation methods of MG-5** (TEI@I, CISS-style weighting) — surveyed, never committed as comparators.
12. `[v2]` **True limit-order-book microstructure** — spread and imbalance are estimated, not observed. Any claim about queue dynamics, depth, or quote-level behaviour is out of scope and must not appear in the report.
13. `[v2]` **Per-instrument retail/institutional attribution** — impossible without counterparty tags. Only the market-level aggregate (FR45) is in scope.
14. `[v3]` **Public hosting of any dashboard rendering Kite-derived data.** Zerodha's FAQ states that displaying or redistributing Kite Connect data on external platforms violates exchange data vending policies — Kite is an execution platform, not a data vendor. The dashboard is **local / demo-only** unless cleared otherwise. Raise with Dr. Satisha before building a hosted demo.
15. `[v3]` **Official Nifty 500 index membership as the analytical universe.** Superseded by the FR52 liquidity-rank reconstruction. The official index level is retained only as a benchmark series (FR41) and validation reference (FR54).
16. **Any empirical result.** Every number in the paper's Chapter 9 is a design target.

---

## 8. OPEN QUESTIONS

### Resolved since v1
- ~~OQ1 Tick data availability~~ → **Resolved:** unobtainable. Jump-event reformulation adopted (§2 O2, FR50).
- ~~OQ2 2005–2023 vs 2025–26 conflict~~ → **Resolved:** corpus extended to 2026; six epochs.
- ~~OQ3 Level-1 vs Level-2~~ → **Superseded:** neither. Estimators from OHLCV (FR6).

### Closed in v3
- ~~OQ1 Broker API~~ → **Kite Connect, ₹500/month, confirmed.** Intraday from ~2015–16.
- ~~OQ2 Bar frequency~~ → **5-minute.** Matches paper [3]; 5× less data.
- ~~OQ4 Point-in-time membership~~ → **Sidestepped.** FR52 liquidity-rank reconstruction.
- ~~OQ8 FII/DII availability~~ → **Available** via nsepython / MrChartist-fii-dii-data / NSE reports. Depth of history still to be established (now OQ8b below).
- ~~OQ11 Sector taxonomy~~ → **NSE sectoral indices**, hand-mapped to the five tariff-exposed sectors. GICS excluded (licensed).

### Open — blocking
1. **Hawkes symbol universe** — Nifty 50 only, or Nifty 50 + tariff-exposed sector representatives? The latter is required for §5.1.1 micro-flags and the tariff demo. **Solar has no clean NSE sectoral index** — a hand-picked basket is needed, with the selection rule documented.
2. **India VIX pre-Nov 2007** — drop the covariate for the 2008 epoch, or substitute Nifty realised volatility? Confirmed: India VIX starts Nov 2007.
3. **Epoch 6 break date** — which of the candidate 2025–26 tariff dates is the label (50% peak, Feb 2026 interim framework, solar CVD single-day drop)? Pick one and justify.
4. **Corpus end date** — where exactly does 2026 stop? Fix now for reproducibility.
5. **OQ8b — how far back does FII/DII actually retrieve?** MrChartist's store is described as *rolling* history, not a deep archive. If it does not reach the early epochs, FR45 is cut. Establish the earliest retrievable date before building on it.
6. **Rebalance dates for FR52** — which two calendar dates per year, and does the universe change take effect immediately or with a lag? Affects reproducibility.

### Closed in v3.1 by ARCHITECTURE.md §7
- ~~**OQ10 `R^(p)(·)` functional form**~~ → **RESOLVED, decision D4.** Bounded saturating exponential `R^(p)(Δ) = R₀ + (R_max − R₀)(1 − e^(−Δ/h_p))`, with `h_p` initialised to each pillar's native update cadence and `R_max = 100·R₀`. Boundedness is decisive: an unbounded form sends `P → ∞` for a structurally-absent pillar and makes the FR29 confidence interval infinite. `h_p`, `R_max/R₀` and `Q_proc` remain MLE-fitted.
- ~~**OQ12 Train/test protocol**~~ → **RESOLVED, decision D5.** Leave-one-epoch-out; 4 folds for Table A (Hawkes-covered epochs only), 6 for Table B; mean AUC reported **with per-fold spread**; the ablation is descriptive, not selective — no configuration is chosen as best and then reported as a headline; a pre-registered, hashed configuration file gates reportability.

### Open — carried from v1
9. **Undefined constants** — kernel order `K`; the `MTS_t` asymmetry threshold; spectral weights `w^p_F, w^p_V, w^p_Ω`; tariff decay constant `η`; severity-score scale; `κ_miss/κ_fa`; amber and red CSRS thresholds; the sector micro-flag threshold. *(`Q_proc` retained here — D4 fixes the form of `R^(p)(·)`, not the process-noise magnitude.)*
11. **Sector taxonomy** — GICS is licensed. NSE sectoral indices, NIC codes, or a hand mapping? Who maintains it?
13. **Crisis labelling source** — who fixes the exact break date for each epoch, against what source?
14. **Cross-market index** — IDX, Ibovespa or JSE?
15. **Streamlit or Plotly Dash** — decide now; FR43–FR49 are non-trivial UI and porting later is wasted work.

### New `[v2]`
16. **Jump-test parameters** — Lee–Mykland significance level and local bipower window length.
17. **Is the mock-data contract written before the pillars?** Recommend yes — half a day of work that unblocks three people (§10).

---

## 9. BUILD ORDER `[v2]`

**O3 → O4 → O1/O5 (fusion skeleton on two pillars) → O6 → O2 → O7 → O8 → O10 → O9**

Rationale: a working two-pillar CSRS early means there is always a submittable system. Pillar 1 is the highest-risk component under the new data regime and must not be the thing blocking the first end-to-end run. O9 is the cheapest to descope honestly if time runs short — running it on the RMT+Hamilton subset and saying so is a legitimate result.

---

## 10. TEAM ALLOCATION `[v2]`

| Owner | Scope | Starts |
|---|---|---|
| **A** | Data platform + **Pillar 2 (RMT)**. Bhavcopy ingestion, TimescaleDB, corporate actions, point-in-time membership, spread/volatility estimators, then RMT. **Owns the §6 data contract.** | Day 1 |
| **B** | **Pillar 1 (Hawkes).** Broker API, Lee–Mykland jump extraction, `S_t` construction, MLE + regularisation, sector micro-flags, contagion network. Hardest component, full-time. | Day 1 (API + theory), real work once A ships bars |
| **C** | **Pillar 3 (Hamilton) + Kalman reconciliation.** Both are predict/update recursions over a latent state — one body of theory, learned once, applied twice. | Day 1 |
| **D** | **Fusion + validation harness + dashboard + decision support.** Harness first against mock pillar outputs, then fusion, then the ablation runner, then dashboard and the §5.6 artefacts. **Owns the ablation runner**, which doubles as the system integration test. | Day 1 |

**Mechanism:** everyone codes against the §6 data contract with **mock pillar outputs from day one**. D can build and test the entire fusion, ablation and dashboard stack on synthetic E6/E7/E8 rows before a real pillar exists. Nobody blocks on anybody.

---

## 11. REPORT AMENDMENTS REQUIRED `[v2]`

| Section | Edit |
|---|---|
| Abstract | "tick-to-monthly" → "bar-to-monthly"; Pillar 1 framing "order-flow toxicity" → "return-jump contagion" |
| §1 Introduction | Same terminology change; state the observation-channel constraint |
| §5.1 | Rewrite event definition; **add §5.1.2 "Observation-Channel Constraint and Jump-Event Reformulation"** |
| §5.1.1 | Unchanged — sector micro-flags work identically on jump events |
| §5.3 | `RV_t` and `BAS_t` estimator substitution |
| §5.5 | **Add the "absent pillar = `R → ∞`" argument.** Strongest paragraph available and it does not exist yet. |
| §7.1 Stage 1 | Full rewrite — new sources (EOD2/bhavcopy, Kite Connect), no Lee–Ready, no LOB reconstruction. `[v3]` **Add the liquidity-rank universe reconstruction and its survivorship-bias justification** (FR52). Note that `TOTTRDVAL` is split-invariant, so the pre-2005 adjustment caveat does not affect universe selection. Wording throughout: **"a Nifty-500-analogous universe of the 500 most liquid NSE equities, reconstructed point-in-time by rolling six-month median traded value"** — not "the Nifty 500". |
| §7.5 | Six epochs |
| §8.1–8.3 | Amended FRs and NFRs; the hardware section shrinks substantially |
| §9 | Latency targets; add the coverage-asymmetry ablation as an expected outcome |
| Bibliography | Add [17]–[22] below |

### Citations to add
- **[17]** Aït-Sahalia, Cacho-Díaz & Laeven, "Modeling financial contagion using mutually exciting jump processes," *Journal of Financial Economics*, 2015. — foundational jump-contagion.
- **[18]** Lee & Mykland, "Jumps in financial markets: a new nonparametric test and jump dynamics," *Review of Financial Studies*, 2008. — the event-extraction method.
- **[19]** Abdi & Ranaldo, "A simple estimation of bid-ask spreads from daily close, high, and low prices," *Review of Financial Studies*, 2017.
- **[20]** Corwin & Schultz, "A simple way to estimate bid-ask spreads from daily high and low prices," *Journal of Finance*, 2012.
- **[21]** Roll, "A simple implicit measure of the effective bid-ask spread in an efficient market," *Journal of Finance*, 1984.
- **[22]** Yang & Zhang, "Drift-independent volatility estimation based on high, low, open, and close prices," *Journal of Business*, 2000.

**The primary in-report justification for the reframing is already in your bibliography:** [3] Nyawa, Ceccarelli & Tiozzo Pezzoli estimate mutually-exciting price- and volatility-jump Hawkes contagion on **five-minute** equity data with no order book. Cite it first, then [17] and [18].

---

## 12. TRACEABILITY

| Objective | Functional Requirements | Non-Functional Requirements |
|---|---|---|
| **O1** Unified composite framework | FR1, FR2, FR7, FR8, FR28, FR31, FR38, FR39 | NFR1, NFR2, NFR5, NFR6, NFR9, NFR21, NFR28, NFR30–NFR36 |
| **O2** Hawkes contagion (jump-event) | FR1, FR5, FR6, FR10, FR11, FR12, FR13, FR14, FR15, **FR50** | NFR4, NFR5, NFR14, NFR30 |
| **O3** RMT spectral monitoring | FR2, FR7, FR16, FR17, FR18, FR19, **FR52**, **FR54** | NFR4, NFR5, NFR15, NFR31 |
| **O4** Hamilton filter | FR3, FR20, FR21, FR23, FR24, **FR53** | NFR4, NFR5, NFR16, NFR17 |
| **O5** Regime-conditional fusion | FR28, FR30, FR31, FR32, FR33, FR34, FR40, FR41 | NFR11, NFR12, NFR13 |
| **O6** Kalman reconciliation | FR23, FR25, FR26, FR27, FR29, FR36 | NFR1, NFR3, NFR18, NFR19, NFR21 |
| **O7** Ablation study | FR27, FR33, FR35, FR36, FR38 | NFR11, NFR12, NFR13, NFR34 |
| **O8** Tariff covariate | FR4, FR22, FR32 | NFR12 |
| **O9** Cross-market transferability | FR37, FR38 | NFR20 `[GAP]` |
| **O10** Decision-support layer | FR14, FR15, FR31, FR39, FR42, FR43, FR44, FR45, FR46, FR47, FR48, FR49, **FR51** | NFR21, NFR22, NFR23–NFR27, NFR28 |

### Coverage check
- Every live FR maps to ≥1 objective. ✔
- Every objective maps to ≥1 FR. ✔
- **FR9 is a tombstone** (cut in v2); partially rescued by FR51 → O10.
- **NFR8 and NFR27** trace to no objective — NFR8 is now specified `[v2]`, NFR27 remains a `[GAP]`.
- **NFR32/33** cut in v2 — they served only the out-of-scope EEMD–LSTM overlay and tick ingestion.
- **O9 remains the weakest-covered objective** — two FRs, one `[GAP]` NFR, and a scoped-down deliverable. Descope candidate if schedule slips.
