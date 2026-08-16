# SPEC-DELTA-01 — Data Feasibility Amendment

**Amends:** `SPEC.md` (v1)
**Trigger:** Open Question 1 answered — tick-level NSE order-flow data is not obtainable (≈₹20 lakh subscription, out of budget). Open Question 2 answered — corpus extends to 2026. Open Question 3 superseded.
**Status:** ⚠ **SUPERSEDED — do not build from this document.**

> Every amendment proposed here was merged into **SPEC.md v2**, and SPEC has since advanced to **v3.1**. This file is retained only as a record of *why* the jump-event reformulation happened; it is useful for writing report §5.1.2 and for the viva, and for nothing else.
>
> **Two sections are now actively wrong:**
> - **D7 (FR52)** described point-in-time index membership curation. SPEC v3 replaced this with liquidity-rank universe reconstruction. Do not curate membership.
> - **D5/D9** quote pre-acquisition estimates (1-minute bars, ~90M rows, unknown broker cost). Superseded by DATA-ACQUISITION: 5-minute bars, ~19M rows, Kite at ₹500/month.
>
> **Authoritative chain:** `SPEC.md` v3.1 → `ARCHITECTURE.md` v2 → `DATA-ACQUISITION.md`.

---

## D0 — SUMMARY OF THE DOWNGRADE

| | SPEC v1 | SPEC v2 (this delta) |
|---|---|---|
| Pillar 1 event type | Aggressive sell **market orders** (tick) | Negative **price-jump events** (1-min / 5-min bars) |
| Pillar 1 state variable `S_t` | Spread quintile × queue-imbalance quintile from reconstructed LOB | Spread-**estimator** quintile × signed-volume-**imbalance-proxy** quintile from OHLCV |
| Pillar 2 (RMT) | Daily returns | **Unchanged** — was never tick-dependent |
| Pillar 3 inputs | 5-min realised vol, AR, cross-sectional median BAS | Yang–Zhang OHLC vol (pre-2015) / 5-min RV (post-2015), AR unchanged, BAS via Abdi–Ranaldo estimator |
| Corpus | 2005–2023, 5 epochs | **2005–2026, 6 epochs** |
| Hawkes coverage | Full corpus | **~2015/16–2026 only** (intraday data availability) |
| Data cost | ≈₹20,00,000 | **₹0 – ₹30,000** |
| Hardware | 16 cores / 64 GB / 10 Gbps / 4 TB | **8 cores / 16–32 GB / ordinary broadband / 250 GB** |

**The methodological claim of the project does not change.** MG-1 (no Hawkes–RMT–Hamilton fusion exists) is still open, still unaddressed in the literature, and still the contribution. You are changing the *observation channel* for Pillar 1, not the pillar.

---

## D1 — JUSTIFICATION FOR THE JUMP-BASED HAWKES REFRAMING

This is not a compromise you need to apologise for in the report. It is the dominant approach in the mutually-exciting-contagion literature:

- **[3] Nyawa, Ceccarelli & Tiozzo Pezzoli (EJOR 2025)** — already in your bibliography — estimates a *double Hawkes* model of mutually exciting **price jumps** and **volatility jumps** on **five-minute** US equity data via GMM. No order book. This paper is your primary citation for the reframing, and you already reviewed it in §2.1.
- **Aït-Sahalia, Cacho-Díaz & Laeven, "Modeling financial contagion using mutually exciting jump processes," Journal of Financial Economics (2015)** — the foundational jump-contagion paper. Daily and intraday returns, no order flow. **Add to bibliography as [17].**
- **Lee & Mykland, Review of Financial Studies (2008)** — the standard non-parametric jump detection test using bipower variation. This is your event-extraction method. **Add as [18].**

Reframe the sentence in §5.1 from "order-flow toxicity" to **"return-jump contagion"** and the pillar keeps every property you claimed for it: self-excitation, cross-excitation, state-dependence, sell/buy (now negative/positive) asymmetry, criticality at `n → 1`, and a directed contagion network from `Φ`.

**Write the constraint into the report explicitly.** A design report that names its data constraint and adapts its method with citation is stronger than one that specifies an unobtainable pipeline. Add a short subsection §5.1.2 "Observation-Channel Constraint and Jump-Event Reformulation."

---

## D2 — REVISED DATA STRATEGY (TWO TIERS)

### Tier A — Daily, full corpus 2005–2026, free

| Series | Source | Cost | Notes |
|---|---|---|---|
| Nifty 500 constituent OHLCV | NSE bhavcopy archives (daily ZIP, full history) | Free | Bulk-download and parse once |
| Nifty 500 index + sector indices | NSE indices | Free | For sector taxonomy and benchmark overlay |
| Point-in-time index membership | NSE index-reconstitution announcements | Free | **Manual curation — budget real hours for this** |
| Corporate actions / adjustment factors | NSE + cross-check | Free | Required by FR7 |
| India VIX | NSE | Free | **Available from Nov 2007 only** — 2005–07 gap, see D6 |
| RBI repo rate, bank credit growth | RBI DBIE | Free | |
| INR trade-weighted / USD-INR | RBI, FRED | Free | |
| Brent crude | FRED / EIA | Free | |
| Second-market index constituents (O9) | `yfinance` | Free | Ibovespa / JSE / IDX all available daily |

**Powers:** Pillar 2 fully, Pillar 3 fully, all six epochs, the whole validation harness, and the cross-market test.

### Tier B — Intraday bars, ~2015/16–2026, cheap

| Option | Approx. cost | Lookback | Notes |
|---|---|---|---|
| Zerodha Kite Connect + historical add-on | ~₹2,500/month (verify) | ~2015 onwards, 1-min | Most reliable lookback; ₹7,500 for a 3-month build window |
| Upstox / Angel One SmartAPI / Dhan / Fyers | Free tiers exist | Shorter, varies | Free but lookback and rate limits vary sharply |
| `yfinance` | Free | 1-min: last ~7 days only; 60-min: ~2 years | Useless for history, fine for a live demo |

> **Verify current pricing and lookback limits this week before committing.** Broker API terms change and my figures may be stale. This is the single number that determines Pillar 1's start date.

**Powers:** Pillar 1 (jump events), true 5-minute realised volatility post-2015, and the live sector micro-flag demo.

**Symbol scope for Tier B:** do **not** pull 1-minute bars for all 500 names. 500 × 375 min × 250 days × 10 years ≈ 470M rows. Restrict to **Nifty 50 + tariff-exposed sector representatives (~100 symbols)** — around 90M rows, comfortable in TimescaleDB on a laptop. Your report only ever requires market-level and sector-level branching ratios (§5.1.1), never all 500 instrument-level ones.

---

## D3 — REVISED PILLAR 1 SPECIFICATION

### Event extraction (new)
Negative jump events are extracted per symbol from intraday log-returns using the **Lee–Mykland bipower-variation test** at a stated significance level. A detected jump with negative sign becomes an event in the counting process `N_i(t)`. Positive jumps feed the asymmetry indicator only.

`[GAP]` — jump-test significance level and local-window length are undefined. Add to Open Questions.

### State variable `S_t` (revised)
LOB reconstruction is impossible. `S_t` becomes the joint quintile of two estimators computable from OHLCV:

1. **Spread proxy** — **Abdi & Ranaldo (2017)** close-high-low estimator (preferred), with **Corwin & Schultz (2012)** two-day high-low as fallback and **Roll (1984)** as a sanity cross-check. Replaces bid–ask spread.
2. **Order-imbalance proxy** — bar-level tick test: sign each bar's volume by the sign of its close-to-close return, then compute signed-volume imbalance over a rolling window. Replaces queue imbalance.

Equation 5.2's structure `φ_ij(u, S_t) = Σ_k α_ij^(k)(S_t) e^{−β_ij^(k)(S_t) u}` is **unchanged**. `S_t` still takes 25 values. Nothing downstream is affected.

**Add to bibliography:** Abdi & Ranaldo (2017) as [19], Corwin & Schultz (2012) as [20], Roll (1984) as [21].

### What dies
- **`MTS_t` update granularity** drops from 1 minute to the bar frequency you choose (1-min or 5-min). Sub-second CSRS (NFR1/NFR2) is now meaningless — see D5.
- **Lee–Ready trade signing (FR5)** is replaced by the bar-level tick test. Lee–Ready needs quotes.
- **True LOB reconstruction (FR6)** is cut entirely.

### What survives untouched
Branching ratio `n(S_t)`, spectral-radius regularisation, `MTS_t` formula (Eq. 5.4), negative/positive asymmetry indicator, sector-level micro-flags (§5.1.1), directed contagion network from `Φ(S_t)`, sector-weighted tariff variant.

---

## D4 — REVISED PILLAR 3 INPUTS

`y_t = (RV_t, AR_t, BAS_t)ᵀ` is retained. Only the estimators change:

- **`RV_t`** — Yang–Zhang OHLC realised-volatility estimator for the whole corpus (drift-independent, handles overnight gaps, best of the OHLC family). Where Tier B intraday data exists, additionally compute true 5-minute RV and **report the correlation between the two as a validation check**. That comparison is a small, publishable methodological note in its own right.
- **`AR_t`** — Absorption Ratio, top k = 10 PCs of the Nifty 500 daily covariance matrix. **Completely unaffected by the downgrade.**
- **`BAS_t`** — cross-sectional median of the Abdi–Ranaldo spread estimator, replacing the median quoted spread.

**Add to bibliography:** Yang & Zhang (2000) as [22].

---

## D5 — REVISED NON-FUNCTIONAL REQUIREMENTS

| ID | v1 | v2 |
|---|---|---|
| NFR1 | tick→dashboard < 1 s | **Bar-close → dashboard < 5 s.** Sub-second is no longer meaningful when the fastest genuine input is a 1-minute bar. |
| NFR2 | Sub-second CSRS | **CSRS refresh at bar frequency (1-min or 5-min).** Kalman predict may still run more often, but the report must not claim informational value it doesn't have. |
| NFR4 | Hawkes 1 min / RMT daily / Hamilton daily | **Hawkes at bar frequency / RMT daily / Hamilton daily.** Unchanged in structure. |
| NFR6 | > 500 securities | **500 for RMT (daily); ~100 for Hawkes (intraday).** |
| NFR8 | `[GAP]` data volume | **≈2.6M daily rows + ≈90M intraday rows ≈ 30–60 GB in TimescaleDB.** Now specified. |
| NFR30 | ≥ 16 physical cores | **≥ 8 cores.** Per-symbol Hawkes on ~100 symbols is still embarrassingly parallel but far smaller. |
| NFR31 | ≥ 64 GB RAM | **16 GB minimum, 32 GB comfortable.** The 500×500 correlation matrix is ~2 MB. The 64 GB figure existed only to hold 20 days of tick data. |
| NFR32 | GPU 16 GB VRAM | **Cut.** Only ever served the out-of-scope EEMD–LSTM overlay. |
| NFR33 | 10 Gbps NIC | **Cut.** No live tick ingestion. Ordinary broadband. |
| NFR34 | 4 TB redundant SSD | **250 GB.** |
| NFR35 | 2-node replication, 99.9% uptime, auto-failover | **Downgrade to single-node with documented restart procedure.** Retain 99.9% as a *design target for a hypothetical production deployment* and state it as such. A four-person student project should not claim a failover cluster it will not build. |
| NFR9 | 99.9% uptime | Reframe as design target, not deliverable. |

**This is a laptop project now.** That is the single biggest practical win of the downgrade.

---

## D6 — COVERAGE ASYMMETRY: TURN IT INTO A FEATURE

Pillar 1 will not exist before ~2015. Epoch coverage becomes:

| Epoch | RMT | Hamilton | Hawkes |
|---|---|---|---|
| 2008 GFC | ✔ | ✔ (no India VIX pre-Nov 2007 — substitute or drop covariate) | ✘ |
| 2013 taper tantrum | ✔ | ✔ | ✘ |
| 2018 IL&FS | ✔ | ✔ | ✔ |
| 2020 COVID-19 | ✔ | ✔ | ✔ |
| 2022 tightening | ✔ | ✔ | ✔ |
| **2025–26 tariff sequence (NEW, epoch 6)** | ✔ | ✔ | ✔ |

**Do not present this as damage.** Present it as follows, because it is true:

1. **The Kalman layer already handles it.** Equation 5.18 models each pillar as an *intermittently-arriving* observation with age-inflating noise `R^(p)(t − τ^(p))`. A pillar that never arrives is the limiting case `R → ∞`. Pre-2015 epochs are handled by the existing machinery with **zero new mathematics**. Say this explicitly in §5.5 — it is a genuine strength of the design and it is the best argument for Objective 6 you have.
2. **The ablation study (Objective 7) becomes a natural experiment.** You now have 2 epochs where only the RMT+Hamilton pair is available and 4 where the full triple is. That is *exactly* the comparison MG-1 demands, handed to you by the data constraint.
3. **Forward-fill has no defensible answer here.** Forward-filling a Hawkes score that has never existed is nonsense. The Kalman layer degrades gracefully. This is your cleanest demonstration that Objective 6 matters.

**India VIX gap (2005–Nov 2007):** either drop the 2008 epoch's VIX covariate and note it, or substitute realised volatility of the Nifty index for that window. Decide and document.

---

## D7 — CHANGED FUNCTIONAL REQUIREMENTS

| ID | Change |
|---|---|
| FR1 | **REPLACED.** "The system shall ingest daily OHLCV for Nifty 500 constituents (2005–2026) from NSE bhavcopy archives, and 1-minute or 5-minute OHLCV bars for a curated ~100-symbol subset from a broker historical API for the available lookback period." |
| FR5 | **REPLACED.** "The system shall sign each intraday bar's volume by the sign of its close-to-close return (bar-level tick test)." Lee–Ready cut. |
| FR6 | **REPLACED.** "The system shall compute the Abdi–Ranaldo spread estimator and a rolling signed-volume imbalance, and discretise the pair into the joint-quintile state variable `S_t`." LOB reconstruction cut. |
| FR9 | **CONDITIONAL → LIKELY CUT.** Per-trade counterparty tags do not exist in bar data. See FR45 below for a partial rescue. |
| **FR10a** | **NEW.** "The system shall extract negative and positive jump events per symbol from intraday log-returns using the Lee–Mykland bipower-variation test." |
| FR10 | **AMENDED.** Hawkes MLE now fits the jump-event point process on rolling one-day windows conditional on `S_t`. |
| FR13 | **AMENDED.** `MTS_t` emitted at bar frequency; the indicator condition becomes negative-vs-positive jump asymmetry. |
| FR20 | **AMENDED.** `RV_t` via Yang–Zhang OHLC estimator (full corpus) and 5-min RV where available; `BAS_t` via Abdi–Ranaldo. |
| FR32 | **AMENDED.** Six epochs, corpus 2005–2026. |
| FR35 | **AMENDED.** Subset ablation runs on the 4 Hawkes-covered epochs for full 7-subset comparison; the 2 pre-2015 epochs contribute a 3-subset comparison (RMT, Hamilton, RMT+Hamilton). Report both tables. |
| FR37 | **AMENDED.** Cross-market transferability runs on the **RMT + Hamilton subset only** unless intraday data for the second market proves free and available. State this scoping explicitly rather than silently omitting it. |
| FR45 | **DOWNGRADED, NOT CUT.** Per-instrument retail/institutional decomposition is impossible. **Partial rescue:** NSE publishes daily **FII/DII net cash-market activity** and (verify) **category-wise turnover**, both free. Reframe the herd-share indicator as a *market-level* institutional-vs-residual flow diagnostic at daily frequency. Weaker than specified, but not vapour, and it keeps the retail-protection narrative in Chapter 10 honest. |

**Unchanged by the downgrade:** FR2, FR3, FR4, FR7, FR8, FR11, FR12, FR14, FR15, **FR16–FR19 (all of Pillar 2)**, FR21–FR31, FR33, FR34, FR36, FR38–FR44, FR46–FR49.

That is 38 of 49 requirements untouched. The blast radius is smaller than it feels right now.

---

## D8 — DATA CONTRACT CHANGES

- **E1 `tick_order_flow` → `intraday_bars`.** Fields: `symbol, ts, open, high, low, close, volume, signed_volume, log_return, is_jump, jump_sign`. Frequency: 1-min or 5-min. Source: broker API. Coverage: ~2015–2026, ~100 symbols.
- **E2 `lob_state` → `microstructure_state`.** Fields: `symbol, ts, spread_ar (Abdi–Ranaldo), spread_cs (Corwin–Schultz, cross-check), amihud_illiq, signed_vol_imbalance, spread_quintile, imbalance_quintile, S_t`. Derived, not ingested.
- **E3 `daily_prices`** — add `open, high, low, volume` (needed for the spread and volatility estimators, which v1 didn't require), and `yz_volatility`.
- **NEW E14 `flow_aggregates`** — `date, fii_net, dii_net, total_turnover, residual_flow_share`. Daily, from NSE. Powers the downgraded FR45.
- **E6 `hawkes_output`** — add `jump_count_neg`, `jump_count_pos`, `asymmetry_ratio`. Drop `retail_toxicity_share` (moves to E14).
- **E4 `macro_covariates`** — add `vix_available` boolean flag for the 2005–07 window.
- **E7, E8, E9, E10, E11, E12, E13** — unchanged.

---

## D9 — BUILD ORDER AND FEASIBILITY (per your "deliver on all, in order" instruction)

| Obj | Feasibility under v2 | Notes |
|---|---|---|
| O1 | **Green** | Unaffected |
| O2 | **Amber** | Deliverable, but as jump-contagion. Needs the §5.1.2 rewrite and 3 new citations. Highest technical risk of the three pillars. |
| O3 | **Green** | Completely unaffected. **Build this first — it is your fastest path to a working pillar and it de-risks the whole schedule.** |
| O4 | **Green** | Proxy estimators are standard and well-cited |
| O5 | **Green** | Unaffected |
| O6 | **Green — and strengthened.** | The coverage asymmetry is now the best argument for this objective |
| O7 | **Green** | Cheap once the harness exists; it is re-running, not new code |
| O8 | **Green** | Now testable thanks to the 2026 extension. Epoch 6 is your only fully-instrumented recent epoch. |
| O9 | **Amber** | Deliverable on the RMT+Hamilton subset for free via `yfinance`. Full three-pillar cross-market needs a second market's intraday data — treat as stretch. |
| O10 | **Amber** | 5 of 7 artefacts deliverable. Herd-share downgraded (FR45). Feedback logging already flagged long-term in the paper itself. |

**Recommended build order:** O3 → O4 → O1/O5 (fusion skeleton on two pillars) → O6 → O2 → O7 → O8 → O10 → O9.

Rationale: getting a working two-pillar CSRS early means you always have a submittable system. Pillar 1 is the riskiest component and should not be the thing that blocks the first end-to-end run.

---

## D10 — REVISED TEAM SPLIT

Your proposed split — data+Hawkes / RMT+Hamilton / reconciliation+fusion / dashboard — has three structural problems:

1. **"Data + Hawkes" is a bottleneck.** That person owns shared infrastructure everyone waits on *and* the hardest pillar. Both slip together.
2. **"RMT + Hamilton" is badly paired.** RMT is a rolling eigendecomposition — genuinely small. Hamilton with TVTP and EM is heavy. Uneven, and the two share no theory.
3. **Fusion and dashboard people idle for weeks**, then crunch at the end.

### Proposed instead

| Owner | Scope | Starts |
|---|---|---|
| **A — Data platform + Pillar 2 (RMT)** | Bhavcopy ingestion, TimescaleDB, corporate actions, point-in-time membership, spread/volatility estimators, **then RMT.** | Day 1 |
| **B — Pillar 1 (Hawkes)** | Broker API integration, Lee–Mykland jump extraction, `S_t` construction, MLE + regularisation, sector micro-flags, contagion network. Full-time, hardest component. | Day 1 (API + literature), real work once A ships bars |
| **C — Pillar 3 (Hamilton) + Kalman reconciliation** | Both are state-space filters with predict/update recursions. **Same theory, learned once, applied twice.** Far better pairing than RMT+Hamilton. | Day 1 |
| **D — Fusion + validation harness + dashboard + decision support** | Validation harness first (against mock pillar outputs), then fusion, then ablation runner, then dashboard and the §5.6 artefacts. | Day 1 |

**The mechanism that makes this work:** everyone codes against the §5 data contract with **mock pillar outputs from day one**. D can build and test the entire fusion, ablation and dashboard stack on synthetic `E6/E7/E8` rows before a single real pillar exists. Nobody blocks on anybody.

Two shared responsibilities that need a named owner:
- **A owns the data contract.** Any schema change goes through them.
- **D owns the ablation runner**, which is also the integration test for the whole system.

Given your backend and full-stack background, **D is the natural fit for you** — it is the widest integration surface and the one that most needs someone who can hold the whole pipeline in their head. A is the alternative if you'd rather own the foundation.

---

## D11 — REPORT SECTIONS REQUIRING EDITS

| Section | Edit |
|---|---|
| Abstract | "tick-to-monthly" → "bar-to-monthly"; drop "order-flow toxicity" framing for Pillar 1, use "return-jump contagion" |
| §1 (Intro) | Same terminology change; note the observation-channel constraint |
| §5.1 | Rewrite event definition; **add new §5.1.2 "Observation-Channel Constraint and Jump-Event Reformulation"** with citations [3], [17], [18] |
| §5.1.1 | Unchanged — sector micro-flags work identically on jump events |
| §5.3 | `RV_t` and `BAS_t` estimator substitution, with citations [19]–[22] |
| §5.5 | **Add the "missing pillar = `R → ∞`" argument.** This is your strongest paragraph and it doesn't exist yet. |
| §7.1 (Stage 1) | ~~Full rewrite — new sources, no Lee–Ready, no LOB reconstruction, add point-in-time membership curation~~ ⚠ **SUPERSEDED** — see SPEC v3 §11: membership curation is replaced by liquidity-rank reconstruction (FR52). |
| §7.5 | Six epochs |
| §8.1 | Amended FRs per D7 |
| §8.2, §8.3 | Amended NFRs per D5 — hardware section shrinks dramatically |
| §9 | Latency targets; add the coverage-asymmetry ablation as an expected outcome |
| Bibliography | Add [17]–[22] |

---

## D12 — NEW AND REVISED OPEN QUESTIONS

Superseding OQ1, OQ2, OQ3 from SPEC.md v1:

1. **Which broker API, and what is its verified lookback?** This single answer fixes Pillar 1's start date and therefore the epoch coverage table in D6. Check pricing and per-symbol historical limits before committing. Highest priority this week.
2. **1-minute or 5-minute bars?** 5-min gives you [3]'s exact frequency (best citation alignment) and 5× less data; 1-min gives more jump events and better Hawkes identification. Recommend 5-min unless jump counts come back too sparse.
3. **Jump-test parameters** — Lee–Mykland significance level and local bipower window length. Undefined.
4. **Hawkes symbol universe** — Nifty 50 only, or Nifty 50 + tariff-exposed sector representatives? The latter is needed for §5.1.1 and the tariff demo.
5. **India VIX pre-Nov 2007** — drop the covariate for the 2008 epoch, or substitute Nifty realised volatility?
6. **Point-in-time index membership** — who curates it, and how many hours are we budgeting? This is the most tedious task in the project and it silently breaks the RMT pillar if done wrong.
7. **Epoch 6 break date** — what is the labelled break for the 2025–26 tariff sequence? Multiple candidate dates (50% peak, Feb 2026 interim framework, the solar CVD single-day drop). Pick one and justify.
8. **Corpus end date** — where exactly does 2026 stop? Fix it now so every run is reproducible.
9. **Does NSE category-wise cash turnover data exist and is it downloadable in bulk?** Determines whether FR45 survives in downgraded form or is cut.
10. **Does D own the ablation runner, and is the mock-data contract written before or alongside the pillars?** Recommend before — it is a half-day of work that unblocks three people.

Carried forward unchanged from SPEC.md v1: OQ4 (undefined constants), OQ5 (`R^(p)(·)` form), OQ6 (sector taxonomy), OQ7 (train/test protocol), OQ8 (crisis labelling), OQ12 (cross-market index), OQ14 (Streamlit vs Dash).
