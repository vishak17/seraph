# DATA-ACQUISITION.md — SERAPH

**Companion to:** `SPEC.md` v2
**Purpose:** Resolve Open Questions 1, 4 and 8. Concrete sources, costs, limits and a week-one checklist.
**Verified:** 15 Aug 2026. Broker terms change — re-check anything marked ⚠ at signup.

> **STATUS — reconciled with SPEC v3.1 and ARCHITECTURE v2.**
> - Kite Connect at **₹500/month is confirmed by the team** — the ⚠ pricing ambiguity in §1 is closed.
> - §7's SPEC amendments were **all applied in SPEC v3**. That table is now a record of what was done, not a to-do list.
> - The §6 checklist remains live except item 5, whose *design* is settled (SPEC FR52 + ARCHITECTURE D3 buffer rule); the *execution* is still outstanding.
> - §4's universe-reconstruction proposal is now binding as SPEC FR52. ARCHITECTURE D3 adds the buffer rule (incumbent ≤ 550, entrant ≤ 450) to reduce rebalance churn — a refinement of this section, not a contradiction.
> - Two items in §5/§6 are still genuinely open: the **solar sector basket** (no clean NSE sectoral index exists) and **FII/DII retrievable depth**.

## Bottom line

| Item | v1 assumption | Verified reality |
|---|---|---|
| Intraday data cost | ₹20,00,000 (tick vendor) | **₹500/month** ⚠ |
| Intraday lookback | Unknown | **~2015 (Nifty 50) / ~2016 (other stocks)** — coverage matrix in SPEC §4 holds |
| Daily 2005–2026 | Manual bhavcopy scraping | **Solved by an existing open-source repo** |
| Point-in-time membership | Manual curation, unbounded hours | **Sidestep it** — see §4 |

**Total project data cost: ~₹2,000** for a four-month build window. Everything else is free.

---

## 1. INTRADAY BARS — Zerodha Kite Connect

### Pricing ⚠
Zerodha restructured in Feb–Mar 2025. Historical data is no longer a separate add-on — it is bundled with the paid Connect plan.

- **Kite Connect Personal (free)** — order execution only. **No market data, no historical data.** Useless for you.
- **Kite Connect paid** — includes live WebSocket data *and* historical candles at no extra charge.

⚠ **Sources disagree on the paid price.** Zerodha's products page and an April 2026 support article both state **₹500/month per API key**; an earlier Z-Connect post states ₹2,000/month for data. Assume ₹500 and **confirm on the signup page before budgeting**. Either figure is trivial against ₹20 lakh — at ₹2,000/month a four-month window is still only ₹8,000.

**Prerequisite:** a Zerodha trading account. If nobody on the team has one, open it now — KYC takes a day or two and it gates everything.

**Static IP:** required only for *order placement* from 1 Apr 2025. Data endpoints (historical, WebSocket) remain accessible from any IP. **You are not affected.**

### Lookback — this fixes your coverage matrix
Zerodha holds **intraday candles from ~2015**; daily candles go back to the 1990s for some instruments. Forum reports confirm minute data returns empty before 2015 for Nifty 50 and before ~2016 for individual stocks.

**This confirms SPEC §4 as written.** Epochs 2008 and 2013 are RMT+Hamilton only; 2018, 2020, 2022 and 2025–26 have all three pillars. No revision needed.

### Per-request limits
Days of data returnable in a single call:

| Interval | Max days/request |
|---|---|
| `minute` | 60 |
| `3minute` / `5minute` / `10minute` | 100 |
| `15minute` / `30minute` | 200 |
| `60minute` | 400 |
| `day` | 2000 |

There is **no cap on total volume** with an active subscription — you just loop.

### Fetch cost for your universe (100 symbols, ~2016–2026)

| Interval | Requests | Wall time @ ~3 req/s ⚠ | Rows |
|---|---|---|---|
| **5-minute** | ~3,700 | ~20 min (budget 1–2 h with backoff) | ~19M |
| **1-minute** | ~6,100 | ~35 min (budget 1–2 h with backoff) | ~94M |

Either is a one-evening job. **The data pull is not a bottleneck** — plan the ingest as a resumable, checkpointed script and let it run overnight.

⚠ Confirm the historical-endpoint rate limit in Kite's rate-limits FAQ before tuning concurrency.

**Recommendation: 5-minute bars.** Matches the frequency of paper [3] exactly (your primary citation for the jump reformulation), 5× less data, same per-request efficiency. Drop to 1-minute only if jump counts come back too sparse for Hawkes identification.

### ⚠ COMPLIANCE FLAG — affects the dashboard
Zerodha's FAQ states that **displaying or redistributing Kite Connect data on external platforms violates exchange data vending policies.** Kite is an execution platform, not a data vendor.

**Implication for SPEC FR39–FR44:** a publicly-hosted SERAPH dashboard rendering Kite-derived intraday signals is a licensing problem, not a technical one.

**Mitigations, pick one:**
1. Keep the dashboard **local / demo-only** — run it on a laptop for the viva. Simplest, almost certainly sufficient for a college project.
2. Drive any public-facing view from **EOD/delayed data only** (freely redistributable from NSE reports).
3. Publish **derived signals** (CSRS, branching ratio) rather than the underlying candles — likely acceptable, but you would be making that judgement yourself.

Raise this with Dr. Satisha before anyone builds a hosted demo. It costs nothing to know now and is embarrassing to discover at the demo.

---

## 2. DAILY DATA 2005–2026 — mostly solved

### Primary: the EOD2 repository
`github.com/BennyThadikaran/eod2` — an automated NSE EOD downloader that already handles the three things that would otherwise eat a week of your time:

- Daily OHLCV + delivery data for **2,000+ NSE stocks since 1995**
- **Adjusts for splits and bonuses** (satisfies SPEC FR7)
- **Tracks ISIN across symbol/company-name changes** and applies them — this is the failure mode that silently corrupts long-horizon panels
- Handles the **July 2024 UDiFF transition** (see below)
- Syncs to current date, tracks NSE holidays

**Documented caveat, and it matters:** *"Stock data before 2005 may not be fully adjusted as NSE does not provide adjustment data before this year."* Your corpus starts in 2005, so this lands exactly on your boundary. **Document it in §7.1 of the report** and treat early-2005 windows with care — the RMT pillar's first usable `C_t` needs T = 1000 trading days of history anyway, so your first spectral output is ~2009 regardless. That absorbs most of the risk.

### If you scrape it yourself
NSE changed the bhavcopy format on **8 July 2024** (Circular 62424). The old path — `nsearchives.nseindia.com/content/historical/EQUITIES/YYYY/MMM/cmDDMMMYYYYbhav.csv.zip` — is discontinued. Post-July-2024 you need **CM-UDiFF Common Bhavcopy Final (zip)** from `nseindia.com/all-reports`.

Any homegrown scraper must handle **two different schemas across the corpus**. This is precisely why using EOD2 is the better call.

### Index levels and India VIX
- Historical index data: `nseindia.com/reports-indices-historical-index-data`
- **India VIX starts Nov 2007.** Confirmed. SPEC Open Question 5 stands — for the 2008 epoch, either drop the VIX covariate or substitute Nifty realised volatility. **Decide and write it into §5.3.**

---

## 3. FII/DII FLOW — FR45 survives

**Official source:** `nseindia.com/reports/fii-dii` — combined FII/FPI and DII trading activity across NSE, BSE and MSEI, cash segment, buy/sell/net in ₹ crore, published daily after close (~8:30–9:30 PM IST).

**Availability:** the official page serves current data; bulk history requires either date-looping the archive or pulling from an aggregator. Several free aggregators carry multi-year daily series.

**What you actually get:** market-level daily FII and DII net cash-market activity. **Not** per-instrument, **not** intraday, **not** a retail tag.

**Verdict: FR45 survives in its downgraded form as specified in SPEC v2.** `residual_flow_share = 1 − (FII_gross + DII_gross)/total_turnover` is a defensible market-level proxy for non-institutional participation. It is weaker than the herd-share indicator the paper describes, and the report must say so plainly rather than implying per-instrument attribution.

**Separately available and worth grabbing:** NSE publishes **participant-wise open interest** for F&O with four categories — FII, DII, PRO, CLIENT. CLIENT is retail + HNI. This is derivatives-only and so does not directly serve Pillar 1, but it is the closest thing to a genuine retail-participation series in free NSE data and could strengthen the decision-support narrative in Chapter 10. Low priority, cheap to add later.

---

## 4. POINT-IN-TIME MEMBERSHIP — don't fight this, sidestep it

### Why the direct route is bad
- `archives.nseindia.com/content/indices/IndexInclExcl.xls` — the official inclusion/exclusion file — **stopped updating around July 2020**. Confirmed by multiple developer reports.
- The niftyindices.com monthly *"Indices Market Capitalization"* report archive gives month-by-month constituent lists, but you would be downloading and parsing **250+ monthly reports** back to 2005 and reconciling company names against symbols by hand. Names change. Companies merge. Symbols get reused.
- Community consensus is that reliable free point-in-time Nifty 500 composition is simply not available.

Back-applying today's constituent list to 2005 is the alternative, and it is **survivorship bias in its purest form** — you would be running a crisis-detection study on a universe of companies selected for having survived every crisis in the sample. A sharp examiner will catch that, and they should.

### Recommended: reconstruct the universe yourself
**Your RMT pillar does not need the Nifty 500 as NSE defines it.** It needs *N = 500 broadly representative NSE stocks with sufficient liquidity and history at each point in time*. Nothing in Equations 5.5–5.9 references index membership.

**Proposed rule:** at each semi-annual rebalance date, rank all NSE equities by **six-month median daily traded value** (available directly from bhavcopy as `TOTTRDVAL`) and take the top 500, subject to a minimum listing-history filter of 1000 trading days so `C_t` is well-defined.

**Why this is better, not just easier:**
- **Survivorship-bias-free by construction** — dead companies are in the universe for exactly the period they were liquid
- **Fully reproducible** from free data with a stated rule, which is more than you could say for a hand-curated membership list
- **Zero curation hours** — it is a `groupby` over data you already have
- Preserves N = 500, so `Q = T/N = 2` and every constant in the paper survives untouched
- Liquidity rank is *closer* to what the RMT pillar actually cares about than free-float market cap is

**Cost:** the report can no longer say "the Nifty 500." Use **"a Nifty-500-analogous universe of the 500 most liquid NSE equities, reconstructed point-in-time by rolling six-month median traded value."** Add a paragraph in §7.1 justifying it on survivorship grounds. That paragraph makes your methodology section stronger, not weaker.

**Keep the official Nifty 500 index level** as the benchmark series for the CSRS overlay (FR41) and as a sanity check that your reconstructed universe tracks it. If your universe's equal-weighted return correlates >0.95 with the official index, you have evidence the reconstruction is sound — and that is a validation figure worth reporting.

**Revised SPEC FR52:** *"The system shall reconstruct a point-in-time universe of the 500 most liquid NSE equities at each semi-annual rebalance date by six-month median traded value, subject to a 1000-trading-day minimum listing history."*

**This converts Open Question 4 from an unbounded curation task into an afternoon of code.** It is the single biggest schedule saving available to you right now.

---

## 5. SECTOR TAXONOMY (SPEC Open Question 11)

GICS is licensed — do not use it. Two free routes:

1. **NSE sectoral indices** — Nifty Auto, Nifty Metal, Nifty Pharma, Nifty Energy etc. Constituent lists are published and these map cleanly onto your tariff-exposed sectors (steel/metal, auto components, pharma, solar/energy). Coarse but official and defensible.
2. **NSE's own industry classification** — the Nifty 500 constituent file carries an `Industry` column; NSE Indices maintains a macro/sector/industry/basic-industry hierarchy.

**Recommendation:** NSE sectoral index membership as the primary taxonomy, hand-mapped to the five tariff-exposed sectors (steel/aluminium, auto components, solar/renewables, pharmaceuticals). Small, auditable table. **Owner: A**, alongside the universe reconstruction.

---

## 6. WEEK-ONE CHECKLIST

| # | Task | Owner | Blocks |
|---|---|---|---|
| 1 | Confirm Zerodha account exists; open one if not (KYC lead time) | B | Everything intraday |
| 2 | Confirm Kite Connect paid price at signup ⚠ and subscribe | B | Pillar 1 |
| 3 | Probe the earliest available 5-min candle for 3 test symbols (a Nifty 50 name, a mid-cap, a solar name) — **this pins the real Pillar 1 start date** | B | SPEC §4 |
| 4 | Clone EOD2, run a full historical sync, spot-check 5 known split events for correct adjustment | A | Pillars 2 and 3 |
| 5 | Build the point-in-time universe by liquidity rank; validate correlation vs official Nifty 500 index | A | Pillar 2 |
| 6 | Pull India VIX, confirm Nov 2007 start, decide the 2008-epoch substitution | C | Pillar 3 |
| 7 | Pull RBI repo + credit growth from DBIE; Brent + INR from FRED | C | Pillar 3 |
| 8 | Test-pull FII/DII history; establish how far back bulk retrieval is practical | D | FR45 |
| 9 | Write the mock-data generator for E6/E7/E8 per SPEC §6 | D | Unblocks the whole fusion/dashboard stack |
| 10 | Raise the Kite redistribution constraint with Dr. Satisha | Aarya | Dashboard hosting decision |

**Item 3 is the one that matters most.** Until someone actually calls the endpoint and sees where the data stops, SPEC §4 is an assumption. It is a ten-minute test.

**Item 9 is the highest-leverage item on the list.** Half a day of work by D means three people stop waiting on each other.

---

## 7. SPEC AMENDMENTS ARISING

| SPEC ref | Change |
|---|---|
| FR52 | Rewritten — liquidity-rank universe reconstruction, not membership curation (§4 above) |
| FR1 | Name the sources: EOD2/bhavcopy for daily, Kite Connect for 5-min |
| §6 E3 | `index_member_flag` → `universe_member_flag`, derived from the liquidity rule |
| §5 NFR8 | Confirmed: ~19M intraday rows at 5-min (was estimated ~90M at 1-min) |
| OQ1 | **Resolved** — Kite Connect, ~₹500/month ⚠, intraday from ~2015 |
| OQ2 | **Resolved** — 5-minute bars |
| OQ4 | **Resolved** — sidestepped via liquidity-rank reconstruction |
| OQ8 | **Resolved** — FII/DII available; FR45 survives downgraded |
| OQ11 | **Resolved** — NSE sectoral indices |
| §7 Out of Scope | **Add:** public hosting of any dashboard rendering Kite-derived data, pending the licensing question |
| §11 | Add the survivorship-bias justification paragraph to the §7.1 rewrite |

Six of the fifteen open questions close on this document.
