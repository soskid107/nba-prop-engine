# NBA Player Props Prediction Engine

An NBA player-props research and daily prediction engine built around a multi-stage pipeline:
data refresh, context gathering, market selection, prediction, trust scoring, calibration, reporting, audit, and learning.

This repository is no longer points-only. The live workflow currently supports multi-market prediction with the strongest production support around points, assists, and rebounds.

## What The Engine Does

- Refreshes NBA schedule, player data, opponent stats, injuries, news, and sportsbook props
- Builds player, matchup, and lineup context for the target slate date
- Selects market candidates and evaluates them with player- and market-level trust checks
- Generates predictions with the live predictor stack and calibrated uncertainty
- Produces a daily markdown report with `BET`, `LEAN`, and `WATCH` outputs
- Audits prior predictions against actual results
- Feeds miss patterns back into the learning loop and Guardian safety checks

## Current Live Architecture

The production path is centered on [run_daily.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_daily.py) and [src/agents/orchestrator.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/orchestrator.py).

High-level flow:

1. Data refresh
2. Player context and lineup context gathering
3. Market candidate selection
4. Consensus and trust scoring
5. Prediction and reference-model comparison
6. Edge scoring
7. Market calibration and final bet gating
8. Report generation
9. Audit and learning updates

Important notes:

- The repo contains several experimental or reference components. Not every model in `src/` is a primary live decision-maker.
- The current live predictor stack is more accurate to describe as an orchestrated hybrid system than a single end-to-end model.
- Unsupported fallback-only markets are intentionally skipped instead of being forced through degraded logic.

## Core Components

### Daily runner

- [run_daily.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_daily.py)
- Main entry point for refresh, prediction, report generation, and audit workflows

### Orchestration

- [src/agents/orchestrator.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/orchestrator.py)
- Coordinates context gathering, market selection, prediction, trust, calibration, and rejection logging

### Consensus and trust

- [src/agents/consensus/engine.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/consensus/engine.py)
- [src/agents/consensus/data_validator.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/consensus/data_validator.py)
- [src/agents/consensus/narrative_validator.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/consensus/narrative_validator.py)

These components decide whether a player and market are trustworthy enough to consider, and they now emit numeric trust scores instead of only blunt yes/no gates.

### Prediction and calibration

- [src/models/simple_predictor.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/models/simple_predictor.py)
- [src/agents/market_calibrator.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/market_calibrator.py)
- [src/agents/edge_scorer.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/edge_scorer.py)
- [src/simulation/monte_carlo.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/simulation/monte_carlo.py)
- [src/agents/mechanistic_modeler.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/mechanistic_modeler.py)

The engine uses a primary live predictor with reference-model support, market-aware calibration, volatility handling, and trust-aware gating.

### Audit and learning

- [src/simulation/audit.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/simulation/audit.py)
- [src/agents/learning_loop.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/learning_loop.py)
- [src/audit/guardian.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/audit/guardian.py)

These components evaluate historical performance, classify miss causes, update policy flags, and keep the engine defensive when calibration degrades.

## 9-Component Reality Check

The repo historically described a "9-Component Deep ML Pipeline." That is no longer the best way to describe the live system. A more truthful framing is:

- `Keep`: active and worth keeping on the live path
- `Defer`: implemented or partially used, but should earn stronger production status with evidence
- `Cut from README claims`: present in the repo, but not currently core to daily live decisioning

| Component | Status | Reality in current codebase |
| --- | --- | --- |
| Fat-Tailed Distribution Engine | `Defer` | Implemented and reachable through variance/Monte Carlo paths, but mostly secondary/reference rather than the main live predictor spine |
| Hierarchical Bayesian Minutes | `Defer` | Implemented and used inside Monte Carlo/reference logic, not the main predictor for every live pick |
| Matchup Cascade | `Defer` | Real matchup and defensive-scheme logic exists, but not as a single dominant production cascade |
| CLV Feedback Loop | `Cut from README claims` | Implemented as a module, but not clearly wired into the daily live orchestration path |
| Bayesian Calibration | `Keep` | Actively used in final calibration and regime-aware gating |
| RL Betting Agent | `Cut from README claims` | Implemented in shadow-mode code, but not part of normal live decision-making |
| Teammate Impact Graph | `Keep` | Actively used in the orchestrator when lineup/injury context is present |
| Guardian Auto-Recovery | `Keep` | Guardian is live and actively gates freshness, calibration, and output quality |
| Edge Scoring Gate | `Keep` | Actively used in the orchestrator before final bet approval |

## Recommended Production Framing

Before pushing or sharing the project externally, the engine is best described as a 5-part live system:

1. Data and context ingestion
2. Core market predictor
3. Lineup and matchup enrichment
4. Edge scoring and Bayesian calibration
5. Guardian, audit, and learning loop

That description matches the code much better than claiming all nine components are equally active production pillars.

## Quick Start

If you want the shortest path from clone to a working slate run:

```bash
git clone <your-repo-url>
cd nba-props-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe verify_config.py
.venv\Scripts\python.exe run_daily.py
```

You will still need valid API keys in `.env` for live odds and supporting data refreshes.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Secrets should live in a local `.env` file, not in tracked config.

Example:

```env
ODDS_API_KEY=your_odds_key
BALLDONTLIE_API_KEY=your_balldontlie_key
API_SPORTS_KEY=
```

Tracked non-secret settings live in [config.yaml](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/config.yaml).

You can start by copying the template:

```bash
copy .env.example .env
```

### 3. Verify config

```bash
.venv\Scripts\python.exe verify_config.py
```

### 4. Understand what ships in this repo

This repo includes the core production model artifacts needed for the current live workflow:

- `models/model_points.pkl`
- `models/model_assists.pkl`
- `models/model_rebounds.pkl`
- `models/minutes_model.pkl`
- `models/ppm_model.pkl`
- `models/residual_model.pkl`

That means a fresh clone can run the live pipeline without retraining first.

This repo can also ship with a sanitized starter database:

- `data/nba_props_starter.db`

On a fresh clone, if `data/nba_props.db` does not exist but the starter DB does,
the engine will automatically bootstrap a writable local `data/nba_props.db`
from that starter copy.

### 5. Optional first-time data preparation

If you want fuller history, cleaner audits, or retraining support after clone:

```bash
.venv\Scripts\python.exe run_full_backfill.py
```

Then, if needed:

```bash
.venv\Scripts\python.exe retrain_pipeline.py
```

## Daily Workflow

Recommended order:

1. Audit yesterday's predictions
2. Review the audit and Guardian status
3. Run today's slate

### Run today's prediction flow

```bash
.venv\Scripts\python.exe run_daily.py
```

### Run an audit-only pass

```bash
.venv\Scripts\python.exe run_daily.py --audit --audit-days 1 --no-refresh --no-predict
```

### Run a specific date

```bash
.venv\Scripts\python.exe run_daily.py --date 2026-03-18
```

### Force retraining

```bash
.venv\Scripts\python.exe run_daily.py --train
```

## Fresh Clone Workflow

For someone cloning the repo for the first time, the best order is:

1. Create `.env` from [`.env.example`](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/.env.example)
2. Run [verify_config.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/verify_config.py)
3. Let the engine bootstrap from `data/nba_props_starter.db` if present
4. Run [run_full_backfill.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_full_backfill.py) if they want fuller local history
5. Run [run_daily.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_daily.py)
6. Audit the previous day before trusting new output patterns

## Reproducibility Notes

A fresh clone will get:

- the same code
- the same shipped production model artifacts
- the same workflow and report structure

A fresh clone will **not** automatically get:

- the same local SQLite history
- the same audit archive state
- the same live odds snapshot
- the same injuries/news state at the exact moment you ran yours

If you include the starter DB, a fresh clone will get a safe baseline of core
historical basketball data and schema state, but still not your full private
operating history.

So the right expectation is:

- same engine behavior and logic
- similar results when the same live inputs exist
- not guaranteed byte-for-byte identical daily outputs unless the local DB state and live data timing also match

## Outputs

The engine writes:

- Daily reports such as `report_YYYY-MM-DD_v2.md`
- Audit reports such as `audit_report_YYYY-MM-DD.md`
- Historical prediction rows to SQLite tables such as:
  - `predictions_archive`
  - `prediction_log`
  - `prediction_rejections`

The daily report distinguishes:

- `BET`
- `LEAN`
- `WATCH`

The engine may also mark a slate as `PARTIAL` when not every in-window game yields surviving prediction output after filters and gating. That is an honest coverage signal, not necessarily a pipeline failure.

## Best Use

This engine is best used as:

- a daily NBA props research workflow
- an auditable prediction pipeline
- a calibration and learning environment

It should not be treated as:

- guaranteed betting advice
- a promise of identical live results across machines
- a substitute for validating API freshness, audit coverage, and Guardian status

## Current Guardrails

The codebase now includes several operational safeguards:

- Same-day props fallback when live odds refresh fails but a fresh local snapshot exists
- Source-priority injury dedupe
- Rejection logging for dropped candidates
- Trust-score-aware final calibration
- Explicit degraded-mode handling for missing props
- More truthful report wording for unattributed model edges

## Repository Layout

```text
src/
  agents/        orchestration, consensus, selection, calibration, learning
  audit/         Guardian and strict audit components
  ingestion/     odds, injuries, schedule, opponent stats, news
  models/        live and reference model logic
  reporting/     markdown report generation
  simulation/    audit and Monte Carlo tooling
  utils/         config, database, HTTP, logging
tests/           pytest coverage for pipeline slices
models/          serialized model assets and feature metadata
data/            SQLite database and cached data
```

## What Is Still Experimental

Some modules are present for experimentation, benchmarking, or future integration. Before treating a component as live production logic, check whether it is actually wired into [run_daily.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_daily.py) and [src/agents/orchestrator.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/src/agents/orchestrator.py).

Examples of areas that may be reference-only or partially integrated:

- RL-style strategy modules
- some specialty market modules
- some advanced model variants
- CLV analysis modules that are not yet part of the normal daily run path

## Development Notes

- Prefer running the engine through the project virtual environment
- Keep secrets in `.env`
- Avoid committing generated reports, local databases, cache files, and log output
- Audit results should be interpreted together with trust, calibration, and coverage status
- Keep README claims aligned with the actual live path, not just code that exists somewhere in `src/`

## Status

The codebase is in active refinement. The engine is much more truthful and defensive than earlier versions, but calibration quality and market-specific reliability should still be validated continuously before treating outputs as production betting advice.
