# NBA Prop Engine

NBA Prop Engine is a multi-market NBA player props workflow for daily research, prediction, reporting, audit, and feedback-driven model improvement.

It refreshes live slate data, builds player and matchup context, generates market predictions, applies trust-aware calibration, produces daily reports, and audits prior results against actual outcomes.

## Features

- Multi-market player prop workflow with strongest live support for points, assists, and rebounds
- Daily refresh of schedule, props, injuries, news, and opponent context
- Market selection, trust scoring, edge scoring, and final calibration
- Daily markdown reports with `BET`, `LEAN`, and `WATCH` outputs
- Historical archiving to SQLite for audit and learning
- Audit-first operating flow with miss attribution and policy feedback
- Guardian safety checks for calibration and degraded input conditions
- Shipped production model artifacts for clone-and-run setup
- Sanitized starter database for faster onboarding

## Quick Start

```bash
git clone https://github.com/Soskid107/nba-prop-engine.git
cd nba-prop-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe verify_config.py
.venv\Scripts\python.exe run_daily.py
```

## Requirements

- Python 3.11+
- Windows PowerShell workflow is currently the most tested path
- API keys for live data refresh

## Configuration

Create a local `.env` file from the template:

```bash
copy .env.example .env
```

Populate the keys you intend to use:

```env
ODDS_API_KEY=your_api_key_here
BALLDONTLIE_API_KEY=
API_SPORTS_KEY=
```

Tracked non-secret settings live in [config.yaml](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/config.yaml).

Verify the local setup:

```bash
.venv\Scripts\python.exe verify_config.py
```

## Included Assets

This repository includes the production model files required by the current live workflow:

- `models/model_points.pkl`
- `models/model_assists.pkl`
- `models/model_rebounds.pkl`
- `models/minutes_model.pkl`
- `models/ppm_model.pkl`
- `models/residual_model.pkl`

It also includes a sanitized starter database:

- `data/nba_props_starter.db`

On a fresh clone, if `data/nba_props.db` does not exist, the engine will automatically bootstrap a writable local database from the starter copy.

## Daily Workflow

Recommended operating order:

1. Audit yesterday’s predictions
2. Review audit results and Guardian status
3. Run today’s slate

Run today’s full workflow:

```bash
.venv\Scripts\python.exe run_daily.py
```

Run audit only:

```bash
.venv\Scripts\python.exe run_daily.py --audit --audit-days 1 --no-refresh --no-predict
```

Run a specific slate date:

```bash
.venv\Scripts\python.exe run_daily.py --date 2026-03-18
```

Force retraining:

```bash
.venv\Scripts\python.exe run_daily.py --train
```

## First-Time Data Setup

The shipped starter database is enough to get the engine running, but a fuller local history is better for retraining and long-horizon audits.

Optional first-time preparation:

```bash
.venv\Scripts\python.exe run_full_backfill.py
```

Optional retraining after backfill:

```bash
.venv\Scripts\python.exe retrain_pipeline.py
```

## Outputs

The engine generates:

- Daily reports such as `report_YYYY-MM-DD_v2.md`
- Audit reports such as `audit_report_YYYY-MM-DD.md`
- Historical archive rows in SQLite tables such as:
  - `predictions_archive`
  - `prediction_log`
  - `prediction_rejections`

Daily report outputs are grouped as:

- `BET`
- `LEAN`
- `WATCH`

The engine may mark a slate as `PARTIAL` when not every in-window game produces surviving output after filtering and calibration. That is a coverage indicator, not automatically a pipeline failure.

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
tests/           pipeline and phase coverage
models/          serialized model assets and feature metadata
data/            local SQLite data and starter database
```

## Core Entry Points

- [run_daily.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_daily.py)
- [run_full_audit.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_full_audit.py)
- [run_full_backfill.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/run_full_backfill.py)
- [retrain_pipeline.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/retrain_pipeline.py)
- [verify_config.py](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/verify_config.py)

## Notes on Reproducibility

A fresh clone will have:

- the same code
- the same shipped production model artifacts
- the same starter database baseline

A fresh clone will not automatically have:

- the same live odds snapshot
- the same injury/news state at the exact same time
- the same local audit history
- the same full private operating database

Expect similar engine behavior, but not guaranteed identical live outputs unless the local data state and live refresh timing also match.

## Limitations

- The strongest live support is currently around points, assists, and rebounds
- Live outputs depend on third-party data availability and API access
- Calibration quality should still be monitored continuously through the audit workflow
- This project is best used as a research and decision-support engine, not as guaranteed betting advice

## License

This project is released under the MIT License. See [LICENSE](/C:/Users/HP/OneDrive/alade.py/nba-props-engine/LICENSE).
