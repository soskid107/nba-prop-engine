#!/usr/bin/env python3
"""
NBA Player Props Prediction Engine - Daily Runner

This is the main entry point for running daily predictions.
Orchestrates data ingestion, model training/loading, and prediction generation.

Usage:
    python run_daily.py                # Run full pipeline
    python run_daily.py --train        # Force model retraining
    python run_daily.py --backfill     # Backfill historical data
    python run_daily.py --audit        # Run audit on yesterday's predictions
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set output encoding to UTF-8 for Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.utils.config import get_config
from src.utils.database import DatabaseManager
from src.utils.logger import default_logger as logger
from src.ingestion.nba_ingestion import NBAIngestion
from src.ingestion.odds_ingestion import OddsIngestion
from src.ingestion.injury_ingestion import InjuryIngestion
from src.ingestion.opponent_stats import OpponentStatsIngestion
from src.models.training import ModelTrainer
from src.simulation.monte_carlo import SimulationEngine
from src.simulation.audit import PredictionAuditor
from src.reporting.markdown_reporter import generate_report


def print_header():
    """Print application header."""
    logger.info("="*60)
    logger.info(" [NBA] Player Props Prediction Engine")
    logger.info(" " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("="*60)


def run_data_refresh(db: DatabaseManager, target_date: str = None, force_backfill: bool = False, refresh_logs: bool = False):
    """Refresh data from all sources.
    
    Args:
        db: Database manager
        target_date: Target date for refresh
        force_backfill: If True, backfill historical data
        refresh_logs: If True, refresh game logs for today's players
    """
    logger.info("[1/4] REFRESHING DATA...")
    
    # NBA Static Data & Schedule
    nba = NBAIngestion(db)
    nba.load_all_teams()
    nba.load_all_players()
    
    # Sync schedule for the target date to ensure games table is populated
    refresh_date = target_date or datetime.now().strftime('%Y-%m-%d')
    nba.sync_schedule(refresh_date)
    
    # Opponent Advanced Stats (for Style Edge features)
    opp_stats = OpponentStatsIngestion(db)
    opp_stats.refresh_all_team_stats()
    
    # Today's Odds
    odds = OddsIngestion(db)
    odds_data = odds.fetch_todays_odds()
    prop_summary = db.get_player_prop_snapshot_summary(refresh_date)
    # Injuries
    logger.info("[Injuries] Fetching standard injury reports...")
    injury_ingestion = InjuryIngestion(db)
    try:
        injury_ingestion.fetch_injuries_from_web()
    except Exception as e:
        logger.warning(f"  [WARN] Standard injury fetch failed: {e}")

    # [NEW] Run News Scraper for late-breaking updates
    logger.info("[News] Checking for late-breaking injury news...")
    try:
        from src.ingestion.news_scraper import NewsScraperAgent
        news_agent = NewsScraperAgent(db)
        news_agent.run()
    except Exception as e:
        logger.warning(f"  [WARN] News scraper failed: {e}")
    
    # Backfill player logs if requested
    if force_backfill or refresh_logs:
        logger.info("[Logs] Refreshing game logs for TODAY'S players...")
        # Get players playing today to ensure audit passes
        todays_players_ids = nba.get_players_for_todays_games(refresh_date)
        
        if len(todays_players_ids) > 0:
             logger.info(f"  Targeting {len(todays_players_ids)} players from {refresh_date} roster...")
             nba.backfill_player_logs(todays_players_ids)
        else:
             # Fallback if no games or error
             logger.warning("  [WARN] No players found for today, falling back to first 150 active.")
             with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT player_id FROM players WHERE is_active = 1")
                player_ids = [row['player_id'] for row in cursor.fetchall()]
             nba.backfill_player_logs(player_ids[:150])
    
    logger.info("  [OK] Data refresh complete")
    return {
        'odds_games': len(odds_data or []),
        'prop_count': int(prop_summary.get('prop_count') or 0),
        'prop_game_count': int(prop_summary.get('game_count') or 0),
        'latest_prop_snapshot': prop_summary.get('latest_snapshot_time'),
    }


def run_training(db: DatabaseManager, force: bool = False, historical: bool = False) -> bool:
    """Train or load models.
    
    Args:
        db: Database manager
        force: Force retraining even if models exist
        historical: Train on ALL historical data (2000-Present)
        
    Returns:
        True if successful
    """
    print("\n[2/4] LOADING/TRAINING MODELS...")
    
    trainer = ModelTrainer(db)
    
    if not force and not historical and trainer.load_models():
        print("  [OK] Loaded existing models")
        return True
    
    print("  Training new models...")
    seasons = ['ALL'] if historical else None
    metrics = trainer.train_models(min_samples=50, seasons=seasons)
    
    # [NEW] Train Market Models (Points, Assists, Rebounds)
    print("  Training Market Models...")
    trainer.train_market_models(min_samples=50, seasons=seasons)
    
    print(f"\n  [OK] Training complete")
    print(f"    Minutes MAE: {metrics['minutes']['mae']:.2f}")
    print(f"    PPM MAE: {metrics['ppm']['mae']:.3f}")
    
    return True


from src.agents.orchestrator import PredictionOrchestrator

def run_predictions(db: DatabaseManager, target_date: str = None, calibration_flags: list = None):
    """Generate predictions for a specific date using the Multi-Agent Orchestrator.
    
    Args:
        db: Database manager
        target_date: Date to predict for (YYYY-MM-DD)
        calibration_flags: System flags to pass to agents
        
    Returns:
        DataFrame of predictions (or empty DataFrame)
    """



    print(f"\n[3/4] GENERATING PREDICTIONS FOR {target_date}...")
    if calibration_flags:
        print(f"  [SYSTEM] Active Calibration Flags: {calibration_flags}")

    now_local = datetime.now().astimezone()
    
    if target_date:
        try:
             # Parse target date and set time to midnight local
             # Note: This assumes target_date fits local timezone context. 
             # Safe assumption for user running locally.
             dt = datetime.strptime(target_date, '%Y-%m-%d')
             # naive to aware (local)
             # Ideally use proper timezone, but using astimezone result as reference
             window_start = dt.replace(hour=0, minute=0, second=0, microsecond=0).astimezone()
        except Exception:
             window_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        window_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        
    window_end = window_start + timedelta(days=1, hours=5)
    
    # Initialize Orchestrator
    orchestrator = PredictionOrchestrator()
    slate_date = target_date or datetime.now().strftime('%Y-%m-%d')
    orchestrator.agent5.clear_rejections_for_date(slate_date)
    
    # Fetch latest odds (Games + Props)
    # Note: run_data_refresh already called this, but we ensure DB is populated here too
    from src.ingestion.odds_ingestion import OddsIngestion
    odds_ingestion = OddsIngestion(db)
    # [OPTIMIZATION] Removed redundant fetch_todays_odds() - run_data_refresh does this.

    events = odds_ingestion.http.get_odds_api(
        endpoint="/sports/basketball_nba/events",
        params={'regions': 'us'},
        cache_hours=1
    )

    games_in_window = []
    if events:
        for ev in events:
            commence = ev.get('commence_time')
            if not commence:
                continue
            try:
                ev_dt = datetime.fromisoformat(commence.replace('Z', '+00:00')).astimezone()
            except Exception:
                continue

            if window_start <= ev_dt < window_end:
                home = ev.get('home_team', '')
                away = ev.get('away_team', '')
                home_abbr = odds_ingestion._normalize_team_name(home)
                away_abbr = odds_ingestion._normalize_team_name(away)
                ev_id = ev.get('id')
                if ev_id and home_abbr and away_abbr:
                    games_in_window.append({
                        'event_id': ev_id,
                        'home_abbr': home_abbr,
                        'away_abbr': away_abbr,
                        'commence_local': ev_dt.isoformat()
                    })

    if not games_in_window:
        print("  [DIAGNOSTIC] No games found in the required time window.")
        print(f"  Window Start (local): {window_start.isoformat()}")
        print(f"  Window End   (local): {window_end.isoformat()}")
        return pd.DataFrame()

    event_ids = [g['event_id'] for g in games_in_window]
    placeholders = ",".join(["?"] * len(event_ids))

    # Updated Query to fetch odds as well
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT game_id, player_id, line, odds_over, odds_under, bookmaker, snapshot_time, market_key
            FROM player_prop_odds
            WHERE game_id IN ({placeholders})
              -- FILTER REMOVED: Fetch all markets (points, assists, rebounds)
              AND player_id IS NOT NULL
              AND line IS NOT NULL
            ORDER BY snapshot_time DESC
        """, tuple(event_ids))
        prop_rows = cursor.fetchall()

    props_by_game = {}
    book_counts_by_game = {}
    
    # [NEW] Intermediate storage for aggregation
    player_lines_map = {} # f"{pid}_{market_key}" -> list of entries

    for r in prop_rows:
        gid = r['game_id']
        pid = r['player_id']
        book = r['bookmaker']
        line = r['line']
        snap = r['snapshot_time']
        mkey = r['market_key']
        
        if gid not in props_by_game: props_by_game[gid] = 0
        props_by_game[gid] += 1
        
        if gid not in book_counts_by_game: book_counts_by_game[gid] = {}
        book_counts_by_game[gid][book] = book_counts_by_game[gid].get(book, 0) + 1
        
        # [FIX] Key by PID + Market
        map_key = f"{pid}_{mkey}"
        if map_key not in player_lines_map:
            player_lines_map[map_key] = []
            
        # Deduplicate: Since sorted by snapshot_time DESC, first occurrence of a book for a player is the latest
        # Check if book already collected for this player/market
        if any(x['book'] == book for x in player_lines_map[map_key]):
            continue
            
        player_lines_map[map_key].append({
            'line': line,
            'book': book,
            'odds_over': r['odds_over'] if r['odds_over'] is not None else -110,
            'odds_under': r['odds_under'],
            'snapshot_time': snap
        })

    # [NEW] Aggregate lines
    market_lines = {}
    market_odds = {}
    line_info_by_player = {}
    
    preferred_books = ['fanduel', 'draftkings', 'betmgm']

    for map_key, entries in player_lines_map.items():
        if not entries: continue
        
        # map_key is "{pid}_{market_key}"
        parts = map_key.split('_')
        pid = int(parts[0])
        
        # Get all lines available for this player
        valid_lines = [x['line'] for x in entries]
        
        # Calculate Median, Floor, Ceiling
        lines_sorted = sorted(valid_lines)
        mid = len(lines_sorted) // 2
        line_median = lines_sorted[mid]
        line_min = lines_sorted[0]
        line_max = lines_sorted[-1]
        
        # Pick representative entry (matching median)
        # Prefer preferred books if feasible
        candidates = [x for x in entries if x['line'] == line_median]
        candidates.sort(key=lambda x: preferred_books.index(x['book']) if x['book'] in preferred_books else 999)
        bp = candidates[0]
        
        # Store structured context
        # [NOTE] market_lines is legacy (pid -> line). We only store Points here if available, 
        # or just overwrite. It's not critical for orchestrator as it uses DB props directly.
        if 'points' in map_key:
            market_lines[pid] = {
                'line': line_median,
                'floor': line_min,
                'ceiling': line_max
            }
            market_odds[pid] = bp['odds_over']
        
        # [FIX] Store keyed by combo so we can look it up correctly later
        line_info_by_player[map_key] = {
            'line': line_median,
            'floor': line_min,
            'ceiling': line_max,
            'odds_over': bp['odds_over'],
            'odds_under': bp['odds_under'],
            'bookmaker': bp['book'],
            'source': 'AGGREGATED'
        }

    games_missing_props = [g for g in games_in_window if props_by_game.get(g['event_id'], 0) == 0]
    if games_missing_props:
        print("  [DIAGNOSTIC] Missing player_points props for some games in window. Continuing with predictions.")
        print(f"  Games in window: {len(games_in_window)}")
        for g in games_in_window:
            cnt = props_by_game.get(g['event_id'], 0)
            print(f"    - {g['away_abbr']} @ {g['home_abbr']} | props_found={cnt}")

    eligible_games = [g for g in games_in_window if props_by_game.get(g['event_id'], 0) > 0]

    print(f"  Games in window: {len(games_in_window)}")
    print(f"  Eligible games (have player_points props): {len(eligible_games)}")
    print(f"  Loaded {len(market_lines)} unique player lines across eligible games.")

    for g in games_in_window:
        gid = g['event_id']
        cnt = props_by_game.get(gid, 0)
        books = book_counts_by_game.get(gid, {})
        print(f"  [LINES] {g['away_abbr']} @ {g['home_abbr']} | total={cnt} | by_book={books}")

    if not eligible_games:
        print("  [DIAGNOSTIC] No games have usable player props for this slate.")
        return pd.DataFrame()

    results = []
    expected_game_keys = set()
    event_id_by_game_key = {}
    for g in eligible_games:
        game_key = " vs ".join(sorted([g['home_abbr'], g['away_abbr']]))
        expected_game_keys.add(game_key)
        event_id_by_game_key[game_key] = g['event_id']
        game_results = orchestrator.predict_game(
            g['home_abbr'], 
            g['away_abbr'], 
            market_lines=market_lines,
            market_odds=market_odds, # [NEW]
            calibration_flags=calibration_flags, # [NEW]
            game_date=target_date # [FIX] Pass date for correct prop fetching
        )
        results.extend(game_results)
    
    if not results:
        print(f"  [DIAGNOSTIC] No predictions generated for eligible games in window.")
        return pd.DataFrame()
        
    # Convert results to DataFrame for CSV export
    print(f"  Converting {len(results)} agent results to CSV...")
    # import pandas as pd # REMOVED: Use global import

    from src.ingestion.injury_ingestion import InjuryIngestion
    injury_ingestion = InjuryIngestion(db)
    todays_injuries = injury_ingestion.get_todays_injuries() or []
    p_play_by_player = {}
    status_by_player = {}
    injury_source_by_player = {}
    injury_fetched_at_by_player = {}
    for inj in todays_injuries:
        pid = inj.get('player_id')
        if pid:
            p_play_by_player[pid] = inj.get('p_play', 1.0)
            status_by_player[pid] = inj.get('status', 'AVAILABLE')
            if inj.get('source_name'):
                injury_source_by_player[pid] = inj.get('source_name')
            if inj.get('fetched_at'):
                injury_fetched_at_by_player[pid] = inj.get('fetched_at')
    
    data = []
    produced_game_keys = set()
    for r in results:
        if not r.pipeline_success:
            continue
        if r.player_name == 'Unknown' or r.team == 'UNK' or r.opponent == 'UNK':
            continue

        game_key = " vs ".join(sorted([str(r.team), str(r.opponent)]))
        event_id = event_id_by_game_key.get(game_key)

        market_type = r.raw_prediction.get('market_type', 'points')
        
        # [FIX] Lookup Line by Combo Key
        # DB market keys are 'player_points', 'player_assists', etc.
        lookup_key = f"{r.player_id}_player_{market_type}"
        line_info = line_info_by_player.get(lookup_key)
        line_value = None
        line_bookmaker = None
        if line_info:
            line_value = line_info.get('line')
            line_bookmaker = line_info.get('bookmaker')
        if line_value is None:
            continue
        if not r.betting_decision:
            continue

        direction = r.betting_decision.direction
        confidence = r.betting_decision.confidence
        edge_over = r.betting_decision.edge_over
        edge_under = r.betting_decision.edge_under
        edge_source = r.betting_decision.edge_source

        # [FIX] Filter for Points Market ONLY (for now)
        market_type = r.raw_prediction.get('market_type', 'points')
        # if market_type != 'points':
        #     continue

        # [FIX] Deduplicate Players
        # Only take the first (best) entry for each player/market combo
        combo_key = f"{r.player_id}_{market_type}"
        if combo_key in produced_game_keys: 
             continue
        produced_game_keys.add(combo_key)

            
        # Flatten structure
        row = {
            'player_id': r.player_id,
            'player_name': r.player_name,
            'team': r.team,
            'opponent': r.opponent,
            'game_date': target_date,
            'event_id': event_id,
            
            # Agent 2
            'predicted_mean': r.audited_prediction.get('mean'),
            'predicted_std': r.audited_prediction.get('std'),
            'p10': r.audited_prediction.get('p10'),
            'p50': r.audited_prediction.get('mean'), # Approx
            'p90': r.audited_prediction.get('p90'),
            'pure_model_pred': r.raw_prediction.get('pure_model_pred'),
            'market_adjusted_pred': r.raw_prediction.get('market_adjusted_pred'),
            'post_rule_pred': r.raw_prediction.get('post_rule_pred'),
            
            # Context
            'minutes_L10': r.player_context.get('stats', {}).get('minutes_L10'),
            'points_L5': r.player_context.get('points_L5', 0),
            'points_L15': r.player_context.get('points_L15', 0),
            'injury_status': status_by_player.get(r.player_id, 'AVAILABLE'),
            'p_play': p_play_by_player.get(r.player_id, 1.0),
            'injury_source': injury_source_by_player.get(r.player_id),
            'injury_fetched_at': injury_fetched_at_by_player.get(r.player_id),
            
            # Consensus (Phase 1)
            'consensus_verdict': r.player_context.get('consensus_verdict', 'N/A'),
            'consensus_level': r.player_context.get('consensus_level', 'MAJORITY'),
            'consensus_reason': r.player_context.get('consensus_reason', ''),
            
            # Agent 4 (Decision)
            'line': line_value,
            'direction': direction,
            'confidence': confidence,
            'edge_over': edge_over,
            'edge_under': edge_under,
            'bookmaker': line_bookmaker,
            
            # Phase 7 Features
            'edge_source': edge_source,
            'regime_status': r.match_context.get('regime_status', 'stable'),
            'archetype': r.raw_prediction.get('archetype'),
            'market': market_type, # [NEW] output market type
            'h2h_avg': r.raw_prediction.get('h2h_avg'),
            
            # Edge Analysis (Phase 8)
            'edge_tier': r.match_context.get('edge_analysis', {}).get('tier', ''),
            'edge_candidate_tier': r.match_context.get('edge_analysis', {}).get('candidate_tier', r.match_context.get('edge_analysis', {}).get('tier', '')),
            'edge_score': r.match_context.get('edge_analysis', {}).get('score', 0),
            'edge_explanation': r.match_context.get('edge_analysis', {}).get('explanation', ''),
            'edge_direction': r.match_context.get('edge_analysis', {}).get('direction', ''),
            'edge_candidate_direction': r.match_context.get('edge_analysis', {}).get('candidate_direction', r.match_context.get('edge_analysis', {}).get('direction', '')),
            'decision_alignment_status': r.match_context.get('decision_alignment', {}).get('alignment_status', ''),
            'approved_bet': int(bool(r.match_context.get('decision_alignment', {}).get('approved_bet', False))),
            'player_consensus_status': r.match_context.get('decision_trace', {}).get('player_consensus_status', r.match_context.get('player_consensus', {}).get('status', '')),
            'player_consensus_level': r.match_context.get('decision_trace', {}).get('player_consensus_level', r.match_context.get('player_consensus', {}).get('level', '')),
            'market_consensus_status': r.match_context.get('decision_trace', {}).get('market_consensus_status', r.match_context.get('market_consensus', {}).get('status', '')),
            'market_consensus_level': r.match_context.get('decision_trace', {}).get('market_consensus_level', r.match_context.get('market_consensus', {}).get('level', '')),
            'candidate_rank': r.match_context.get('decision_trace', {}).get('candidate_rank', r.match_context.get('selection_reasoning', {}).get('candidate_rank', 1)),
            'candidate_score_gap': r.match_context.get('decision_trace', {}).get('candidate_score_gap', r.match_context.get('selection_reasoning', {}).get('score_gap_to_next')),
            'final_status': r.match_context.get('decision_trace', {}).get('final_status', ''),
            'rejection_stage': r.match_context.get('decision_trace', {}).get('rejection_stage', ''),
            'final_decision_reason': r.match_context.get('decision_trace', {}).get('final_decision_reason', ''),
            
            # Selection Reasoning (from MarketSelector)
            'selection_reason': '; '.join(r.match_context.get('selection_reasoning', {}).get('reasons', [])) if r.match_context.get('selection_reasoning') else '',
            'prediction_health_score': r.raw_prediction.get('prediction_health', {}).get('health_score', 1.0),
            'prediction_degradation_flags': ';'.join(r.raw_prediction.get('prediction_health', {}).get('degradation_flags', [])),
            'used_fallback_model': int(bool(r.raw_prediction.get('prediction_health', {}).get('used_fallback_model', False))),
            'market_anchor_applied': int(bool(r.raw_prediction.get('prediction_health', {}).get('market_anchor_applied', False))),
            'market_anchor_weight': r.raw_prediction.get('prediction_health', {}).get('market_anchor_weight', 0.0),
            'core_predictor': r.raw_prediction.get('model_provenance', {}).get('core_predictor', 'market_predictor'),
            'mechanistic_reference_used': int(bool(r.raw_prediction.get('model_provenance', {}).get('mechanistic_reference_used', False))),
            'mechanistic_reference_mean': r.raw_prediction.get('model_provenance', {}).get('mechanistic_reference_mean'),
            'monte_carlo_reference_used': int(bool(r.raw_prediction.get('model_provenance', {}).get('monte_carlo_reference_used', False))),
            'monte_carlo_reference_mean': r.raw_prediction.get('model_provenance', {}).get('monte_carlo_reference_mean'),
            'residual_model_ready': int(bool(r.raw_prediction.get('model_provenance', {}).get('residual_model_ready', False))),
            'residual_reference_adjustment': r.raw_prediction.get('model_provenance', {}).get('residual_reference_adjustment', 0.0),
            'injury_context_present': int(bool(r.raw_prediction.get('model_provenance', {}).get('injury_context_present', False))),
            'injury_context_size': r.raw_prediction.get('model_provenance', {}).get('injury_context_size', 0),
            'teammate_network_active': int(bool(r.raw_prediction.get('model_provenance', {}).get('teammate_network_active', False))),
            'ensemble_applied': int(bool(r.raw_prediction.get('model_provenance', {}).get('ensemble_applied', False))),
            'ensemble_mean': r.raw_prediction.get('model_provenance', {}).get('ensemble_mean'),
            'reference_consensus_mean': r.raw_prediction.get('model_provenance', {}).get('reference_consensus_mean'),
            'model_disagreement': r.raw_prediction.get('model_provenance', {}).get('model_disagreement'),
            'ensemble_notes': ';'.join(r.raw_prediction.get('model_provenance', {}).get('ensemble_notes', [])),
        }
        
        # Add Style Edges if calculated in context
        if 'efficiency_model' in r.player_context:
             # This might be nested deeper or named differently in Agent 1 output
             pass # Skip for now to avoid key errors, safe default
             
        data.append(row)
        
    df = pd.DataFrame(data)
    
    if df.empty:
        print("  [DIAGNOSTIC] No eligible player predictions (missing lines and/or missing linkage).")
        return pd.DataFrame()

    produced_game_keys = set()
    for _, row in df.iterrows():
        produced_game_keys.add(" vs ".join(sorted([str(row.get('team')), str(row.get('opponent'))])))

    missing_games = [g for g in expected_game_keys if g not in produced_game_keys]
    if missing_games:
        print("  [DIAGNOSTIC] Slate gap: at least one game in window produced zero eligible players.")
        for mg in sorted(missing_games):
            print(f"    - Missing game in output: {mg}")
        print("  Continuing with available games.")
        
    
    # [MODIFIED] Return DataFrame directly (No CSV)
    return df


# Reporting logic moved to src.reporting.markdown_reporter

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NBA Player Props Prediction Engine')
    parser.add_argument('--train', action='store_true', help='Force model retraining')
    parser.add_argument('--backfill', action='store_true', help='Backfill historical data')
    parser.add_argument('--no-predict', action='store_true', help='Skip prediction generation')
    parser.add_argument('--report-only', action='store_true', help='Regenerate report from existing predictions (skip refresh/predictions)')
    parser.add_argument('--audit', action='store_true', help='Audit yesterday predictions vs actuals')
    parser.add_argument('--audit-days', type=int, default=1, help='Days back to audit (default: 1)')
    parser.add_argument('--historical', action='store_true', help='Train on ALL historical data (2000+)')
    parser.add_argument('--refresh-logs', action='store_true', help='Force refresh of player game logs')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--no-refresh', action='store_true', help='Skip data refresh (odds, logs, etc.)')
    
    args = parser.parse_args()
    
    print_header()
    
    try:
        # Initialize database
        db = DatabaseManager()
        auditor = PredictionAuditor(db)
        
        # [NEW] Initialize Guardian
        from src.audit.guardian import ProductionGuardian
        guardian = ProductionGuardian(db)
        
        # Determine target date
        today = args.date if args.date else datetime.now().strftime('%Y-%m-%d')
        
        # [NEW] Report-Only Mode: Skip refresh and predictions, just regenerate report
        if args.report_only:
            print(f"\n[REPORT-ONLY MODE] Regenerating report for {today} from existing predictions...")
            csv_path = f"predictions_{today}.csv"
            
            if not os.path.exists(csv_path):
                print(f"\n[ERROR] Predictions file not found: {csv_path}")
                print("Run without --report-only first to generate predictions.")
                return 1
            
            health_report = guardian.generate_health_report()
            report_ok = generate_report(db, csv_path, target_date=today, guardian_report=health_report)
            
            if report_ok:
                print(f"\n[OK] Report regenerated: report_{today}_v2.md")
            return 0
        
        # Run pipeline
        refresh_summary = {}
        if not args.no_refresh:
            refresh_summary = run_data_refresh(db, target_date=today, force_backfill=args.backfill, refresh_logs=args.refresh_logs) or {}
        else:
            print("\n[INFO] Skipping data refresh (--no-refresh). Using existing DB data.")
        
        # [AUDIT] Gate 1 & 2: Data Integrity
        # We check freshness immediately after refresh
        if not guardian.check_data_freshness():
             print(f"\n[WARN] Guardian raised flags: {guardian.flags}")
             # In WARN mode we continue, but we track the flags

        prop_count_after_refresh = int((refresh_summary or {}).get('prop_count') or 0)
        if not args.no_refresh and prop_count_after_refresh == 0:
            latest_prop_snapshot = (refresh_summary or {}).get('latest_prop_snapshot')
            print("\n[STOP] No usable player props are available for this slate.")
            if latest_prop_snapshot:
                print(f"  Latest stored prop snapshot: {latest_prop_snapshot}")
            print("  Cause: live prop refresh failed and no same-day fallback props were available.")
            print("  Next step: wait for the Odds API quota/auth to recover, then rerun.")
            return 1
        
        # [R9] Gate 5: Graduated Active Calibration Check
        calibration_status = guardian.check_historical_calibration(days_back=7)
        system_flags = []
        defense_labels = {
            'SEVERELY_OVERCONFIDENT': ('🔴 SEVERELY OVERCONFIDENT', 'SYSTEM_SEVERELY_OVERCONFIDENT'),
            'OVERCONFIDENT':         ('🟠 OVERCONFIDENT', 'SYSTEM_OVERCONFIDENT'),
            'SLIGHTLY_OVERCONFIDENT':('🟡 SLIGHTLY OVERCONFIDENT', 'SYSTEM_SLIGHTLY_OVERCONFIDENT'),
            'RECOVERING':           ('🟢 RECOVERING', 'SYSTEM_RECOVERING'),
            'UNDERCONFIDENT':       ('🔵 UNDERCONFIDENT', 'SYSTEM_UNDERCONFIDENT'),
        }
        if calibration_status in defense_labels:
            label, flag = defense_labels[calibration_status]
            system_flags.append(flag)
            print(f"\n[GUARDIAN] ⚠️ System is {label}. Flag: {flag}")
        else:
            print(f"\n[GUARDIAN] ✅ Calibration STABLE. No defensive adjustments.")

        # [NEW] Learning-loop policy flags from recurring miss patterns
        from src.agents.learning_loop import LearningLoopAgent
        learning_policy = LearningLoopAgent(db)
        policy_flags = learning_policy.get_policy_flags(lookback_days=30)
        if policy_flags:
            system_flags.extend([flag for flag in policy_flags if flag not in system_flags])
            print(f"\n[LEARNING] Active Policy Flags: {policy_flags}")
        
        # Initialize engine for audit and predictions
        engine = SimulationEngine(db)
        
        # [AUDIT] Mandatory Feature Variability Check
        # We use the Guardian's check now if possible, but keep engine's for back-compat if needed
        # Actually, let's use engine's deep check but have Guardian log it
        if not engine.run_feature_audit(auditor):
             print("\n[STOP] Feature variability check FAILED. See output for details.")
             print("Fix data integrity issues before proceeding.")
             return 1
        
        run_training(db, force=args.train, historical=args.historical)
        
        df_preds = pd.DataFrame()
        if not args.no_predict:
            # today is already set above
            
            print(f"\n>>> PROCESSING SLATE FOR {today} <<<")
            
            # [AUDIT] Gate 1 (Revised): Slate Integrity
            # Check if we have games before trying to run
            # Fetch events first? orchestrator does it inside. 
            # We trust run_predictions to handle empty slate, but Guardian can check after.
            
            df_preds = run_predictions(db, target_date=today, calibration_flags=system_flags)
            report_ok = False

            if not df_preds.empty:
                # [NEW] Portfolio Manager (Phase 2)
                # Optimize amounts and apply risk controls
                from src.agents.portfolio_manager import PortfolioManagerAgent
                pm_agent = PortfolioManagerAgent()
                
                print("\n[Portfolio] Optimizing bet sizing and risk exposure...")
                raw_predictions = df_preds.to_dict('records')
                actionable_bets = [
                    bet for bet in raw_predictions
                    if str(bet.get('direction', '')).upper() in ('OVER', 'UNDER')
                ]
                optimized_bets = pm_agent.optimize_portfolio(actionable_bets)

                sized_lookup = {
                    (
                        bet.get('prediction_date'),
                        bet.get('player_id'),
                        bet.get('market'),
                        str(bet.get('direction', '')).upper(),
                    ): bet
                    for bet in optimized_bets
                }

                df_preds = df_preds.copy()
                df_preds['units'] = 0.0
                df_preds['risk_rationale'] = ''

                for idx, row in df_preds.iterrows():
                    key = (
                        row.get('prediction_date'),
                        row.get('player_id'),
                        row.get('market'),
                        str(row.get('direction', '')).upper(),
                    )
                    sized_bet = sized_lookup.get(key)
                    if sized_bet:
                        df_preds.at[idx, 'units'] = sized_bet.get('units', 0.0)
                        df_preds.at[idx, 'risk_rationale'] = sized_bet.get('risk_rationale', '')

                if actionable_bets and not optimized_bets:
                    print("  [WARN] All final bets filtered out by risk rules!")
                else:
                    print(
                        f"  Approved {len(optimized_bets)} final bets "
                        f"(from {len(actionable_bets)} actionable, {len(raw_predictions)} total predictions)."
                    )

                # [AUDIT] Mandatory Prediction Distribution Check
                
                # Check with Guardian (Structure + Drift)
                guardian.check_prediction_reasons(df_preds)
                guardian.check_distribution_drift(df_preds, days_back=7) # [NEW]
                
                if not auditor.check_prediction_distribution(df_preds):
                     print(f"\n[STOP] Prediction distribution check FAILED for {today}. Predictions are structurally wrong.")
                     return 1

                # Archive predictions for later auditing
                print(f"\n[5/5] ARCHIVING PREDICTIONS FOR {today}...")
                auditor.archive_predictions(df_preds)
                
                # Generate Guardian Report
                health_report = guardian.generate_health_report()
                health_report.metrics['active_policy_flags'] = system_flags
                print(f"\n[GUARDIAN] System Health: {health_report.status}")
                if health_report.flags:
                    print("  Flags raised:")
                    for f in health_report.flags:
                        print(f"  - {f}")
                    
                    # [NEW] Persist Alerts
                    guardian.save_alerts()
                    print(f"  [ALERT] Critical issues logged to ALERTS.log")

                report_ok = generate_report(db, df_preds, target_date=today, guardian_report=health_report)

            if report_ok:
                slate_scope = "UNKNOWN"
                if not df_preds.empty:
                    with db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT DISTINCT home_team_abbr, away_team_abbr
                            FROM games
                            WHERE game_date = ?
                        """, (today,))
                        expected_games = {
                            " vs ".join(sorted([row['home_team_abbr'], row['away_team_abbr']]))
                            for row in cursor.fetchall()
                            if row['home_team_abbr'] and row['away_team_abbr']
                        }
                    reported_games = set()
                    if {'home_team', 'away_team'}.issubset(df_preds.columns):
                        reported_games = {
                            " vs ".join(sorted([str(row['home_team']), str(row['away_team'])]))
                            for _, row in df_preds[['home_team', 'away_team']].dropna().drop_duplicates().iterrows()
                        }
                    missing_games = sorted(expected_games - reported_games)
                    slate_scope = "COMPLETE" if expected_games and reported_games == expected_games else "PARTIAL"

                print(f"\nThis report represents a {slate_scope} slate for the defined time window.")
                if slate_scope == "PARTIAL":
                    if missing_games:
                        preview = ", ".join(missing_games[:5])
                        suffix = " ..." if len(missing_games) > 5 else ""
                        print(f"  Missing reported game coverage: {preview}{suffix}")
                    else:
                        print("  Partial because some in-window game sides produced no surviving predictions after filtering.")
            else:
                print("\nNo valid report generated — conditions unmet.")
        
        # Run audit if requested
        if args.audit:
            print("\n[AUDIT] RUNNING PREDICTION AUDIT...")
            audit_result = auditor.run_daily_audit(days_back=args.audit_days)
            auditor.generate_audit_report()
            
            # [NEW] Run Learning Loop after audit
            learning = LearningLoopAgent(db)
            audit_date = (datetime.now() - timedelta(days=args.audit_days)).strftime('%Y-%m-%d')
            updated = learning.update_with_actuals(audit_date)
            if updated > 0:
                print(f"  [Learning] Updated {updated} predictions with actuals")
                report = learning.generate_daily_report(audit_date)
                for line in report.get('summary', [])[:5]:  # Show first 5 summary lines
                    print(f"  {line}")
        
        print("\n" + "="*60)
        print(" [OK] PIPELINE COMPLETE")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
