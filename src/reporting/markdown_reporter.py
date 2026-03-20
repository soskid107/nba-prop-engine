import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback
from typing import Optional, Dict, Any

from ..utils.config import get_config
from ..utils.database import DatabaseManager
from ..utils.logger import default_logger as logger
from ..agents.parlay_builder import ParlayBuilder
from ..agents.sgp_engine import SGPEngine
from ..agents.edge_tracker import EdgeTracker

def generate_report(db: DatabaseManager, predictions_source: Any, target_date: str = None, guardian_report = None) -> bool:
    """Generate markdown report.
    
    Args:
        db: Database manager
        predictions_source: Path to predictions CSV (str) or DataFrame
        target_date: Date of report
        guardian_report: Optional GuardianReport object
    """
    logger.info("[4/4] GENERATING REPORT...")
    
    try:
        config = get_config()
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        report_path = config.project_root / f'report_{target_date}_v2.md' # Force v2 name
        
        df = pd.DataFrame()
        
        # Handle Input Source
        if isinstance(predictions_source, pd.DataFrame):
            df = predictions_source
            logger.info(f"  Using DataFrame with {len(df)} rows")
        elif isinstance(predictions_source, str) and predictions_source:
             if Path(predictions_source).exists():
                 try:
                     df = pd.read_csv(predictions_source)
                     logger.info(f"  Loaded {len(df)} predictions from CSV")
                 except Exception as e:
                     logger.warning(f"  [WARN] Failed to read CSV: {e}")
        else:
             # Auto-discovery fallback
             default_path = config.project_root / f'predictions_{target_date}.csv'
             if default_path.exists():
                 try:
                    df = pd.read_csv(default_path)
                    logger.info(f"  [INFO] Auto-discovered predictions CSV: {default_path}")
                 except Exception as e:
                     logger.warning(f"  [WARN] Failed to read CSV {default_path}: {e}")
        
        # === INSTRUMENTATION ===
        # Get API usage stats
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT api_name, COUNT(*) as calls, SUM(cached) as cached_calls
                FROM api_usage WHERE call_date = ?
                GROUP BY api_name
            """, (datetime.now().strftime('%Y-%m-%d'),)) # Usage is always "today"
            api_stats = {row['api_name']: {'calls': row['calls'], 'cached': row['cached_calls']} 
                        for row in cursor.fetchall()}
        
        # Generate report content
        report_lines = []
        
        # Header
        report_lines.append(f"# NBA Props Prediction Report")
        report_lines.append(f"## {target_date}")
        report_lines.append("")
        
        # [NEW] System Health Section
        if guardian_report:
            icon = "🟢"
            if guardian_report.status == 'YELLOW': icon = "🟡"
            elif guardian_report.status == 'RED': icon = "🔴"
            
            report_lines.append(f"### {icon} System Health: {guardian_report.status}")
            if guardian_report.flags:
                report_lines.append("Analysis Gates Triggered:")
                for flag in guardian_report.flags:
                    report_lines.append(f"- {flag}")
            else:
                report_lines.append("All integrity checks passed.")
            active_policy_flags = guardian_report.metrics.get('active_policy_flags', []) if guardian_report.metrics else []
            if active_policy_flags:
                report_lines.append(f"Active Policy Flags: {', '.join(active_policy_flags)}")
            report_lines.append("")

        
        if df.empty:
            logger.info("  [DIAGNOSTIC] Report not generated: predictions CSV is empty.")
            return False
        else:
            # Filter "Unknown" players explicitly
            df = df[df['player_name'] != 'Unknown'].copy()

            # Normalize archive-vs-live column names so historical reports can be rebuilt
            if 'direction' not in df.columns and 'pick_direction' in df.columns:
                df['direction'] = df['pick_direction']
            if 'line' not in df.columns and 'sportsbook_line' in df.columns:
                df['line'] = df['sportsbook_line']
            if 'market' not in df.columns and 'market_key' in df.columns:
                df['market'] = df['market_key']
            if 'confidence' not in df.columns and 'bet_confidence' in df.columns:
                df['confidence'] = df['bet_confidence']
            if 'edge_direction' not in df.columns and 'edge_candidate_direction' in df.columns:
                df['edge_direction'] = df['edge_candidate_direction']
            if 'edge_tier' not in df.columns and 'edge_candidate_tier' in df.columns:
                df['edge_tier'] = df['edge_candidate_tier']
            if 'direction' in df.columns and 'final_status' in df.columns:
                watch_or_lean = df['final_status'].fillna('').astype(str).str.upper().isin(['WATCH', 'LEAN'])
                non_bet_dir = ~df['direction'].fillna('').isin(['OVER', 'UNDER'])
                df.loc[watch_or_lean & non_bet_dir, 'direction'] = 'NO_BET'
            
            if 'direction' in df.columns:
                picks = df[df['direction'].isin(['OVER', 'UNDER'])].copy()
                no_plays = df[~df['direction'].isin(['OVER', 'UNDER'])].copy()
            else:
                picks = df.copy()
                no_plays = pd.DataFrame()
            candidate_leans = pd.DataFrame()
            if 'decision_alignment_status' in df.columns:
                candidate_leans = df[
                    (df['decision_alignment_status'] == 'candidate_only') &
                    (~df['direction'].isin(['OVER', 'UNDER']))
                ].copy()

            # === MARKET OVERVIEW ===
            # Calculate heuristics for the "Market Overview"
            # Use map to ensure all values are converted, coercing errors to NaN
            conf_map = {'none': 0, 'low': 0.3, 'medium': 0.6, 'high': 0.9}
            avg_conf = df['confidence'].map(conf_map).mean()
            if pd.isna(avg_conf): avg_conf = 0
            
            action_rate = len(picks) / len(df) if len(df) > 0 else 0
            
            overview = "The market is efficient."
            if action_rate < 0.05:
                overview = "Pricing is highly compressed. The model is finding very few discrepancies, suggesting efficient lines or high uncertainty."
            elif avg_conf < 0.4:
                overview = "Model confidence is generally low, likely due to rotation volatility or insufficient sample size (early season/backfill)."
            elif action_rate > 0.2:
                overview = "Multiple discrepancies detected. The model sees significant value, possibly due to recent role changes not yet priced in."
                
            report_lines.append(f"### 🧠 Market Overview")
            report_lines.append(f"{overview}")
            report_lines.append("")

            if 'prediction_health_score' in df.columns:
                avg_health = df['prediction_health_score'].fillna(1.0).mean()
                fallback_rate = df['used_fallback_model'].fillna(0).mean() if 'used_fallback_model' in df.columns else 0
                anchor_rate = df['market_anchor_applied'].fillna(0).mean() if 'market_anchor_applied' in df.columns else 0
                report_lines.append("### Model Health")
                report_lines.append(f"- Average prediction health: {avg_health:.2f}")
                report_lines.append(f"- Fallback-model usage: {fallback_rate:.0%}")
                report_lines.append(f"- Market-anchor usage: {anchor_rate:.0%}")
                if 'core_predictor' in df.columns:
                    report_lines.append(f"- Core predictor mix: {df['core_predictor'].fillna('unknown').value_counts().to_dict()}")
                if 'mechanistic_reference_used' in df.columns:
                    mech_rate = df['mechanistic_reference_used'].fillna(0).mean()
                    report_lines.append(f"- Mechanistic reference coverage: {mech_rate:.0%}")
                if 'monte_carlo_reference_used' in df.columns:
                    mc_rate = df['monte_carlo_reference_used'].fillna(0).mean()
                    report_lines.append(f"- Monte Carlo reference coverage: {mc_rate:.0%}")
                if 'injury_context_present' in df.columns:
                    injury_rate = df['injury_context_present'].fillna(0).mean()
                    report_lines.append(f"- Injury-context coverage: {injury_rate:.0%}")
                low_health = df[df['prediction_health_score'].fillna(1.0) < 0.60]
                if not low_health.empty:
                    report_lines.append(f"- Low-health predictions on slate: {len(low_health)}")
                report_lines.append("")

            if any(col in df.columns for col in ['mechanistic_reference_used', 'ensemble_applied', 'model_disagreement']):
                report_lines.append("### Model Alignment")
                if 'mechanistic_reference_used' in df.columns:
                    mech_rate = df['mechanistic_reference_used'].fillna(0).mean()
                    report_lines.append(f"- Mechanistic reference attached: {mech_rate:.0%}")
                if 'residual_model_ready' in df.columns:
                    residual_rate = df['residual_model_ready'].fillna(0).mean()
                    report_lines.append(f"- Residual model ready: {residual_rate:.0%}")
                if 'ensemble_applied' in df.columns:
                    ensemble_rate = df['ensemble_applied'].fillna(0).mean()
                    report_lines.append(f"- Ensemble applied: {ensemble_rate:.0%}")
                if 'model_disagreement' in df.columns:
                    disagreement_series = pd.to_numeric(df['model_disagreement'], errors='coerce')
                    valid_disagreement = disagreement_series.dropna()
                    if not valid_disagreement.empty:
                        report_lines.append(f"- Avg core/reference disagreement: {valid_disagreement.mean():.2f}")
                        high_disagreement = (valid_disagreement >= 7.0).sum()
                        report_lines.append(f"- High-disagreement picks (>= 7.0): {high_disagreement}")
                if 'ensemble_notes' in df.columns:
                    notes = (
                        df['ensemble_notes']
                        .fillna('')
                        .astype(str)
                        .str.split(';')
                        .explode()
                        .str.strip()
                    )
                    notes = notes[notes != '']
                    if not notes.empty:
                        top_notes = notes.value_counts().head(3).to_dict()
                        report_lines.append(f"- Top ensemble notes: {top_notes}")
                report_lines.append("")

            if any(col in df.columns for col in ['player_consensus_status', 'market_consensus_status', 'final_status', 'rejection_stage']):
                report_lines.append("### Decision Flow")
                if 'player_consensus_status' in df.columns:
                    report_lines.append(f"- Player viability: {df['player_consensus_status'].fillna('UNKNOWN').value_counts().to_dict()}")
                if 'market_consensus_status' in df.columns:
                    report_lines.append(f"- Market trust: {df['market_consensus_status'].fillna('UNKNOWN').value_counts().to_dict()}")
                if 'player_consensus_trust_score' in df.columns:
                    player_trust = pd.to_numeric(df['player_consensus_trust_score'], errors='coerce').dropna()
                    if not player_trust.empty:
                        report_lines.append(f"- Avg player trust score: {player_trust.mean():.1f}")
                if 'market_consensus_trust_score' in df.columns:
                    market_trust = pd.to_numeric(df['market_consensus_trust_score'], errors='coerce').dropna()
                    if not market_trust.empty:
                        report_lines.append(f"- Avg market trust score: {market_trust.mean():.1f}")
                if 'final_status' in df.columns:
                    report_lines.append(f"- Final statuses: {df['final_status'].fillna('UNKNOWN').value_counts().to_dict()}")
                if 'rejection_stage' in df.columns:
                    rejection_series = df['rejection_stage'].fillna('').astype(str)
                    rejection_series = rejection_series[rejection_series != '']
                    if not rejection_series.empty:
                        report_lines.append(f"- Rejection stages: {rejection_series.value_counts().to_dict()}")
                report_lines.append("")

        # === SECTION 1: TOP PICKS (Key Insights) ===
            report_lines.append(f"### 🔥 Final Bets")
            report_lines.append("")
            
            if not picks.empty:
                # Add sort score: High=3, Med=2, Low=1, None=0
                conf_priority = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
                picks['conf_score'] = picks['confidence'].map(conf_priority)
                if all(col in picks.columns for col in ['edge_over', 'edge_under']):
                    picks['abs_edge'] = picks[['edge_over', 'edge_under']].max(axis=1)
                elif 'edge_score' in picks.columns:
                    picks['abs_edge'] = pd.to_numeric(picks['edge_score'], errors='coerce').fillna(0)
                else:
                    picks['abs_edge'] = 0
                
                # Sort by Confidence (Desc) then Edge (Desc)
                top_picks = picks.sort_values(['conf_score', 'abs_edge'], ascending=[False, False]).head(5)
                
                for _, row in top_picks.iterrows():
                    p_name = row.get('player_name')
                    matchup = f"{row.get('team')} vs {row.get('opponent')}"
                    direction = row.get('direction', '') or 'NO_BET'
                    line = row.get('line')
                    model = f"{row.get('predicted_mean', 0):.1f}"
                    conf = row.get('confidence').upper()
                    edge = f"{row.get('abs_edge', 0):.1%}"
                    source = row.get('edge_source', 'MODEL_DIV')
                    
                    market = row.get('market', 'PROP').upper()
                    final_reason = str(row.get('final_decision_reason', '') or '').strip()
                    selection_reason = str(row.get('selection_reason', '') or '').strip()
                    
                    # Edge data from flat columns (flattened by run_daily.py)
                    edge_explanation = row.get('edge_explanation', '')
                    edge_tier = row.get('edge_tier', '')
                    edge_score_val = row.get('edge_score', 0)
                    
                    # Tier badge
                    if edge_tier == 'parlay_core':
                        tier_badge = f"🔥 **PARLAY CORE** ({edge_score_val:.0f}/100)"
                    elif edge_tier == 'playable':
                        tier_badge = f"✅ **PLAYABLE** ({edge_score_val:.0f}/100)"
                    else:
                        tier_badge = f"Edge Score: {edge_score_val:.0f}"
                    
                    # Generate Insight: only use final gate language for an actual
                    # final-decision blocker/reason. Market-selection reasons are
                    # useful context, but they are not direction-aware enough to be
                    # labeled as the final decision for an approved bet.
                    if final_reason and final_reason.lower() not in ('nan', 'none'):
                        insight = f"Final decision: {final_reason}."
                    elif edge_explanation and edge_score_val > 0:
                        insight = edge_explanation
                    elif selection_reason and selection_reason.lower() not in ('nan', 'none'):
                        insight = f"Context: {selection_reason}."
                    else:
                        # Fallback to legacy insight
                        insight = f"Model sees {edge} edge."
                        
                        h2h_avg = row.get('h2h_avg')
                        if pd.notnull(h2h_avg) and h2h_avg > 0:
                            is_supporting = (direction == 'OVER' and h2h_avg > line) or \
                                            (direction == 'UNDER' and h2h_avg < line)
                            if is_supporting:
                                insight = f"**H2H**: Avg {h2h_avg:.1f} vs {row.get('opponent')} supports {direction}."
                        
                        if source in ('USAGE_REDISTRIBUTION', 'USAGE_REDISTRIBUTION_EDGE'):
                            insight = f"**Usage Alert**: Key teammates missing; {p_name} taking more volume."
                        elif source == 'LINEUP_SHIFT_EDGE':
                            insight = f"**Lineup Shift**: Rotation changes materially altered expected role or minutes."
                        elif source == 'ROLE_CHANGE_EDGE':
                            insight = f"**Role Change**: Current role looks different from the baseline the line implies."
                        elif source == 'VOLATILITY_UNDER_EDGE':
                            insight = f"**Volatility Edge**: High-variance profile favors the under distribution."
                        elif source == 'MARKET_ANCHOR_DISAGREEMENT':
                            insight = f"**Market Tension**: Model and sharp movement are disagreeing on this number."
                        elif source in ('UNATTRIBUTED_MODEL_EDGE', 'MARKET_INEFFICIENCY'):
                            insight = f"**Unattributed Edge**: Model diverges from the line, but no single structural driver dominates."
                    
                    report_lines.append(f"- **{p_name} ({market} {direction} {line})** | {matchup}")
                    report_lines.append(f"  - Model: **{model}** | Line: **{line}** | **Confidence**: {conf}")
                    report_lines.append(f"  - {tier_badge}")
                    if row.get('player_consensus_level') or row.get('market_consensus_level'):
                        report_lines.append(
                            f"  - Trust Layer: player={row.get('player_consensus_level', 'n/a')} | "
                            f"market={row.get('market_consensus_level', 'n/a')}"
                        )
                    report_lines.append(f"  - 💡 *{insight}*")
                    report_lines.append("")
            else:
                report_lines.append("_No final approved bets for this slate under the current calibration regime._")
                report_lines.append("")

            if not candidate_leans.empty:
                report_lines.append("### 🧭 Strong Leans")
                report_lines.append("_These cleared the edge layer but were blocked by final calibration._")
                report_lines.append("")
                if all(col in candidate_leans.columns for col in ['edge_over', 'edge_under']):
                    candidate_leans['abs_edge'] = candidate_leans[['edge_over', 'edge_under']].max(axis=1)
                elif 'edge_score' in candidate_leans.columns:
                    candidate_leans['abs_edge'] = pd.to_numeric(candidate_leans['edge_score'], errors='coerce').fillna(0)
                else:
                    candidate_leans['abs_edge'] = 0
                top_leans = candidate_leans.sort_values(['abs_edge', 'edge_score'], ascending=[False, False]).head(8)
                for _, row in top_leans.iterrows():
                    p_name = row.get('player_name')
                    matchup = f"{row.get('team')} vs {row.get('opponent')}"
                    lean_direction = row.get('edge_candidate_direction', row.get('edge_direction', 'NO_BET'))
                    line = row.get('line')
                    model = f"{row.get('predicted_mean', 0):.1f}"
                    tier = str(row.get('edge_candidate_tier', 'lean')).upper()
                    market = str(row.get('market', 'PROP')).upper()
                    insight = row.get('final_decision_reason', '') or row.get('edge_explanation', '') or 'Candidate edge blocked by final calibration.'
                    report_lines.append(f"- **{p_name} ({market} {lean_direction} {line})** | {matchup}")
                    report_lines.append(f"  - Candidate tier: **{tier}** | Model: **{model}** | Line: **{line}**")
                    report_lines.append(f"  - Blocked at: **{str(row.get('rejection_stage', '') or 'calibrator').upper()}**")
                    report_lines.append(f"  - 💡 *{insight}*")
                    report_lines.append("")

            # === PARLAY TICKETS ===
            if not picks.empty:
                try:
                    parlay_picks = []
                    for _, row in picks.iterrows():
                        # Use flat columns (edge_score, edge_tier, edge_direction)
                        # These are flattened from edge_analysis in run_daily.py
                        p_direction = row.get('direction', '') or row.get('edge_direction', '')
                        
                        parlay_picks.append({
                            'player_name': row.get('player_name', 'Unknown'),
                            'team': row.get('team', ''),
                            'opponent': row.get('opponent', ''),
                            'market': row.get('market', 'PTS'),
                            'line': row.get('line', 0),
                            'direction': p_direction,
                            'edge_score': row.get('edge_score', 0),
                            'edge_tier': row.get('edge_tier', 'reject'),
                            'edge_explanation': row.get('edge_explanation', ''),
                            'kill_count': 0,
                        })
                    
                    builder = ParlayBuilder()
                    tickets = builder.build_tickets(parlay_picks)
                    parlay_md = builder.format_tickets_markdown(tickets)
                    report_lines.append(parlay_md)
                except Exception as e:
                    logger.warning(f"Parlay generation failed: {e}")
                    report_lines.append("\n### 🎫 Parlay Tickets")
                    report_lines.append(f"_Parlay generation error: {e}_")
                report_lines.append("")

            # === SAME-GAME PARLAYS ===
            if not picks.empty:
                try:
                    # Group picks by game for SGP
                    picks_by_game = {}
                    for _, row in picks.iterrows():
                        game_key = f"{row.get('team', '')} vs {row.get('opponent', '')}"
                        if game_key not in picks_by_game:
                            picks_by_game[game_key] = []
                        picks_by_game[game_key].append({
                            'player_name': row.get('player_name', ''),
                            'team': row.get('team', ''),
                            'market': row.get('market', 'PTS'),
                            'line': row.get('line', 0),
                            'direction': row.get('direction', ''),
                            'edge_score': row.get('edge_score', 0),
                            'edge_tier': row.get('edge_tier', ''),
                        })
                    
                    sgp_engine = SGPEngine()
                    sgps = sgp_engine.build_sgps(picks_by_game)
                    if sgps:
                        sgp_md = sgp_engine.format_sgps_markdown(sgps)
                        report_lines.append(sgp_md)
                        report_lines.append("")
                except Exception as e:
                    logger.warning(f"SGP generation failed: {e}")

            # === EDGE PERFORMANCE TRACKER ===
            try:
                edge_tracker = EdgeTracker(db)
                perf_report = edge_tracker.get_performance_report(lookback_days=30)
                if perf_report and perf_report.get('total_picks', 0) > 0:
                    edge_md = edge_tracker.format_report_markdown(perf_report)
                    report_lines.append(edge_md)
                    report_lines.append("")
            except Exception as e:
                logger.warning(f"Edge performance report failed: {e}")

            # === DIAGNOSTICS ===
            report_lines.append("### ✅ Diagnostics")
            report_lines.append(f"- Games represented: {df.apply(lambda r: ' vs '.join(sorted([str(r.get('team')), str(r.get('opponent'))])), axis=1).nunique()}")
            lined_players = df[df['line'].notnull()] if 'line' in df.columns else pd.DataFrame()
            report_lines.append(f"- Players with lines: {len(lined_players)}")
            if not lined_players.empty and 'bookmaker' in lined_players.columns:
                report_lines.append(f"- Line bookmakers used: {lined_players['bookmaker'].fillna('unknown').value_counts().to_dict()}")
            report_lines.append("")

            # === SECTION 2: GAME-BY-GAME BREAKDOWN ===
            report_lines.append(f"### 📅 Game-by-Game Breakdown")
            
            # Group by Matchup
            # We need a unique matchup key. Usually Team vs Opponent.
            # But we have two rows per game (Team A vs Team B, Team B vs Team A).
            # We want to group them together: "LAL vs DAL"
            
            def get_game_key(row):
                teams = sorted([str(row.get('team')), str(row.get('opponent'))])
                return f"{teams[0]} vs {teams[1]}"
            
            df['game_key'] = df.apply(get_game_key, axis=1)
            games = df['game_key'].unique()
                
            for game in sorted(games):
                game_rows = df[df['game_key'] == game].copy()

                # === GROUP BY PLAYER ===
                player_group_col = 'player_id' if 'player_id' in game_rows.columns else 'player_name'
                players_in_game = game_rows[player_group_col].unique()
                
                output_rows = []
                for pid in players_in_game:
                    player_markets = game_rows[game_rows[player_group_col] == pid]
                    if player_markets.empty:
                        continue
                    
                    # Base info
                    base = player_markets.iloc[0]
                    p_name = str(base.get('player_name', 'Unknown'))[:16]
                    team = base.get('team', 'UNK')
                    opp = base.get('opponent', 'UNK')
                    p_play = base.get('p_play', 1.0)
                    book = str(base.get('bookmaker', ''))[:8]
                    
                    # Iterate through all markets for this player
                    for _, row in player_markets.iterrows():
                        market_key = row.get('market', 'UNK').upper()
                        
                        # Skip if market is unknown or weird
                        if not market_key: continue
                        
                        pred = row.get('predicted_mean', 0)
                        line = row.get('line', None)
                        
                        # Use flat columns — edge_direction from EdgeScorer, plus edge_score/explanation
                        direction = row.get('direction', 'NO_BET')
                        edge_lean = row.get('edge_direction', '') or row.get('direction', 'NO_BET')
                        edge_score_val = row.get('edge_score', 0)
                        edge_expl = str(row.get('edge_explanation', '') or '')
                        
                        # Calculate edge
                        edge_over = row.get('edge_over', 0)
                        edge_under = row.get('edge_under', 0)
                        edge = max(edge_over, edge_under) if pd.notnull(edge_over) and pd.notnull(edge_under) else 0
                        
                        conf = row.get('confidence', 'low')
                        
                        # Build concise reason — prefer edge explanation, and always show score
                        reason = ''
                        sel_reason = str(row.get('selection_reason', '') or '')
                        e_source = str(row.get('edge_source', '') or '')
                        
                        final_reason = str(row.get('final_decision_reason', '') or '')
                        if final_reason and final_reason not in ('nan', 'None'):
                            reason = final_reason.split(';')[0][:40]
                        elif direction in ('OVER', 'UNDER') and sel_reason and sel_reason != 'nan':
                            reason = sel_reason.split(';')[0][:40]
                        elif edge_expl and str(edge_expl) != 'nan':
                            reason = str(edge_expl).split('.')[0][:40]
                        elif sel_reason and sel_reason != 'nan':
                            reason = sel_reason.split(';')[0][:40]
                        elif e_source and e_source not in ('nan', 'None', ''):
                            reason = e_source.replace('_', ' ').title()[:30]
                        elif edge_lean != 'NO_BET' and edge > 0.05:
                            reason = 'Model divergence'
                        else:
                            reason = f'No edge identified (score: {edge_score_val:.0f}/100)'

                        if direction == 'NO_BET' and edge_lean and edge_lean != 'NO_BET':
                            reason = f"Lean {edge_lean}: {reason}"[:40]
                        
                        if line is not None:
                            if direction in ('OVER', 'UNDER'):
                                dir_label = str(direction).upper()
                            elif edge_lean and edge_lean != 'NO_BET':
                                dir_label = f"LEAN {str(edge_lean).upper()}"
                            else:
                                dir_label = 'NO_BET'

                            conf_value = str(conf).upper() if conf else 'LOW'
                            conf_str = 'NONE' if conf_value in ('NONE', 'NO_BET') else conf_value
                            
                            output_rows.append({
                                'player_name': p_name,
                                'team': team,
                                'opp': opp,
                                'p_play': p_play,
                                'market': market_key,
                                'pred': pred,
                                'line': line,
                                'dir': dir_label,
                                'edge': edge,
                                'conf': conf_str,
                                'book': book,
                                'reason': reason
                            })
                
                # Sort by edge descending
                output_rows.sort(key=lambda x: x['edge'], reverse=True)
                
                report_lines.append(f"#### 🏀 {game}")
                report_lines.append("| Player | Team | Mkt | Pred | Line | Dir | Edge | Conf | Reason |")
                report_lines.append("|--------|------|-----|------|------|-----|------|------|--------|")
                
                for row in output_rows:
                    edge_str = f"{row['edge']:.0%}" if row['edge'] > 0 else '-'
                    pred_str = f"{row['pred']:.1f}"
                    line_str = f"{row['line']:.1f}" if row['line'] else '-'
                    reason_str = row.get('reason', '')[:40]
                    report_lines.append(f"| {row['player_name']} | {row['team']} | {row['market']} | {pred_str} | {line_str} | {row['dir']} | {edge_str} | {row['conf']} | {reason_str} |")

                report_lines.append("")
            
            report_lines.append("")

            # === SECTION 2: WATCHLIST (Top 20 Near Misses) ===
            if not no_plays.empty:
                # Calculate "Near Miss" score = absolute edge (even if not confident)
                # Ensure we handle missing columns gracefully
                def get_max_edge(x):
                    e_over = x.get('edge_over')
                    e_under = x.get('edge_under')
                    if pd.notnull(e_over) and pd.notnull(e_under):
                        return max(e_over, e_under)
                    e_score = x.get('edge_score')
                    if pd.notnull(e_score):
                        return e_score
                    return 0
                
                no_plays['abs_edge'] = no_plays.apply(get_max_edge, axis=1)
                
                # Filter trivial edges (< 2%) to reduce noise
                watchlist = no_plays[no_plays['abs_edge'] > 0.02].sort_values('abs_edge', ascending=False).head(20)
                
                report_lines.append(f"### 🟡 Watchlist ({len(watchlist)})")
                report_lines.append("_Top 20 dismissed opportunities with >2% edge_")
                report_lines.append("")
                report_lines.append("| Player | Matchup | Model | Line | Edge | Reason |")
                report_lines.append("|--------|---------|-------|------|------|--------|")
                
                for _, row in watchlist.iterrows():
                    p_name = row.get('player_name')[:18]
                    team = row.get('team', 'UNK')
                    opp = row.get('opponent', 'UNK')
                    model = f"{row.get('predicted_mean', 0):.1f}"
                    line = row.get('line', 'N/A')
                    edge_val = row.get('abs_edge', 0)
                    edge_str = f"{edge_val:.1%}"
                    
                    # Semantic Reason
                    source = str(row.get('edge_source', 'nan'))
                    regime = str(row.get('regime_status', 'stable'))
                    
                    reason = "Unattributed Edge"
                    if source == 'nan' or source == 'None':
                        reason = "No Signal"
                    elif regime != 'stable':
                        reason = f"Regime: {regime}"
                    elif row.get('confidence') == 'none':
                        reason = "Low Confidence"
                    
                    report_lines.append(f"| {p_name} | {team} vs {opp} | {model} | {line} | {edge_str} | {reason} |")
                
                report_lines.append("")

        # API Stats
        report_lines.append("### API Usage")
        report_lines.append("| API | Calls | Cached |")
        report_lines.append("|-----|-------|--------|")
        for api, stats in api_stats.items():
            report_lines.append(f"| {api} | {stats['calls']} | {stats['cached']} |")
        
        report_lines.append("")
        report_lines.append(f"*Generated at {datetime.now().strftime('%H:%M:%S')}*")
        
        # Save Report
        report_filename = f"report_{target_date}_v2.md"
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            logger.info(f"  [OK] Report generated: {report_filename}")
            return True
        except Exception as e:
            logger.error(f"  [ERROR] Failed to write report: {e}")
            return False

    except Exception as e:
        logger.error(f"[CRITICAL ERROR] Report generation failed: {e}")
        # traceback.print_exc() # Logger handles trace usually, or we can explicit
        logger.error(traceback.format_exc())
        return False
