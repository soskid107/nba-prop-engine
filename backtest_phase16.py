"""
Phase 16 Backtest: Replay past game dates from DB
===================================================
Fetches historical data already in the database, re-runs predictions
with the R1-R9 enhanced pipeline, and compares to actual results.
"""

import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.database import DatabaseManager
from src.simulation.monte_carlo import SimulationEngine

def get_backtest_dates(db, n_dates=5):
    """Get recent game dates that have actual results in the DB."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT game_date 
            FROM player_logs 
            WHERE points IS NOT NULL AND minutes > 5
            ORDER BY game_date DESC 
            LIMIT ?
        """, (n_dates,))
        return [r['game_date'] for r in cursor.fetchall()]

def get_players_for_date(db, game_date):
    """Get players who played on a given date with their actual stats."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pl.player_id, p.full_name as player_name, pl.team_abbreviation,
                   pl.points, pl.minutes, pl.fga, pl.fta,
                   pl.opponent_abbreviation
            FROM player_logs pl
            LEFT JOIN players p ON pl.player_id = p.player_id
            WHERE pl.game_date = ? AND pl.minutes > 15
            ORDER BY pl.points DESC
        """, (game_date,))
        return [dict(r) for r in cursor.fetchall()]

def get_market_line(db, player_id, game_date):
    """Try to get historical market line for a player on a date."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            # Try player_props table first
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='player_props'
            """)
            if cursor.fetchone():
                cursor.execute("""
                    SELECT line FROM player_props
                    WHERE player_id = ? AND game_date = ? AND market = 'points'
                    LIMIT 1
                """, (player_id, game_date))
                row = cursor.fetchone()
                if row:
                    return row['line']
            
            # Try predictions table as fallback
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'
            """)
            if cursor.fetchone():
                cursor.execute("""
                    SELECT market_line FROM predictions
                    WHERE player_id = ? AND game_date = ? AND market = 'points'
                    LIMIT 1
                """, (player_id, game_date))
                row = cursor.fetchone()
                if row:
                    return row['market_line']
    except Exception:
        pass
    return None

def run_backtest():
    print("=" * 70)
    print("  Phase 16 Backtest: R1-R9 Enhanced Pipeline vs Actual Results")
    print("=" * 70)
    
    db = DatabaseManager()
    
    # 1. Get available dates
    dates = get_backtest_dates(db, n_dates=5)
    if not dates:
        print("\n❌ No game dates with actual results found in DB.")
        return
    
    print(f"\nFound {len(dates)} game dates to backtest:")
    for d in dates:
        print(f"  📅 {d}")
    
    # Initialize simulation engine
    print("\n[INIT] Loading simulation engine...")
    engine = SimulationEngine(db=db)
    engine.load_models()
    
    # PyMC now uses 200 draws / 1 chain (~4s per sparse-data player)
    # Only fires for players with data_weight < 0.7 (typically <10 games)
    
    # Track overall accuracy
    all_errors = []   # absolute errors
    all_results = []  # detailed results
    hit_count = 0     # within ±5 pts
    tight_hits = 0    # within ±3 pts
    total = 0
    
    for game_date in dates:
        print(f"\n{'─' * 60}")
        print(f"  📅 Backtesting: {game_date}")
        print(f"{'─' * 60}")
        
        players = get_players_for_date(db, game_date)
        if not players:
            print("  No players found for this date.")
            continue
        
        # Limit to top 20 scorers per date for speed
        players = players[:20]
        
        date_errors = []
        
        for p in players:
            try:
                market_line = get_market_line(db, p['player_id'], game_date)
                opponent = p.get('opponent_abbreviation', 'UNK')
                
                if not opponent or opponent == 'UNK':
                    continue
                
                # Run enhanced simulation
                result = engine.simulate_player_points(
                    player_id=p['player_id'],
                    team_abbr=p['team_abbreviation'],
                    opponent_abbr=opponent,
                    game_date=game_date,
                    p_play=1.0,
                    team_injuries={},
                    market_line=market_line
                )
                
                if not result or 'predicted_mean' not in result:
                    continue
                
                predicted = result['predicted_mean']
                actual = p['points']
                error = abs(predicted - actual)
                
                all_errors.append(error)
                date_errors.append(error)
                total += 1
                
                if error <= 5:
                    hit_count += 1
                if error <= 3:
                    tight_hits += 1
                
                # Direction check (if we have market line)
                direction_correct = None
                if market_line:
                    pred_over = predicted > market_line
                    actual_over = actual > market_line
                    direction_correct = pred_over == actual_over
                
                marker = "✅" if error <= 5 else "⚠️" if error <= 8 else "❌"
                dir_mark = ""
                if direction_correct is not None:
                    dir_mark = " 🎯" if direction_correct else " ✗"
                
                all_results.append({
                    'date': game_date,
                    'player': p['player_name'],
                    'predicted': predicted,
                    'actual': actual,
                    'error': error,
                    'market_line': market_line,
                    'direction_correct': direction_correct,
                })
                
                line_str = f"(line: {market_line:.1f}{dir_mark})" if market_line else ""
                print(f"  {marker} {p['player_name']:<22s} | Pred: {predicted:5.1f} | Actual: {actual:5.1f} | Err: {error:4.1f} {line_str}")
                
            except Exception as e:
                print(f"  ⚠️  {p['player_name']}: Error - {e}")
        
        if date_errors:
            print(f"\n  Date MAE: {np.mean(date_errors):.1f} pts | "
                  f"Median Err: {np.median(date_errors):.1f} | "
                  f"Hit Rate (±5): {sum(1 for e in date_errors if e<=5)/len(date_errors)*100:.0f}%")
    
    # ─── Overall Summary ───
    print("\n" + "=" * 70)
    print("  BACKTEST SUMMARY")
    print("=" * 70)
    
    if total > 0:
        mae = np.mean(all_errors)
        median_err = np.median(all_errors)
        hit_rate = hit_count / total * 100
        tight_rate = tight_hits / total * 100
        
        # Direction accuracy (where we had market lines)
        dir_results = [r for r in all_results if r['direction_correct'] is not None]
        dir_acc = sum(1 for r in dir_results if r['direction_correct']) / len(dir_results) * 100 if dir_results else 0
        
        print(f"""
  Players backtested:  {total}
  Game dates:          {len(dates)}
  
  Mean Abs Error:      {mae:.1f} pts
  Median Error:        {median_err:.1f} pts
  
  Hit Rate (±5 pts):   {hit_rate:.0f}% ({hit_count}/{total})
  Tight Hits (±3 pts): {tight_rate:.0f}% ({tight_hits}/{total})
  
  Direction Accuracy:  {dir_acc:.0f}% ({sum(1 for r in dir_results if r['direction_correct'])}/{len(dir_results)} with market lines)
""")
        
        # Top 5 best predictions
        sorted_results = sorted(all_results, key=lambda x: x['error'])
        print("  🏆 Best Predictions:")
        for r in sorted_results[:5]:
            print(f"     {r['player']:<22s} | Pred: {r['predicted']:5.1f} | Actual: {r['actual']:5.1f} | Err: {r['error']:.1f}")
        
        # Bottom 5 worst
        print("\n  💀 Worst Predictions:")
        for r in sorted_results[-5:]:
            print(f"     {r['player']:<22s} | Pred: {r['predicted']:5.1f} | Actual: {r['actual']:5.1f} | Err: {r['error']:.1f}")
    else:
        print("\n  No predictions were generated. Check data availability.")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    run_backtest()
