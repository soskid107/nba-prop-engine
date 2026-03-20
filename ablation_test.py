
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from src.utils.database import DatabaseManager
from src.simulation.monte_carlo import SimulationEngine

def run_ablation_tests():
    print("="*60)
    print("  ROBUSTNESS CHECK: FEATURE ABLATION")
    print("="*60)
    
    db = DatabaseManager()
    
    # Define scenarios
    scenarios = {
        'Baseline': {},
        'No Bayes Minutes': {'disable_bayes_minutes': True},
        'No Market Adj': {'disable_market_adjustment': True},
        'No Teammate Impact': {'disable_teammate_impact': True},
        'No Matchup Model': {'disable_matchup_model': True}
    }
    
    # Reuse logical from backtest - Get recent dates
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT game_date FROM player_logs WHERE points IS NOT NULL AND minutes > 5 ORDER BY game_date DESC LIMIT 5")
        dates = [r['game_date'] for r in cursor.fetchall()]
        
    print(f"Test Dates: {dates}")
    
    results = []
    
    for name, flags in scenarios.items():
        print(f"\nRunning Scenario: {name} (Flags: {flags})")
        
        # Init engine with flags
        engine = SimulationEngine(db=db, ablation_flags=flags)
        engine.load_models()
        
        errors = []
        hits_5 = 0
        total = 0
        
        for date in dates:
            # Get players with actuals
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pl.player_id, p.full_name, pl.points, pl.minutes, pl.team_abbreviation, pl.opponent_abbreviation
                    FROM player_logs pl
                    JOIN players p ON pl.player_id = p.player_id
                    WHERE pl.game_date = ? AND pl.minutes > 15
                    LIMIT 20
                """, (date,))
                players = [dict(r) for r in cursor.fetchall()]
            
            for p in players:
                # Need market line for baseline comparison? 
                # For ablation, we simulate providing a line to see if system uses it or not
                # But to see pure model performance we might want to pass None?
                # Actually, 'No Market Adj' flag handles ignoring it even if passed.
                # Let's pass a dummy line to ensure code triggers if enabled.
                
                # Fetch real line if possible for fair test
                # (Simple lookup or pass None)
                
                try:
                    res = engine.simulate_player_points(
                        player_id=p['player_id'],
                        team_abbr=p['team_abbreviation'],
                        opponent_abbr=p['opponent_abbreviation'],
                        game_date=date,
                        market_line=20.5 # Dummy line to trigger market logic if active
                    )
                    
                    pred = res['predicted_mean']
                    actual = p['points']
                    err = abs(pred - actual)
                    errors.append(err)
                    if err <= 5: hits_5 += 1
                    total += 1
                    
                except Exception:
                    pass
        
        if total > 0:
            mae = np.mean(errors)
            hit_rate = (hits_5 / total) * 100
            print(f"  -> MAE: {mae:.2f} | Hit Rate: {hit_rate:.1f}%")
            results.append({
                'Scenario': name,
                'MAE': mae,
                'HitRate': hit_rate
            })
    
    # Summary
    print("\n" + "="*60)
    print("  ABLATION RESULTS SUMMARY")
    print("="*60)
    df = pd.DataFrame(results).sort_values('MAE')
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_ablation_tests()
