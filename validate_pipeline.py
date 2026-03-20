
import sys
import os
import argparse
from datetime import datetime, timedelta

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from src.utils.database import DatabaseManager
from src.agents.learning_loop import LearningLoopAgent

def run_validation(days_back=30):
    print("="*60)
    print("  PIPELINE VALIDATION REPOT")
    print("="*60)
    
    db = DatabaseManager()
    agent = LearningLoopAgent(db)
    
    # 1. Update with actuals (in case any are pending)
    # Iterate through last few days to catch up
    print("\n[1] Updating pending actuals...")
    total_updated = 0
    for i in range(days_back):
        d = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        total_updated += agent.update_with_actuals(d)
    print(f"    Updated {total_updated} records.")

    # 2. Profitability
    print("\n[2] Profitability Analysis (ROI)")
    prof = agent.analyze_profitability(lookback_days=days_back)
    print(f"    Period:       {prof.get('period')}")
    print(f"    Total Bets:   {prof.get('total_bets')}")
    print(f"    Win Rate:     {prof.get('win_rate', 0):.1f}%")
    print(f"    ROI:          {prof.get('roi', 0):.1f}%")
    print(f"    Units Profit: {prof.get('units_profit', 0):.2f}u")
    
    # 3. Calibration
    print("\n[3] Calibration Check")
    cal = agent.get_calibration_stats(lookback_days=days_back)
    print(f"    Sample Size:  {cal.get('sample_size')}")
    print(f"    1-Sigma:      {cal.get('calibration_1std')*100:.1f}% (Target 68%)")
    print(f"    2-Sigma:      {cal.get('calibration_2std')*100:.1f}% (Target 95%)")
    
    cr = agent.generate_calibration_report(lookback_days=days_back)
    if cr.get('buckets'):
        print("\n    Reliability Buckets:")
        print(f"    {'Conf':<10} {'Bets':<6} {'Win%':<6}")
        print(f"    {'-'*24}")
        for b in cr['buckets']:
            print(f"    {b['bucket']:<10} {b['total']:<6} {b['win_rate']:.1f}%")
    else:
        print("\n    (No confidence buckets available yet)")
        
    # 4. Edge Performance
    print("\n[4] Edge Source Performance")
    edges = agent.analyze_edge_performance(lookback_days=days_back)
    if edges:
        print(f"    {'Source':<20} {'Bets':<5} {'Win%':<6} {'ROI':<6}")
        print(f"    {'-'*40}")
        for e in edges[:10]: # Top 10
            print(f"    {e['source']:<20} {e['bets']:<5} {e['win_rate']:5.1f}% {e['roi']:5.1f}%")
    else:
        print("    (No edge data available)")

    # 5. Error Decomposition
    print("\n[5] Error Decomposition")
    decomp = agent.analyze_error_decomposition(lookback_days=days_back)
    if decomp:
        print(f"    Sample Size:      {decomp.get('sample_size')}")
        print(f"    Mean Abs Error:   {decomp.get('mean_abs_error', 0):.1f} pts")
        print(f"    Minutes Share:    {decomp.get('minutes_error_share', 0):.1f}%")
        print(f"    Efficiency Share: {decomp.get('efficiency_error_share', 0):.1f}%")
    else:
        print("    (No decomposition data available)")

    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='Lookback days')
    args = parser.parse_args()
    
    run_validation(args.days)
