
import pandas as pd
import sqlite3
from src.utils.database import DatabaseManager

def main():
    # 1. Load Predictions CSV
    csv_path = 'predictions_2026-01-24.csv'
    try:
        df_pred = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # Normalize columns
    df_pred.columns = [c.lower() for c in df_pred.columns]
    
    # Filter for High Confidence and Valid Bet
    # Check distinct values for confidence to be safe
    # print(df_pred['confidence'].unique()) 
    
    high_conf_df = df_pred[
        (df_pred['confidence'].str.lower() == 'high') & 
        (df_pred['direction'].isin(['OVER', 'UNDER']))
    ].copy()
    
    if high_conf_df.empty:
        print("No High Confidence predictions found in CSV.")
        return

    # 2. Connect to DB to get Actuals
    db = DatabaseManager()
    
    results = []
    wins = 0
    losses = 0
    pushes = 0
    pending = 0
    
    with db.get_connection() as conn:
        for _, row in high_conf_df.iterrows():
            player_name = row['player_name']
            line = float(row['line'])
            direction = row['direction']
            
            # Fetch actuals from predictions_archive
            # We use the archive because we just populated it with the correct data
            query = """
                SELECT actual_points 
                FROM predictions_archive 
                WHERE player_name = ? AND prediction_date = '2026-01-24'
            """
            cursor = conn.execute(query, (player_name,))
            row_db = cursor.fetchone()
            
            actual = None
            status = 'PENDING'
            
            if row_db and row_db['actual_points'] is not None:
                actual = float(row_db['actual_points'])
                
                if direction == 'OVER':
                    if actual > line:
                        status = 'WIN'
                        wins += 1
                    elif actual < line:
                        status = 'LOSS'
                        losses += 1
                    else:
                        status = 'PUSH'
                        pushes += 1
                elif direction == 'UNDER':
                    if actual < line:
                        status = 'WIN'
                        wins += 1
                    elif actual > line:
                        status = 'LOSS'
                        losses += 1
                    else:
                        status = 'PUSH'
                        pushes += 1
            else:
                # Try fallback to direct player_logs if archive missed it (shouldn't happen now but safety net)
                # This handles the case where name matching might have been tricky in archive process
                # but we can try fuzzy match or just report pending
                pending += 1
            
            results.append({
                'Player': player_name,
                'Team': row['team'],
                'Line': line,
                'Direction': direction,
                'Actual': actual,
                'Result': status,
                'Predicted': row['predicted_mean'],
                'Edge': row['edge_over'] if direction == 'OVER' else row['edge_under']
            })

    # 3. Generate Report
    total_finished = wins + losses + pushes
    win_rate = (wins / total_finished * 100) if total_finished > 0 else 0
    
    print(f"# High Confidence Prediction Audit (2026-01-24)")
    print(f"\n## Summary")
    print(f"- **Total High Confidence Bets**: {len(high_conf_df)}")
    print(f"- **Wins**: {wins}")
    print(f"- **Losses**: {losses}")
    print(f"- **Pushes**: {pushes}")
    print(f"- **Pending/Unknown**: {pending}")
    print(f"- **Win Rate**: {win_rate:.1f}%")
    
    print(f"\n## Detailed Results")
    print(f"| Player | Team | Bet | Line | Predicted | Actual | Result |")
    print(f"|--------|------|-----|------|-----------|--------|--------|")
    
    for r in results:
        actual_str = f"{r['Actual']:.0f}" if r['Actual'] is not None else "N/A"
        result_icon = "✅" if r['Result'] == 'WIN' else "❌" if r['Result'] == 'LOSS' else "➖" if r['Result'] == 'PUSH' else "❓"
        print(f"| {r['Player']} | {r['Team']} | {r['Direction']} | {r['Line']} | {r['Predicted']:.1f} | {actual_str} | {result_icon} {r['Result']} |")

if __name__ == "__main__":
    main()
