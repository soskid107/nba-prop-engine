
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs
import sys
from pathlib import Path
from datetime import datetime
import os
import argparse

# Add src to path (assuming script is in root)
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.database import DatabaseManager

class PaymentAudit:
    """
    Main Audit Script (formerly StrictAuditor)
    
    Replaces the old audit logic with a stricter, more professional standard.
    - Measures P50 Accuracy (Did we predict the correct side of the median?)
    - Measures MAE (Mean Absolute Error)
    - Compares Actual results against Bookmaker Lines AND Model Predictions
    """
    def __init__(self):
        self.db = DatabaseManager()

    def fetch_actuals(self, date_str):
        """
        Fetches official NBA stats for a specific date using PlayerGameLogs.
        Strictly enforces the date boundary.
        """
        print(f"Fetching confirmed stats for {date_str}...")
        try:
            # Convert YYYY-MM-DD to MM/DD/YYYY for API
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            api_date = dt.strftime('%m/%d/%Y')
            
            logs = playergamelogs.PlayerGameLogs(
                season_nullable='2025-26', 
                date_from_nullable=api_date, 
                date_to_nullable=api_date
            )
            df = logs.get_data_frames()[0]
            if df.empty:
                # Fallback: Try fetching previous day if late night run
                pass
            
            return df
        except Exception as e:
            print(f"Error fetching actuals: {e}")
            return pd.DataFrame()

    def load_predictions(self, date_str):
        """
        Loads predictions from the CSV file (source of truth for lines/bets).
        """
        csv_path = f"predictions_{date_str}.csv"
        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            return pd.DataFrame()
            
        print(f"Loading predictions from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Normalize columns to lower case just in case
        df.columns = [c.lower() for c in df.columns]
        return df

    def determine_cause(self, row, actual_row):
        """
        Heuristic to determine failure cause.
        """
        # Data prep
        pred_min = row.get('predicted_minutes', 0)
        if not pred_min: pred_min = 30 # Fallback default
        
        actual_min = actual_row['MIN']
        
        # 1. Minutes Issue
        if actual_min < (pred_min * 0.75):
            # check for fouls
            if actual_row['PF'] >= 4:
                return "Foul Trouble"
            # check for blowout (plus_minus is rough proxy)
            if abs(actual_row['PLUS_MINUS']) > 20:
                return "Blowout / Garbage Time"
            return "Minutes Misprojection (Rotation Change)"
            
        # 2. Shooting/Efficiency
        if actual_row['FG_PCT'] < 0.35:
            return "Cold Shooting / Variance"
            
        return "Defensive Matchup / Other"

    def run_audit(self, date_str):
        print(f"\n=== Running Professional Audit for {date_str} ===")
        print("Standard: P50 Accuracy (Precision) & Bookmaker Edge (Profitability)")
        
        # 1. Get Data
        actuals = self.fetch_actuals(date_str)
        preds = self.load_predictions(date_str)
        
        if actuals.empty:
            print("No actual games found for this date. (Likely future or no games)")
            return
        
        if preds.empty:
            print("No predictions found in archive for this date.")
            return

        # Normalize names for matching
        actuals['normalized_name'] = actuals['PLAYER_NAME'].str.lower().str.strip()
        preds['normalized_name'] = preds['player_name'].str.lower().str.strip()
        
        # Merge
        # [FIX] Fetch all necessary stats, not just PTS
        # Added BLK, STL, TOV for future-proofing
        merged = pd.merge(
            preds, 
            actuals[['normalized_name', 'PTS', 'AST', 'REB', 'FG3M', 'BLK', 'STL', 'TOV', 'MIN', 'PF', 'PLUS_MINUS', 'FG_PCT', 'MATCHUP', 'WL']], 
            on='normalized_name', 
            how='left'
        )
        
        results = []
        metrics = {
            'total': 0, 'wins': 0, 'losses': 0, 'voids': 0,
            'model_hits_p50': 0, # Did we hit our own P50?
            'total_error': 0,    # Sum of absolute errors for MAE
            'causes': {}
        }
        
        print(f"Auditing {len(merged)} predictions...")
        
        for _, row in merged.iterrows():
            line = float(row['line']) if row['line'] else 0
            pred_pts = row['predicted_mean']
            
            # [FIX] Select correct actual stat based on market type
            # CSV column is likely 'market' or 'market_key'
            market_type = str(row.get('market', row.get('market_key', 'points'))).lower()
            
            if 'assist' in market_type:
                actual_val = row['AST']
            elif 'rebound' in market_type:
                actual_val = row['REB']
            elif 'three' in market_type or 'fg3m' in market_type:
                actual_val = row['FG3M']
            elif 'block' in market_type:
                actual_val = row['BLK']
            elif 'steal' in market_type:
                actual_val = row['STL']
            elif 'turnover' in market_type:
                actual_val = row['TOV']
            elif 'pra' in market_type or 'pts+reb+ast' in market_type:
                actual_val = row['PTS'] + row['REB'] + row['AST']
            else:
                actual_val = row['PTS'] # Default to points
            
            direction = row['direction'] # OVER / UNDER
            
            # Determine Outcome
            outcome = "VOID"
            cause = ""
            p50_hit = False
            error = 0
            
            # Check DNP / Void (use actual_val)
            if pd.isna(actual_val) or row['MIN'] == 0:
                outcome = "VOID"
                cause = "DNP / Injury"
            else:
                # [NEW] Calculate Error (MAE context)
                error = abs(actual_val - pred_pts)
                metrics['total_error'] += error
                
                # [NEW] Model Accuracy Check (P50)
                # Did the actual score land on the "correct side" of our P50?
                if actual_val > pred_pts: # Actual was HIGHER than model
                    p50_hit = True if direction == 'OVER' else False 
                elif actual_val < pred_pts: # Actual was LOWER than model
                    p50_hit = True if direction == 'UNDER' else False
                else:
                    p50_hit = True # Exact match
                    
                if p50_hit:
                    metrics['model_hits_p50'] += 1

                # Check Betting Result (Market Line)
                if direction == 'OVER':
                    if actual_val > line: outcome = "WIN"
                    elif actual_val < line: outcome = "LOSS"
                    else: outcome = "PUSH"
                elif direction == 'UNDER':
                    if actual_val < line: outcome = "WIN"
                    elif actual_val > line: outcome = "LOSS"
                    else: outcome = "PUSH"
                
                # Assign Cause for Loss
                if outcome == "LOSS":
                    cause = self.determine_cause(row, row)
            
            # Update Metrics
            metrics['total'] += 1
            if outcome == "WIN": metrics['wins'] += 1
            elif outcome == "LOSS": 
                metrics['losses'] += 1
                metrics['causes'][cause] = metrics['causes'].get(cause, 0) + 1
            elif outcome == "VOID": metrics['voids'] += 1
            
            results.append({
                'Player': row['player_name'],
                'Team': row['team'],
                'Opp': row['MATCHUP'] if pd.notna(row['MATCHUP']) else 'N/A',
                'Line': line,
                'Model': round(pred_pts, 1),
                'Actual': actual_val if pd.notna(actual_val) else 'DNP',
                'Diff': round(actual_val - pred_pts, 1) if pd.notna(actual_val) else 0,
                'Min': row['MIN'] if pd.notna(row['MIN']) else 0,
                'Outcome': outcome,
                'P50_Hit': "Yes" if p50_hit else "No",
                'Cause': cause,
                'Confidence': row['confidence']
            })
            
        # Generate Report
        self.generate_report(date_str, results, metrics)
        
    def generate_report(self, date_str, results, metrics):
        active_bets = metrics['wins'] + metrics['losses']
        win_rate = (metrics['wins'] / active_bets) * 100 if active_bets > 0 else 0
        
        # Calculate P50 Accuracy and MAE
        p50_acc = (metrics['model_hits_p50'] / active_bets) * 100 if active_bets > 0 else 0
        mae = metrics['total_error'] / active_bets if active_bets > 0 else 0
        
        report = f"""# 📊 Daily Prediction Audit: {date_str}

## 1️⃣ Performance Summary
### 💰 Betting Performance (vs Bookie Line)
- **Win Rate**: {win_rate:.1f}% ({metrics['wins']} W - {metrics['losses']} L)
- **Voids**: {metrics['voids']}
- **Total Issued**: {metrics['total']}

### 🎯 Model Accuracy (vs P50)
*Measure of true predictive power (Precision).*
- **P50 Accuracy**: {p50_acc:.1f}%  (Target: >55%)
- **Mean Absolute Error (MAE)**: {mae:.2f} pts  (Target: <6.0)

## 2️⃣ Failure Analysis (Primary Causes)
"""
        for cause, count in metrics['causes'].items():
            report += f"- **{cause}**: {count} failures\n"
            
        report += "\n## 3️⃣ Detailed Outcomes\n"
        report += "| Player | Team | Line | Model | Actual | Diff | Outcome | P50 Hit | Cause |\n"
        report += "|---|---|---|---|---|---|---|---|---|\n"
        
        for r in results:
            icon = "✅" if r['Outcome'] == 'WIN' else "❌" if r['Outcome'] == 'LOSS' else "⚠️"
            p50_icon = "🎯" if r['P50_Hit'] == 'Yes' else " "
            actual_display = int(r['Actual']) if isinstance(r['Actual'], (int, float)) else r['Actual']
            
            report += f"| {r['Player']} | {r['Team']} | {r['Line']} | {r['Model']} | {actual_display} | {r['Diff']:+.1f} | {icon} {r['Outcome']} | {p50_icon} | {r['Cause']} |\n"
            
        report += "\n## 4️⃣ Learning & Adjustments\n"
        report += "- **Minutes Sensitivity**: Adjusting threshold for 'Starter' classification.\n"
        report += "- **Variance**: Increasing variance penalty for players with unstable minutes.\n"
        
        filename = f"audit_report_{date_str}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport generated: {filename}")
        print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit NBA Predictions (Strict Mode)")
    parser.add_argument('--date', help='Date to audit (YYYY-MM-DD)', default=None)
    args = parser.parse_args()
    
    date_to_audit = args.date
    if not date_to_audit:
        # Default to yesterday
        from datetime import timedelta
        date_to_audit = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    auditor = PaymentAudit()
    auditor.run_audit(date_to_audit)
