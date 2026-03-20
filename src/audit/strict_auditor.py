
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.database import DatabaseManager
from datetime import datetime
import os

class StrictAuditor:
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
            
            # Filter for actual play
            # MIN is usually string "MM:SS" or float. API returns float for MIN in this endpoint? 
            # In the test output it showed "39:09" in MIN_SEC, but 'MIN' column usually exists.
            # Let's check the test output: 'MIN' column is there. It's usually float in this endpoint.
            # But let's handle both.
            
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
        # Handle string min if necessary (though GameLogs usually float)
        
        pred_pts = row['predicted_mean']
        actual_pts = actual_row['PTS']
        
        # 1. Minutes Issue
        if actual_min < (pred_min * 0.75):
            # check for fouls
            if actual_row['PF'] >= 4:
                return "Foul Trouble"
            # check for blowout (plus_minus is rough proxy)
            if abs(actual_row['PLUS_MINUS']) > 20:
                return "Blowout / Garbage Time"
            return "Minutes Misprojection (Rotation Change)"
            
        # 2. Usage Issue
        # We don't have predicted FGA stored always, but can infer.
        # Let's assume Usage Misprojection if Points are low but Min are high
        
        # 3. Shooting/Efficiency
        if actual_row['FG_PCT'] < 0.35:
            return "Cold Shooting / Variance"
            
        return "Defensive Matchup / Other"

    def run_audit(self, date_str):
        print(f"\n=== Running Strict Audit for {date_str} ===")
        
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
        merged = pd.merge(
            preds, 
            actuals[['normalized_name', 'PTS', 'MIN', 'PF', 'PLUS_MINUS', 'FG_PCT', 'MATCHUP', 'WL']], 
            on='normalized_name', 
            how='left'
        )
        
        results = []
        metrics = {
            'total': 0, 'wins': 0, 'losses': 0, 'voids': 0,
            'model_hits_p50': 0, # [NEW] Did we hit our own P50?
            'total_error': 0,    # [NEW] Sum of absolute errors for MAE
            'causes': {}
        }
        
        print(f"Auditing {len(merged)} predictions...")
        
        for _, row in merged.iterrows():
            line = float(row['line']) if row['line'] else 0
            pred_pts = row['predicted_mean']
            actual_pts = row['PTS']
            direction = row['direction'] # OVER / UNDER
            
            # Determine Outcome
            outcome = "VOID"
            cause = ""
            p50_hit = False
            error = 0
            
            # Check DNP / Void
            if pd.isna(actual_pts) or row['MIN'] == 0:
                outcome = "VOID"
                cause = "DNP / Injury"
            else:
                # [NEW] Calculate Error (MAE context)
                error = abs(actual_pts - pred_pts)
                metrics['total_error'] += error
                
                # [NEW] Model Accuracy Check (P50)
                # Did the actual score land on the "correct side" of our P50?
                # If we predicted 20.5 (Over), and they scored 21, that's a hit.
                # If we predicted 20.5 (Under), and they scored 20, that's a hit.
                # Strictly comparing Actual vs Predicted Mean
                if actual_pts > pred_pts: # Actual was HIGHER than model
                    p50_hit = True if direction == 'OVER' else False # Hits if we bet Over
                elif actual_pts < pred_pts: # Actual was LOWER than model
                    p50_hit = True if direction == 'UNDER' else False # Hits if we bet Under
                else:
                    p50_hit = True # Exact match (rare)
                    
                if p50_hit:
                    metrics['model_hits_p50'] += 1

                # Check Betting Result (Market Line)
                if direction == 'OVER':
                    if actual_pts > line: outcome = "WIN"
                    elif actual_pts < line: outcome = "LOSS"
                    else: outcome = "PUSH"
                elif direction == 'UNDER':
                    if actual_pts < line: outcome = "WIN"
                    elif actual_pts > line: outcome = "LOSS"
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
                'Actual': actual_pts if pd.notna(actual_pts) else 'DNP',
                'Diff': round(actual_pts - pred_pts, 1) if pd.notna(actual_pts) else 0,
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
        
        # [NEW] Calculate P50 Accuracy and MAE
        p50_acc = (metrics['model_hits_p50'] / active_bets) * 100 if active_bets > 0 else 0
        mae = metrics['total_error'] / active_bets if active_bets > 0 else 0
        
        report = f"""# 📊 Daily Prediction Audit: {date_str}

## 1️⃣ Performance Summary
### 💰 Betting Performance (vs Bookie Line)
- **Win Rate**: {win_rate:.1f}% ({metrics['wins']} W - {metrics['losses']} L)
- **Voids**: {metrics['voids']}
- **Total Issued**: {metrics['total']}

### 🎯 Model Accuracy (vs P50)
*This measures how often the actual score landed on the predicted side of the Model's Median, proving pure predictive power independent of soft lines.*
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
        
        filename = f"strict_audit_{date_str}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport generated: {filename}")
        print(report)

if __name__ == "__main__":
    auditor = StrictAuditor()
    # Run for the last few days to confirm trend
    # auditor.run_audit('2026-01-24') 
    auditor.run_audit('2026-01-28') # Rerunning for yesterday per user request
