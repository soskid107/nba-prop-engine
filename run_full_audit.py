
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.audit import PredictionAuditor

def main():
    from datetime import datetime, timedelta
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Starting Full Audit for {yesterday}...")
    
    auditor = PredictionAuditor()
    
    # 1. Archive predictions if not already in DB
    print("\nStep 1: Archiving predictions...")
    predictions_file = f"predictions_{yesterday}.csv"
    if os.path.exists(predictions_file):
        auditor.archive_predictions(predictions_file)
    else:
        print(f"Warning: {predictions_file} not found! Attempting sync from database logs...")
        auditor.sync_from_log(yesterday)
    
    # 2. Run the audit (fetches actuals and compares)
    print("\nStep 2: Running daily audit logic...")
    # days_back=1 implies yesterday. If today is Jan 25, this audits Jan 24.
    results = auditor.run_daily_audit(days_back=1)
    
    # 3. Generate Report
    print("\nStep 3: Generating report...")
    report_path = auditor.generate_audit_report()
    print(f"Report generated at: {report_path}")
    
    # Read and print the report content for the user
    if report_path and os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + "="*50)
            print(f.read())
            print("="*50)

if __name__ == "__main__":
    main()
