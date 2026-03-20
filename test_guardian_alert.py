
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.getcwd())

from src.utils.database import DatabaseManager
from src.audit.guardian import ProductionGuardian

def test_drift_alert():
    print("Testing Guardian Drift Alert...")
    
    db = DatabaseManager()
    guardian = ProductionGuardian(db)
    
    # 1. Create a dummy dataframe with EXTREME values to trigger drift
    # Historical mean is likely ~15-20. We'll set this to 50.
    df = pd.DataFrame({
        'predicted_mean': [50.0, 51.0, 49.0, 50.5, 49.5] * 10
    })
    
    print(f"Injecting dummy data with Mean: {df['predicted_mean'].mean()}")
    
    # 2. Check Drift
    # Note: This relies on DB having SOME history. If DB is empty, it returns True.
    # We assume 'backtest_phase16.py' populated something or previous runs did.
    # If not, we might need to mock history, but let's try.
    is_ok = guardian.check_distribution_drift(df, days_back=30)
    
    print(f"Drift Check Result: {is_ok}")
    
    if not is_ok:
        print(" -> Drift detected (Success)")
        
        # 3. Save Alerts
        if os.path.exists("ALERTS.log"):
            os.remove("ALERTS.log")
            
        guardian.save_alerts("ALERTS.log")
        
        if os.path.exists("ALERTS.log"):
            print(" -> ALERTS.log created (Success)")
            with open("ALERTS.log", "r") as f:
                print("Content:")
                print(f.read())
        else:
            print(" -> ALERTS.log NOT created (Fail)")
    else:
        print(" -> Drift NOT detected (Fail or No History)")
        # Check if history exists
        stats = guardian._get_historical_stats(30)
        print(f"Historical Stats: {stats}")

if __name__ == "__main__":
    test_drift_alert()
