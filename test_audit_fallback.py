
from src.utils.database import DatabaseManager
from src.simulation.audit import PredictionAuditor

def test_audit_fallback():
    db = DatabaseManager()
    auditor = PredictionAuditor(db)
    
    test_date = "2026-03-07"
    print(f"Testing audit fallback (sync_from_log) for {test_date}...")
    
    # 1. Clear archive for this date to ensure we are testing correctly
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions_archive WHERE prediction_date = ?", (test_date,))
        conn.commit()
    
    # 2. Sync from log
    count = auditor.sync_from_log(test_date)
    print(f"Synced {count} predictions from log.")
    
    # 3. Verify
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions_archive WHERE prediction_date = ?", (test_date,))
        archived_count = cursor.fetchone()[0]
        print(f"Predictions in archive: {archived_count}")

if __name__ == "__main__":
    test_audit_fallback()
