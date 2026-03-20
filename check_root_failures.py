import sqlite3

db_path = "nba_props.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print(f"--- FAILED Odds API Calls in {db_path} ---")
cursor.execute("""
    SELECT call_date, response_status, endpoint, COUNT(*) 
    FROM api_usage 
    WHERE api_name = 'the_odds_api' AND response_status != 200
    GROUP BY call_date, response_status, endpoint
    ORDER BY call_date DESC
""")

rows = cursor.fetchall()
if not rows:
    print("No non-200 calls found for the_odds_api.")
else:
    for row in rows:
        print(f"Date: {row[0]} | Status: {row[1]} | Endpoint: {row[2]} | Count: {row[3]}")

print("\n--- Summary of all statuses ---")
cursor.execute("""
    SELECT response_status, COUNT(*) 
    FROM api_usage 
    WHERE api_name = 'the_odds_api'
    GROUP BY response_status
""")
for row in cursor.fetchall():
    print(f"Status {row[0]}: {row[1]} calls")

conn.close()
