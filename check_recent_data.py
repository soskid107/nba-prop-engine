import sqlite3

db_path = "data/nba_props.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("--- Recent Games (March 2026) ---")
cursor.execute("""
    SELECT game_date, home_team_abbr, away_team_abbr, status 
    FROM games 
    WHERE game_date >= '2026-03-01'
    ORDER BY game_date DESC
    LIMIT 20
""")

for row in cursor.fetchall():
    print(f"{row['game_date']} | {row['away_team_abbr']} @ {row['home_team_abbr']} | Status: {row['status']}")

print("\n--- Recent Odds Snapshots ---")
cursor.execute("""
    SELECT game_date, home_team, away_team, bookmaker, total 
    FROM odds_snapshots 
    WHERE game_date >= '2026-03-01'
    ORDER BY snapshot_time DESC
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"{row['game_date']} | {row['away_team']} @ {row['home_team']} | Book: {row['bookmaker']} | Total: {row['total']}")

conn.close()
