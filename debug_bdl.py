"""Quick diagnostic for balldontlie API."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('BALLDONTLIE_API_KEY')
print(f"API Key loaded: {key[:8]}..." if key else "NO KEY FOUND")

# Test 1: Teams endpoint
print("\n--- Test 1: GET /teams ---")
try:
    r = requests.get("https://api.balldontlie.io/v1/teams", 
                      headers={"Authorization": key}, timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Teams found: {len(data.get('data', []))}")
    else:
        print(f"Response: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Stats endpoint with a date
print("\n--- Test 2: GET /stats?dates[]=2026-02-20 ---")
try:
    r = requests.get("https://api.balldontlie.io/v1/stats", 
                      headers={"Authorization": key},
                      params={"dates[]": "2026-02-20", "per_page": 5},
                      timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Stats returned: {len(data.get('data', []))}")
        if data.get('data'):
            s = data['data'][0]
            p = s.get('player', {})
            print(f"Sample: {p.get('first_name')} {p.get('last_name')} - {s.get('pts')} PTS, {s.get('min')} MIN")
            print(f"Meta: {data.get('meta', {})}")
    else:
        print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"ERROR: {e}")
