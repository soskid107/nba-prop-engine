
import logging
import sys
import os

# Add root to verify package structure
sys.path.append(os.getcwd())

from src.ingestion.odds_ingestion import OddsIngestion
from src.utils.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_odds_fetch():
    try:
        db = DatabaseManager()
        ingestor = OddsIngestion(db)
        
        print(f"API Key present: {bool(ingestor.config.odds_api_key)}")
        if ingestor.config.odds_api_key:
            print(f"API Key starting with: {ingestor.config.odds_api_key[:4]}...")
            
        print("Clearing HTTP Cache...")
        ingestor.http.clear_cache()
        
        print("Testing Direct API Call...")
        response = ingestor.http.get_odds_api(
            endpoint="/sports/basketball_nba/odds",
            params={'regions': 'us', 'markets': 'spreads,totals', 'oddsFormat': 'american'},
            cache_hours=0
        )
        
        if response:
            print(f"Direct Call Success. Items: {len(response)}")
            if len(response) > 0:
                print(f"First item: {response[0]}")
        else:
            print("Direct Call Returned None")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_odds_fetch()
