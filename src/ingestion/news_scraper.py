
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from ..utils.config import get_config
from ..utils.database import DatabaseManager

class NewsScraperAgent:
    """
    Subagent that monitors NBA player news for late-breaking injury updates.
    Sources: Rotowire (via Web), ESPN (via Web).
    """
    
    ROTOWIRE_URL = "https://www.rotowire.com/basketball/news.php" 
    # Alternative: https://www.nbcsports.com/nba/player-news
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.config = get_config()
        self.db = db or DatabaseManager()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def run(self):
        """Main execution method."""
        print("\n[NewsAgent] Scanning for late-breaking injury news...")
        news_items = self.fetch_rotowire_news()
        
        if not news_items:
            print("  [NewsAgent] No news items found or scrape failed.")
            return

        print(f"  [NewsAgent] Found {len(news_items)} recent news items.")
        
        updates = 0
        for item in news_items:
            if self._process_news_item(item):
                updates += 1
                
        print(f"  [NewsAgent] Processed {updates} distinct injury updates.")

    def fetch_rotowire_news(self) -> List[Dict[str, Any]]:
        """Scrape Rotowire NBA News page."""
        try:
            resp = requests.get(self.ROTOWIRE_URL, headers=self.headers, timeout=15)
            if resp.status_code != 200:
                print(f"  [Warn] Rotowire fetch failed: {resp.status_code}")
                return []
                
            soup = BeautifulSoup(resp.content, 'html.parser')
            news_items = []
            
            # Rotowire news items are usually in 'news-update' divs
            cards = soup.find_all('div', class_='news-update')
            
            for card in cards:
                # Player Name
                name_tag = card.find('a', class_='news-update__player-link')
                if not name_tag:
                    continue
                player_name = name_tag.get_text(strip=True)
                
                # Report
                report_tag = card.find('div', class_='news-update__news')
                report_text = report_tag.get_text(strip=True) if report_tag else ""
                
                # Headline/Status (often contains "Ruled Out", "Available", etc.)
                headline_tag = card.find('div', class_='news-update__headline')
                headline = headline_tag.get_text(strip=True) if headline_tag else ""
                
                # Timestamp (Crucial for "Up-to-Date" requirement)
                time_tag = card.find('div', class_='news-update__timestamp')
                timestamp_text = time_tag.get_text(strip=True) if time_tag else "Now"
                
                # Parse relative time into ISO format
                report_time = self._parse_relative_time(timestamp_text)
                
                # Filter out stale news (> 24 hours old)
                if self._is_stale(report_time):
                    continue

                news_items.append({
                    'player': player_name,
                    'headline': headline,
                    'report': report_text,
                    'source': 'Rotowire',
                    'timestamp': report_time.isoformat()
                })
                
            return news_items
            
        except Exception as e:
            print(f"  [Error] News scrape exception: {e}")
            return []

    def _process_news_item(self, item: Dict[str, Any]) -> bool:
        """
        Parse a news item and update injury status if relevant.
        Returns True if an update was made (or would be made).
        """
        player_name = item['player']
        text = (item['headline'] + " " + item['report']).lower()
        
        # 1. Determine Status from Text
        status = self._parse_status_from_text(text)
        if not status:
            return False
            
        # 2. Get Player ID
        # leveraging existing injury ingestion logic if possible, or manual lookup
        # For now, simple DB lookup
        player = self.db.search_players(player_name)
        if not player:
            # Try fuzzy match manually if needed, or skip
            return False
            
        # Take the best match (search_players returns list)
        # Ideally we check for exact match
        matched_player = None
        for p in player:
            if p['full_name'].lower() == player_name.lower():
                matched_player = p
                break
        
        if not matched_player and player:
             matched_player = player[0] # Fallback to first result
             
        if not matched_player:
            return False
            
        pid = matched_player['player_id']
        
        # 3. Calculate Probability & Insert
        # We assume news is the TRUTH, so we map directly
        # Check if we already have this status today to avoid dupes?
        # The database unique constraint (date, player, reason) might not catch it if reason differs
        # But `insert_injury_snapshot` handles it.
        
        injury_record = {
            'player_id': pid,
            'player_name': matched_player['full_name'],
            'team_abbreviation': matched_player['team_abbreviation'],
            'status': status,
            'reason': f"[News] {item['headline']}",
            'source_name': 'Rotowire',
            'fetched_at': item.get('timestamp', datetime.now().isoformat()),
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'p_play': self._get_prob_from_status(status)
        }
        
        print(f"    -> Update: {player_name} is {status} (prob {injury_record['p_play']})")
        self.db.insert_injury_snapshot(injury_record)
        return True

    def _parse_status_from_text(self, text: str) -> Optional[str]:
        """NLP-lite to determine status."""
        if 'ruled out' in text or 'will not play' in text or 'out for' in text:
            return 'OUT'
        if 'available' in text or 'will play' in text or 'active' in text:
            return 'AVAILABLE'
        if 'game-time' in text or 'game time' in text or 'gtd' in text:
            return 'GTD'
        if 'doubtful' in text or 'unlikely' in text:
            return 'DOUBTFUL'
        if 'questionable' in text:
            return 'QUESTIONABLE'
        if 'probable' in text or 'likely' in text:
            return 'PROBABLE'
        return None

    def _get_prob_from_status(self, status: str) -> float:
        """Map status to config probability."""
        if hasattr(self.config, 'injury_probabilities'):
            return self.config.injury_probabilities.get(status, 0.5)
        return 0.5

    def _parse_relative_time(self, time_str: str) -> datetime:
        """Parse relative time strings like '5 mins ago', '2 hours ago'."""
        now = datetime.now()
        time_str = time_str.lower()
        
        try:
            if 'min' in time_str:
                digit = int(re.search(r'\d+', time_str).group())
                return now - timedelta(minutes=digit)
            elif 'hour' in time_str or 'hr' in time_str:
                digit = int(re.search(r'\d+', time_str).group())
                return now - timedelta(hours=digit)
            elif 'day' in time_str:
                digit = int(re.search(r'\d+', time_str).group())
                return now - timedelta(days=digit)
            elif 'yesterday' in time_str:
                return now - timedelta(days=1)
            else:
                return now # Default to now if unparseable
        except:
            return now

    def _is_stale(self, report_time: datetime, max_hours: int = 24) -> bool:
        """Check if news is older than max_hours."""
        age = datetime.now() - report_time
        return age.total_seconds() > (max_hours * 3600)

if __name__ == "__main__":
    # Fix path for direct execution
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from datetime import timedelta # Needed for main execution context too if not imported top level
    
    # Test run
    agent = NewsScraperAgent()
    agent.run()
