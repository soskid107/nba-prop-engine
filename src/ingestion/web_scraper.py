import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import time
import random

class WebScraper:
    """Helper for scraping web data."""
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

    @staticmethod
    def _get_soup(url: str, retries: int = 3) -> BeautifulSoup:
        """Robust fetch with retries."""
        for i in range(retries):
            try:
                # Add random jitter
                if i > 0:
                    time.sleep(random.uniform(1, 3))
                    
                response = requests.get(url, headers=WebScraper.HEADERS, timeout=15)
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
                elif response.status_code == 404:
                    print(f"  [Scraper] 404 Not Found: {url}")
                    return None
                else:
                    print(f"  [Scraper] {response.status_code} error for {url} (Attempt {i+1})")
            except Exception as e:
                print(f"  [Scraper] Error fetching {url}: {e}")
        return None

    @staticmethod
    def scrape_espn_injuries() -> List[Dict[str, Any]]:
        """Scrape NBA injuries from ESPN."""
        url = "https://www.espn.com/nba/injuries"
        soup = WebScraper._get_soup(url)
        if not soup:
            return []
            
        injuries = []
        try:
            # ESPN organizes by one table per team
            tables = soup.find_all('table', class_='Table')
            
            for table in tables:
                rows = table.find_all('tr')
                if not rows:
                    continue
                    
                # Skip header row (index 0)
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        name_cell = cols[0]
                        player_name = name_cell.get_text(strip=True)
                        status_text = cols[3].get_text(strip=True)
                        comment = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                        
                        status = WebScraper._normalize_status(status_text, comment)
                        injuries.append({
                            'name': player_name,
                            'status': status,
                            'reason': f"{status_text} - {comment}",
                            'team': 'UNK'
                        })
            
            return injuries
        except Exception as e:
            print(f"  [Scraper] Parse error ESPN: {e}")
            return []

    @staticmethod
    def scrape_yahoo_injuries() -> List[Dict[str, Any]]:
        """Scrape NBA injuries from Yahoo Sports."""
        url = "https://sports.yahoo.com/nba/injuries/"
        soup = WebScraper._get_soup(url)
        if not soup:
            return []
            
        injuries = []
        try:
            # Yahoo usually has a big table
            # Look for table rows with player links
            # Structure: <tr> <td>Player</td> <td>Pos</td> <td>Status</td> <td>Date</td> <td>Injury</td> </tr>
            # But classes are dynamic. Look for 'tr'
            
            rows = soup.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    # Heuristic: First col is player
                    col0_text = cols[0].get_text(strip=True)
                    if not col0_text or "Player" in col0_text: # Header
                        continue
                        
                    player_name = col0_text
                    
                    # Look for status in other cols
                    # Usually col 2 or 3 is status/injury
                    full_row_text = row.get_text(" ", strip=True)
                    
                    status = "UNKNOWN"
                    if 'Out' in full_row_text or 'Injured Reserve' in full_row_text:
                        status = 'OUT'
                    elif 'Day-to-Day' in full_row_text or 'Questionable' in full_row_text:
                        status = 'GTD' # Yahoo uses Day-to-Day often for GTD
                    elif 'Doubtful' in full_row_text:
                        status = 'DOUBTFUL'
                        
                    if status != "UNKNOWN":
                        injuries.append({
                            'name': player_name,
                            'status': status,
                            'reason': full_row_text,
                            'team': 'UNK'
                        })
            return injuries
        except Exception as e:
            print(f"  [Scraper] Parse error Yahoo: {e}")
            return []

    @staticmethod
    def _normalize_status(status_text: str, comment: str = "") -> str:
        """Normalize injury status text."""
        full_text = (status_text + " " + comment).lower()
        
        # Check for explicit availability signals first
        # Prioritize 'Probable' over 'Out' because text often says "Out last game, now Probable"
        if 'probable' in full_text:
            return 'PROBABLE'
            
        if 'questionable' in full_text:
            return 'QUESTIONABLE'
            
        if 'doubtful' in full_text:
            return 'DOUBTFUL'
            
        if 'game time decision' in full_text or 'gtd' in full_text:
            return 'GTD'
            
        if 'out' in full_text and 'without' not in full_text: 
            return 'OUT'
        
        if 'injured reserve' in full_text:
            return 'OUT'
            
        if 'day-to-day' in full_text:
            return 'DAY-TO-DAY'
            
        if 'miss' in full_text or 'surgery' in full_text:
            return 'OUT'
            
        return 'UNKNOWN'

    @staticmethod
    def scrape_cbs_injuries() -> List[Dict[str, Any]]:
        """Scrape NBA injuries from CBS Sports."""
        url = "https://www.cbssports.com/nba/injuries/"
        soup = WebScraper._get_soup(url)
        if not soup:
            return []
            
        injuries = []
        try:
            for table in soup.find_all('table', class_='TableBase-table'):
                for row in table.find_all('tr', class_='TableBase-bodyTr'):
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        player_cell = cols[0]
                        player_name = player_cell.get_text(strip=True)
                        name_span = player_cell.find('span', class_='CellPlayerName--long')
                        if name_span:
                            player_name = name_span.get_text(strip=True)
                        
                        status_text = cols[3].get_text(strip=True)
                        comment = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                        
                        status = WebScraper._normalize_status(status_text, comment)
                        
                        injuries.append({
                            'name': player_name,
                            'status': status,
                            'reason': f"{status_text} - {comment}",
                            'team': 'UNK'
                        })
                        
            return injuries
            
        except Exception as e:
            print(f"  [Scraper] Parse error CBS: {e}")
            return []

if __name__ == "__main__":
    # Test
    data = WebScraper.scrape_cbs_injuries()
    print(f"Found {len(data)} injuries.")
    for i in data[:5]:
        print(i)
