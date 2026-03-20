"""

Smart HTTP Client



Centralized HTTP client with:

- Automatic caching (SQLite-backed)

- Rate limiting with jitter

- Exponential backoff retries

- API usage tracking

"""



import hashlib
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from urllib.parse import urlencode



import requests

from tenacity import (

    retry,

    RetryError,

    stop_after_attempt,

    wait_exponential,

    retry_if_exception_type
)




from .config import get_config

from .database import DatabaseManager





class RateLimitExceeded(Exception):

    """Raised when API rate limit is exceeded."""

    pass





class SmartHttpClient:

    """HTTP client with caching, rate limiting, and retries."""

    

    def __init__(self, db: Optional[DatabaseManager] = None):

        """Initialize the smart HTTP client.

        

        Args:

            db: Optional database manager. Creates new one if not provided.

        """

        self.config = get_config()

        self.db = db or DatabaseManager()

        self._last_request_time: Dict[str, float] = {}

        self.session = requests.Session()

        self.logger = logging.getLogger('nba_engine')
        self.session.headers.update({
            'User-Agent': 'NBA-Props-Engine/1.0'
        })

    

    def _generate_cache_key(self, url: str, params: Optional[Dict] = None) -> str:

        """Generate a unique cache key for a request.

        

        Args:

            url: Request URL

            params: Optional query parameters

            

        Returns:

            MD5 hash of URL + sorted params

        """

        key_data = url

        if params:

            key_data += "?" + urlencode(sorted(params.items()))

        return hashlib.md5(key_data.encode()).hexdigest()

    

    def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:

        """Get cached response if available and not expired.

        

        Args:

            cache_key: Cache key to lookup

            

        Returns:

            Parsed JSON response or None

        """

        if not self.config.cache_enabled:

            return None

        

        cached = self.db.get_cached_response(cache_key)

        if cached:

            try:

                return json.loads(cached)

            except json.JSONDecodeError:

                return None

        return None

    

    def _set_cached(self, cache_key: str, url: str, params: Optional[Dict],

                   response_data: Any, expiry_hours: Optional[int] = None) -> None:

        """Store response in cache.

        

        Args:

            cache_key: Cache key

            url: Original URL

            params: Original params

            response_data: Response to cache

            expiry_hours: Optional custom expiry (uses config default if None)

        """

        if not self.config.cache_enabled:

            return

        

        expiry = expiry_hours or self.config.cache_expiry_hours

        expires_at = (datetime.now() + timedelta(hours=expiry)).isoformat()

        

        self.db.set_cached_response(

            cache_key=cache_key,

            url=url,

            params=json.dumps(params) if params else "",

            response_data=json.dumps(response_data),

            expires_at=expires_at

        )

    

    def _throttle(self, api_name: str, min_delay: float, max_jitter: float) -> None:

        """Apply rate limiting with jitter.

        

        Args:

            api_name: Name of API for tracking

            min_delay: Minimum delay between requests

            max_jitter: Maximum random jitter to add

        """

        last_time = self._last_request_time.get(api_name, 0)

        elapsed = time.time() - last_time

        

        if elapsed < min_delay:

            # Calculate sleep time with jitter

            sleep_time = min_delay - elapsed + random.uniform(0, max_jitter)

            time.sleep(sleep_time)

        

        self._last_request_time[api_name] = time.time()

    

    def _check_daily_limit(self, api_name: str, max_calls: int) -> None:

        """Check if daily API limit is exceeded.

        

        Args:

            api_name: Name of API

            max_calls: Maximum allowed calls per day

            

        Raises:

            RateLimitExceeded: If limit is exceeded

        """

        current_calls = self.db.get_api_usage_today(api_name)

        if current_calls >= max_calls:

            raise RateLimitExceeded(

                f"Daily limit for {api_name} exceeded ({current_calls}/{max_calls})"

            )

    

    @retry(

        retry=retry_if_exception_type(requests.exceptions.RequestException),

        stop=stop_after_attempt(3),

        wait=wait_exponential(multiplier=1, min=2, max=10)

    )

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:

        """Make HTTP request with retries.

        

        Args:

            method: HTTP method (GET, POST, etc.)

            url: Request URL

            **kwargs: Additional request arguments

            

        Returns:

            Response object

        """

        response = self.session.request(method, url, **kwargs)

        response.raise_for_status()

        return response

    

    # =====================

    # NBA API Methods

    # =====================

    def get_nba_api(self, url: str, params: Optional[Dict] = None,

                    cache_hours: int = 24) -> Optional[Dict[str, Any]]:

        """Make a request to NBA API with rate limiting and caching.

        

        Args:

            url: API endpoint URL

            params: Query parameters

            cache_hours: How long to cache (default 24 hours)

            

        Returns:

            JSON response or None on error

        """

        api_name = "nba_api"

        cache_key = self._generate_cache_key(url, params)

        

        # Check cache first

        cached = self._get_cached(cache_key)

        if cached is not None:
            self.logger.info(f"  [CACHE] {api_name} {url}")
            self.db.log_api_call(api_name, url, 200, cached=True)
            return cached

        

        # Apply rate limiting

        self._throttle(

            api_name,

            self.config.nba_api_delay,

            self.config.nba_api_jitter

        )

        

        try:

            response = self._make_request("GET", url, params=params, timeout=30)

            self.logger.info(f"  [NETWORK] {api_name} {url} - Status: {response.status_code}")
            # Cache successful response
            self._set_cached(cache_key, url, params, data, cache_hours)
            self.db.log_api_call(api_name, url, response.status_code, cached=False)
            
            return data

            

        except requests.exceptions.RequestException as e:
            status_code = getattr(e.response, 'status_code', 0)
            self.logger.error(f"  [ERROR] {api_name} {url} - Status: {status_code} - Error: {e}")
            self.db.log_api_call(api_name, url, status_code, cached=False)
            return None

    

    # =====================

    # The Odds API Methods

    # =====================

    def get_odds_api(self, endpoint: str, params: Optional[Dict] = None,
                     cache_hours: int = 6) -> Optional[Dict[str, Any]]:
        """Make a request to The Odds API with daily limit checking.

        

        Args:

            endpoint: API endpoint (e.g., "/sports/basketball_nba/odds")

            params: Query parameters (api_key added automatically)

            cache_hours: How long to cache (default 6 hours for odds)

            

        Returns:

            JSON response or None on error

        """

        api_name = "the_odds_api"

        base_url = "https://api.the-odds-api.com/v4"

        url = f"{base_url}{endpoint}"

        

        # Add API key to params

        params = params or {}

        params['apiKey'] = self.config.odds_api_key

        

        cache_key = self._generate_cache_key(url, params)

        

        # Check cache first

        cached = self._get_cached(cache_key)

        if cached is not None:
            self.logger.info(f"  [CACHE] {api_name} {endpoint}")
            self.db.log_api_call(api_name, endpoint, 200, cached=True)
            return cached

        

        # Check daily limit before making request

        try:

            self._check_daily_limit(api_name, self.config.odds_api_max_daily_calls)

        except RateLimitExceeded as e:

            print(f"[WARN] {e}")

            return None

        

        try:

            response = self._make_request("GET", url, params=params, timeout=30)

            data = response.json()

            

            # Log remaining requests from headers

            remaining = response.headers.get('x-requests-remaining', 'unknown')

            self.logger.info(f"  [NETWORK] {api_name} {endpoint} - Status: {response.status_code} - Rem: {remaining}")
            
            # Cache successful response
            self._set_cached(cache_key, url, params, data, cache_hours)
            self.db.log_api_call(api_name, endpoint, response.status_code, cached=False)
            
            return data

            

        except (requests.exceptions.RequestException, RetryError) as e:
            status_code = 0
            response = getattr(e, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", 0) or 0
            elif hasattr(e, "last_attempt") and e.last_attempt.exception():
                inner = e.last_attempt.exception()
                inner_resp = getattr(inner, "response", None)
                if inner_resp is not None:
                    status_code = getattr(inner_resp, "status_code", 0) or 0

            self.logger.error(f"  [ERROR] {api_name} {endpoint} - Status: {status_code} - Error: {e}")
            self.db.log_api_call(api_name, endpoint, status_code, cached=False)
            return None

    

    # =====================

    # balldontlie API Methods

    # =====================

    def get_balldontlie_api(self, endpoint: str, params: Optional[Dict] = None,

                            cache_hours: int = 12) -> Optional[Dict[str, Any]]:

        """Make a request to balldontlie API with rate limiting and caching.

        

        Args:

            endpoint: API endpoint (e.g., "/players")

            params: Query parameters

            cache_hours: How long to cache (default 12 hours)

            

        Returns:

            JSON response or None on error

        """

        api_name = "balldontlie"

        base_url = "https://api.balldontlie.io/v1"

        url = f"{base_url}{endpoint}"

        

        # Add API key to headers

        headers = {'Authorization': self.config.balldontlie_api_key}

        

        cache_key = self._generate_cache_key(url, params)

        

        # Check cache first

        cached = self._get_cached(cache_key)

        if cached is not None:

            self.db.log_api_call(api_name, endpoint, 200, cached=True)

            return cached

        

        # Apply rate limiting (60 req/min -> 1.0s delay)

        self._throttle(api_name, 1.0, 0.5)

        

        try:

            response = self._make_request("GET", url, params=params, headers=headers, timeout=30)

            data = response.json()

            

            # Cache successful response

            self._set_cached(cache_key, url, params, data, cache_hours)

            self.db.log_api_call(api_name, endpoint, response.status_code, cached=False)

            

            return data

            

        except requests.exceptions.RequestException as e:

            self.db.log_api_call(api_name, endpoint, getattr(e.response, 'status_code', 0), cached=False)

            print(f"[ERROR] balldontlie API request failed: {e}")

            return None

    # =====================

    # Generic Methods

    # =====================

    def get(self, url: str, params: Optional[Dict] = None,

            api_name: str = "generic", cache_hours: int = 24,

            throttle_delay: float = 0) -> Optional[Dict[str, Any]]:

        """Generic GET request with optional caching and throttling.

        

        Args:

            url: Request URL

            params: Query parameters

            api_name: Name for tracking/throttling

            cache_hours: Cache duration (0 to disable)

            throttle_delay: Minimum delay between requests

            

        Returns:

            JSON response or None on error

        """

        cache_key = self._generate_cache_key(url, params)

        

        # Check cache if enabled

        if cache_hours > 0:

            cached = self._get_cached(cache_key)

            if cached is not None:

                return cached

        

        # Apply throttling if specified

        if throttle_delay > 0:

            self._throttle(api_name, throttle_delay, 0.5)

        

        try:

            response = self._make_request("GET", url, params=params, timeout=30)

            data = response.json()

            

            # Cache if enabled

            if cache_hours > 0:

                self._set_cached(cache_key, url, params, data, cache_hours)

            

            return data

            

        except requests.exceptions.RequestException as e:

            print(f"[ERROR] Request to {url} failed: {e}")

            return None

    

    def download_file(self, url: str, save_path: str, 

                      api_name: str = "generic") -> bool:

        """Download a file (e.g., PDF) with throttling.

        

        Args:

            url: URL to download

            save_path: Local path to save file

            api_name: Name for throttling

            

        Returns:

            True if successful, False otherwise

        """

        self._throttle(api_name, 1.0, 0.5)

        

        try:

            response = self._make_request("GET", url, stream=True, timeout=60)

            

            with open(save_path, 'wb') as f:

                for chunk in response.iter_content(chunk_size=8192):

                    f.write(chunk)

            

            return True

            

        except Exception as e:

            print(f"[ERROR] Download failed: {e}")

            return False

    

    # =====================

    # Cache Management

    # =====================

    def clear_cache(self) -> int:

        """Clear all expired cache entries.

        

        Returns:

            Number of entries cleared

        """

        return self.db.clear_expired_cache()

    

    def get_cache_stats(self) -> Dict[str, Any]:

        """Get cache statistics.

        

        Returns:

            Dict with cache stats

        """

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            cursor.execute("SELECT COUNT(*) as total FROM http_cache")

            total = cursor.fetchone()['total']

            

            cursor.execute("""

                SELECT COUNT(*) as expired FROM http_cache WHERE expires_at < ?

            """, (datetime.now().isoformat(),))

            expired = cursor.fetchone()['expired']

            

            return {

                'total_entries': total,

                'expired_entries': expired,

                'active_entries': total - expired

            }





# Convenience function

def get_http_client() -> SmartHttpClient:

    """Get a SmartHttpClient instance."""

    return SmartHttpClient()

