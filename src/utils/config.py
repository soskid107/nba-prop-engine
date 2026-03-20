"""

Configuration Loader



Loads and validates the config.yaml file, providing typed access to all settings.

"""



import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()





class Config:

    """Central configuration manager for the NBA Props Engine."""

    

    _instance: Optional['Config'] = None

    _config: Dict[str, Any] = {}

    

    def __new__(cls):

        """Singleton pattern to ensure one config instance."""

        if cls._instance is None:

            cls._instance = super().__new__(cls)

        return cls._instance

    

    def __init__(self):

        if not self._config:

            self._load_config()

    

    def _load_config(self) -> None:

        """Load configuration from config.yaml."""

        # Find project root (where config.yaml lives)

        current = Path(__file__).resolve()

        project_root = current.parent.parent.parent

        config_path = project_root / "config.yaml"

        

        if not config_path.exists():

            raise FileNotFoundError(f"Config file not found at {config_path}")

        

        with open(config_path, 'r') as f:

            self._config = yaml.safe_load(f)

        

        # Store project root for path resolution

        self._config['_project_root'] = str(project_root)

    

    @property

    def project_root(self) -> Path:

        """Get the project root directory."""

        return Path(self._config['_project_root'])

    

    # =====================

    # API Keys

    # =====================

    @property

    def odds_api_key(self) -> str:

        """The Odds API key."""

        return os.getenv('ODDS_API_KEY') or self._config['api_keys'].get('the_odds_api')

    

    @property
    def api_sports_key(self) -> Optional[str]:
        """API-SPORTS key (fallback)."""
        return self._config['api_keys'].get('api_sports')

    @property
    def balldontlie_api_key(self) -> Optional[str]:
        """balldontlie.io API key."""
        return os.getenv('BALLDONTLIE_API_KEY') or self._config['api_keys'].get('balldontlie')

    

    # =====================

    # Database

    # =====================

    @property

    def database_path(self) -> Path:

        """Full path to SQLite database."""

        return self.project_root / self._config['database']['path']

    

    # =====================

    # Rate Limits

    # =====================

    @property

    def nba_api_delay(self) -> float:

        """Minimum delay between NBA API calls (seconds)."""

        return self._config['rate_limits']['nba_api']['min_delay_seconds']

    

    @property

    def nba_api_jitter(self) -> float:

        """Max jitter to add to NBA API delay (seconds)."""

        return self._config['rate_limits']['nba_api']['max_jitter_seconds']

    

    @property

    def odds_api_max_daily_calls(self) -> int:

        """Maximum daily calls to The Odds API."""

        return self._config['rate_limits']['the_odds_api']['max_daily_calls']

    

    # =====================

    # Cache

    # =====================

    @property

    def cache_enabled(self) -> bool:

        """Whether caching is enabled."""

        return self._config['cache']['enabled']

    

    @property

    def cache_directory(self) -> Path:

        """Path to cache directory."""

        return self.project_root / self._config['cache']['directory']

    

    @property

    def cache_expiry_hours(self) -> int:

        """Cache expiry time in hours."""

        return self._config['cache']['expiry_hours']

    

    # =====================

    # Model Settings

    # =====================

    @property

    def model_type(self) -> str:

        """ML model type (lightgbm or xgboost)."""

        return self._config['model']['minutes']['type']

    

    @property

    def rolling_windows(self) -> list:

        """Rolling window sizes for feature engineering."""

        return self._config['model']['minutes']['rolling_windows']

    

    # =====================

    # Simulation

    # =====================

    @property

    def n_simulations(self) -> int:

        """Number of Monte Carlo simulations."""

        return self._config['simulation']['n_simulations']

    

    @property

    def random_seed(self) -> int:

        """Random seed for reproducibility."""

        return self._config['simulation']['random_seed']

    

    # =====================

    # Injury Probabilities

    # =====================

    @property

    def injury_probabilities(self) -> Dict[str, float]:

        """Mapping of injury status to play probability."""

        return self._config['injury_probabilities']

    

    def get_play_probability(self, status: str) -> float:

        """Get probability of playing given injury status."""

        status_upper = status.upper().strip()

        return self.injury_probabilities.get(status_upper, 1.0)

    

    # =====================

    # Logging

    # =====================

    @property

    def log_level(self) -> str:

        """Logging level."""

        return self._config['logging']['level']

    

    @property

    def log_file(self) -> Path:

        """Path to log file."""

        return self.project_root / self._config['logging']['file']

    

    # =====================

    # NBA Settings

    # =====================

    @property

    def current_season(self) -> str:

        """Current NBA season in YYYY-YY format."""

        return self._config['nba']['current_season']

    

    @property

    def season_start_date(self) -> str:

        """Season start date (YYYY-MM-DD)."""

        return self._config['nba']['season_start_date']





# Convenience function for quick access

def get_config() -> Config:

    """Get the singleton Config instance."""

    return Config()

