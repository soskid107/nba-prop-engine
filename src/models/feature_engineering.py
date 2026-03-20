"""

Feature Engineering



Builds pre-game feature vectors for the two-stage model:

- Minutes Model Features: Rolling minutes, starter flag, spread (blowout risk), rest days

- PPM Model Features: Rolling PPM, opponent defense, total (pace proxy), usage



Ensures no data leakage by using only pre-game information.

"""



import pandas as pd

import numpy as np

from datetime import datetime, timedelta

from typing import Any, Dict, List, Optional, Tuple



from ..utils.config import get_config

from ..utils.database import DatabaseManager





class FeatureEngineer:

    """Builds feature vectors for ML models."""

    

    def __init__(self, db: Optional[DatabaseManager] = None):

        """Initialize feature engineer.

        

        Args:

            db: Optional database manager

        """

        self.config = get_config()

        self.db = db or DatabaseManager()

        self.rolling_windows = self.config.rolling_windows  # [3, 5, 10]

    

    def _get_player_history(self, player_id: int, 

                            before_date: str = None,

                            limit: int = 50) -> pd.DataFrame:

        """Get player's game history as DataFrame.

        

        Args:

            player_id: NBA player ID

            before_date: Only include games before this date (leak prevention)

            limit: Max games to retrieve

            

        Returns:

            DataFrame of game logs sorted by date descending

        """

        with self.db.get_connection() as conn:

            query = """

                SELECT * FROM player_logs 

                WHERE player_id = ?

            """

            params = [player_id]

            

            if before_date:

                query += " AND game_date < ?"

                params.append(before_date)

            

            # Exclude DNPs (0 minutes) from history averages

            query += " AND minutes > 0"



            

            query += " ORDER BY game_date DESC LIMIT ?"

            params.append(limit)

            

            df = pd.read_sql_query(query, conn, params=params)

        

        return df

    

    def _calculate_rolling_stats(self, df: pd.DataFrame, 

                                  column: str,

                                  windows: List[int] = None) -> Dict[str, float]:

        """Calculate rolling averages for a column.

        

        Args:

            df: DataFrame with game history (sorted by date desc)

            column: Column name to calculate rolling stats for

            windows: List of window sizes (default: from config)

            

        Returns:

            Dict with rolling stats (e.g., {'min_L3': 32.5, 'min_L5': 31.2})

        """

        windows = windows or self.rolling_windows

        stats = {}

        

        for w in windows:

            key = f"{column}_L{w}"

            if len(df) >= w:

                stats[key] = df[column].head(w).mean()

            else:

                # Use available data if less than window

                stats[key] = df[column].mean() if len(df) > 0 else 0.0

        

        return stats

    

    def _calculate_rest_days(self, df: pd.DataFrame, 

                              current_date: str) -> int:

        """Calculate days since last game.

        

        Args:

            df: DataFrame with game history

            current_date: Current game date

            

        Returns:

            Days since last game (0 if B2B, capped at 7)

        """

        if df.empty:

            return 3  # Default for new players

        

        last_game = df['game_date'].iloc[0]

        try:

            current = datetime.strptime(current_date, '%Y-%m-%d')

            last = datetime.strptime(last_game, '%Y-%m-%d')

            rest = (current - last).days - 1  # Subtract 1 for rest days

            return min(max(rest, 0), 7)  # Cap at 7

        except (ValueError, TypeError):

            return 3

    

    def _is_starter(self, df: pd.DataFrame, threshold: float = 0.6) -> int:

        """Determine if player is typically a starter.

        

        Args:

            df: DataFrame with game history

            threshold: Fraction of games to be considered starter

            

        Returns:

            1 if starter, 0 otherwise

        """

        if df.empty:

            return 0

        

        # Use minutes as proxy - starters typically play 25+ minutes

        avg_minutes = df['minutes'].head(10).mean()

        return 1 if avg_minutes >= 25 else 0

    

    def _get_odds_context(self, team_abbr: str, 

                          game_date: str) -> Dict[str, Optional[float]]:

        """Get spread and total for a team's game.

        

        Args:

            team_abbr: Team abbreviation

            game_date: Game date

            

        Returns:

            Dict with 'spread' and 'total' (None if not available)

        """

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            # Try as home team

            cursor.execute("""

                SELECT spread_home, total FROM odds_snapshots 

                WHERE game_date = ? AND home_team = ?

                ORDER BY snapshot_time DESC LIMIT 1

            """, (game_date, team_abbr))

            row = cursor.fetchone()

            

            if row:

                return {

                    'spread': row['spread_home'],

                    'total': row['total'],

                    'is_home': 1

                }

            

            # Try as away team

            cursor.execute("""

                SELECT spread_away, total FROM odds_snapshots 

                WHERE game_date = ? AND away_team = ?

                ORDER BY snapshot_time DESC LIMIT 1

            """, (game_date, team_abbr))

            row = cursor.fetchone()

            

            if row:

                return {

                    'spread': row['spread_away'],

                    'total': row['total'],

                    'is_home': 0

                }

        

        return {'spread': None, 'total': None, 'is_home': None}

    

    def _get_opponent_defense(self, opponent_abbr: str) -> float:

        """Get opponent's defensive rating (points allowed per game).

        

        Args:

            opponent_abbr: Opponent team abbreviation

            

        Returns:

            Average points allowed (default 110 if no data)

        """

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            

            # Calculate average points scored against this team

            cursor.execute("""

                SELECT AVG(points) as avg_pts FROM player_logs 

                WHERE opponent_abbreviation = ?

                AND game_date > date('now', '-30 days')

            """, (opponent_abbr,))

            

            row = cursor.fetchone()

            if row and row['avg_pts']:

                return row['avg_pts']

        

        return 15.0  # Default average points per player vs team

    

    def build_features_for_player(self, player_id: int,
                                   team_abbr: str,
                                   opponent_abbr: str,
                                   game_date: str,
                                   p_play: float = 1.0,
                                   full_history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Build complete feature vector for a player's upcoming game.
        
        This is the main entry point for pre-game feature generation.
        Ensures no data leakage by only using data before game_date.
        
        Args:
            player_id: NBA player ID
            team_abbr: Player's team abbreviation
            opponent_abbr: Opponent team abbreviation
            game_date: Upcoming game date
            p_play: Probability of playing (from injury status)
            full_history_df: Optional pre-loaded history (OPTIMIZATION)
            
        Returns:
            Dict of features ready for model input
        """
        # Get historical data (before this game - no leakage!)
        if full_history_df is not None:
            # Optimization: Filter in memory
            history = full_history_df[full_history_df['game_date'] < game_date].head(50)
        else:
            # Standard: Query DB
            history = self._get_player_history(player_id, before_date=game_date)
        
        # Base features
        features = {
            'player_id': player_id,
            'team': team_abbr,
            'opponent': opponent_abbr,
            'game_date': game_date,
            'p_play': p_play,
        }

        

        # =====================

        # Minutes Model Features

        # =====================

        

        # Rolling minutes

        min_stats = self._calculate_rolling_stats(history, 'minutes')

        features.update(min_stats)
        
        # [NEW] Season/Long-term Minutes Context (Fix for Bench Bias)
        if not history.empty:
            features['minutes_season'] = history['minutes'].mean()
        else:
            features['minutes_season'] = 0.0

        

        # Recent minutes variance (for uncertainty estimation)

        if len(history) >= 5:

            features['min_std_L5'] = history['minutes'].head(5).std()

        else:

            features['min_std_L5'] = 5.0  # Default variance

        

        # Starter flag

        features['is_starter'] = self._is_starter(history)

        

        # Rest days

        features['rest_days'] = self._calculate_rest_days(history, game_date)
        features['is_b2b'] = 1 if features['rest_days'] == 0 else 0

        

        # =====================

        # PPM Model Features

        # =====================

        

        # Rolling PPM

        ppm_stats = self._calculate_rolling_stats(history, 'ppm')

        features.update(ppm_stats)

        

        # PPM variance

        if len(history) >= 5:

            features['ppm_std_L5'] = history['ppm'].head(5).std()

        else:

            features['ppm_std_L5'] = 0.2  # Default PPM variance

        

        # Rolling points (fallback)

        pts_stats = self._calculate_rolling_stats(history, 'points')

        features.update(pts_stats)
        
        # Create ppg_L5 alias (model expects this name)
        if 'points_L5' in features:
            features['ppg_L5'] = features['points_L5']

        # =====================
        # Recency-Weighted Composites (#5)
        # =====================
        # Weight: L3=50%, L5=30%, L10=20% — reacts faster to form changes
        for stat in ['points', 'minutes', 'ppm']:
            l3 = features.get(f'{stat}_L3', features.get(f'{stat}_L5', 0))
            l5 = features.get(f'{stat}_L5', l3)
            l10 = features.get(f'{stat}_L10', l5)
            features[f'{stat}_weighted'] = (l3 * 0.50) + (l5 * 0.30) + (l10 * 0.20)
        
        # Minutes trend: L3/L10 ratio — detects workload shifts
        l3_min = features.get('minutes_L3', 0)
        l10_min = features.get('minutes_L10', 1)
        features['minutes_trend'] = l3_min / l10_min if l10_min > 0 else 1.0

        
        # =====================
        # Expanded Market Features (Assists, Rebounds, FGA)
        # =====================
        
        # Assists
        ast_stats = self._calculate_rolling_stats(history, 'assists')
        features.update(ast_stats)
        if len(history) >= 5:
            features['ast_std_L5'] = history['assists'].head(5).std()
        else:
            features['ast_std_L5'] = 1.0
        
        # Create ast_L5 alias
        if 'assists_L5' in features:
            features['ast_L5'] = features['assists_L5']

        # Rebounds
        reb_stats = self._calculate_rolling_stats(history, 'rebounds')
        features.update(reb_stats)
        if len(history) >= 5:
            features['reb_std_L5'] = history['rebounds'].head(5).std()
        else:
            features['reb_std_L5'] = 2.0
            
        # Create reb_L5 alias
        if 'rebounds_L5' in features:
            features['reb_L5'] = features['rebounds_L5']

        # Shot Volume (FGA, FTA)
        fga_stats = self._calculate_rolling_stats(history, 'fga')
        features.update(fga_stats)
        
        fta_stats = self._calculate_rolling_stats(history, 'fta')
        features.update(fta_stats)
        
        # [PHASE 13] New Markets Features
        # Threes (FG3M)
        threes_stats = self._calculate_rolling_stats(history, 'fg3m')
        features.update(threes_stats)
        if 'fg3m_L5' in features:
            features['threes_L5'] = features['fg3m_L5'] # Alias
            
        # Blocks (BLK)
        blk_stats = self._calculate_rolling_stats(history, 'blocks')
        features.update(blk_stats)
        if 'blocks_L5' in features:
            features['blk_L5'] = features['blocks_L5'] # Alias for model
        
        # Steals (STL)
        stl_stats = self._calculate_rolling_stats(history, 'steals')
        features.update(stl_stats)
        if 'steals_L5' in features:
            features['stl_L5'] = features['steals_L5'] # Alias for model
            
        # Field Goals Made (FGM)
        fgm_stats = self._calculate_rolling_stats(history, 'fgm')
        features.update(fgm_stats)
        if 'fgm_L5' in features:
            pass # Name matches prediction config 'fgm_L5'

        

        # =====================

        # Market Context Features

        # =====================

        

        odds = self._get_odds_context(team_abbr, game_date)

        features['spread'] = odds['spread']

        features['total'] = odds['total']

        features['is_home'] = odds['is_home']

        

        # Derived context features

        if odds['spread'] is not None:

            # Blowout risk: absolute spread value

            features['blowout_risk'] = abs(odds['spread'])

            # Is favorite?

            features['is_favorite'] = 1 if odds['spread'] < 0 else 0

        else:

            features['blowout_risk'] = 0.0

            features['is_favorite'] = None

        

        if odds['total'] is not None:

            # Pace proxy: high total = fast game

            features['pace_proxy'] = odds['total'] / 220.0  # Normalize around 220

        else:

            features['pace_proxy'] = 1.0

        

        # =====================

        # Matchup Features

        # =====================

        

        features['opp_def_rating'] = self._get_opponent_defense(opponent_abbr)

        

        # Games played (sample size indicator)

        features['games_played'] = len(history)

        

        # =====================

        # STYLE & MATCHUP EDGE FEATURES (NEW)

        # =====================

        style_features = self._calculate_style_edge_features(

            history, team_abbr, opponent_abbr, game_date

        )

        features.update(style_features)

        

        return features

    

    def _get_opponent_advanced_stats(self, opponent_abbr: str,

                                      window: str = 'L10') -> Dict[str, Optional[float]]:

        """Get opponent's advanced stats from team_advanced_stats table.

        

        Args:

            opponent_abbr: Opponent team abbreviation

            window: 'Season', 'L10', or 'L5'

            

        Returns:

            Dict of advanced stats

        """

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM team_advanced_stats 

                WHERE team_abbreviation = ? AND window_type = ?

                ORDER BY stat_date DESC LIMIT 1

            """, (opponent_abbr, window))

            

            row = cursor.fetchone()

            if row:

                return dict(row)

        

        # Return defaults if no data

        return {

            'pace': 100.0,

            'def_rating': 110.0,

            'opp_fg3_pct': 0.36,

            'opp_fg_pct': 0.46,

            'opp_oreb_pct': 0.25,

        }

    

    def _get_team_pace(self, team_abbr: str, window: str = 'L10') -> float:

        """Get team's pace from advanced stats."""

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT pace FROM team_advanced_stats 

                WHERE team_abbreviation = ? AND window_type = ?

                ORDER BY stat_date DESC LIMIT 1

            """, (team_abbr, window))

            

            row = cursor.fetchone()

            return row['pace'] if row and row['pace'] else 100.0

    

    def _get_league_average(self, stat_name: str, window: str = 'L10') -> float:

        """Get league average for a stat."""

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(f"""

                SELECT AVG({stat_name}) as avg_val FROM team_advanced_stats 

                WHERE window_type = ?

            """, (window,))

            

            row = cursor.fetchone()

            return row['avg_val'] if row and row['avg_val'] else 100.0

    

    def _calculate_player_shooting_profile(self, history: pd.DataFrame) -> Dict[str, float]:

        """Calculate player's shooting style profile.

        

        Returns:

            Dict with 3PA rate, paint points ratio, etc.

        """

        if history.empty or len(history) < 3:

            return {

                'fg3a_rate': 0.3,

                'paint_ratio': 0.3,

                'fta_rate': 0.2

            }

        

        recent = history.head(10)

        

        # 3-Point Attempt Rate: 3PA / FGA

        total_fga = recent['fga'].sum()

        total_fg3a = recent['fg3a'].sum()

        fg3a_rate = total_fg3a / total_fga if total_fga > 0 else 0.3

        

        # Paint ratio: (FGM - 3PM) / FGM (proxy for 2PT paint scoring)

        total_fgm = recent['fgm'].sum()

        total_fg3m = recent['fg3m'].sum()

        two_pt_makes = total_fgm - total_fg3m

        paint_ratio = two_pt_makes / total_fgm if total_fgm > 0 else 0.3

        

        # Free throw rate: FTA / FGA

        total_fta = recent['fta'].sum()

        fta_rate = total_fta / total_fga if total_fga > 0 else 0.2

        

        return {

            'fg3a_rate': fg3a_rate,

            'paint_ratio': paint_ratio,

            'fta_rate': fta_rate

        }

    

    def _calculate_style_edge_features(self, history: pd.DataFrame,

                                        team_abbr: str,

                                        opponent_abbr: str,

                                        game_date: str) -> Dict[str, Optional[float]]:

        """Calculate Style & Matchup Edge features.

        

        These capture interaction effects between player style and opponent defense.

        

        Args:

            history: Player's game history

            team_abbr: Player's team

            opponent_abbr: Opponent team

            game_date: Game date

            

        Returns:

            Dict of style edge features

        """

        features = {}

        

        # Get opponent advanced stats

        opp_stats = self._get_opponent_advanced_stats(opponent_abbr, 'L10')

        

        # Get team pace

        team_pace = self._get_team_pace(team_abbr, 'L10')

        opp_pace = opp_stats.get('pace', 100.0)

        

        # Get league averages

        league_avg_def_rtg = self._get_league_average('def_rating', 'L10')

        league_avg_pace = self._get_league_average('pace', 'L10')

        

        # =====================

        # 1. PACE EDGE

        # Is this game faster than player is used to?

        # =====================

        expected_game_pace = (team_pace + opp_pace) / 2

        player_season_avg_pace = league_avg_pace  # Proxy: assume player used to avg

        

        features['pace_edge'] = expected_game_pace - player_season_avg_pace

        features['game_pace'] = expected_game_pace

        features['opp_pace_L10'] = opp_pace

        

        # =====================

        # 2. EFFICIENCY EDGE (Defensive Rating)

        # How much worse is this defense than average?

        # =====================

        opp_def_rating = opp_stats.get('def_rating', 110.0)

        features['efficiency_edge'] = opp_def_rating - league_avg_def_rtg

        features['opp_def_rating_L10'] = opp_def_rating

        

        # =====================

        # 3. THREE-POINT STYLE MATCH (Guards/Wings)

        # High volume shooter vs defense that allows 3s

        # =====================

        player_profile = self._calculate_player_shooting_profile(history)

        opp_fg3_pct_allowed = opp_stats.get('opp_fg3_pct', 0.36)

        

        # Style match: player 3PA rate * opponent 3P% allowed

        # Higher = good matchup for shooter

        features['three_pt_style_match'] = player_profile['fg3a_rate'] * opp_fg3_pct_allowed

        features['player_fg3a_rate'] = player_profile['fg3a_rate']

        features['opp_fg3_pct_allowed'] = opp_fg3_pct_allowed

        

        # =====================

        # 4. PAINT/REBOUND MATCH (Bigs)

        # Paint scorer vs weak interior defense

        # =====================

        opp_fg_pct_allowed = opp_stats.get('opp_fg_pct', 0.46)

        opp_oreb_allowed = opp_stats.get('opp_oreb_pct', 0.25)

        

        # Style match: paint ratio * opponent FG% allowed

        features['paint_style_match'] = player_profile['paint_ratio'] * opp_fg_pct_allowed

        features['player_paint_ratio'] = player_profile['paint_ratio']

        features['opp_paint_fg_allowed'] = opp_fg_pct_allowed

        

        # Rebounding edge for bigs

        features['opp_oreb_allowed'] = opp_oreb_allowed

        

        # =====================

        # 5. FREE THROW OPPORTUNITY

        # Aggressive players vs fouling teams

        # =====================

        features['ft_opportunity'] = player_profile['fta_rate']

        

        return features

    

    def build_training_dataset(self, player_ids: List[int] = None,

                                min_games: int = 10,

                                seasons: List[str] = None) -> pd.DataFrame:

        """Build training dataset from historical game logs.

        

        For each game in history, builds features using only prior data.

        

        Args:

            player_ids: List of player IDs (default: all with sufficient data)

            min_games: Minimum games required for a player

            seasons: Seasons to include (default: current)

            

        Returns:

            DataFrame with features and targets (actual minutes, points, ppm)

        """

        if seasons is None:
            seasons = [self.config.current_season]
        
        # Get players with sufficient data
        if player_ids is None:
            with self.db.get_connection() as conn:
                if seasons == ['ALL'] or seasons == 'ALL':
                     # Fetch ALL active players with enough games, regardless of season
                     query = """
                        SELECT player_id, COUNT(*) as game_count 
                        FROM player_logs 
                        GROUP BY player_id 
                        HAVING game_count >= ?
                    """
                     df = pd.read_sql_query(query, conn, params=[min_games])
                else:
                    query = f"""
                        SELECT player_id, COUNT(*) as game_count 
                        FROM player_logs 
                        WHERE season IN ({','.join('?' * len(seasons))})
                        GROUP BY player_id 
                        HAVING game_count >= ?
                    """
                    df = pd.read_sql_query(query, conn, params=seasons + [min_games])

                player_ids = df['player_id'].tolist()

        

        print(f"[Features] Building training data for {len(player_ids)} players...")

        

        all_rows = []

        

        for i, player_id in enumerate(player_ids):

            # Get all games for this player
            with self.db.get_connection() as conn:
                query = """
                    SELECT * FROM player_logs 
                    WHERE player_id = ? 
                    ORDER BY game_date DESC -- Optim: Sort DESC for history slicing
                """
                all_logs = pd.read_sql_query(query, conn, params=[player_id])
            
            # For training loop, we need them ASCENDING to build timeline
            games_to_process = all_logs.sort_values('game_date', ascending=True)
            
            if len(games_to_process) < min_games:
                continue
            
            # For each game (starting from min_games), build features
            # history_df needs to be DESC for head(50) logic in build_features_for_player
            history_source = all_logs 

            for idx in range(min_games, len(games_to_process)):
                game = games_to_process.iloc[idx]
                
                # Build features using only prior games
                # Pass full history (descending) for optimization
                features = self.build_features_for_player(
                    player_id=player_id,
                    team_abbr=game['team_abbreviation'] or '',
                    opponent_abbr=game['opponent_abbreviation'] or '',
                    game_date=game['game_date'],
                    p_play=1.0,  # We know they played
                    full_history_df=history_source
                )

                

                # Add targets (actual values from this game)

                features['target_minutes'] = game['minutes']

                features['target_points'] = game['points']

                features['target_ppm'] = game['ppm']
                
                features['target_assists'] = game['assists']
                
                features['target_rebounds'] = game['rebounds']

                

                all_rows.append(features)

            

            if (i + 1) % 50 == 0:

                print(f"  Processed {i + 1}/{len(player_ids)} players...")

        

        df = pd.DataFrame(all_rows)
        
        # Create aliases for model-expected feature names
        if 'points_L5' in df.columns:
            df['ppg_L5'] = df['points_L5']
        if 'assists_L5' in df.columns:
            df['ast_L5'] = df['assists_L5']
        if 'rebounds_L5' in df.columns:
            df['reb_L5'] = df['rebounds_L5']

        print(f"   Built {len(df)} training samples")

        

        return df

    

    def get_feature_columns(self, model_type: str = 'minutes') -> List[str]:

        """Get list of feature columns for a model.

        

        Args:

            model_type: 'minutes' or 'ppm'

            

        Returns:

            List of feature column names

        """

        base_features = [

            'is_starter',

            'rest_days',
            'is_b2b',

            'is_home',

            'games_played',

        ]

        

        if model_type == 'minutes':

            return base_features + [

                'minutes_L3', 'minutes_L5', 'minutes_L10',

                'min_std_L5',

                'spread', 'blowout_risk', 'is_favorite',

            ]

        elif model_type == 'ppm':

            return base_features + [

                'ppm_L3', 'ppm_L5', 'ppm_L10',

                'ppm_std_L5',

                'total', 'pace_proxy',

                'opp_def_rating',

            ]

        else:

            raise ValueError(f"Unknown model type: {model_type}")





# Convenience function

def get_feature_engineer() -> FeatureEngineer:

    """Get feature engineer instance."""

    return FeatureEngineer()

