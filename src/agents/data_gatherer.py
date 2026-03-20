"""
Agent 1: Data & Context Gatherer

Role: Reality anchor
Failure mode prevented: Outdated stats, fake rotations, fantasy minutes

This agent does ZERO prediction.
It only gathers and validates data for downstream agents.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..utils.database import DatabaseManager
from ..utils.config import get_config


class DataGathererAgent:
    """
    Reality anchor agent that gathers and validates all context.
    
    Does NOT make predictions - only collects validated data.
    Filters to rotation players only (starters + 6-8th man).
    """
    
    # Minimum minutes to be considered rotation player
    MIN_ROTATION_MINUTES = 10
    
    # Minimum games to have reliable data
    MIN_GAMES_REQUIRED = 5
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize data gatherer agent."""
        self.db = db or DatabaseManager()
        self.config = get_config()
    
    # ========================================
    # PLAYER CONTEXT GATHERING
    # ========================================
    
    def gather_player_context(self, player_id: int, window_long: int = 15, 
                               window_short: int = 5, date_limit: str = None) -> Dict[str, Any]:
        """
        Collect comprehensive player context for prediction.
        
        Args:
            player_id: NBA player ID
            window_long: Long-term window (default 15 games)
            window_short: Short-term window (default 5 games)
            date_limit: [NEW] Optional date cutoff (YYYY-MM-DD) for data leakage prevention.
            
        Returns:
            Dict with all player context
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Apply strict date filtering
            if date_limit:
                query = """
                    SELECT 
                        game_date, minutes, points, assists, rebounds,
                        fga, fta, turnovers, is_home, is_starter,
                        team_abbreviation, opponent_abbreviation,
                        fg3m, blocks, steals, fgm
                    FROM player_logs
                    WHERE player_id = ?
                    AND minutes > 0
                    AND game_date < ? 
                    ORDER BY game_date DESC
                    LIMIT ?
                """
                params = (player_id, date_limit, window_long)
            else:
                query = """
                    SELECT 
                        game_date, minutes, points, assists, rebounds,
                        fga, fta, turnovers, is_home, is_starter,
                        team_abbreviation, opponent_abbreviation,
                        fg3m, blocks, steals, fgm
                    FROM player_logs
                    WHERE player_id = ?
                    AND minutes > 0
                    ORDER BY game_date DESC
                    LIMIT ?
                """
                params = (player_id, window_long)
                
            cursor.execute(query, params)
            
            games = [dict(row) for row in cursor.fetchall()]
        
        if len(games) < self.MIN_GAMES_REQUIRED:
            return self._default_player_context(player_id)
        
        # Use Helper
        return self._compute_context_from_games(player_id, games, reference_date=date_limit)

    def _get_player_position(self, player_id: int) -> str:
        """Get player position from database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT position FROM players WHERE player_id = ?", (player_id,))
            row = cursor.fetchone()
            return row['position'] if row else 'G' # Default to G if unknown
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction from recent values.
        
        Returns: 'up', 'down', or 'stable'
        """
        if len(values) < 3:
            return 'stable'
        
        # Split into halves and compare
        mid = len(values) // 2
        recent_half = np.mean(values[:mid])
        older_half = np.mean(values[mid:])
        
        diff = recent_half - older_half
        threshold = np.std(values) * 0.3  # Significant if > 0.3 std
        
        if diff > threshold:
            return 'up'
        elif diff < -threshold:
            return 'down'
        return 'stable'
    
    def _calculate_usage_proxy(self, games: List[Dict]) -> float:
        """Calculate usage proxy: (FGA + 0.44*FTA + TOV) / Minutes."""
        total_usage = 0
        total_minutes = 0
        
        for g in games:
            fga = g.get('fga') or 0
            fta = g.get('fta') or 0
            tov = g.get('turnovers') or 0
            minutes = g.get('minutes') or 1
            
            total_usage += fga + 0.44 * fta + tov
            total_minutes += minutes
        
        return total_usage / total_minutes if total_minutes > 0 else 0.15
    
    def _calculate_rest_days(self, games: List[Dict], reference_date: Optional[str] = None) -> int:
        """Calculate days since last game relative to the target slate date when available."""
        if not games or not games[0].get('game_date'):
            return 1
        
        try:
            last_game = datetime.strptime(games[0]['game_date'], '%Y-%m-%d')
            if reference_date:
                slate_date = datetime.strptime(reference_date, '%Y-%m-%d')
            else:
                slate_date = datetime.now()
            return max(0, (slate_date - last_game).days)
        except (ValueError, TypeError, KeyError) as e:
            print(f"  [WARN] rest_days calc failed for game_date={games[0].get('game_date')}: {e}")
            return 1
    
    def _default_player_context(self, player_id: int) -> Dict[str, Any]:
        """Return default context for players with insufficient data."""
        return {
            'player_id': player_id,
            'games_available': 0,
            'minutes_L5': 0,
            'minutes_L15': 0,
            'minutes_trend': 'stable',
            'minutes_std': 0,
            'points_L5': 0,
            'points_L15': 0,
            'points_trend': 'stable',
            'points_std': 0,
            'usage_proxy_L5': 0,
            'usage_proxy_L15': 0,
            'starter_rate': 0,
            'is_starter': False,
            'home_ppg': None,
            'away_ppg': None,
            'rest_days': 1,
            'team': None,
            'insufficient_data': True,
        }
    
    # ========================================
    # CONTEXT INFERENCE (PHASE 6)
    # ========================================
    
    def infer_context_signals(self, player_id: int, reference_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Infer context signals from data patterns (Instruction 3.1).
        
        Detects:
        - Rotation Tightening (bench minutes dropping)
        - Usage Spikes (L3 vs L15 abnormality)
        - Role Changes (Starter <-> Bench)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get last 15 games
            cursor.execute("""
                SELECT game_date, minutes, points, is_starter, fga, fta, turnovers
                FROM player_logs
                WHERE player_id = ? AND minutes > 0
                ORDER BY game_date DESC
                LIMIT 15
            """, (player_id,))
            
            games = [dict(row) for row in cursor.fetchall()]
        

        
        signals = {
            'rotation_tightening': False,
            'usage_spike': False,
            'role_change': False,
            'minutes_volatility': 'normal'
        }
        
        if len(games) < 5:
            return signals
            
        recent = games[:3]
        older = games[3:]
        
        if not older:
            return signals
            
        # 1. Detect Rotation Tightening
        # If player is bench and minutes are dropping significantly
        is_bench = np.mean([g['is_starter'] for g in recent]) < 0.5
        avg_min_recent = np.mean([g['minutes'] for g in recent])
        avg_min_older = np.mean([g['minutes'] for g in older])
        
        if is_bench and avg_min_recent < avg_min_older * 0.75:
            signals['rotation_tightening'] = True
            
        # 2. Detect Usage Spike (Abnormal increase)
        # Calculate usage proxy for each game
        recent_usage = [self._calculate_usage_proxy([g]) for g in recent]
        older_usage = [self._calculate_usage_proxy([g]) for g in older]
        
        avg_usage_recent = np.mean(recent_usage)
        avg_usage_older = np.mean(older_usage)
        
        if avg_usage_recent > avg_usage_older * 1.25:
            signals['usage_spike'] = True
            
        # 3. Detect Role Change
        recent_starter = np.mean([g['is_starter'] for g in recent]) > 0.5
        older_starter = np.mean([g['is_starter'] for g in older]) > 0.5
        
        if recent_starter != older_starter:
            signals['role_change'] = True
            
        return signals
    
    # ========================================
    # MATCH CONTEXT GATHERING
    # ========================================
    
    def gather_match_context(self, team_abbr: str, opponent_abbr: str,
                              game_date: str = None) -> Dict[str, Any]:
        """
        Collect match-level context.
        
        Gathers:
        - Opponent defensive rating
        - Pace (both teams)
        - Vegas spread & total (if available)
        - Blowout probability
        
        Args:
            team_abbr: Player's team
            opponent_abbr: Opponent team
            game_date: Game date (default: today)
            
        Returns:
            Dict with match context
        """
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        
        context = {
            'team': team_abbr,
            'opponent': opponent_abbr,
            'game_date': game_date,
        }
        
        # Get opponent defensive stats
        opp_defense = self._get_team_defense(opponent_abbr)
        context.update({
            'opp_def_rating': opp_defense.get('def_rating', 110.0),
            'opp_pace': opp_defense.get('pace', 100.0),
            'opp_fg_pct_allowed': opp_defense.get('opp_fg_pct', 0.46),
            'opp_3pt_pct_allowed': opp_defense.get('opp_fg3_pct', 0.36),
            'dvp_stats': self._get_dvp_stats(opponent_abbr, game_date) # New Position-Specific Defense
        })
        
        # Get team pace
        team_stats = self._get_team_defense(team_abbr)
        context['team_pace'] = team_stats.get('pace', 100.0)
        
        # Expected game pace (average of both)
        context['expected_pace'] = (context['team_pace'] + context['opp_pace']) / 2
        
        # Get Vegas lines if available
        vegas = self._get_vegas_context(team_abbr, opponent_abbr, game_date)
        context.update(vegas)
        
        # Calculate blowout probability from spread
        spread = context.get('spread', 0)
        context['blowout_probability'] = self._estimate_blowout_prob(spread)
        
        return context
    
    def _get_team_defense(self, team_abbr: str) -> Dict[str, float]:
        """Get team's defensive stats from database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for window_type in ('L10', 'L5', 'Season'):
                cursor.execute("""
                    SELECT def_rating, pace, opp_fg_pct, opp_fg3_pct
                    FROM team_advanced_stats
                    WHERE team_abbreviation = ?
                    AND window_type = ?
                    ORDER BY stat_date DESC
                    LIMIT 1
                """, (team_abbr, window_type))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'def_rating': row['def_rating'] or 110.0,
                        'pace': row['pace'] or 100.0,
                        'opp_fg_pct': row['opp_fg_pct'] or 0.46,
                        'opp_fg3_pct': row['opp_fg3_pct'] or 0.36,
                    }
            
            return {'def_rating': 110.0, 'pace': 100.0, 'opp_fg_pct': 0.46, 'opp_fg3_pct': 0.36}

    def _get_dvp_stats(self, opponent_abbr: str, game_date: str = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate Defense vs Position (DvP) Multipliers for Pts, Ast, Reb.
        Delegates to DvPCalculator model.
        """
        from ..models.dvp_model import DvPCalculator
        if not hasattr(self, '_dvp_calculator'):
            self._dvp_calculator = DvPCalculator(self.db)
        calculator = self._dvp_calculator
        
        # Get all multipliers
        # Note: dvp_model returns {Stat: {Team: {Pos: Mult}}}
        # We need {Stat: {Pos: Mult}} for this specific opponent
        
        all_mults = calculator.get_dvp_multipliers()
        
        result = {}
        for stat, team_map in all_mults.items():
            if opponent_abbr in team_map:
                result[stat] = team_map[opponent_abbr]
            else:
                # Default if team not found
                result[stat] = {'G': 1.0, 'F': 1.0, 'C': 1.0}
                
        return result
    
    def _get_vegas_context(self, team: str, opponent: str, game_date: str) -> Dict[str, Any]:
        """Get Vegas lines if available."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT spread_home, total
                FROM odds_snapshots
                WHERE game_date = ?
                AND (home_team = ? OR away_team = ?)
                ORDER BY snapshot_time DESC
                LIMIT 1
            """, (game_date, team, team))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'spread': row['spread_home'] or 0,
                    'total': row['total'] or 220,
                    'has_vegas': True,
                }
            
            return {'spread': 0, 'total': 220, 'has_vegas': False}
    
    def _estimate_blowout_prob(self, spread: float) -> float:
        """Estimate blowout probability from spread."""
        # Larger spread = higher blowout chance
        abs_spread = abs(spread or 0)
        
        if abs_spread < 5:
            return 0.05
        elif abs_spread < 10:
            return 0.15
        elif abs_spread < 15:
            return 0.30
        else:
            return 0.45
            
    def get_player_vs_opponent_history(self, player_id: int, opponent_abbr: str, limit: int = 3) -> List[Dict]:
        """
        Get player's recent game logs against a specific opponent.
        Crucial for 'Knowledge' reasoning (e.g. 'Avg 28 PPG vs BOS last 3 games').
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_date, minutes, points, assists, rebounds, fga, is_home
                FROM player_logs
                WHERE player_id = ?
                AND opponent_abbreviation = ?
                AND minutes > 0
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, opponent_abbr, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # ========================================
    # ROTATION FILTERING (CRITICAL)
    # ========================================
    
    def filter_rotation_players(self, roster: List[Dict], 
                                  team_abbr: str,
                                  injuries: Dict[int, float] = None) -> List[Dict]:
        """
        Filter roster to only rotation players.
        
        Only pass forward:
        - Starters
        - 6th-8th man
        - Confirmed injury replacements
        
        If a book wouldn't hang a prop → exclude the player.
        
        Args:
            roster: Full team roster
            team_abbr: Team abbreviation
            injuries: Dict of {player_id: probability_of_playing}
            
        Returns:
            Filtered list of rotation players only
        """
        injuries = injuries or {}
        rotation = []
        out_starters = []
        
        for player in roster:
            player_id = player.get('player_id')
            if not player_id:
                continue
            
            # Get player's recent context
            context = self.gather_player_context(player_id, window_long=10, window_short=5)
            roster_starter = bool(player.get('is_starter')) if 'is_starter' in player else False
            context['roster_is_starter'] = roster_starter
            is_starter = bool(context.get('is_starter')) or roster_starter

            if player_id in injuries:
                p_play = injuries[player_id]
                if p_play < 0.3:
                    if is_starter:
                        out_starters.append({
                            'player_id': player_id,
                            'minutes_L5': context.get('minutes_L5', 0),
                            'position': context.get('position', 'G')
                        })
                    print(f"  [FILTER] Excluding {player.get('full_name')} (Injured, p_play={p_play})")
                    continue
            
            # Filter criteria
            is_rotation = self._is_rotation_player(context)
            
            if is_rotation:
                player['context'] = context
                rotation.append(player)

        starters = [p for p in rotation if (p.get('context', {}).get('is_starter') or p.get('context', {}).get('roster_is_starter'))]
        bench = [p for p in rotation if not (p.get('context', {}).get('is_starter') or p.get('context', {}).get('roster_is_starter'))]
        bench_sorted = sorted(bench, key=lambda p: p.get('context', {}).get('minutes_L5', 0), reverse=True)

        selected = starters + bench_sorted[:3]
        selected_ids = {p.get('player_id') for p in selected if p.get('player_id')}

        if out_starters:
            out_starters_sorted = sorted(out_starters, key=lambda x: x.get('minutes_L5', 0), reverse=True)
            needed_pos = out_starters_sorted[0].get('position', 'G')
            pos_group = 'G'
            if 'C' in needed_pos:
                pos_group = 'C'
            elif 'F' in needed_pos:
                pos_group = 'F'

            replacement = None
            for p in bench_sorted[3:]:
                if p.get('player_id') in selected_ids:
                    continue
                p_pos = p.get('context', {}).get('position', 'G')
                p_group = 'G'
                if 'C' in p_pos:
                    p_group = 'C'
                elif 'F' in p_pos:
                    p_group = 'F'
                if p_group == pos_group:
                    replacement = p
                    break

            if replacement is None:
                for p in bench_sorted[3:]:
                    if p.get('player_id') not in selected_ids:
                        replacement = p
                        break

            if replacement is not None and replacement.get('player_id') not in selected_ids:
                selected.append(replacement)

        return selected
    
    def _is_rotation_player(self, context: Dict) -> bool:
        """
        Determine if player is a rotation player worthy of prediction.
        
        Criteria:
        - Has sufficient games
        - Average minutes >= MIN_ROTATION_MINUTES
        - Reasonable starter rate or consistent bench role
        """
        if context.get('insufficient_data'):
            return False
        
        if context.get('games_available', 0) < self.MIN_GAMES_REQUIRED:
            return False
        
        # Check minutes threshold
        avg_minutes = context.get('minutes_L5', 0)
        if avg_minutes < self.MIN_ROTATION_MINUTES:
            return False
        
        # Must have some role stability
        minutes_std = context.get('minutes_std', 10)
        if minutes_std > 12:  # Too volatile
            return False
        
        return True
    
    def get_eligible_players(self, team_id: int, team_abbr: str,
                              injuries: Dict[int, float] = None) -> List[Dict]:
        """
        Get all eligible players for prediction from a team.
        Also updates player positions in DB from the roster.
        
        This is the main entry point for Agent 1.
        
        Args:
            team_id: NBA team ID
            team_abbr: Team abbreviation
            injuries: Injury map {player_id: p_play}
            
        Returns:
            List of eligible players with full context
        """
        from ..ingestion.nba_ingestion import NBAIngestion
        
        nba = NBAIngestion(self.db)
        roster = nba.get_team_roster(team_id)
        
        # [NEW] Update positions in DB
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for p in roster:
                # [FIX] Inject team abbreviation for downstream usage
                p['team'] = team_abbr
                
                if p.get('position') and p.get('player_id'):
                    cursor.execute(
                        "UPDATE players SET position = ? WHERE player_id = ?", 
                        (p['position'], p['player_id'])
                    )
            conn.commit()
        
        return self.filter_rotation_players(roster, team_abbr, injuries)


    def gather_batch_player_contexts(self, player_ids: List[int], date_limit: str = None) -> Dict[int, Dict]:
        """
        Batch optimized version of gather_player_context.
        Fetch all logs for all players in ONE query.
        """
        contexts = {}
        if not player_ids: return contexts
        
        placeholders = ",".join(["?"] * len(player_ids))
        window_long = 15
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Complex query: We need top N rows per group. 
            # SQLite doesn't have PARTITION BY in older versions, but typically we can just fetch potentially more data and process in python for stability.
            # Fetching last 20 games for ALL players in list.
            
            if date_limit:
                 query = f"""
                    SELECT player_id, game_date, minutes, points, assists, rebounds, fga, fta, turnovers, is_home, is_starter, team_abbreviation, fg3m, blocks, steals, fgm, opponent_abbreviation
                    FROM player_logs
                    WHERE player_id IN ({placeholders})
                    AND minutes > 0
                    AND game_date < ?
                    ORDER BY game_date DESC
                 """
                 params = tuple(player_ids) + (date_limit,)
            else:
                 query = f"""
                    SELECT player_id, game_date, minutes, points, assists, rebounds, fga, fta, turnovers, is_home, is_starter, team_abbreviation, fg3m, blocks, steals, fgm, opponent_abbreviation
                    FROM player_logs
                    WHERE player_id IN ({placeholders})
                    AND minutes > 0
                    ORDER BY game_date DESC
                 """
                 params = tuple(player_ids)
            
            cursor.execute(query, params)
            all_rows = cursor.fetchall()
            
        # Group by player
        games_by_player = {}
        for row in all_rows:
            pid = row['player_id']
            if pid not in games_by_player: games_by_player[pid] = []
            if len(games_by_player[pid]) < window_long:
                 games_by_player[pid].append(dict(row))
                 
        # Compute context for each
        for pid in player_ids:
            games = games_by_player.get(pid, [])
            # Re-use the computation logic (extract to helper if needed, or dupe for speed)
            # Refactoring gather_player_context to use a helper `_compute_context_from_games`
            contexts[pid] = self._compute_context_from_games(pid, games, reference_date=date_limit)
            
        return contexts

    def _compute_context_from_games(self, player_id: int, games: List[Dict], reference_date: str = None) -> Dict[str, Any]:
        """Compute context stats from list of game logs"""
        window_short = 5
        window_long = 15
        
        if len(games) < self.MIN_GAMES_REQUIRED:
            return self._default_player_context(player_id)
            
        recent_games = games[:window_short]
        all_games = games[:window_long]
        older_games = games[window_short:window_long] if len(games) > window_short else all_games
        usage_recent = self._calculate_usage_proxy(recent_games)
        usage_long = self._calculate_usage_proxy(all_games)
        usage_older = self._calculate_usage_proxy(older_games) if older_games else usage_long
        
        # Calculate metrics (Same as gather_player_context)
        context = {
            'player_id': player_id,
            'games_available': len(games),
            'minutes_L5': np.mean([g['minutes'] for g in recent_games]),
            'minutes_L15': np.mean([g['minutes'] for g in all_games]),
            'minutes_season': np.mean([g['minutes'] for g in games]) if games else 0, # [NEW] Full history avg
            'minutes_trend': self._calculate_trend([g['minutes'] for g in all_games]),
            'minutes_std': np.std([g['minutes'] for g in all_games]),
            'points_L5': np.mean([g['points'] for g in recent_games]),
            'points_L15': np.mean([g['points'] for g in all_games]),
            'points_max_L5': np.max([g['points'] for g in recent_games]) if recent_games else 0,
            'points_max_L15': np.max([g['points'] for g in all_games]) if all_games else 0,
            'points_trend': self._calculate_trend([g['points'] for g in all_games]),
            'points_std': np.std([g['points'] for g in all_games]),
            
            # [FIX] Added missing Assits/Rebounds Context
            'ast_L5': np.mean([g['assists'] for g in recent_games]),
            'ast_L15': np.mean([g['assists'] for g in all_games]),
            'ast_std_L5': np.std([g['assists'] for g in recent_games]),
            
            'reb_L5': np.mean([g['rebounds'] for g in recent_games]),
            'reb_L15': np.mean([g['rebounds'] for g in all_games]),
            'reb_std_L5': np.std([g['rebounds'] for g in recent_games]),
            
            # New Market Stats
            'threes_L5': np.mean([g.get('fg3m', 0) or 0 for g in recent_games]),
            'threes_L15': np.mean([g.get('fg3m', 0) or 0 for g in all_games]),
            'threes_std_L5': np.std([g.get('fg3m', 0) or 0 for g in recent_games]),
            
            'blocks_L5': np.mean([g.get('blocks', 0) or 0 for g in recent_games]),
            'blocks_L15': np.mean([g.get('blocks', 0) or 0 for g in all_games]),
            'blocks_std_L5': np.std([g.get('blocks', 0) or 0 for g in recent_games]),
            
            'steals_L5': np.mean([g.get('steals', 0) or 0 for g in recent_games]),
            'steals_L15': np.mean([g.get('steals', 0) or 0 for g in all_games]),
            'steals_std_L5': np.std([g.get('steals', 0) or 0 for g in recent_games]),
            
            'fgm_L5': np.mean([g.get('fgm', 0) or 0 for g in recent_games]),
            'fgm_L15': np.mean([g.get('fgm', 0) or 0 for g in all_games]),
            'fgm_std_L5': np.std([g.get('fgm', 0) or 0 for g in recent_games]),
            
            # Recent Logs for Hit Rate Calculation (Phase 11)
            'recent_logs': [
                {
                    'date': g['game_date'],
                    'points': g['points'],
                    'assists': g['assists'],
                    'rebounds': g['rebounds'],
                    'fg3m': g.get('fg3m', 0) or 0,
                    'blk': g.get('blocks', 0) or 0,
                    'stl': g.get('steals', 0) or 0,
                    'fgm': g.get('fgm', 0) or 0,
                    'minutes': g['minutes']
                }
                for g in games[:10] # Last 10 games
            ],
            
            'usage_proxy_L5': usage_recent,
            'usage_proxy_L15': usage_long,
            'usage_proxy_delta': usage_recent - usage_older,
            'starter_rate': np.mean([g['is_starter'] or 0 for g in all_games]),
            'is_starter': np.mean([g['is_starter'] or 0 for g in recent_games]) > 0.5,
            'home_ppg': np.mean([g['points'] for g in all_games if g['is_home']]) if any(g['is_home'] for g in all_games) else None,
            'away_ppg': np.mean([g['points'] for g in all_games if not g['is_home']]) if any(not g['is_home'] for g in all_games) else None,
            'rest_days': self._calculate_rest_days(games, reference_date=reference_date),
            'team': games[0]['team_abbreviation'] if games else None,
            # Position fetch is fast/cached elsewhere usually, but here we invoke it
            'position': self._get_player_position(player_id),
            'role_minutes_delta': np.mean([g['minutes'] for g in recent_games]) - np.mean([g['minutes'] for g in older_games]) if older_games else 0,
            'inferred_signals': self.infer_context_signals(player_id) 
        }
        
        # Create aliases to match training data field names
        context['ppg_L5'] = context['points_L5']
        context['ppg_season'] = context['points_L15']  # Best proxy for season avg
        
        # ==========================================
        # SUB-AGENT DATA ENRICHMENT
        # ==========================================
        
        # A) recent_games — NarrativeDetector needs this for bounce-back detection
        context['recent_games'] = [
            {
                'points': g['points'],
                'assists': g['assists'],
                'rebounds': g['rebounds'],
                'minutes': g['minutes'],
                'fg3m': g.get('fg3m', 0) or 0,
                'date': g['game_date']
            }
            for g in recent_games  # Last 5
        ]
        
        # B) player_role — StatFragilityAgent uses this for kill script analysis
        #    Derived from usage proxy + minutes
        usage = context.get('usage_proxy_L5', 0)
        mins = context.get('minutes_L5', 0)
        pts = context.get('points_L5', 0)
        
        if pts >= 22 and mins >= 32:
            context['player_role'] = 'star'
        elif pts >= 15 and mins >= 28:
            context['player_role'] = 'starter'
        elif mins >= 20:
            context['player_role'] = 'rotation'
        else:
            context['player_role'] = 'bench'
        
        # C) foul_rate — StatFragilityAgent uses this for foul trouble script
        foul_counts = []
        for g in all_games:
            # Estimate fouls from turnovers + FTA pattern (no direct PF column)
            # Use turnovers as proxy — higher turnovers correlate with foul trouble
            foul_counts.append(g.get('turnovers', 0) or 0)
        context['foul_rate'] = np.mean(foul_counts) if foul_counts else 2.0
        
        # D) stats dict — some sub-agents look in context['stats'] for data
        context['stats'] = {
            'l5_ppg': context['points_L5'],
            'season_ppg': context['ppg_season'],
            'l5_std_pts': context['points_std'],
            'l5_ast': context['ast_L5'],
            'l5_reb': context['reb_L5'],
            'fouls_L5': context['foul_rate'],
            'l5_minutes': context['minutes_L5'],
        }
        
        return context

# Convenience function
def get_data_gatherer() -> DataGathererAgent:
    """Get data gatherer agent instance."""
    return DataGathererAgent()
