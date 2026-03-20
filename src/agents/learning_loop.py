"""
Agent 5: Post-Game Learning Loop
=================================
Role: Self-learning system that updates priors from completed games

Core Principles (Bengio - Uncertainty & Generalization):
- "Learn from every prediction, especially the wrong ones"
- Track systematic biases in each component
- Update priors without touching live predictions
- Maintain uncertainty until certain

Learning Tracks:
1. Minutes Model: Over/under-predicting playing time
2. Usage Model: Role-based usage accuracy
3. Efficiency Model: Matchup adjustment calibration
4. Variance Model: Archetype volatility accuracy
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PredictionError:
    """Error decomposition for a single prediction"""
    player_id: int
    game_date: str
    predicted_points: float
    actual_points: float
    total_error: float
    
    # Component errors (how much each contributed)
    minutes_error: float  # Due to minutes prediction
    usage_error: float    # Due to usage prediction
    efficiency_error: float  # Due to efficiency prediction
    variance_captured: bool  # Was actual within 1σ?
    
    # Context
    archetype: str
    opponent: str
    spread: float


@dataclass
class BiasReport:
    """Systematic bias analysis"""
    component: str
    sample_size: int
    mean_error: float
    std_error: float
    bias_direction: str  # 'over', 'under', 'neutral'
    is_significant: bool  # p < 0.05
    recommended_adjustment: float


class LearningLoopAgent:
    """
    Agent 5: The Learning Loop
    
    Post-game analysis that tracks errors, detects biases,
    and updates model priors for future predictions.
    
    NEVER touches live predictions - only updates parameters
    for FUTURE use based on PAST performance.
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        self._ensure_learning_tables()

    @staticmethod
    def _add_column_if_missing(cursor, table_name: str, column_name: str, column_type: str) -> None:
        """Add a column unless it already exists."""
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" in str(exc).lower():
                return
            raise
    
    def _ensure_learning_tables(self):
        """Create learning tracking tables if they don't exist"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prediction tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date TEXT,
                    player_id INTEGER,
                    player_name TEXT,
                    game_date TEXT,
                    
                    -- Predictions
                    pred_points REAL,
                    pred_std REAL,
                    pred_minutes REAL,
                    pred_usage REAL,
                    pred_efficiency REAL,
                    
                    -- Market Data (New for Phase 3)
                    market_line REAL,
                    market_odds INTEGER,
                    bet_direction TEXT,
                    bet_confidence TEXT,
                    
                    -- Actuals (filled after game)
                    actual_points REAL,
                    actual_minutes REAL,
                    
                    -- Context
                    archetype TEXT,
                    opponent TEXT,
                    spread REAL,
                    edge_source TEXT,
                    regime_status TEXT, -- Added missing column definition
                    
                    -- Analysis
                    error REAL,
                    within_1std INTEGER,
                    within_2std INTEGER,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add columns if table exists (Migration)
            self._add_column_if_missing(cursor, "prediction_log", "market_line", "REAL")
            self._add_column_if_missing(cursor, "prediction_log", "market_odds", "INTEGER")
            self._add_column_if_missing(cursor, "prediction_log", "bet_direction", "TEXT")
            self._add_column_if_missing(cursor, "prediction_log", "bet_confidence", "TEXT")
            self._add_column_if_missing(cursor, "prediction_log", "regime_status", "TEXT")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date TEXT,
                    player_id INTEGER,
                    player_name TEXT,
                    team TEXT,
                    opponent TEXT,
                    rejection_stage TEXT,
                    reason TEXT,
                    proposed_market TEXT,
                    proposed_line REAL,
                    consensus_level TEXT,
                    player_trust_score REAL,
                    market_trust_score REAL,
                    candidate_rank INTEGER,
                    validator_vote_summary TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self._add_column_if_missing(cursor, "prediction_rejections", "player_trust_score", "REAL")
            self._add_column_if_missing(cursor, "prediction_rejections", "market_trust_score", "REAL")
            self._add_column_if_missing(cursor, "prediction_rejections", "candidate_rank", "INTEGER")
            self._add_column_if_missing(cursor, "prediction_rejections", "validator_vote_summary", "TEXT")
            
            # Bias tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bias_tracker (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TEXT,
                    component TEXT,
                    segment TEXT,  -- e.g., 'volume_star', 'vs_elite_defense', 'OVER_BET', 'HIGH_CONF'
                    sample_size INTEGER,
                    mean_bias REAL,
                    std_error REAL,
                    is_significant INTEGER,
                    adjustment_applied REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Calibration Buckets table (New for Phase 6)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibration_buckets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TEXT,
                    prob_bucket TEXT,  -- '55-60', '60-65', '65-70', '70+'
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    actual_hit_rate REAL,
                    target_hit_rate REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Edge Performance table (New for Phase 7)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS edge_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TEXT,
                    edge_source TEXT,
                    bets_placed INTEGER,
                    wins INTEGER,
                    roi REAL,
                    avg_edge REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Regime Log table (New for Phase 7)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detection_date TEXT,
                    regime_status TEXT,  -- 'stable', 'shifting', 'decaying'
                    warning_flags TEXT,  -- JSON list of flags
                    action_taken TEXT,   -- 'tighten_gates', 'reduce_risk'
                    over_under_bias REAL,
                    archetype_decay TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Prior updates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prior_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    update_date TEXT,
                    component TEXT,
                    parameter TEXT,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    def log_rejection(self,
                      player_id: int,
                      player_name: str,
                      team: str,
                      opponent: str,
                      rejection_stage: str,
                      reason: str,
                      proposed_market: Optional[str] = None,
                      proposed_line: Optional[float] = None,
                      consensus_level: Optional[str] = None,
                      player_trust_score: Optional[float] = None,
                      market_trust_score: Optional[float] = None,
                      candidate_rank: Optional[int] = None,
                      validator_vote_summary: Optional[str] = None) -> int:
        """Persist pre-prediction rejects so trust-layer failures are auditable."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_rejections (
                    prediction_date, player_id, player_name, team, opponent,
                    rejection_stage, reason, proposed_market, proposed_line, consensus_level,
                    player_trust_score, market_trust_score, candidate_rank, validator_vote_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                player_id,
                player_name,
                team,
                opponent,
                rejection_stage,
                reason,
                proposed_market,
                proposed_line,
                consensus_level,
                player_trust_score,
                market_trust_score,
                candidate_rank,
                validator_vote_summary,
            ))
            conn.commit()
            return cursor.lastrowid

    def clear_rejections_for_date(self, prediction_date: str) -> None:
        """Clear prior reject logs for a rerun of the same slate date."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM prediction_rejections WHERE prediction_date = ?",
                (prediction_date,)
            )
            conn.commit()
    
    def log_prediction(self, 
                       player_id: int,
                       player_name: str,
                       prediction: Dict[str, Any],
                       match_context: Dict[str, Any],
                       betting_decision: Optional[Any] = None) -> int:
        """
        Log a prediction for post-game analysis
        
        Returns: prediction_log ID
        """
        # Extract Betting Info if available
        market_line = None
        market_odds = None
        bet_direction = None
        bet_confidence = None
        
        if betting_decision:
            market_line = betting_decision.line
            # We assume betting_decision might have odds, or we get it from context
            # For now, let's look at match_context if available or default
            market_odds = match_context.get('market_context', {}).get('odds', -110)
            
            if betting_decision.direction != 'NO_BET':
                bet_direction = betting_decision.direction
                bet_confidence = betting_decision.confidence
        
        # Fallback if just passed as context
        if market_line is None:
            market_line = match_context.get('market_context', {}).get('line')

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_log (
                    prediction_date, player_id, player_name, game_date,
                    pred_points, pred_std, pred_minutes, pred_usage, pred_efficiency,
                    archetype, opponent, spread, edge_source, regime_status,
                    market_line, market_odds, bet_direction, bet_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                player_id,
                player_name,
                match_context.get('game_date', datetime.now().strftime('%Y-%m-%d')),
                prediction.get('mean', 0),
                prediction.get('std', 0),
                prediction.get('minutes_mean', 0),
                prediction.get('usage_mean', 0),
                prediction.get('efficiency_mult', 1.0),
                prediction.get('archetype', 'role_player'),
                match_context.get('opponent', 'UNK'),
                match_context.get('spread', 0),
                prediction.get('edge_source', 'UNKNOWN'),
                match_context.get('regime_status', 'stable'),
                market_line,
                market_odds,
                bet_direction,
                bet_confidence
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def update_with_actuals(self, game_date: str) -> int:
        """
        Update predictions with actual game results
        
        Args:
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            Number of predictions updated
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get actual stats from player_logs
            cursor.execute('''
                UPDATE prediction_log
                SET 
                    actual_points = (
                        SELECT pl.points 
                        FROM player_logs pl 
                        WHERE pl.player_id = prediction_log.player_id
                        AND date(pl.game_date) = date(prediction_log.game_date)
                    ),
                    actual_minutes = (
                        SELECT pl.minutes 
                        FROM player_logs pl 
                        WHERE pl.player_id = prediction_log.player_id
                        AND date(pl.game_date) = date(prediction_log.game_date)
                    )
                WHERE game_date = ?
                AND actual_points IS NULL
            ''', (game_date,))
            
            updated = cursor.rowcount
            
            # Calculate errors
            cursor.execute('''
                UPDATE prediction_log
                SET 
                    error = actual_points - pred_points,
                    within_1std = CASE 
                        WHEN ABS(actual_points - pred_points) <= pred_std THEN 1 
                        ELSE 0 
                    END,
                    within_2std = CASE 
                        WHEN ABS(actual_points - pred_points) <= 2 * pred_std THEN 1 
                        ELSE 0 
                    END
                WHERE game_date = ?
                AND actual_points IS NOT NULL
            ''', (game_date,))
            
            conn.commit()
            
            return updated
    
    def analyze_component_bias(self, 
                               component: str,
                               segment: Optional[str] = None,
                               lookback_days: int = 30) -> BiasReport:
        """
        Analyze systematic bias in a prediction component
        
        Args:
            component: 'minutes', 'usage', 'efficiency', 'total'
            segment: Optional filter (e.g., 'volume_star', 'vs_BOS')
            lookback_days: Days to analyze
            
        Returns:
            BiasReport with findings
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Build query based on component
            if component == 'minutes':
                error_col = '(actual_minutes - pred_minutes) AS comp_error'
            elif component == 'usage':
                # Can't directly measure usage error, use proxy
                error_col = '(error / pred_minutes) AS comp_error'  # Error per minute
            elif component == 'total':
                error_col = 'error AS comp_error'
            else:
                error_col = 'error AS comp_error'
            
            query = f'''
                SELECT {error_col}, archetype, opponent
                FROM prediction_log
                WHERE actual_points IS NOT NULL
                AND game_date >= ?
            '''
            params = [cutoff_date]
            
            if segment:
                if segment.startswith('vs_'):
                    query += ' AND opponent = ?'
                    params.append(segment[3:])
                else:
                    query += ' AND archetype = ?'
                    params.append(segment)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        if len(rows) < 10:
            return BiasReport(
                component=component,
                sample_size=len(rows),
                mean_error=0,
                std_error=0,
                bias_direction='insufficient_data',
                is_significant=False,
                recommended_adjustment=0
            )
        
        # Calculate statistics
        errors = [row['comp_error'] for row in rows if row['comp_error'] is not None]
        
        if not errors:
            return BiasReport(
                component=component,
                sample_size=0,
                mean_error=0,
                std_error=0,
                bias_direction='no_data',
                is_significant=False,
                recommended_adjustment=0
            )
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        n = len(errors)
        
        # T-test for significance (is mean significantly different from 0?)
        t_stat = mean_error / (std_error / np.sqrt(n)) if std_error > 0 else 0
        is_significant = abs(t_stat) > 2.0  # Rough p < 0.05
        
        # Determine bias direction
        if abs(mean_error) < std_error * 0.2:
            bias_direction = 'neutral'
        elif mean_error > 0:
            bias_direction = 'under_predicting'  # Actuals higher than predictions
        else:
            bias_direction = 'over_predicting'  # Actuals lower than predictions
        
        # Recommended adjustment (conservative - only partial correction)
        adjustment = mean_error * 0.5 if is_significant else 0
        
        return BiasReport(
            component=component,
            sample_size=n,
            mean_error=mean_error,
            std_error=std_error,
            bias_direction=bias_direction,
            is_significant=is_significant,
            recommended_adjustment=adjustment
        )
    
    def get_calibration_stats(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Calculate calibration statistics
        
        Good calibration: ~68% within 1σ, ~95% within 2σ
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(within_1std) as in_1std,
                    SUM(within_2std) as in_2std,
                    AVG(error) as mean_error,
                    AVG(ABS(error)) as mean_abs_error
                FROM prediction_log
                WHERE actual_points IS NOT NULL
                AND game_date >= ?
            ''', (cutoff_date,))
            
            row = cursor.fetchone()
        
        if not row or row['total'] == 0:
            return {
                'sample_size': 0,
                'calibration_1std': None,
                'calibration_2std': None,
                'mean_error': None,
                'mae': None,
                'is_well_calibrated': False
            }
        
        pct_1std = row['in_1std'] / row['total']
        pct_2std = row['in_2std'] / row['total']
        
        # Check if calibration is good
        # Target: 68% ± 5% for 1σ, 95% ± 3% for 2σ
        well_calibrated = (0.63 <= pct_1std <= 0.73) and (0.92 <= pct_2std <= 0.98)
        
        return {
            'sample_size': row['total'],
            'calibration_1std': pct_1std,
            'calibration_2std': pct_2std,
            'target_1std': 0.68,
            'target_2std': 0.95,
            'mean_error': row['mean_error'],
            'mae': row['mean_abs_error'],
            'is_well_calibrated': well_calibrated
        }
    
    def generate_calibration_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Generate detailed calibration report (Phase 6 Requirement)
        Tracks if X% confidence actually means X% win rate.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # 1. Bucket Analysis
            # We need to store probability in prediction_log first to do this historically
            # For now, we'll calculate it from current data if available, or just prep the structure
            
            # 2. Bias by Market Side (Over vs Under)
            cursor.execute('''
                SELECT 
                    CASE WHEN pred_points > 0 THEN 'OVER' ELSE 'UNDER' END as side, -- Simplified, need actual prob
                    COUNT(*) as count,
                    AVG(error) as mean_error
                FROM prediction_log
                WHERE actual_points IS NOT NULL AND game_date >= ?
                GROUP BY side
            ''', (cutoff_date,))
            side_biases = cursor.fetchall()
            
            # 3. Bias by Confidence Band
            # Assuming we add a 'confidence_score' to prediction_log later
            
        return {
            'period': f"Last {lookback_days} days",
            'buckets': [], # To be filled when we start logging probabilities
            'side_biases': [{'side': r['side'], 'bias': r['mean_error']} for r in side_biases]
        }

    def analyze_profitability(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze ROI and Win Rate (Phase 3 Requirement)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Fetch settled bets
            cursor.execute('''
                SELECT 
                    market_line, market_odds, bet_direction, bet_confidence,
                    actual_points, pred_points, edge_source
                FROM prediction_log
                WHERE actual_points IS NOT NULL 
                AND bet_direction IS NOT NULL
                AND game_date >= ?
            ''', (cutoff_date,))
            
            rows = cursor.fetchall()
            
        if not rows:
            return {'roi': 0.0, 'win_rate': 0.0, 'total_bets': 0, 'units_profit': 0.0}
            
        total_bets = 0
        wins = 0
        units_staked = 0.0
        units_returned = 0.0
        
        for r in rows:
            line = r['market_line']
            actual = r['actual_points']
            direction = r['bet_direction']
            odds = r['market_odds'] or -110
            
            if line is None: continue
            
            # Determine Outcome
            is_win = False
            is_push = False
            
            if direction == 'OVER':
                if actual > line: is_win = True
                elif actual == line: is_push = True
            elif direction == 'UNDER':
                if actual < line: is_win = True
                elif actual == line: is_push = True
                
            # Assume 1 unit flat stake for basic ROI calc (Portfolio Manager handles real sizing)
            # This is "Model ROI", not "Portfolio ROI"
            stake = 1.0
            
            if is_push:
                units_returned += stake
                # Push doesn't count as bet in some books, but usually valid denominator
                # We'll count it as a "void" bet for win rate? 
                # Standards vary. Let's exclude from Win Rate denominator but keep in ROI (0 profit).
            else:
                total_bets += 1
                units_staked += stake
                
                if is_win:
                    wins += 1
                    profit = self._calculate_payout(odds, stake)
                    units_returned += (stake + profit)
        
        roi = ((units_returned - units_staked) / units_staked * 100) if units_staked > 0 else 0.0
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0.0
        units_profit = units_returned - units_staked
        
        return {
            'roi': roi,
            'win_rate': win_rate,
            'total_bets': total_bets,
            'units_profit': units_profit,
            'period': f"Last {lookback_days} days"
        }

    def _calculate_payout(self, odds: int, stake: float) -> float:
        """Calculate profit for a winning bet (American Odds)"""
        if odds > 0:
            return stake * (odds / 100.0)
        else:
            return stake * (100.0 / abs(odds))

    def analyze_edge_performance(self, lookback_days: int = 30) -> List[Dict]:
        """
        Analyze ROI and Hit Rate by Edge Source (Phase 7)
        "Know why you win."
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT 
                    edge_source,
                    market_line, market_odds, bet_direction,
                    actual_points
                FROM prediction_log
                WHERE actual_points IS NOT NULL 
                AND bet_direction IS NOT NULL
                AND edge_source IS NOT NULL
                AND game_date >= ?
            ''', (cutoff_date,))
            
            rows = cursor.fetchall()
            
        # Group by Edge Source
        source_stats = {}
        
        for r in rows:
            source = r['edge_source']
            if source not in source_stats:
                source_stats[source] = {'bets': 0, 'wins': 0, 'staked': 0.0, 'returned': 0.0}
                
            line = r['market_line']
            actual = r['actual_points']
            direction = r['bet_direction']
            odds = r['market_odds'] or -110
            
            if line is None: continue
            
            # Determine Outcome (Duplicate logic from profitability, could be refactored)
            is_win = False
            is_push = False
            
            if direction == 'OVER':
                if actual > line: is_win = True
                elif actual == line: is_push = True
            elif direction == 'UNDER':
                if actual < line: is_win = True
                elif actual == line: is_push = True
            
            if is_push:
                source_stats[source]['returned'] += 1.0 # Void
                # Don't increment bets/wins for hit rate?
            else:
                source_stats[source]['bets'] += 1
                source_stats[source]['staked'] += 1.0
                
                if is_win:
                    source_stats[source]['wins'] += 1
                    profit = self._calculate_payout(odds, 1.0)
                    source_stats[source]['returned'] += (1.0 + profit)

        results = []
        for source, stats in source_stats.items():
            bets = stats['bets']
            if bets == 0: continue
            
            win_rate = (stats['wins'] / bets) * 100
            roi = ((stats['returned'] - stats['staked']) / stats['staked']) * 100
            
            results.append({
                'source': source,
                'bets': bets,
                'win_rate': win_rate,
                'roi': roi
            })
            
        return sorted(results, key=lambda x: x['bets'], reverse=True)

    def generate_calibration_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Generate detailed calibration report (Reliability Diagram Data)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT 
                    bet_confidence,
                    market_line, bet_direction, actual_points
                FROM prediction_log
                WHERE actual_points IS NOT NULL 
                AND bet_direction IS NOT NULL
                AND game_date >= ?
            ''', (cutoff_date,))
            
            rows = cursor.fetchall()
            
        # Group by Confidence
        bucket_stats = {}
        
        for r in rows:
            conf = r['bet_confidence'] or 'UNKNOWN'
            if conf not in bucket_stats:
                bucket_stats[conf] = {'total': 0, 'wins': 0}
                
            line = r['market_line']
            actual = r['actual_points']
            direction = r['bet_direction']
            
            if line is None: continue
            
            is_win = False
            is_push = False
            if direction == 'OVER':
                if actual > line: is_win = True
                elif actual == line: is_push = True
            elif direction == 'UNDER':
                if actual < line: is_win = True
                elif actual == line: is_push = True
                
            if not is_push:
                bucket_stats[conf]['total'] += 1
                if is_win:
                    bucket_stats[conf]['wins'] += 1
        
        buckets = []
        for conf, stats in bucket_stats.items():
            total = stats['total']
            if total > 0:
                buckets.append({
                    'bucket': conf,
                    'total': total,
                    'win_rate': (stats['wins'] / total) * 100
                })
                
        return {
            'period': f"Last {lookback_days} days",
            'buckets': buckets
        }
    def check_regime_health(self) -> Dict[str, Any]:
        """
        Detect Regime Shift (Phase 7).
        Is the system's edge decaying? Are we biased?
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Look at last 50 outcomes
            cursor.execute('''
                SELECT pred_points, actual_points, error, archetype
                FROM prediction_log
                WHERE actual_points IS NOT NULL
                ORDER BY game_date DESC
                LIMIT 50
            ''')
            rows = cursor.fetchall()
            
        if len(rows) < 20:
            return {'status': 'insufficient_data'}
            
        flags = []
        
        # 1. Check Over/Under Bias Drift
        # If >60% of misses are consistently on one side
        errors = [r['error'] for r in rows]
        # Error = Actual - Pred
        # Positive Error = Actual > Pred (Under was wrong)
        # Negative Error = Actual < Pred (Over was wrong)
        
        misses = [e for e in errors if abs(e) > 3] # Significant misses
        if misses:
            over_miss_rate = len([e for e in misses if e < 0]) / len(misses)
            under_miss_rate = len([e for e in misses if e > 0]) / len(misses)
            
            if over_miss_rate > 0.65:
                flags.append('OVER_BIAS_DETECTED') # We are betting Overs and losing
            elif under_miss_rate > 0.65:
                flags.append('UNDER_BIAS_DETECTED') # We are betting Unders and losing
        
        # 2. Check Hit Rate Stability (Last 20 vs Previous 30)
        recent = rows[:20]
        older = rows[20:]
        
        def calc_mae(data):
            return np.mean([abs(r['error']) for r in data]) if data else 0
            
        mae_recent = calc_mae(recent)
        mae_older = calc_mae(older)
        
        if mae_recent > mae_older * 1.2:
            flags.append('ACCURACY_DECAY')
            
        # Determine Status
        status = 'stable'
        actions = []
        
        if flags:
            status = 'decaying' if 'ACCURACY_DECAY' in flags else 'shifting'
            actions.append('tighten_gates')
            if 'OVER_BIAS_DETECTED' in flags:
                actions.append('stop_overs')
        
        # Log it
        self._log_regime_check(status, flags, actions)
        
        return {
            'status': status,
            'flags': flags,
            'actions': actions,
            'mae_trend': f"{mae_older:.1f} -> {mae_recent:.1f}"
        }
    
    def _log_regime_check(self, status: str, flags: List[str], actions: List[str]):
        """Log regime check results"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO regime_log (detection_date, regime_status, warning_flags, action_taken)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                status,
                ",".join(flags),
                ",".join(actions)
            ))
            conn.commit()

    def analyze_error_decomposition(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Decompose error into Minutes, Usage, and Efficiency components.
        
        Error = (Pred - Actual)
        
        We want to know:
        - How much error is due to Minutes miss?
        - How much due to Usage miss? (PPM)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT 
                    pl.pred_points, pl.actual_points,
                    pl.pred_minutes, pl.actual_minutes,
                    log.points, log.minutes,
                    (log.fga + 0.44 * log.fta + log.turnovers) as implied_usage_count
                FROM prediction_log pl
                JOIN player_logs log ON pl.player_id = log.player_id AND date(pl.game_date) = date(log.game_date)
                WHERE pl.actual_points IS NOT NULL
                AND pl.game_date >= ?
                AND pl.pred_minutes > 0
                AND pl.actual_minutes > 0
            ''', (cutoff,))
            
            rows = cursor.fetchall()
            
        if not rows:
            return {}
            
        total_error = 0
        minutes_contrib = 0
        ppm_contrib = 0
        n = 0
        
        for r in rows:
            pred_pts = r['pred_points']
            act_pts = r['actual_points']
            pred_min = r['pred_minutes']
            act_min = r['actual_minutes']
            
            # Derived PPM
            pred_ppm = pred_pts / pred_min if pred_min > 0 else 0
            act_ppm = act_pts / act_min if act_min > 0 else 0
            
            error = pred_pts - act_pts
            
            # Decomposition:
            # Error ~= (Limit_Delta * PPM) + (PPM_Delta * Minutes)
            # 1. Minutes Component: (PredMin - ActMin) * PredPPM
            mix_minutes = (pred_min - act_min) * pred_ppm
            
            # 2. Efficiency/Usage Component: (PredPPM - ActPPM) * ActMin
            mix_ppm = (pred_ppm - act_ppm) * act_min
            
            total_error += abs(error)
            minutes_contrib += abs(mix_minutes)
            ppm_contrib += abs(mix_ppm)
            n += 1
            
        if n == 0: return {}
        
        return {
            'sample_size': n,
            'mean_abs_error': total_error / n,
            'minutes_error_share': (minutes_contrib / (minutes_contrib + ppm_contrib)) * 100,
            'efficiency_error_share': (ppm_contrib / (minutes_contrib + ppm_contrib)) * 100
        }

    def generate_daily_report(self, game_date: str) -> Dict[str, Any]:
        """
        Generate learning report for a game day
        """
        # First update with actuals
        updated = self.update_with_actuals(game_date)
        
        # Get calibration
        calibration = self.get_calibration_stats(lookback_days=30)
        
        # Phase 6: Deep Dive Calibration Report
        calibration_deep_dive = self.generate_calibration_report(lookback_days=30)
        
        # Phase 7: Edge Attribution Analysis
        edge_analysis = self.analyze_edge_performance(lookback_days=30)
        
        # Phase 7: Regime Detection
        regime_health = self.check_regime_health()
        
        # [NEW] Error Decomposition (Phase 3)
        error_breakdown = self.analyze_error_decomposition(lookback_days=30)

        # [NEW] Miss Pattern Analysis from audited archive
        miss_patterns = self.analyze_miss_patterns(lookback_days=30)
        
        # Analyze biases
        biases = {
            'total': self.analyze_component_bias('total'),
            'minutes': self.analyze_component_bias('minutes'),
        }
        
        # Archetype-specific biases
        for archetype in ['volume_star', 'secondary_star', 'role_player']:
            biases[f'total_{archetype}'] = self.analyze_component_bias('total', segment=archetype)
            
        # [NEW Phase 4] Scheme Biases
        scheme_biases = self.analyze_scheme_bias(lookback_days=45)
        biases.update(scheme_biases)
            
        # [NEW] Persist Biases to DB for Recursive Learning
        self._persist_biases(game_date, biases)
        
        return {
            'game_date': game_date,
            'predictions_updated': updated,
            'calibration': calibration,
            'calibration_deep_dive': calibration_deep_dive,
            'edge_analysis': edge_analysis,
            'error_breakdown': error_breakdown,
            'miss_patterns': miss_patterns,
            'regime_health': regime_health,
            'biases': biases,
            'summary': self._create_summary(calibration, biases, error_breakdown, miss_patterns)
        }
    
    def _persist_biases(self, date: str, biases: Dict[str, BiasReport]):
        """Save bias reports to database for other agents to read"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for name, report in biases.items():
                if report.is_significant:
                    cursor.execute('''
                        INSERT INTO bias_tracker 
                        (analysis_date, component, segment, sample_size, mean_bias, 
                         std_error, is_significant, adjustment_applied)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        date,
                        report.component,
                        name, # using key as segment identifier (e.g. 'total_volume_star')
                        report.sample_size,
                        report.mean_error,
                        report.std_error,
                        1 if report.is_significant else 0,
                        report.recommended_adjustment
                    ))
            conn.commit()
    
    def _create_summary(self, calibration: Dict, biases: Dict, error_breakdown: Dict = None,
                        miss_patterns: Dict[str, Any] = None) -> List[str]:
        """Create human-readable summary"""
        summary = []
        
        # Error Decomposition (New)
        if error_breakdown and 'minutes_error_share' in error_breakdown:
            min_share = error_breakdown['minutes_error_share']
            eff_share = error_breakdown['efficiency_error_share']
            summary.append(f"Error Source: {min_share:.1f}% Minutes / {eff_share:.1f}% Efficiency")

        # Calibration assessment
        if calibration['sample_size'] > 0:
            cal_1std = calibration['calibration_1std']
            summary.append(f"Calibration: {cal_1std*100:.1f}% within 1σ (target: 68%)")
            
            if calibration['is_well_calibrated']:
                summary.append("✓ Model is well-calibrated")
            elif cal_1std > 0.73:
                summary.append("⚠️ Overconfident - std too narrow, consider widening")
            elif cal_1std < 0.63:
                summary.append("⚠️ Underconfident - std too wide, consider tightening")
        
        # Bias alerts
        for name, bias in biases.items():
            if bias.is_significant and abs(bias.recommended_adjustment) > 1:
                summary.append(
                    f"🔴 {name}: {bias.bias_direction} by {abs(bias.mean_error):.1f} pts, "
                    f"adjust by {bias.recommended_adjustment:+.1f}"
                )
        
        if miss_patterns:
            for item in miss_patterns.get('top_reasons', [])[:3]:
                reason = str(item.get('reason', 'unclear')).replace('_', ' ')
                count = item.get('count', 0)
                summary.append(f"Miss Pattern: {reason} appeared {count} times")

            fallback_rate = miss_patterns.get('fallback_miss_rate')
            if fallback_rate is not None and fallback_rate >= 0.20:
                summary.append(
                    f"Fallback Risk: {fallback_rate*100:.1f}% of audited misses used fallback models"
                )

        return summary
    
    def analyze_miss_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze recurring miss reasons from predictions_archive and convert
        them into actionable recommendations.
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    COALESCE(miss_reason_primary, 'unclear') AS miss_reason_primary,
                    COUNT(*) AS miss_count
                FROM predictions_archive
                WHERE prediction_date >= ?
                  AND actual_points IS NOT NULL
                  AND ABS(COALESCE(prediction_error, 0)) >= 4
                GROUP BY COALESCE(miss_reason_primary, 'unclear')
                ORDER BY miss_count DESC
            ''', (cutoff,))
            top_reasons = [
                {'reason': row['miss_reason_primary'], 'count': row['miss_count']}
                for row in cursor.fetchall()
            ]

            cursor.execute('''
                SELECT
                    COUNT(*) AS total_misses,
                    SUM(CASE WHEN used_fallback_model = 1 THEN 1 ELSE 0 END) AS fallback_misses,
                    AVG(prediction_health_score) AS avg_health_score
                FROM predictions_archive
                WHERE prediction_date >= ?
                  AND actual_points IS NOT NULL
                  AND ABS(COALESCE(prediction_error, 0)) >= 4
            ''', (cutoff,))
            row = cursor.fetchone()

        total_misses = row['total_misses'] if row and row['total_misses'] else 0
        fallback_misses = row['fallback_misses'] if row and row['fallback_misses'] else 0
        avg_health_score = row['avg_health_score'] if row else None
        fallback_rate = (fallback_misses / total_misses) if total_misses else None

        recommended_actions = []
        if top_reasons:
            top_reason = top_reasons[0]['reason']
            if top_reason == 'missing_injury_context':
                recommended_actions.append('Tighten no-bet gate when injury context is missing.')
            elif top_reason == 'large_stat_miss':
                recommended_actions.append('Widen uncertainty bands and reduce confidence on volatile high-edge projections.')
            elif top_reason == 'moderate_stat_miss':
                recommended_actions.append('Slightly widen uncertainty and review confidence calibration for mid-tier picks.')
            elif top_reason == 'unclear':
                recommended_actions.append('Expand miss attribution coverage before trusting confidence tiers further.')
            elif top_reason == 'market_anchor_shift':
                recommended_actions.append('Reduce market anchor weight for affected predictions.')
            elif top_reason == 'minutes_miss':
                recommended_actions.append('Increase minutes uncertainty for volatile roles.')
            elif top_reason == 'minutes_role_miss':
                recommended_actions.append('Revisit lineup-aware minutes allocation and role-based minute ceilings.')
            elif top_reason == 'usage_volume_miss':
                recommended_actions.append('Penalize points overs when projected usage is not supported by actual shot volume.')
            elif top_reason == 'efficiency_miss':
                recommended_actions.append('Widen scoring uncertainty and reduce confidence on efficiency-driven edges.')
            elif top_reason == 'blowout_script_miss':
                recommended_actions.append('Discount full-game ceilings when blowout script risk can suppress minutes.')
            elif top_reason == 'playmaking_volume_miss':
                recommended_actions.append('Lower assist confidence when expected on-ball creation is not stable.')
            elif top_reason == 'ceiling_creation_miss':
                recommended_actions.append('Raise assist ceiling estimates for creators gaining unexpected playmaking load.')
            elif top_reason == 'rebound_role_miss':
                recommended_actions.append('Recheck rebound role assumptions and lineup rebounding share.')
            elif top_reason == 'rebound_ceiling_miss':
                recommended_actions.append('Increase rebound ceiling for players with rising frontcourt opportunity.')
            elif top_reason == 'fallback_model':
                recommended_actions.append('Suppress or downgrade picks that rely on fallback models.')

        if fallback_rate is not None and fallback_rate >= 0.20:
            recommended_actions.append('Escalate model-health warnings when fallback predictions miss repeatedly.')

        return {
            'lookback_days': lookback_days,
            'top_reasons': top_reasons,
            'total_misses': total_misses,
            'fallback_misses': fallback_misses,
            'fallback_miss_rate': fallback_rate,
            'avg_health_score_on_misses': avg_health_score,
            'recommended_actions': recommended_actions,
        }

    def get_policy_flags(self, lookback_days: int = 30) -> List[str]:
        """
        Convert historical miss patterns into lightweight system flags for the
        live pipeline.
        """
        miss_patterns = self.analyze_miss_patterns(lookback_days=lookback_days)
        minutes_bias = self.analyze_component_bias('minutes', lookback_days=lookback_days)
        flags: List[str] = []

        top_reason = miss_patterns.get('top_reasons', [{}])[0].get('reason') if miss_patterns.get('top_reasons') else None
        fallback_rate = miss_patterns.get('fallback_miss_rate')

        if top_reason == 'missing_injury_context':
            flags.append('SYSTEM_STRICT_INJURY_CONTEXT')
        if top_reason in ('large_stat_miss', 'moderate_stat_miss'):
            flags.append('SYSTEM_WIDEN_MINUTES_UNCERTAINTY')
        if top_reason == 'market_anchor_shift':
            flags.append('SYSTEM_REDUCE_MARKET_ANCHOR')
        if top_reason == 'minutes_miss':
            flags.append('SYSTEM_WIDEN_MINUTES_UNCERTAINTY')
        if top_reason == 'minutes_role_miss':
            flags.append('SYSTEM_WIDEN_MINUTES_UNCERTAINTY')
        if top_reason in ('usage_volume_miss', 'efficiency_miss'):
            flags.append('SYSTEM_REDUCE_MARKET_ANCHOR')
        if top_reason == 'blowout_script_miss':
            flags.append('SYSTEM_WIDEN_MINUTES_UNCERTAINTY')
        if top_reason in ('playmaking_volume_miss', 'rebound_role_miss'):
            flags.append('SYSTEM_STRICT_INJURY_CONTEXT')
        if top_reason == 'fallback_model':
            flags.append('SYSTEM_SUPPRESS_FALLBACK_MODELS')
        if fallback_rate is not None and fallback_rate >= 0.20 and 'SYSTEM_SUPPRESS_FALLBACK_MODELS' not in flags:
            flags.append('SYSTEM_SUPPRESS_FALLBACK_MODELS')
        if minutes_bias.is_significant and minutes_bias.mean_error > 1.5:
            flags.append('SYSTEM_RAISE_MINUTES_BASELINE')

        return flags

    def apply_prior_update(self, component: str, parameter: str,
                           old_value: float, new_value: float, reason: str):
        """Log and apply a prior update"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prior_updates (update_date, component, parameter, 
                                           old_value, new_value, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                component,
                parameter,
                old_value,
                new_value,
                reason
            ))
            
            conn.commit()

    def analyze_scheme_bias(self, lookback_days: int = 45) -> Dict[str, BiasReport]:
        """
        Phase 4: Analyze bias by (Archetype, Defense Scheme).
        "Do Rim Runners struggle vs Paint Protectors?"
        """
        from .defensive_schemes import DefensiveSchemeAnalyzer
        analyzer = DefensiveSchemeAnalyzer(db=self.db)
        
        # 1. Fetch recent predictions with errors
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT 
                    pl.player_id, pl.game_date, pl.opponent, pl.archetype,
                    pl.pred_points, pl.actual_points, pl.error,
                    pl.pred_minutes, pl.actual_minutes
                FROM prediction_log pl
                WHERE pl.actual_points IS NOT NULL
                AND pl.game_date >= ?
                AND pl.pred_minutes > 15 -- Filter garbage time
            ''', (cutoff,))
            
            rows = cursor.fetchall()
            
        if not rows: return {}
        
        # 2. Group errors by (Archetype, Scheme)
        # We must re-classify the scheme for each game date
        scheme_errors = {}  # key: (archetype, scheme) -> list of errors
        
        # Cache scheme lookups to avoid DB hammer
        scheme_cache = {} # key: (opponent, date) -> schemes list
        
        for r in rows:
            opp = r['opponent']
            date = r['game_date']
            archetype = r['archetype']
            error = r['error'] # Actual - Pred
            
            # Get Scheme (Cached)
            cache_key = (opp, date)
            if cache_key not in scheme_cache:
                analysis = analyzer.analyze_defense(opp, game_date=date)
                scheme_cache[cache_key] = analysis.get('schemes', [])
                
            schemes = scheme_cache[cache_key]
            
            # Map error to each active scheme
            for scheme in schemes:
                key = (archetype, scheme)
                if key not in scheme_errors:
                    scheme_errors[key] = []
                
                # Use % error for cleaner aggregation? 
                # Or just raw point error. Raw is safer for now.
                scheme_errors[key].append(error)
                
        # 3. Calculate Bias Reports
        results = {}
        for (archetype, scheme), errors in scheme_errors.items():
            n = len(errors)
            if n < 5: continue # Minimum sample
            
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            
            # T-test logic
            t_stat = mean_error / (std_error / np.sqrt(n)) if std_error > 0 else 0
            is_sig = abs(t_stat) > 1.5 # Lower threshold for discovery
            
            if is_sig:
                # Rec adjustment: If error is +2.0 (Underpredicted), add +1.0
                adjustment = mean_error * 0.5
                
                report = BiasReport(
                    component='matchup',
                    segment=f"{archetype}_vs_{scheme}",
                    sample_size=n,
                    mean_error=mean_error,
                    std_error=std_error,
                    bias_direction='under' if mean_error > 0 else 'over',
                    is_significant=True,
                    recommended_adjustment=adjustment
                )
                results[f"{archetype}_vs_{scheme}"] = report
                
        return results
