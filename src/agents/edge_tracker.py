"""
Historical Edge Tracker
========================
Tracks edge scores vs actual results to calibrate thresholds.

After 2-3 weeks of data, this tells you:
  - What edge score threshold ACTUALLY wins at what rate
  - Which edge tier (parlay_core vs playable) is producing ROI
  - Which sub-agent (narrative, fragility, market) is most predictive
  - Whether the system's edge is decaying over time

Data Source: learning_loop prediction_logs + edge scores from match_context
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("EDGE_TRACKER")


class EdgeTracker:
    """
    Tracks historical edge scores vs actual outcomes.
    
    This is the calibration engine — without it, threshold
    choices (70/80) are just guesses.
    """
    
    TABLE_NAME = 'edge_tracking'
    
    def __init__(self, db=None):
        self.db = db
        if db:
            self._ensure_table()
    
    def _ensure_table(self):
        """Create edge tracking table if it doesn't exist."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS edge_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_date TEXT NOT NULL,
                        player_name TEXT NOT NULL,
                        player_id INTEGER,
                        market_type TEXT,
                        line REAL,
                        opening_line REAL,
                        current_line REAL,
                        closing_line REAL,
                        direction TEXT,
                        edge_score REAL,
                        edge_tier TEXT,
                        narrative_score REAL,
                        fragility_score REAL,
                        market_score REAL,
                        kill_count INTEGER,
                        actual_value REAL,
                        hit INTEGER,  -- 1 = pick won, 0 = pick lost
                        margin REAL,  -- how much the pick won/lost by
                        clv REAL,
                        beat_close INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(game_date, player_name, market_type)
                    )
                """)
                for column_def in [
                    "ALTER TABLE edge_tracking ADD COLUMN opening_line REAL",
                    "ALTER TABLE edge_tracking ADD COLUMN current_line REAL",
                    "ALTER TABLE edge_tracking ADD COLUMN closing_line REAL",
                    "ALTER TABLE edge_tracking ADD COLUMN clv REAL",
                    "ALTER TABLE edge_tracking ADD COLUMN beat_close INTEGER",
                ]:
                    try:
                        cursor.execute(column_def)
                    except Exception as e:
                        if "duplicate column name" not in str(e).lower():
                            raise
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to create edge_tracking table: {e}")
    
    def log_edge_pick(self, 
                      game_date: str,
                      player_name: str,
                      player_id: int,
                      market_type: str,
                      line: float,
                      direction: str,
                      edge_score: float,
                      edge_tier: str,
                      opening_line: float = None,
                      current_line: float = None,
                      narrative_score: float = 0,
                      fragility_score: float = 0,
                      market_score: float = 0,
                      kill_count: int = 0) -> bool:
        """Log an edge pick for future tracking."""
        if not self.db:
            return False
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO edge_tracking 
                    (game_date, player_name, player_id, market_type, line,
                     opening_line, current_line, direction, edge_score, edge_tier, narrative_score,
                     fragility_score, market_score, kill_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_date, player_name, player_id, market_type, line,
                      opening_line, current_line, direction, edge_score, edge_tier, narrative_score,
                      fragility_score, market_score, kill_count))
                conn.commit()
                return True
        except Exception as e:
            logger.warning(f"Failed to log edge pick: {e}")
            return False
    
    def update_with_actuals(self, game_date: str) -> int:
        """
        Update edge picks with actual results from player_logs.
        Call this after games complete.
        
        Returns number of picks updated.
        """
        if not self.db:
            return 0
        
        updated = 0
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get pending picks for this date
                cursor.execute("""
                    SELECT id, player_name, player_id, market_type, line, direction
                    FROM edge_tracking
                    WHERE game_date = ? AND actual_value IS NULL
                """, (game_date,))
                pending = cursor.fetchall()
                
                for pick in pending:
                    pick_id = pick['id']
                    player_id = pick['player_id']
                    market = pick['market_type']
                    line = pick['line']
                    direction = pick['direction']
                    
                    # Get actual value from player_logs
                    stat_col = self._market_to_stat(market)
                    if not stat_col:
                        continue
                    
                    cursor.execute(f"""
                        SELECT {stat_col} as actual
                        FROM player_logs
                        WHERE player_id = ? AND game_date = ?
                        LIMIT 1
                    """, (player_id, game_date))
                    
                    result = cursor.fetchone()
                    if not result:
                        continue
                    
                    actual = result['actual']
                    
                    # Determine if pick hit
                    if direction == 'OVER':
                        hit = 1 if actual > line else 0
                        margin = actual - line
                    elif direction == 'UNDER':
                        hit = 1 if actual < line else 0
                        margin = line - actual
                    else:
                        continue
                    
                    closing_line = self._get_closing_line(
                        player_name=pick['player_name'],
                        player_id=pick['player_id'],
                        market_type=market,
                        game_date=game_date,
                    )
                    beat_close = None
                    clv = None
                    if closing_line is not None:
                        if direction == 'OVER':
                            clv = closing_line - line
                            beat_close = 1 if line < closing_line else 0
                        elif direction == 'UNDER':
                            clv = line - closing_line
                            beat_close = 1 if line > closing_line else 0

                    cursor.execute("""
                        UPDATE edge_tracking 
                        SET actual_value = ?, hit = ?, margin = ?,
                            closing_line = ?, clv = ?, beat_close = ?
                        WHERE id = ?
                    """, (actual, hit, margin, closing_line, clv, beat_close, pick_id))
                    updated += 1
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update actuals: {e}")
        
        return updated
    
    def get_performance_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Generate edge performance report.
        
        Returns win rates and ROI by:
          - Edge score buckets (60-70, 70-80, 80-90, 90-100)
          - Edge tier (parlay_core, playable)
          - Sub-agent scores
          - Kill count
        """
        if not self.db:
            return {}
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                
                # Overall stats
                cursor.execute("""
                    SELECT COUNT(*) as total, 
                           SUM(hit) as wins,
                           AVG(margin) as avg_margin,
                           AVG(clv) as avg_clv,
                           AVG(beat_close) as beat_close_rate
                    FROM edge_tracking
                    WHERE game_date >= ? AND hit IS NOT NULL
                """, (cutoff,))
                overall = dict(cursor.fetchone())
                
                # By edge score bucket
                buckets = {}
                for low, high in [(60, 70), (70, 80), (80, 90), (90, 100)]:
                    cursor.execute("""
                        SELECT COUNT(*) as total,
                               SUM(hit) as wins,
                               AVG(margin) as avg_margin,
                               AVG(clv) as avg_clv,
                               AVG(beat_close) as beat_close_rate
                        FROM edge_tracking
                        WHERE game_date >= ? AND hit IS NOT NULL
                              AND edge_score >= ? AND edge_score < ?
                    """, (cutoff, low, high))
                    row = dict(cursor.fetchone())
                    total = row['total'] or 0
                    wins = row['wins'] or 0
                    buckets[f"{low}-{high}"] = {
                        'total': total,
                        'wins': wins,
                        'win_rate': wins / total if total > 0 else 0,
                        'avg_margin': row['avg_margin'] or 0,
                        'avg_clv': row.get('avg_clv') or 0,
                        'beat_close_rate': row.get('beat_close_rate') or 0,
                    }
                
                # By tier
                tiers = {}
                for tier in ['parlay_core', 'playable']:
                    cursor.execute("""
                        SELECT COUNT(*) as total,
                               SUM(hit) as wins,
                               AVG(margin) as avg_margin
                        FROM edge_tracking
                        WHERE game_date >= ? AND hit IS NOT NULL
                              AND edge_tier = ?
                    """, (cutoff, tier))
                    row = dict(cursor.fetchone())
                    total = row['total'] or 0
                    wins = row['wins'] or 0
                    tiers[tier] = {
                        'total': total,
                        'wins': wins,
                        'win_rate': wins / total if total > 0 else 0,
                        'avg_margin': row['avg_margin'] or 0,
                    }
                
                # By kill count
                kill_perf = {}
                for kc in range(0, 6):
                    cursor.execute("""
                        SELECT COUNT(*) as total, SUM(hit) as wins
                        FROM edge_tracking
                        WHERE game_date >= ? AND hit IS NOT NULL
                              AND kill_count = ?
                    """, (cutoff, kc))
                    row = dict(cursor.fetchone())
                    total = row['total'] or 0
                    wins = row['wins'] or 0
                    if total > 0:
                        kill_perf[f"{kc}_kills"] = {
                            'total': total,
                            'wins': wins,
                            'win_rate': wins / total if total > 0 else 0,
                        }
                
                total = overall.get('total', 0) or 0
                wins = overall.get('wins', 0) or 0
                
                return {
                    'lookback_days': lookback_days,
                    'total_picks': total,
                    'total_wins': wins,
                    'overall_win_rate': wins / total if total > 0 else 0,
                    'avg_margin': overall.get('avg_margin', 0) or 0,
                    'avg_clv': overall.get('avg_clv', 0) or 0,
                    'beat_close_rate': overall.get('beat_close_rate', 0) or 0,
                    'by_score_bucket': buckets,
                    'by_tier': tiers,
                    'by_kill_count': kill_perf,
                    'optimal_threshold': self._find_optimal_threshold(buckets),
                }
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}
    
    def _find_optimal_threshold(self, buckets: Dict) -> Dict:
        """Find the edge score threshold that maximizes win rate."""
        best_threshold = 70
        best_rate = 0
        
        for bucket_name, stats in buckets.items():
            if stats['total'] >= 10 and stats['win_rate'] > best_rate:
                best_rate = stats['win_rate']
                best_threshold = int(bucket_name.split('-')[0])
        
        return {
            'threshold': best_threshold,
            'win_rate': best_rate,
            'recommendation': f"Edge score ≥{best_threshold} has {best_rate:.0%} win rate"
        }
    
    def _market_to_stat(self, market_type: str) -> Optional[str]:
        """Map market type to player_logs column."""
        mapping = {
            'points': 'points',
            'player_points': 'points',
            'assists': 'assists',
            'player_assists': 'assists',
            'rebounds': 'rebounds',
            'player_rebounds': 'rebounds',
            'blocks': 'blocks',
            'player_blocks': 'blocks',
            'steals': 'steals',
            'player_steals': 'steals',
            'threes': 'fg3m',
            'player_threes': 'fg3m',
            'field_goals': 'fgm',
            'player_field_goals': 'fgm',
        }
        return mapping.get(market_type)

    def _get_closing_line(self, player_name: str, market_type: str, game_date: str,
                          player_id: Optional[int] = None) -> Optional[float]:
        """Fetch the latest observed line for CLV tracking."""
        if not self.db:
            return None

        market_key = market_type if market_type.startswith('player_') else f'player_{market_type}'
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if player_id:
                    cursor.execute("""
                        SELECT line
                        FROM player_prop_odds
                        WHERE player_id = ? AND market_key = ? AND game_date = ?
                        ORDER BY snapshot_time DESC
                        LIMIT 1
                    """, (player_id, market_key, game_date))
                    row = cursor.fetchone()
                    if row:
                        return row['line']

                cursor.execute("""
                    SELECT line
                    FROM player_prop_odds
                    WHERE player_name LIKE ? AND market_key = ? AND game_date = ?
                    ORDER BY snapshot_time DESC
                    LIMIT 1
                """, (f"%{player_name}%", market_key, game_date))
                row = cursor.fetchone()
                return row['line'] if row else None
        except Exception as e:
            logger.warning(f"Failed to fetch closing line: {e}")
            return None
    
    def format_report_markdown(self, report: Dict) -> str:
        """Format performance report as markdown for daily report."""
        if not report or report.get('total_picks', 0) == 0:
            return "\n### 📊 Edge Performance\n_No historical edge data yet. Run the pipeline for 2-3 weeks to calibrate._\n"
        
        lines = []
        lines.append("\n### 📊 Edge Performance Tracker")
        lines.append(f"_Last {report['lookback_days']} days | {report['total_picks']} picks tracked_")
        lines.append("")
        
        win_rate = report['overall_win_rate']
        lines.append(f"**Overall: {report['total_wins']}/{report['total_picks']} ({win_rate:.0%}) | Avg Margin: {report['avg_margin']:.1f}**")
        lines.append("")
        
        # By tier
        lines.append("| Tier | Picks | Wins | Win Rate | Avg Margin |")
        lines.append("|------|-------|------|----------|------------|")
        for tier, stats in report.get('by_tier', {}).items():
            icon = "🔥" if tier == 'parlay_core' else "✅"
            lines.append(
                f"| {icon} {tier} | {stats['total']} | {stats['wins']} "
                f"| {stats['win_rate']:.0%} | {stats['avg_margin']:.1f} |"
            )
        lines.append("")
        
        # Optimal threshold recommendation
        opt = report.get('optimal_threshold', {})
        if opt:
            lines.append(f"> 💡 **{opt.get('recommendation', '')}**")
            lines.append("")
        
        return "\n".join(lines)
