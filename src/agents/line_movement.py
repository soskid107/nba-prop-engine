"""
Line Movement Tracker
======================
Detects line moves between odds snapshots to identify sharp money.

Key Signals:
  1. Reverse Line Movement (RLM): line moves AGAINST public action
  2. Steam Moves: rapid 1+ point shifts in short time window
  3. Opening vs Current: drift from opening line indicates information
  4. Cross-Book Divergence: when one book moves and others don't

Data Source: player_prop_odds table (multiple snapshots per day per bookmaker)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("LINE_MOVEMENT")


class LineMovementTracker:
    """
    Tracks line movement patterns to detect sharp money signals.
    
    Sharp money = informed bettors who move lines.
    If a line moves 0.5-1.0 points, someone with information bet into it.
    """
    
    # Movement thresholds
    SIGNIFICANT_MOVE = 0.5     # 0.5 point move = notable
    STEAM_MOVE = 1.0           # 1.0+ point move = steam (sharp action)
    JUICE_SHIFT_THRESHOLD = 15 # 15+ cent juice shift = significant
    
    def __init__(self, db=None):
        self.db = db
    
    def analyze_movement(self, 
                         player_name: str,
                         market_key: str,
                         game_date: str,
                         current_line: float,
                         current_odds_over: int = -110,
                         current_odds_under: int = -110,
                         player_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze line movement for a player prop.
        
        Args:
            player_name: Player name
            market_key: e.g. 'player_points'
            game_date: Game date YYYY-MM-DD
            current_line: Current line value
            current_odds_over/under: Current juice
            
        Returns:
            Dict with movement analysis
        """
        snapshots = self._get_snapshots(player_name, market_key, game_date, player_id=player_id)
        
        if not snapshots or len(snapshots) < 2:
            return self._no_movement_result(current_line)
        
        # Sort by time
        snapshots.sort(key=lambda s: s.get('snapshot_time', ''))
        
        # Opening line (earliest snapshot)
        opening = snapshots[0]
        opening_line = opening.get('line', current_line)
        
        # Calculate movement
        total_move = current_line - opening_line
        
        # Detect movement patterns
        signals = []
        sharp_direction = None
        movement_score = 0  # 0-100
        
        # === 1. Opening vs Current ===
        if abs(total_move) >= self.SIGNIFICANT_MOVE:
            if total_move > 0:
                # Line moved UP → books adjusting to sharp OVER action
                signals.append(f"Line opened {opening_line:.1f}, moved UP to {current_line:.1f} (+{total_move:.1f})")
                sharp_direction = 'OVER'
                movement_score += min(40, int(abs(total_move) * 20))
            else:
                # Line moved DOWN → books adjusting to sharp UNDER action
                signals.append(f"Line opened {opening_line:.1f}, moved DOWN to {current_line:.1f} ({total_move:.1f})")
                sharp_direction = 'UNDER'
                movement_score += min(40, int(abs(total_move) * 20))
        
        # === 2. Steam Move Detection ===
        steam_detected = False
        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]
            curr = snapshots[i]
            move = curr.get('line', 0) - prev.get('line', 0)
            if abs(move) >= self.STEAM_MOVE:
                steam_detected = True
                direction = "UP" if move > 0 else "DOWN"
                signals.append(f"⚡ Steam move: {direction} {abs(move):.1f} pts between snapshots")
                movement_score += 25
                break
        
        # === 3. Juice Analysis ===
        opening_over = opening.get('odds_over', -110) or -110
        opening_under = opening.get('odds_under', -110) or -110
        
        juice_shift_over = (current_odds_over or -110) - opening_over
        juice_shift_under = (current_odds_under or -110) - opening_under
        
        if abs(juice_shift_over) >= self.JUICE_SHIFT_THRESHOLD:
            if juice_shift_over < 0:
                # Over odds got worse (more negative) → sharp under action
                signals.append(f"Juice shifted: OVER odds {opening_over} → {current_odds_over} (sharps on UNDER)")
                if sharp_direction is None:
                    sharp_direction = 'UNDER'
                movement_score += 15
            else:
                signals.append(f"Juice shifted: OVER odds {opening_over} → {current_odds_over} (sharps on OVER)")
                if sharp_direction is None:
                    sharp_direction = 'OVER'
                movement_score += 15
        
        # === 4. Cross-Book Divergence ===
        book_lines = self._get_cross_book_lines(player_name, market_key, game_date, player_id=player_id)
        if book_lines and len(book_lines) > 1:
            lines = [b['line'] for b in book_lines]
            line_spread = max(lines) - min(lines)
            if line_spread >= 1.0:
                signals.append(f"Cross-book spread: {min(lines):.1f} - {max(lines):.1f} ({line_spread:.1f} pts)")
                movement_score += 10
                # The lowest line book is likely the sharpest
                if current_line > min(lines) + 0.5:
                    signals.append("Current line above sharpest book → potential UNDER value")
                    if sharp_direction is None:
                        sharp_direction = 'UNDER'
        
        # === 5. Reverse Line Movement (most powerful signal) ===
        # If line moves UP but juice favors UNDER → RLM
        if total_move > 0 and juice_shift_under < -10:
            signals.append("🔄 Reverse Line Movement: line UP but UNDER juice improving → SHARP UNDER")
            sharp_direction = 'UNDER'
            movement_score += 20
        elif total_move < 0 and juice_shift_over < -10:
            signals.append("🔄 Reverse Line Movement: line DOWN but OVER juice improving → SHARP OVER")
            sharp_direction = 'OVER'
            movement_score += 20
        
        # Clamp score
        movement_score = min(100, movement_score)
        
        return {
            'opening_line': opening_line,
            'current_line': current_line,
            'total_move': round(total_move, 1),
            'sharp_direction': sharp_direction,
            'steam_detected': steam_detected,
            'movement_score': movement_score,
            'signals': signals,
            'num_snapshots': len(snapshots),
            'cross_book_lines': book_lines,
        }
    
    def _get_snapshots(self, player_name: str, market_key: str, 
                       game_date: str,
                       player_id: Optional[int] = None) -> List[Dict]:
        """Get all odds snapshots for a player prop from DB."""
        if not self.db:
            return []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if player_id:
                    cursor.execute("""
                        SELECT line, odds_over, odds_under, bookmaker, snapshot_time
                        FROM player_prop_odds
                        WHERE player_id = ? AND market_key = ? AND game_date = ?
                        ORDER BY snapshot_time ASC
                    """, (player_id, market_key, game_date))
                    rows = cursor.fetchall()
                    if rows:
                        return [dict(row) for row in rows]

                cursor.execute("""
                    SELECT line, odds_over, odds_under, bookmaker, snapshot_time
                    FROM player_prop_odds
                    WHERE player_name LIKE ? AND market_key = ? AND game_date = ?
                    ORDER BY snapshot_time ASC
                """, (f"%{player_name}%", market_key, game_date))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get snapshots: {e}")
            return []
    
    def _get_cross_book_lines(self, player_name: str, market_key: str,
                              game_date: str,
                              player_id: Optional[int] = None) -> List[Dict]:
        """Get latest line from each bookmaker."""
        if not self.db:
            return []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if player_id:
                    cursor.execute("""
                        SELECT bookmaker, line, odds_over, odds_under, 
                               MAX(snapshot_time) as latest
                        FROM player_prop_odds
                        WHERE player_id = ? AND market_key = ? AND game_date = ?
                        GROUP BY bookmaker
                        ORDER BY line ASC
                    """, (player_id, market_key, game_date))
                    rows = cursor.fetchall()
                    if rows:
                        return [dict(row) for row in rows]

                cursor.execute("""
                    SELECT bookmaker, line, odds_over, odds_under, 
                           MAX(snapshot_time) as latest
                    FROM player_prop_odds
                    WHERE player_name LIKE ? AND market_key = ? AND game_date = ?
                    GROUP BY bookmaker
                    ORDER BY line ASC
                """, (f"%{player_name}%", market_key, game_date))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get cross-book lines: {e}")
            return []
    
    def _no_movement_result(self, current_line: float) -> Dict[str, Any]:
        """Default result when no movement data available."""
        return {
            'opening_line': current_line,
            'current_line': current_line,
            'total_move': 0,
            'sharp_direction': None,
            'steam_detected': False,
            'movement_score': 0,
            'signals': ['No line movement data available'],
            'num_snapshots': 0,
            'cross_book_lines': [],
        }
