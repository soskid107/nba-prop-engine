"""
Edge Agent E3: Line-Ceiling Optimizer
=======================================
Purpose: Analyze where the market is steering bettors 
and where real edge hides in line positioning.

This agent asks:
- What is the highest UNDER line that still wins most of the time?
- Where does juice flip from protection → greed?
- Is the main line set as a trap?

Output: best_line recommendation + market steering detection
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("LINE_CEILING")


class LineCeilingOptimizer:
    """
    Analyzes market line positioning to find optimal entry points.
    
    Core Insight (Azramund):
    "The main line is where the book WANTS you to bet.
    The real edge lives in the lines they try to hide."
    """
    
    def __init__(self, db_manager=None):
        self.db = db_manager
    
    def analyze(self,
                player_context: Dict[str, Any],
                match_context: Dict[str, Any],
                market_line: float,
                market_type: str = 'points',
                market_floor: Optional[float] = None,
                market_ceiling: Optional[float] = None,
                odds_over: float = -110,
                odds_under: float = -110) -> Dict[str, Any]:
        """
        Analyze line positioning and find optimal bet placement.
        
        Args:
            player_context: Player stats
            match_context: Game context
            market_line: Main sportsbook line
            market_type: 'points', 'assists', 'rebounds'
            market_floor: Lowest available line across books
            market_ceiling: Highest available line across books
            odds_over: American odds for OVER
            odds_under: American odds for UNDER
            
        Returns:
            Dict with best line, market steering, and score
        """
        stats = player_context.get('stats', {})
        ppg_L5 = (stats.get('l5_ppg') or
                  player_context.get('ppg_L5') or
                  player_context.get('points_L5') or 15)
        ppg_std = (stats.get('l5_std_pts') or
                   player_context.get('ppg_std') or
                   player_context.get('points_std') or 5)
        
        # ========================================
        # 1. DETECT MARKET STEERING
        # ========================================
        # Where is the book pushing public money?
        steering = self._detect_steering(
            market_line, odds_over, odds_under,
            market_floor, market_ceiling
        )
        
        # ========================================
        # 2. LINE SPREAD ANALYSIS
        # ========================================
        # If there's a wide gap between floor and ceiling across books,
        # the market is uncertain → more edge opportunity
        line_spread = self._analyze_line_spread(
            market_line, market_floor, market_ceiling
        )
        
        # ========================================
        # 3. FIND OPTIMAL LINE 
        # ========================================
        best_line = self._find_best_line(
            ppg_L5, ppg_std, market_line,
            market_floor, market_ceiling,
            steering['direction']
        )
        
        # ========================================
        # 4. VARIANCE vs LINE GAP
        # ========================================
        # How much variance protection does this line offer?
        variance_protection = self._assess_variance_protection(
            ppg_L5, ppg_std, market_line, market_type
        )
        
        # ========================================
        # Calculate Score
        # ========================================
        score = 0
        
        # Steering detection value: 0-30
        if steering['strength'] == 'strong':
            score += 30
        elif steering['strength'] == 'moderate':
            score += 20
        elif steering['strength'] == 'mild':
            score += 10
        
        # Line spread opportunity: 0-25
        if line_spread['opportunity'] == 'high':
            score += 25
        elif line_spread['opportunity'] == 'medium':
            score += 15
        
        # Variance protection: 0-25
        if variance_protection['level'] == 'strong':
            score += 25
        elif variance_protection['level'] == 'moderate':
            score += 15
        elif variance_protection['level'] == 'weak':
            score += 5
        
        # Best line quality: 0-20
        if best_line.get('quality') == 'excellent':
            score += 20
        elif best_line.get('quality') == 'good':
            score += 12
        elif best_line.get('quality') == 'fair':
            score += 5
        
        return {
            'main_line': market_line,
            'best_line': best_line,
            'market_steering': steering,
            'line_spread': line_spread,
            'variance_protection': variance_protection,
            'score': min(100, score),
        }
    
    def _detect_steering(self, line: float,
                         odds_over: float, odds_under: float,
                         floor: Optional[float],
                         ceiling: Optional[float]) -> Dict:
        """
        Detect where the book is steering public money.
        
        Key signals:
        - Heavy juice on one side → book protecting against that side
        - Line shifted toward ceiling → public is on OVER
        """
        # Convert to implied probabilities
        ip_over = self._implied_prob(odds_over)
        ip_under = self._implied_prob(odds_under)
        
        # Juice differential
        juice_diff = ip_over - ip_under  # Positive = OVER is heavier juice
        
        direction = "neutral"
        strength = "none"
        reason = "Balanced odds"
        
        if juice_diff > 0.04:
            # OVER has more juice → book is protecting OVER side
            # → Public is on OVER → edge may be on UNDER
            direction = "toward_over"
            strength = "strong" if juice_diff > 0.08 else "moderate"
            reason = f"OVER juice {ip_over:.1%} vs UNDER {ip_under:.1%} → public on OVER"
        elif juice_diff < -0.04:
            direction = "toward_under"
            strength = "strong" if juice_diff < -0.08 else "moderate"
            reason = f"UNDER juice {ip_under:.1%} vs OVER {ip_over:.1%} → public on UNDER"
        elif abs(juice_diff) > 0.02:
            direction = "toward_over" if juice_diff > 0 else "toward_under"
            strength = "mild"
            reason = f"Slight juice imbalance ({juice_diff:+.1%})"
        
        # Cross-book line positioning
        if floor and ceiling and ceiling > floor:
            mid = (floor + ceiling) / 2
            if line > mid + 0.5:
                # Main line set high → books expect UNDER traffic
                if direction == "neutral":
                    direction = "toward_over"
                    strength = "mild"
                    reason += f" | Line above cross-book midpoint ({mid:.1f})"
        
        return {
            'direction': direction,
            'strength': strength,
            'reason': reason,
            'juice_differential': juice_diff,
        }
    
    def _analyze_line_spread(self, line: float,
                             floor: Optional[float],
                             ceiling: Optional[float]) -> Dict:
        """
        Analyze how wide the line spread is across books.
        Wide spread = market uncertainty = more edge opportunity.
        """
        if not floor or not ceiling:
            return {
                'spread': 0,
                'opportunity': 'unknown',
                'reason': 'Single-book data only'
            }
        
        spread = ceiling - floor
        
        if spread >= 3.0:
            return {
                'spread': spread,
                'opportunity': 'high',
                'reason': f'{spread:.1f}pt spread across books → high disagreement'
            }
        elif spread >= 1.5:
            return {
                'spread': spread,
                'opportunity': 'medium',
                'reason': f'{spread:.1f}pt spread → moderate disagreement'
            }
        else:
            return {
                'spread': spread,
                'opportunity': 'low',
                'reason': f'{spread:.1f}pt spread → consensus line'
            }
    
    def _find_best_line(self, ppg_L5: float, ppg_std: float,
                        main_line: float,
                        floor: Optional[float],
                        ceiling: Optional[float],
                        steering_dir: str) -> Dict:
        """
        Find the optimal line to bet given player stats and market context.
        
        For UNDERS: find the HIGHEST line that still clears
        For OVERS: find the LOWEST line that still clears
        """
        # Calculate player's typical range
        p25 = ppg_L5 - 0.67 * ppg_std  # 25th percentile
        p75 = ppg_L5 + 0.67 * ppg_std  # 75th percentile
        median = ppg_L5
        
        # Determine best direction based on line vs performance
        if main_line > ppg_L5 + ppg_std * 0.5:
            # Line is above typical range → UNDER opportunity
            direction = "UNDER"
            
            # Best UNDER line: the highest one available that's still above median
            best_target = main_line  # Use main line (highest we can safely take)
            if ceiling and ceiling > main_line:
                best_target = ceiling  # Even higher UNDER line at another book
            
            # How much buffer above the median?
            buffer = best_target - median
            
            if buffer > ppg_std * 1.0:
                quality = 'excellent'
                reason = f"Line {best_target:.1f} is {buffer:.1f}pts above median → strong variance protection"
            elif buffer > ppg_std * 0.5:
                quality = 'good'
                reason = f"Line {best_target:.1f} gives {buffer:.1f}pts buffer above median"
            else:
                quality = 'fair'
                reason = f"Line {best_target:.1f} is close to median, narrow margin"
            
            # Warn against going too low
            avoid = None
            if floor and floor < median:
                avoid = {
                    'line': floor,
                    'reason': f"UNDER {floor:.1f} is below median ({median:.1f}) → too risky"
                }
            
            return {
                'direction': direction,
                'line': best_target,
                'quality': quality,
                'reason': reason,
                'avoid': avoid
            }
        
        elif main_line < ppg_L5 - ppg_std * 0.3:
            # Line is below typical → OVER opportunity
            direction = "OVER"
            
            best_target = main_line
            if floor and floor < main_line:
                best_target = floor  # Lowest OVER line available
            
            buffer = median - best_target
            
            if buffer > ppg_std * 0.7:
                quality = 'excellent'
                reason = f"Line {best_target:.1f} is {buffer:.1f}pts below median → clear OVER"
            elif buffer > ppg_std * 0.3:
                quality = 'good'
                reason = f"Line {best_target:.1f} gives {buffer:.1f}pts below median"
            else:
                quality = 'fair'
                reason = f"Line {best_target:.1f} needs player to perform near median"
            
            return {
                'direction': direction,
                'line': best_target,
                'quality': quality,
                'reason': reason,
                'avoid': None
            }
        
        else:
            # Line is right at the median → no structural edge
            return {
                'direction': 'NEUTRAL',
                'line': main_line,
                'quality': 'none',
                'reason': f'Line {main_line:.1f} ≈ median {median:.1f} → no structural edge',
                'avoid': None
            }
    
    def _assess_variance_protection(self, ppg_L5: float, ppg_std: float,
                                    line: float, market: str) -> Dict:
        """
        How much variance protection does betting UNDER this line provide?
        
        Variance protection = line is high enough that the player must
        OUTPERFORM their norm to hit the OVER.
        """
        if ppg_L5 <= 0:
            return {'level': 'unknown', 'gap_pct': 0}
        
        gap = line - ppg_L5
        gap_pct = gap / ppg_L5
        
        # For UNDER: higher line → more protection
        if gap_pct > 0.15:
            return {
                'level': 'strong',
                'gap_pct': gap_pct,
                'reason': f'Line is {gap_pct:.0%} above L5 avg → needs outperformance'
            }
        elif gap_pct > 0.05:
            return {
                'level': 'moderate',
                'gap_pct': gap_pct,
                'reason': f'Line is {gap_pct:.0%} above L5 avg → some protection'
            }
        elif gap_pct > -0.05:
            return {
                'level': 'weak',
                'gap_pct': gap_pct,
                'reason': f'Line ≈ L5 avg → coin flip territory'
            }
        else:
            return {
                'level': 'inverted',
                'gap_pct': gap_pct,
                'reason': f'Line is {abs(gap_pct):.0%} BELOW L5 avg → OVER has protection'
            }
    
    def _implied_prob(self, american_odds: float) -> float:
        """Convert American odds to implied probability."""
        if american_odds is None:
            return 0.5238  # Default -110
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
