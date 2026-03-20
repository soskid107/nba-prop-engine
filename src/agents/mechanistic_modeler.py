"""
Agent 2: Mechanistic Modeler

Role: Basketball logic engine
Failure mode prevented: Naive averages, minutes dominance, usage explosions

This agent NEVER sees sportsbook lines.
It models how points are created through component decomposition.

Points = Minutes × Usage × Efficiency ± Variance
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..utils.database import DatabaseManager
from ..utils.config import get_config
from ..models.usage_model import UsageModel
from ..models.efficiency_model import EfficiencyModel, FreeThrowModel
from ..models.variance_model import VarianceModel, PaceAdjuster


class MechanisticModelerAgent:
    """
    Basketball logic engine.
    
    Never sees sportsbook lines.
    Models how points are created via component decomposition.
    Output is a distribution, not a number.
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None, n_sims: int = 5000):
        """Initialize mechanistic modeler agent."""
        self.db = db or DatabaseManager()
        self.config = get_config()
        self.n_sims = n_sims
        
        # Initialize component models
        self.usage_model = UsageModel(self.db)
        self.efficiency_model = EfficiencyModel(self.db)
        self.ft_model = FreeThrowModel(self.db)
        self.variance_model = VarianceModel(self.db)
        self.pace_adjuster = PaceAdjuster(self.db)
        
        # [NEW] Load Active Biases from Learning Loop
        self.active_biases = self._load_active_biases()
        
    def _load_active_biases(self) -> Dict[str, float]:
        """Load latest significant biases from database"""
        biases = {}
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                # Get latest significant bias for each segment
                cursor.execute('''
                    SELECT segment, adjustment_applied
                    FROM bias_tracker
                    WHERE is_significant = 1
                    AND analysis_date = (SELECT MAX(analysis_date) FROM bias_tracker)
                ''')
                for row in cursor.fetchall():
                    biases[row['segment']] = row['adjustment_applied']
        except Exception:
            pass # Fail gracefully if table doesn't exist yet
        return biases
    
    def model_components(self, player_context: Dict, match_context: Dict,
                         team_injuries: Dict[int, float] = None) -> Dict[str, Any]:
        """
        Model all components independently.
        """
        player_id = player_context['player_id']
        team = player_context.get('team') or match_context.get('team')
        opponent = match_context.get('opponent')
        # [FIX] Orchestrator passes injuries in player_context, not as arg.
        # Fallback to player_context if argument is empty
        if not team_injuries:
            team_injuries = player_context.get('team_injuries', {})
        
        # Extract additional context
        position = player_context.get('position') or 'G'
        dvp_stats = match_context.get('dvp_stats', {'G': 1.0, 'F': 1.0, 'C': 1.0})
        
        components = {
            'player_id': player_id,
            'team': team,
            'opponent': opponent,
            'points_L5': player_context.get('points_L5', 0), # [NEW]
            'points_max_L5': player_context.get('points_max_L5', 0), # [NEW]
            'points_trend': player_context.get('points_trend', 'stable'), # [NEW]
            'inferred_signals': player_context.get('inferred_signals', {}), # [NEW]
        }
        
        # ========================================
        # COMPONENT 1: MINUTES (Opportunity)
        # ========================================
        minutes = self.model_minutes(player_context, match_context)
        components['minutes'] = minutes
        
        # ========================================
        # COMPONENT 2: USAGE (Involvement)
        # ========================================
        usage = self.model_usage(player_id, team, team_injuries)
        components['usage'] = usage
        
        # ========================================
        # COMPONENT 3: EFFICIENCY (Conversion)
        # ========================================
        # ========================================
        # COMPONENT 3: EFFICIENCY (Conversion)
        # ========================================
        efficiency = self.model_efficiency(player_id, opponent, position, dvp_stats)
        # [NEW] Get Base PPM from history
        # [CRITICAL FIX] Use FG PPM only, to avoid double-counting FTs
        efficiency['base_ppm'] = self.efficiency_model.get_player_fg_ppm_profile(player_id)
        components['efficiency'] = efficiency
        
        # ========================================
        # COMPONENT 4: FREE THROWS (Separate)
        # ========================================
        ft = self.model_free_throws(player_id, minutes['mean'], opponent)
        components['free_throws'] = ft
        
        # ========================================
        # COMPONENT 5: VARIANCE (Uncertainty)
        # ========================================
        variance = self.model_variance(player_id)
        components['variance'] = variance
        
        # ========================================
        # COMPONENT 6: PACE (Multiplier)
        # ========================================
        pace = self.model_pace(team, opponent)
        components['pace'] = pace
        
        return components
    
    def model_minutes(self, player_context: Dict, match_context: Dict) -> Dict[str, float]:
        """
        Model minutes (opportunity) with role and blowout awareness.
        Includes penalties for "Rust" (stale data) and "Injury Status".
        
        Returns:
            Dict with mean, std, and distribution info
        """
        is_starter = player_context.get('is_starter', False)
        minutes_L5 = player_context.get('minutes_L5', 25)
        minutes_L15 = player_context.get('minutes_L15', 25)
        
        # [FIX] If is_starter is False but they play huge minutes, assume they are a starter.
        # This prevents superstars from getting capped at 26 mins just because of a missing flag.
        if not is_starter and minutes_L5 > 29.0:
            is_starter = True
        minutes_trend = player_context.get('minutes_trend', 'stable')
        blowout_prob = match_context.get('blowout_probability', 0.1)
        
        # [FIX] Dynamic Recency Weighting for Breakout Detection
        # If L5 is significantly higher than L15, player is breaking out -> trust L5 more
        minutes_delta = minutes_L5 - minutes_L15
        if minutes_delta > 5:
            # Player's minutes jumped recently -> weight L5 at 90%
            l5_weight = 0.90
        elif minutes_delta > 2:
            # Moderate increase -> weight L5 at 80%
            l5_weight = 0.80
        else:
            # Normal/stable/declining -> standard weighting
            l5_weight = 0.70
            
        base_minutes = l5_weight * minutes_L5 + (1 - l5_weight) * minutes_L15
        
        # Trend adjustment
        if minutes_trend == 'up':
            base_minutes += 1.5
        elif minutes_trend == 'down':
            base_minutes -= 1.5
            
        # [NEW] Stale Data / Rust Penalty
        # If rest_days is huge (e.g. > 30), it implies the player is returning from long absence
        # OR our data is stale. In either case, we should be conservative.
        rest_days = player_context.get('rest_days', 1)
        if rest_days > 30:
            # Player hasn't played in a month (or data is old)
            # Reduce minutes by 15% to account for rust / ramp-up
            base_minutes *= 0.85
            
        # [NEW] Injury Status Penalty
        # Check if player has an active injury status in context (if available)
        # Note: player_context usually doesn't have 'status' directly unless we enrich it.
        # But DataGatherer usually passes it. If not, we assume healthy.
        # If we added 'injury_status' to player_context in Orchestrator, we use it.
        status = player_context.get('injury_status', 'AVAILABLE')
        if status in ['GTD', 'QUESTIONABLE', 'DAY-TO-DAY']:
            # Reduce minutes expectation for banged up players
            base_minutes *= 0.90
        elif status in ['DOUBTFUL']:
            base_minutes *= 0.50
        
        # Role-based bounds
        if is_starter:
            mu_minutes = max(15, min(base_minutes, 38)) # Lowered floor to 15 to catch blowouts/fouls
            sigma_minutes = 4.5
            min_cap, max_cap = 10, 42
        else:
            mu_minutes = min(base_minutes, 26)
            sigma_minutes = 5.5
            min_cap, max_cap = 8, 30
        
        # Blowout adjustment
        if blowout_prob > 0.2:
            if is_starter:
                mu_minutes -= blowout_prob * 5
            else:
                mu_minutes += blowout_prob * 3
        
        return {
            'mean': mu_minutes,
            'std': sigma_minutes,
            'min': min_cap,
            'max': max_cap,
            'is_starter': is_starter,
        }
    
    def model_usage(self, player_id: int, team: str, 
                    injuries: Dict[int, float]) -> Dict[str, float]:
        """
        Model usage (involvement) with on/off context.
        
        Returns:
            Dict with mean, std, and usage details
        """
        baseline = self.usage_model.get_baseline_usage(player_id)
        role = self.usage_model.classify_player_role(player_id, team)
        
        adjusted_usage, usage_std = self.usage_model.predict_adjusted_usage(
            player_id, team, injuries
        )
        
        usage_bump = adjusted_usage - baseline
        
        return {
            'mean': adjusted_usage,
            'std': usage_std,
            'baseline': baseline,
            'bump': usage_bump,
            'role': role,
        }
    
    def model_efficiency(self, player_id: int, opponent: str, 
                         position: str = 'G', dvp_stats: Dict[str, float] = None) -> Dict[str, float]:
        """
        Model efficiency (conversion) with shot profile × opponent defense × DvP.
        
        Returns:
            Dict with efficiency multiplier and shot profile
        """
        dvp_stats = dvp_stats or {'G': 1.0, 'F': 1.0, 'C': 1.0}
        
        mult, variance = self.efficiency_model.predict_efficiency_multiplier(
            player_id, opponent
        )
        
        # Apply DvP Multiplier based on Position
        # Map player position to G/F/C group
        pos_group = 'G'
        if 'C' in position: pos_group = 'C'
        elif 'F' in position: pos_group = 'F'
        
        dvp_mult = dvp_stats.get(pos_group, 1.0)
        
        # Blend the general multiplier with DvP multiplier
        # 70% General Defense, 30% Position Specific
        final_mult = mult * 0.7 + (mult * dvp_mult) * 0.3
        
        profile = self.efficiency_model.get_player_shot_profile(player_id)
        
        return {
            'multiplier': final_mult,
            'variance': variance,
            'three_rate': profile['three_rate'],
            'rim_rate': profile['rim_rate'],
            'mid_rate': profile['mid_rate'],
        }
    
    def model_free_throws(self, player_id: int, predicted_minutes: float,
                          opponent: str) -> Dict[str, float]:
        """
        Model free throw contribution separately.
        
        FTs are the most stable scoring source.
        """
        ft_pts, ft_var = self.ft_model.predict_ft_points(
            player_id, predicted_minutes, opponent
        )
        
        return {
            'mean': ft_pts,
            'std': ft_var,
        }
    
    def model_variance(self, player_id: int) -> Dict[str, Any]:
        """
        Model player-specific variance (uncertainty).
        
        Returns:
            Dict with adjusted std and archetype
        """
        adjusted_std, archetype = self.variance_model.calculate_player_variance(player_id)
        
        return {
            'std': adjusted_std,
            'archetype': archetype,
            'multiplier': self.variance_model.get_variance_multiplier(player_id),
        }
    
    def model_pace(self, team: str, opponent: str) -> Dict[str, float]:
        """
        Model pace as possession multiplier.
        
        Returns:
            Dict with pace multiplier and expected pace
        """
        multiplier = self.pace_adjuster.calculate_pace_multiplier(team, opponent)
        expected_pace = self.pace_adjuster.get_expected_pace(team, opponent)
        
        return {
            'multiplier': multiplier,
            'expected_pace': expected_pace,
        }
    
    def compose_prediction(self, components: Dict) -> Dict[str, Any]:
        """
        Compose components into final points distribution.
        
        Points = (Minutes × Usage × PPM × Efficiency × Pace) + FT + Noise
        
        Returns:
            Dict with mean, std, distribution samples, and skew direction
        """
        minutes = components['minutes']
        usage = components['usage']
        efficiency = components['efficiency']
        ft = components['free_throws']
        variance = components['variance']
        pace = components['pace']
        
        # Sample distributions for Monte Carlo
        minutes_samples = np.random.normal(
            minutes['mean'], minutes['std'], self.n_sims
        )
        minutes_samples = np.clip(minutes_samples, minutes['min'], minutes['max'])
        
        usage_samples = self.usage_model.predict_distribution(
            components['player_id'],
            components['team'],
            {},  # Will be filled from context
            self.n_sims
        )
        
        # Base PPM (from efficiency model - now ACTUAL player PPM)
        base_ppm = efficiency.get('base_ppm', 0.55)
        ppm_samples = np.random.normal(
            base_ppm * efficiency['multiplier'],
            0.08,
            self.n_sims
        )
        
        # Free throw samples
        ft_samples = np.random.normal(ft['mean'], ft['std'], self.n_sims)
        ft_samples = np.maximum(0, ft_samples)
        
        # Noise samples (player-specific)
        noise_samples = np.random.normal(0, variance['std'], self.n_sims)
        
        # Compose: Points = (Minutes * PPM * Pace) + UsageImpact + FT + Noise
        # [FIX] Simplified Composition. Usage is implicitly usually in PPM, 
        # but we track usage BUMP separately.
        
        # We model "Scoring derived from FG" separately from FT
        # Core FG Points = Minutes * PPM_FG * Pace * Usage_Scaling
        
        # Usage multiplier: (Sampled Usage / Baseline Usage) 
        # If usage goes up 20%, points should go up ~20% (simplified)
        baseline_usage = usage.get('baseline', 0.20)
        if baseline_usage < 0.05: baseline_usage = 0.20
        
        # [Active] Dynamic Usage Scaling
        # We simulate the ratio of Projected Usage / Baseline Usage
        # e.g., 0.30 / 0.25 = 1.2x multiplier on scoring volume
        usage_scaler = usage_samples / baseline_usage
        
        # Clamp to avoid explosions (e.g. if baseline is near zero)
        # We cap upside at +35% usage bump, downside at -20%
        usage_scaler = np.clip(usage_scaler, 0.8, 1.35) 
        
        # We remove the 0.82 dampener. We trust the PPM.
        fg_points = minutes_samples * ppm_samples * usage_scaler * pace['multiplier']
        points_samples = np.maximum(0, fg_points + ft_samples + noise_samples)
        
        # [NEW] Reality Check / Ceiling Constraint (Phase 4)
        points_samples = self._apply_ceiling_constraint(points_samples, components)
        
        # [NEW] Apply Self-Correcting Bias Adjustment
        # 1. Global Bias
        global_bias = self.active_biases.get('total', 0.0)
        points_samples += global_bias
        
        # 2. Archetype Bias
        archetype = variance.get('archetype', 'role_player')
        archetype_bias = self.active_biases.get(f'total_{archetype}', 0.0)
        points_samples += archetype_bias
        
        # Ensure non-negative after adjustment
        points_samples = np.maximum(0, points_samples)
        
        # Calculate statistics
        mean_points = np.mean(points_samples)
        std_points = np.std(points_samples)
        
        # Determine skew direction
        skewness = np.mean((points_samples - mean_points) ** 3) / (std_points ** 3)
        if skewness > 0.3:
            skew = 'positive'  # Can explode
        elif skewness < -0.3:
            skew = 'negative'  # Has floor
        else:
            skew = 'symmetric'
        
        return {
            'mean': mean_points,
            'std': std_points,
            'skew': skew,
            'p10': np.percentile(points_samples, 10),
            'p25': np.percentile(points_samples, 25),
            'p50': np.percentile(points_samples, 50),
            'p75': np.percentile(points_samples, 75),
            'p90': np.percentile(points_samples, 90),
            'samples': points_samples,
            'components': {
                'minutes_mean': minutes['mean'],
                'usage_mean': usage['mean'],
                'efficiency_mult': efficiency['multiplier'],
                'ft_mean': ft['mean'],
                'variance_std': variance['std'],
                'pace_mult': pace['multiplier'],
                'archetype': variance['archetype'],
            }
        }
    
    def _apply_ceiling_constraint(self, points_samples: np.ndarray, components: Dict) -> np.ndarray:
        """
        Phase 4: Reality Check / Ceiling Constraint.
        
        If the model predicts a mean significantly higher than the player's recent ceiling,
        we dampen the projection UNLESS there is a structural reason for the breakout.
        
        Args:
            points_samples: Raw Monte Carlo samples
            components: Context components
            
        Returns:
            Adjusted samples
        """
        mean_proj = np.mean(points_samples)
        max_L5 = components.get('points_max_L5', 0)
        avg_L5 = components.get('points_L5', 0)
        signals = components.get('inferred_signals', {})
        
        # Criteria for dampening:
        # 1. Projecting > Max L5 + Buffer
        # 2. Projecting > Avg_L5 * 1.3 (30% jump)
        # 3. No structural reason for improvement (Usage Spike or Role Change or Trend Up)
        
        if max_L5 == 0: return points_samples # No data
        
        # Calculate thresholds
        # Buffer: max(3pts, 15%)
        buffer = max(3.0, max_L5 * 0.15)
        ceiling_threshold = max_L5 + buffer
        
        breakout_detected = (mean_proj > ceiling_threshold)
        
        if breakout_detected:
            # Check for valid reasons to breakout
            valid_reasons = []
            if signals.get('usage_spike'): valid_reasons.append('usage_spike')
            if signals.get('role_change'): valid_reasons.append('role_change')
            if components.get('points_trend') == 'up': valid_reasons.append('trend_up')
            
            # If no valid reasons, DAMPEN
            if not valid_reasons:
                # Dampen factor: Pull mean halfway back to threshold
                # New Mean = Threshold + (Diff * 0.5)
                # Ideally we scalarly shift the samples
                shift = (mean_proj - ceiling_threshold) * 0.6 # Remove 60% of the excess
                points_samples = np.maximum(0, points_samples - shift)
                
        return points_samples

    def predict(self, player_context: Dict, match_context: Dict,
                team_injuries: Dict[int, float] = None) -> Dict[str, Any]:
        """
        Main entry point: Generate full prediction for a player.
        
        Args:
            player_context: From Agent 1
            match_context: From Agent 1
            team_injuries: Teammate injuries
            
        Returns:
            Full prediction with distribution
        """
        # Model each component
        components = self.model_components(
            player_context, match_context, team_injuries
        )
        
        # Compose into final prediction
        prediction = self.compose_prediction(components)
        
        # Add context
        prediction['player_id'] = player_context['player_id']
        prediction['team'] = player_context.get('team')
        prediction['opponent'] = match_context.get('opponent')
        
        return prediction


# Convenience function
def get_mechanistic_modeler() -> MechanisticModelerAgent:
    """Get mechanistic modeler agent instance."""
    return MechanisticModelerAgent()
