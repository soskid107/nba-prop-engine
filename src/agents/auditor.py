"""
Agent 3: Auditor & Skeptic
==========================
Role: Challenge unrealistic predictions, suppress optimism bias, widen uncertainty

Core Principles (LeCun - Structure & Constraints):
- "Never trust unchecked model output"
- Apply hard basketball constraints
- Detect and flag anomalies
- Widen uncertainty when skeptical

Skeptical Filters:
1. Hard caps: No player > 65 points, < 2 points if playing
2. Historical ceiling: Player's mean + 2.5σ is absolute max
3. Role consistency: Bench players can't suddenly score 35+
4. Blowout awareness: Reduce confidence in extreme spreads
5. Minutes sanity: Points can't exceed realistic PPM × minutes
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class AuditFlag:
    """Flag raised by the auditor"""
    severity: str  # 'warning', 'error', 'critical'
    code: str  # Short identifier
    message: str  # Human readable
    adjustment: Optional[Dict[str, float]] = None  # Suggested fix


class AuditorAgent:
    """
    Agent 3: The Skeptic
    
    Reviews predictions from Agent 2 and applies skeptical adjustments.
    Never makes things MORE extreme - only dampens outliers.
    
    Mental Model:
    - If it seems too good to be true, it probably is
    - Uncertainty should grow with prediction magnitude
    - Historical patterns constrain future outcomes
    """
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        
        # Hard basketball constraints
        self.ABSOLUTE_MAX_POINTS = 65  # Wilt's record
        self.ABSOLUTE_MIN_POINTS = 0
        self.MIN_POINTS_IF_PLAYING = 2  # Even bad nights get FTs
        
        # Role-based ceilings (realistic maximums)
        self.ROLE_CEILINGS = {
            'volume_star': 60,
            'star': 60,             # [FIX] Added mapping for UsageModel 'star'
            'secondary_star': 45,
            'third_option': 32,
            'microwave_scorer': 30,
            'bench_scorer': 35,     # [FIX] Added mapping for UsageModel 'bench_scorer'
            'catch_and_shoot': 25,
            'rim_runner': 22,
            'floor_general': 20,
            'role_player': 18,
        }
        
        # Minimum thresholds for rotation players
        self.ROLE_FLOORS = {
            'volume_star': 12,
            'star': 10,             # [FIX] Added mapping
            'secondary_star': 8,
            'third_option': 5,
            'microwave_scorer': 4,
            'bench_scorer': 4,      # [FIX] Added mapping
            'catch_and_shoot': 3,
            'rim_runner': 3,
            'floor_general': 2,
            'role_player': 2,
        }
        
        # Skepticism multipliers for uncertainty
        self.UNCERTAINTY_INFLATION = {
            'high_spread': 1.25,  # Blowout expected
            'back_to_back': 1.15,  # Fatigue factor
            'injury_return': 1.40,  # Minutes uncertain
            'role_change': 1.30,  # New situation
            'outlier_prediction': 1.35,  # Model says something unusual
        }
    
    def audit(self, 
              prediction: Dict[str, Any],
              player_context: Dict[str, Any],
              match_context: Dict[str, Any],
              market_type: str = 'points') -> Dict[str, Any]:
        """
        Main audit function - review and adjust prediction
        
        Args:
            prediction: Output from Agent 2 (mean, std, distribution)
            player_context: From Agent 1
            match_context: From Agent 1
            
        Returns:
            Audited prediction with adjusted stats
        """
        flags = []
        adjustments = {}
        
        # Get key values
        mean = prediction.get('mean', 15)
        std = prediction.get('std', 5)
        archetype = prediction.get('archetype', 'role_player')
        
        # 1. Check absolute bounds
        bound_flags = self._check_absolute_bounds(mean, std, archetype)
        flags.extend(bound_flags)
        
        # 2. Check historical consistency
        hist_flags, hist_adj = self._check_historical_consistency(
            mean, std, player_context
        )
        flags.extend(hist_flags)
        adjustments.update(hist_adj)
        
        # 3. Check role consistency
        role_flags, role_adj = self._check_role_consistency(
            mean, std, archetype, player_context
        )
        flags.extend(role_flags)
        adjustments.update(role_adj)
        
        # 4. Check for blowout scenario
        blowout_flags, blowout_adj = self._check_blowout_scenario(
            std, match_context
        )
        flags.extend(blowout_flags)
        adjustments.update(blowout_adj)
        
        # 5. Check minutes-points consistency
        mins_flags, mins_adj = self._check_minutes_consistency(
            mean, prediction.get('minutes_mean', 25), player_context
        )
        flags.extend(mins_flags)
        adjustments.update(mins_adj)
        
        # 6. Check Narrative Consistency (Phase 6)
        narrative_flags, narrative_adj = self._check_narrative_consistency(
            mean, prediction, player_context
        )
        flags.extend(narrative_flags)
        adjustments.update(narrative_adj)
        
        # 7. Apply uncertainty inflation if needed
        final_std = self._inflate_uncertainty(std, flags, match_context)
        adjustments['std_adjustment'] = final_std - std
        
        # Apply adjustments to create final prediction
        adjusted_mean = mean + adjustments.get('mean_adjustment', 0)
        adjusted_std = final_std
        
        # Clamp to valid range — use market-specific limits
        market_bounds = {
            'points':      (self.ABSOLUTE_MIN_POINTS, self.ABSOLUTE_MAX_POINTS),
            'assists':     (0, 25),
            'rebounds':    (0, 30),
            'threes':      (0, 15),
            'blocks':      (0, 12),
            'steals':      (0, 10),
            'field_goals': (0, 25),
        }
        abs_min, abs_max = market_bounds.get(market_type, (0, 65))
        adjusted_mean = max(abs_min, min(abs_max, adjusted_mean))
        
        # Rebuild distribution with adjusted parameters
        return {
            'mean': adjusted_mean,
            'std': adjusted_std,
            'original_mean': mean,
            'original_std': std,
            'p10': max(0, adjusted_mean - 1.28 * adjusted_std),
            'p25': max(0, adjusted_mean - 0.67 * adjusted_std),
            'p50': adjusted_mean,
            'p75': adjusted_mean + 0.67 * adjusted_std,
            'p90': adjusted_mean + 1.28 * adjusted_std,
            'flags': flags,
            'adjustments': adjustments,
            'confidence': self._calculate_confidence(flags),
            'passed_audit': len([f for f in flags if f.severity == 'critical']) == 0,
            'pure_model_pred': prediction.get('pure_model_pred'),
            'market_adjusted_pred': prediction.get('market_adjusted_pred'),
            'post_rule_pred': prediction.get('post_rule_pred'),
            'prediction_health': prediction.get('prediction_health', {}),
        }
    
    def _check_absolute_bounds(self, mean: float, std: float, 
                                archetype: str) -> List[AuditFlag]:
        """Check if prediction violates absolute constraints"""
        flags = []
        
        ceiling = self.ROLE_CEILINGS.get(archetype, 50)
        floor = self.ROLE_FLOORS.get(archetype, 2)
        
        # Check ceiling
        if mean > ceiling:
            flags.append(AuditFlag(
                severity='warning',
                code='CEILING_EXCEEDED',
                message=f"Mean {mean:.1f} exceeds {archetype} ceiling of {ceiling}",
                adjustment={'mean_adjustment': ceiling - mean}
            ))
        
        # Check if p90 is absurdly high
        p90 = mean + 1.28 * std
        if p90 > self.ABSOLUTE_MAX_POINTS:
            flags.append(AuditFlag(
                severity='error',
                code='P90_UNREALISTIC',
                message=f"P90 of {p90:.1f} exceeds all-time record",
                adjustment={'std_reduction': (p90 - self.ABSOLUTE_MAX_POINTS) / 1.28}
            ))
        
        # Check floor for rotation players
        if mean < floor:
            flags.append(AuditFlag(
                severity='warning',
                code='FLOOR_VIOLATION',
                message=f"Mean {mean:.1f} below expected floor of {floor}",
                adjustment={'mean_adjustment': floor - mean}
            ))
        
        return flags
    
    def _check_historical_consistency(self, mean: float, std: float,
                                       player_context: Dict) -> tuple:
        """Ensure prediction aligns with player's historical performance"""
        flags = []
        adjustments = {}
        
        stats = player_context.get('stats', {})
        hist_mean = stats.get('l15_ppg', mean)
        hist_std = stats.get('l15_std_pts', std)
        
        # Check if prediction is too far from history
        z_score = (mean - hist_mean) / max(hist_std, 1)
        
        if abs(z_score) > 2.5:
            severity = 'error' if abs(z_score) > 3.5 else 'warning'
            flags.append(AuditFlag(
                severity=severity,
                code='HISTORICAL_DEVIATION',
                message=f"Prediction {mean:.1f} is {z_score:.1f}σ from L15 mean {hist_mean:.1f}",
                adjustment={'mean_adjustment': -z_score * 0.3 * hist_std}  # Pull back toward mean
            ))
            # Apply dampening - don't let it deviate more than 2σ
            if abs(z_score) > 2.5:
                adjustments['mean_adjustment'] = -z_score * 0.3 * hist_std
        
        # Check for suspiciously low variance prediction
        if std < hist_std * 0.5:
            flags.append(AuditFlag(
                severity='warning',
                code='LOW_VARIANCE',
                message=f"Predicted std {std:.1f} much lower than historical {hist_std:.1f}",
                adjustment={'std_adjustment': hist_std * 0.7 - std}
            ))
            adjustments['std_floor'] = hist_std * 0.7
        
        return flags, adjustments
    
    def _check_role_consistency(self, mean: float, std: float,
                                 archetype: str, 
                                 player_context: Dict) -> tuple:
        """Check if prediction matches player's role"""
        flags = []
        adjustments = {}
        
        # Role-based sanity checks
        role_expectations = {
            'volume_star': (20, 35),  # Expected range
            'secondary_star': (14, 26),
            'third_option': (10, 20),
            'microwave_scorer': (8, 18),
            'catch_and_shoot': (6, 14),
            'rim_runner': (6, 14),
            'floor_general': (4, 12),
            'role_player': (4, 12),
        }
        
        expected_low, expected_high = role_expectations.get(archetype, (5, 20))
        
        if mean > expected_high * 1.5:
            flags.append(AuditFlag(
                severity='warning',
                code='ROLE_MISMATCH_HIGH',
                message=f"{archetype} with {mean:.1f} pts exceeds typical range",
                adjustment={'suppress_optimism': True}
            ))
            # Pull back toward expected range
            adjustments['mean_adjustment'] = -(mean - expected_high) * 0.4
        
        if mean < expected_low * 0.5:
            flags.append(AuditFlag(
                severity='warning',
                code='ROLE_MISMATCH_LOW',
                message=f"{archetype} with {mean:.1f} pts below typical range",
                adjustment={'boost_pessimism': True}
            ))
        
        return flags, adjustments
    
    def _check_blowout_scenario(self, std: float, 
                                 match_context: Dict) -> tuple:
        """Widen uncertainty in potential blowout games"""
        flags = []
        adjustments = {}
        
        spread = abs(match_context.get('spread', 0))
        
        # High spread = more uncertainty in minutes
        if spread >= 10:
            flags.append(AuditFlag(
                severity='warning',
                code='BLOWOUT_RISK',
                message=f"Spread of {spread:.1f} suggests possible blowout",
                adjustment={'uncertainty_multiplier': 1.25}
            ))
            adjustments['blowout_uncertainty'] = self.UNCERTAINTY_INFLATION['high_spread']
        elif spread >= 7:
            adjustments['blowout_uncertainty'] = 1.1
        
        return flags, adjustments
    
    def _check_minutes_consistency(self, points_mean: float,
                                    minutes_mean: float,
                                    player_context: Dict) -> tuple:
        """Ensure points are consistent with expected minutes"""
        flags = []
        adjustments = {}
        
        # Calculate implied PPM
        if minutes_mean > 0:
            implied_ppm = points_mean / minutes_mean
            
            # Check against historical PPM
            stats = player_context.get('stats', {})
            hist_ppm = stats.get('l15_ppm', implied_ppm)
            
            # PPM over 1.5 is elite (Luka, Giannis territory)
            if implied_ppm > 1.5:
                flags.append(AuditFlag(
                    severity='warning',
                    code='PPM_UNREALISTIC',
                    message=f"Implied PPM of {implied_ppm:.2f} is elite-tier",
                    adjustment={'cap_ppm': 1.4}
                ))
                # Cap at 1.4 PPM
                adjustments['mean_adjustment'] = minutes_mean * 1.4 - points_mean
            
            # Check deviation from player's historical PPM
            ppm_deviation = (implied_ppm - hist_ppm) / max(hist_ppm * 0.2, 0.1)
            if abs(ppm_deviation) > 2:
                flags.append(AuditFlag(
                    severity='warning',
                    code='PPM_DEVIATION',
                    message=f"Implied PPM {implied_ppm:.2f} deviates from historical {hist_ppm:.2f}"
                ))
        
        return flags, adjustments
    
    def _check_narrative_consistency(self, mean: float, 
                                   prediction: Dict,
                                   player_context: Dict) -> tuple:
        """
        Phase 6: "What story must be true?"
        
        Checks if the prediction relies on a fragile stack of assumptions.
        """
        flags = []
        adjustments = {}
        fragility_score = 0
        stories = []
        
        stats = player_context.get('stats', {})
        signals = player_context.get('inferred_signals', {})
        
        l15_ppg = stats.get('l15_ppg', 0)
        l15_std = stats.get('l15_std_pts', 1)
        
        # Assumption 1: Significant Overperformance
        if mean > l15_ppg + 1.5 * l15_std:
            stories.append("Breakout scoring night")
            fragility_score += 1
            
        # Assumption 2: Relies on abnormal minutes
        pred_mins = prediction.get('minutes_mean', 0)
        hist_mins = stats.get('l15_minutes', 0)
        if pred_mins > hist_mins * 1.2:
            stories.append("Minutes spike (+20%)")
            fragility_score += 1
            
        # Assumption 3: Relies on usage spike signal to be true
        if signals.get('usage_spike'):
            stories.append("Sustained usage spike")
            fragility_score += 0.5  # Signal exists, so less fragile
            
        # Assumption 4: Ignoring rotation tightening
        # If auditor predicts minutes > X but rotation is tightening
        if signals.get('rotation_tightening') and pred_mins > hist_mins:
            stories.append("Defying rotation tightening trend")
            fragility_score += 2  # High fragility - fighting the coach
            
        # Evaluate Fragility
        if fragility_score >= 2.5:
            story_str = " + ".join(stories)
            flags.append(AuditFlag(
                severity='warning',
                code='FRAGILE_NARRATIVE',
                message=f"Fragile Narrative (Score {fragility_score}): Requires {story_str}",
                adjustment={'confidence_downgrade': 'low'}
            ))
            # Dampen mean slightly to be safe
            adjustments['mean_adjustment'] = -(mean - l15_ppg) * 0.2
            
        return flags, adjustments

    def _inflate_uncertainty(self, base_std: float, 
                             flags: List[AuditFlag],
                             match_context: Dict) -> float:
        """
        Inflate uncertainty based on detected issues
        
        Principle: When in doubt, widen the distribution
        """
        multiplier = 1.0
        
        # Apply based on flags
        warning_count = len([f for f in flags if f.severity == 'warning'])
        error_count = len([f for f in flags if f.severity == 'error'])
        
        if error_count > 0:
            multiplier *= 1.3
        if warning_count >= 2:
            multiplier *= 1.15
        
        # Check for back-to-back (if available)
        if match_context.get('is_back_to_back', False):
            multiplier *= self.UNCERTAINTY_INFLATION['back_to_back']
        
        # Check for injury return
        if match_context.get('is_injury_return', False):
            multiplier *= self.UNCERTAINTY_INFLATION['injury_return']
        
        return base_std * multiplier
    
    def _calculate_confidence(self, flags: List[AuditFlag]) -> str:
        """Calculate overall confidence level"""
        critical = len([f for f in flags if f.severity == 'critical'])
        errors = len([f for f in flags if f.severity == 'error'])
        warnings = len([f for f in flags if f.severity == 'warning'])
        
        if critical > 0:
            return 'very_low'
        elif errors > 0:
            return 'low'
        elif warnings >= 3:
            return 'medium'
        elif warnings >= 1:
            return 'good'
        else:
            return 'high'
    
    def reject_prediction(self, flags: List[AuditFlag]) -> bool:
        """Determine if prediction should be rejected entirely"""
        critical = [f for f in flags if f.severity == 'critical']
        return len(critical) > 0
    
    def summarize_audit(self, audit_result: Dict) -> str:
        """Generate human-readable audit summary"""
        flags = audit_result.get('flags', [])
        
        if not flags:
            return "✓ Prediction passed all skeptical checks"
        
        summary_parts = []
        for flag in flags:
            icon = {'warning': '⚠️', 'error': '❌', 'critical': '🚫'}.get(flag.severity, '•')
            summary_parts.append(f"{icon} {flag.message}")
        
        confidence = audit_result.get('confidence', 'unknown')
        passed = audit_result.get('passed_audit', True)
        
        status = "PASSED" if passed else "REJECTED"
        
        return f"Audit {status} (confidence: {confidence})\n" + "\n".join(summary_parts)
