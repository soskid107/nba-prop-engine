
from typing import Dict, Any, Optional
import logging

from .validator import ValidatorAgent, Vote
from ...utils.database import DatabaseManager
from ...agents.data_gatherer import DataGathererAgent

logger = logging.getLogger(__name__)

class DataValidator(ValidatorAgent):
    """
    Validator A: The Data Purist.
    Votes based purely on statistical significance (L5/L10/H2H vs Line).
    Does NOT check injuries or news (that's NarrativeValidator's job).
    """
    
    def __init__(self, db: DatabaseManager):
        super().__init__(name="DataValidator")
        self.db = db
        self.gatherer = DataGathererAgent(db)

    def validate(self, player_id: int, prop_type: str, line: float, reference_date: str = None) -> Vote:
        """
        Check if the prop is statistically favorable.
        """
        try:
            context = self.gatherer.gather_player_context(player_id, date_limit=reference_date)
            recent_logs = context.get('recent_logs', []) or []
            games_sample = len(recent_logs)

            stat_aliases = {
                'points': ['points'],
                'assists': ['assists', 'ast'],
                'rebounds': ['rebounds', 'reb'],
                'threes': ['threes', 'fg3m'],
                'blocks': ['blocks', 'blk'],
                'steals': ['steals', 'stl'],
                'field_goals': ['field_goals', 'fgm'],
            }
            stat_bases = stat_aliases.get(prop_type, [prop_type])

            if prop_type == 'status_check':
                if games_sample >= 8:
                    return Vote(
                        self.name,
                        "YES",
                        0.75,
                        f"Sufficient live sample available ({games_sample} recent games).",
                        {'recent_games': games_sample}
                    )
                if games_sample >= 4:
                    return Vote(
                        self.name,
                        "YES",
                        0.55,
                        f"Moderate live sample available ({games_sample} recent games).",
                        {'recent_games': games_sample}
                    )
                if games_sample > 0:
                    return Vote(
                        self.name,
                        "ABSTAIN",
                        0.35,
                        f"Thin sample ({games_sample} recent games).",
                        {'recent_games': games_sample}
                    )
                return Vote(
                    self.name,
                    "NO",
                    0.9,
                    "No recent game sample available for viability check.",
                    {'recent_games': 0}
                )
            
            candidate_l5_keys = []
            candidate_l15_keys = []
            for base in stat_bases:
                candidate_l5_keys.append(f"{base}_L5")
                candidate_l15_keys.append(f"{base}_L15")

            val_l5 = next((context.get(key) for key in candidate_l5_keys if context.get(key) is not None), 0.0)
            val_l15 = next((context.get(key) for key in candidate_l15_keys if context.get(key) is not None), 0.0)
            
            if val_l5 == 0 and val_l15 == 0:
                return Vote(self.name, "ABSTAIN", 0.0, "No data available", {})

            blended = val_l5 if val_l15 == 0 else (0.65 * val_l5 + 0.35 * val_l15)
            diff = blended - line
            line_scale = max(abs(line), 1.0)
            pct_diff = diff / line_scale
            trend = val_l5 - val_l15

            # Promote stronger votes when both short- and medium-term form agree.
            if pct_diff >= 0.12:
                confidence = 0.78 if trend >= 0 else 0.68
                reason = f"Blended form ({blended:.1f}) is materially above line ({line:.1f})"
                return Vote(
                    self.name,
                    "YES",
                    confidence,
                    reason,
                    {'L5': val_l5, 'L15': val_l15, 'blended': blended, 'diff': diff, 'trend': trend}
                )

            if pct_diff <= -0.12:
                confidence = 0.78 if trend <= 0 else 0.68
                reason = f"Blended form ({blended:.1f}) is materially below line ({line:.1f})"
                return Vote(
                    self.name,
                    "NO",
                    confidence,
                    reason,
                    {'L5': val_l5, 'L15': val_l15, 'blended': blended, 'diff': diff, 'trend': trend}
                )

            if abs(pct_diff) <= 0.04:
                return Vote(
                    self.name,
                    "ABSTAIN",
                    0.45,
                    f"Blended form ({blended:.1f}) is close to line ({line:.1f})",
                    {'L5': val_l5, 'L15': val_l15, 'blended': blended, 'diff': diff, 'trend': trend}
                )

            lean_verdict = "YES" if diff > 0 else "NO"
            lean_direction = "above" if diff > 0 else "below"
            confidence = 0.60 if (diff > 0 and trend >= 0) or (diff < 0 and trend <= 0) else 0.52
            return Vote(
                self.name,
                lean_verdict,
                confidence,
                f"Blended form ({blended:.1f}) sits modestly {lean_direction} line ({line:.1f})",
                {'L5': val_l5, 'L15': val_l15, 'blended': blended, 'diff': diff, 'trend': trend}
            )

        except Exception as e:
            logger.error(f"DataValidator error: {e}")
            return Vote(self.name, "ABSTAIN", 0.0, f"Error: {e}", {})
