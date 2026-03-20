"""
RL Betting Agent (R7)
======================
Deep Q-Network (DQN) for learning optimal betting decisions.

Replaces static edge thresholds with a learned policy that considers:
- State: model edge, std, confidence, regime flags, CLV history, bankroll
- Actions: PASS, BET_SMALL (1u), BET_MEDIUM (2u), BET_LARGE (3u)
- Reward: actual P&L from each bet

Starts in SHADOW MODE: runs alongside traditional betting logic but
doesn't override decisions. Logs what it would have done for analysis.

After sufficient training (500+ decisions with outcomes), can be 
promoted to ADVISORY or ACTIVE mode.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("RL_AGENT")


class RLBettingAgent:
    """
    DQN-based betting agent operating in shadow mode.
    
    Shadow mode: observes all betting decisions, records what it
    would have done, and learns from outcomes without affecting
    real betting decisions.
    """
    
    # Actions
    ACTIONS = ['PASS', 'BET_1U', 'BET_2U', 'BET_3U']
    N_ACTIONS = len(ACTIONS)
    
    # State vector dimensions
    STATE_DIM = 10  # edge, std, prob, confidence_score, regime, clv_avg, bankroll_frac, vol, streak, time
    
    # Modes
    MODE_SHADOW = 'SHADOW'       # Observe only, no override
    MODE_ADVISORY = 'ADVISORY'   # Suggest to operator, flag disagreements
    MODE_ACTIVE = 'ACTIVE'       # Override traditional logic (future)
    
    def __init__(self, db=None, mode: str = 'SHADOW'):
        from ..utils.database import DatabaseManager
        self.db = db or DatabaseManager()
        self.mode = mode
        
        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=5000)
        
        # Q-table (simple tabular Q-learning for now)
        # Discretized state → action values
        self._q_table: Dict[tuple, np.ndarray] = {}
        
        # Learning parameters
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.95     # Discount factor
        self.epsilon = 0.15   # Exploration rate
        
        # Performance tracking
        self.decisions_made = 0
        self.total_pnl = 0.0
        self._decision_log: List[Dict] = []
    
    def build_state_vector(self, 
                           edge: float,
                           model_std: float,
                           probability: float,
                           confidence: str,
                           regime_flags: List[str],
                           clv_avg: float = 0.0,
                           bankroll_fraction: float = 1.0,
                           recent_volatility: float = 0.0,
                           win_streak: int = 0) -> np.ndarray:
        """
        Build state vector from current bet context.
        
        Returns:
            10-dimensional state vector
        """
        # Encode confidence as numeric
        conf_map = {'high': 1.0, 'good': 0.75, 'medium': 0.5, 'low': 0.25, 'very_low': 0.1}
        conf_score = conf_map.get(confidence, 0.5)
        
        # Encode regime severity (0 = stable, 1 = maximum defense)
        regime_score = 0.0
        if 'SYSTEM_SEVERELY_OVERCONFIDENT' in regime_flags:
            regime_score = 1.0
        elif 'SYSTEM_OVERCONFIDENT' in regime_flags:
            regime_score = 0.7
        elif 'SYSTEM_RECOVERING' in regime_flags:
            regime_score = 0.4
        elif 'SYSTEM_SLIGHTLY_OVERCONFIDENT' in regime_flags:
            regime_score = 0.3
        elif 'SYSTEM_UNDERCONFIDENT' in regime_flags:
            regime_score = -0.3
        
        state = np.array([
            np.clip(edge, -0.20, 0.30),          # Edge (clipped)
            np.clip(model_std, 1.0, 15.0) / 15,   # Normalized std
            probability,                            # Win probability
            conf_score,                             # Confidence score
            regime_score,                           # Regime severity
            np.clip(clv_avg, -5.0, 5.0) / 5.0,    # Normalized CLV
            np.clip(bankroll_fraction, 0.0, 2.0),  # Bankroll fraction
            np.clip(recent_volatility, 0, 10) / 10, # Volatility  
            np.clip(win_streak, -10, 10) / 10,     # Win/loss streak
            0.0,  # Reserved for time-of-season encoding
        ], dtype=np.float32)
        
        return state
    
    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for Q-table lookup."""
        # Round each dimension to 1 decimal place for tractable table
        return tuple(np.round(state, 1))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in this state."""
        key = self._discretize_state(state)
        if key not in self._q_table:
            self._q_table[key] = np.zeros(self.N_ACTIONS)
        return self._q_table[key]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.get_q_values(state)))
    
    def shadow_decision(self,
                         edge: float,
                         model_std: float,
                         probability: float,
                         confidence: str,
                         regime_flags: List[str],
                         traditional_decision: str,
                         player_name: str = '',
                         **kwargs) -> Dict[str, Any]:
        """
        Make a shadow decision (observe only, don't override).
        
        Returns:
            Dict with RL agent's recommendation and comparison to traditional
        """
        state = self.build_state_vector(
            edge=edge,
            model_std=model_std,
            probability=probability,
            confidence=confidence,
            regime_flags=regime_flags,
            **kwargs
        )
        
        action_idx = self.select_action(state, training=False)
        rl_action = self.ACTIONS[action_idx]
        q_values = self.get_q_values(state)
        
        # Log the shadow decision
        decision = {
            'player_name': player_name,
            'rl_action': rl_action,
            'traditional_action': traditional_decision,
            'agree': self._actions_agree(rl_action, traditional_decision),
            'q_values': {a: float(q) for a, q in zip(self.ACTIONS, q_values)},
            'state': state.tolist(),
            'edge': edge,
            'confidence': confidence,
        }
        
        self._decision_log.append(decision)
        self.decisions_made += 1
        
        if not decision['agree']:
            logger.info(
                f"[RL SHADOW] Disagreement for {player_name}: "
                f"RL says {rl_action}, traditional says {traditional_decision} "
                f"(edge={edge:.3f}, conf={confidence})"
            )
        
        return decision
    
    def _actions_agree(self, rl_action: str, traditional: str) -> bool:
        """Check if RL and traditional actions broadly agree."""
        if rl_action == 'PASS' and traditional == 'NO_BET':
            return True
        if rl_action.startswith('BET_') and traditional in ('OVER', 'UNDER'):
            return True
        if rl_action == 'PASS' and traditional in ('OVER', 'UNDER'):
            return False  # RL would pass, traditional bets
        if rl_action.startswith('BET_') and traditional == 'NO_BET':
            return False  # RL would bet, traditional passes
        return True
    
    def update(self, state: np.ndarray, action_idx: int, 
               reward: float, next_state: Optional[np.ndarray] = None):
        """Q-learning update after observing outcome."""
        key = self._discretize_state(state)
        if key not in self._q_table:
            self._q_table[key] = np.zeros(self.N_ACTIONS)
        
        if next_state is not None:
            next_q = np.max(self.get_q_values(next_state))
        else:
            next_q = 0.0  # Terminal state
        
        # Q-learning update
        target = reward + self.gamma * next_q
        self._q_table[key][action_idx] += self.alpha * (target - self._q_table[key][action_idx])
        
        self.total_pnl += reward
    
    def calculate_reward(self, bet_result: str, units_bet: float, 
                          odds: float = -110) -> float:
        """
        Calculate reward from bet outcome.
        
        bet_result: 'WIN', 'LOSS', 'PUSH', or 'PASS'
        """
        if bet_result == 'PASS':
            return 0.0  # No risk, no reward
        
        if bet_result == 'PUSH':
            return 0.0
        
        # Convert American odds to payout
        if odds > 0:
            payout_ratio = odds / 100
        else:
            payout_ratio = 100 / abs(odds)
        
        if bet_result == 'WIN':
            return units_bet * payout_ratio
        elif bet_result == 'LOSS':
            return -units_bet
        
        return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get RL agent's shadow performance."""
        if not self._decision_log:
            return {'decisions': 0, 'agreement_rate': 0}
        
        agreements = sum(1 for d in self._decision_log if d['agree'])
        
        return {
            'decisions': self.decisions_made,
            'agreement_rate': agreements / len(self._decision_log) * 100,
            'total_pnl': self.total_pnl,
            'mode': self.mode,
            'q_table_size': len(self._q_table),
            'disagreements': [
                d for d in self._decision_log[-20:] 
                if not d['agree']
            ],
        }
    
    def should_promote(self) -> Tuple[bool, str]:
        """
        Check if RL agent has earned promotion from shadow to advisory.
        
        Requirements:
        - 500+ decisions observed
        - Agreement rate > 60% (learns market structure)
        - Simulated P&L > 0 (shadow profit)
        """
        if self.decisions_made < 500:
            return False, f"Need 500+ decisions (have {self.decisions_made})"
        
        summary = self.get_performance_summary()
        
        if summary['agreement_rate'] < 60:
            return False, f"Agreement rate too low ({summary['agreement_rate']:.1f}% < 60%)"
        
        if self.total_pnl <= 0:
            return False, f"Shadow P&L negative ({self.total_pnl:.2f}u)"
        
        return True, f"Ready for promotion: {self.decisions_made} decisions, {self.total_pnl:.2f}u shadow P&L"


# Convenience function
def get_rl_agent(db=None, mode: str = 'SHADOW'):
    """Get RL Betting Agent instance."""
    return RLBettingAgent(db=db, mode=mode)
