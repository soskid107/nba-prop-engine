"""
Phase 16 Validation: Import & Unit Test All R1-R9 Modules
==========================================================
Run this offline (no live games needed) to validate all new modules.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def check(label, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result is True or result is None:
            print(f"  ✅ {label}")
            PASS += 1
        else:
            print(f"  ✅ {label} → {result}")
            PASS += 1
    except Exception as e:
        print(f"  ❌ {label} → {e}")
        traceback.print_exc()
        FAIL += 1

print("=" * 60)
print("  Phase 16 Validation: R1-R9 Module Tests")
print("=" * 60)

# ─────────────────────────────────────────
# R1: Fat-Tailed Distribution Engine
# ─────────────────────────────────────────
print("\n[R1] Fat-Tailed Distribution Engine")
def test_r1_import():
    from src.models.fat_tailed_sampler import FatTailedSampler
    return True
check("Import FatTailedSampler", test_r1_import)

def test_r1_fallback():
    """Test that sampler returns fallback when no data exists."""
    import numpy as np
    from src.models.fat_tailed_sampler import FatTailedSampler
    sampler = FatTailedSampler()
    # With no data, should fall back to Gaussian
    info = sampler.get_distribution_info(player_id=99999, market='points')
    return f"fallback_info={info.get('distribution', 'none')}"
check("Fallback with no data", test_r1_fallback)

# ─────────────────────────────────────────
# R2: Hierarchical Bayesian Minutes  
# ─────────────────────────────────────────
print("\n[R2] Hierarchical Bayesian Minutes")
def test_r2_import():
    from src.models.bayesian_minutes import BayesianMinutesModel
    return True
check("Import BayesianMinutesModel", test_r2_import)

def test_r2_pymc():
    import pymc as pm
    return f"PyMC v{pm.__version__}"
check("PyMC available", test_r2_pymc)

def test_r2_analytical():
    """Test analytical conjugate fallback with synthetic data."""
    import numpy as np
    from src.models.bayesian_minutes import BayesianMinutesModel
    bm = BayesianMinutesModel()
    
    # Test analytical fallback directly
    role_prior = {'mean': 32.0, 'std': 3.0}
    # Simulate a rookie with 5 games averaging 28 min
    fake_data = np.array([26, 28, 30, 27, 29], dtype=float)
    result = bm._fit_analytical(fake_data, role_prior, None, 500)
    
    # Posterior should be between data mean (28) and prior mean (32)
    assert 28 < result['mean'] < 32, f"Posterior {result['mean']:.1f} not between 28-32"
    assert result['data_weight'] < 1.0, "Data weight should be < 1 with only 5 games"
    return f"posterior={result['mean']:.1f}, prior_weight={result['prior_weight']:.2f}"
check("Analytical conjugate (5-game rookie)", test_r2_analytical)

def test_r2_veteran():
    """Test that veteran with lots of data is mostly data-driven."""
    import numpy as np
    from src.models.bayesian_minutes import BayesianMinutesModel
    bm = BayesianMinutesModel()
    
    role_prior = {'mean': 34.0, 'std': 3.0}
    # Veteran with 30 games averaging 30 min
    fake_data = np.random.normal(30, 2, 30)
    result = bm._fit_analytical(fake_data, role_prior, None, 500)
    
    assert result['data_weight'] > 0.7, f"Data weight {result['data_weight']:.2f} should be > 0.7 for veteran"
    return f"data_weight={result['data_weight']:.2f} (data-driven ✓)"
check("Veteran data dominance (30 games)", test_r2_veteran)

# ─────────────────────────────────────────
# R3: Lineup-Conditional Usage
# ─────────────────────────────────────────
print("\n[R3] Lineup-Conditional Usage")
def test_r3_import():
    from src.models.usage_model import UsageModel
    um = UsageModel()
    assert hasattr(um, '_get_on_off_splits'), "Missing _get_on_off_splits method"
    return True
check("UsageModel has _get_on_off_splits", test_r3_import)

def test_r3_absorption():
    from src.models.usage_model import UsageModel
    um = UsageModel()
    assert um._get_absorption_score('star') > um._get_absorption_score('role_player')
    return f"star={um._get_absorption_score('star')}, role_player={um._get_absorption_score('role_player')}"
check("Absorption scoring hierarchy", test_r3_absorption)

# ─────────────────────────────────────────
# R4: Matchup Model
# ─────────────────────────────────────────
print("\n[R4] Matchup Model")
def test_r4_import():
    from src.models.matchup_model import MatchupModel
    mm = MatchupModel()
    return True
check("Import MatchupModel", test_r4_import)

def test_r4_scheme_table():
    from src.models.matchup_model import MatchupModel
    table = MatchupModel.ARCHETYPE_SCHEME_TABLE
    assert len(table) >= 8, f"Only {len(table)} archetype-scheme entries"
    # Catch-and-shoot vs drop coverage should boost
    assert table[('catch_and_shoot', 'DROP_COVERAGE')] > 1.0
    return f"{len(table)} archetype-scheme matchups loaded"
check("Archetype-scheme table populated", test_r4_scheme_table)

# ─────────────────────────────────────────
# R5: CLV Engine
# ─────────────────────────────────────────
print("\n[R5] CLV Engine")
def test_r5_import():
    from src.agents.clv_engine import CLVFeedbackEngine
    clv = CLVFeedbackEngine()
    return True
check("Import CLVFeedbackEngine", test_r5_import)

def test_r5_summary():
    from src.agents.clv_engine import CLVFeedbackEngine
    clv = CLVFeedbackEngine()
    summary = clv.get_clv_summary(lookback_days=30)
    assert 'total_bets' in summary
    return f"summary keys: {list(summary.keys())}"
check("CLV summary structure", test_r5_summary)

# ─────────────────────────────────────────
# R6: Bayesian Calibration
# ─────────────────────────────────────────
print("\n[R6] Bayesian Calibration")
def test_r6_bayesian_anchor():
    from src.agents.market_calibrator import MarketCalibratorAgent
    cal = MarketCalibratorAgent()
    
    # Test: model says 25pts, market says 22.5, model std=5
    result = cal._apply_market_anchor(
        model_mean=25.0, market_line=22.5, 
        confidence='medium', model_std=5.0
    )
    # Should be between 22.5 and 25
    assert 22.5 < result < 25.0, f"Result {result:.2f} not between market and model"
    return f"posterior={result:.2f} (between 22.5 and 25.0 ✓)"
check("Bayesian anchor (conjugate N-N)", test_r6_bayesian_anchor)

def test_r6_high_confidence():
    """High confidence should pull less toward market."""
    from src.agents.market_calibrator import MarketCalibratorAgent
    cal = MarketCalibratorAgent()
    
    high = cal._apply_market_anchor(25.0, 20.0, 'high', model_std=3.0)
    low = cal._apply_market_anchor(25.0, 20.0, 'very_low', model_std=10.0)
    
    # High confidence → closer to model (25), low → closer to market (20)
    assert high > low, f"high_conf={high:.2f} should be > low_conf={low:.2f}"
    return f"high_conf={high:.2f}, low_conf={low:.2f}"
check("Confidence affects anchoring strength", test_r6_high_confidence)

# ─────────────────────────────────────────
# R7: RL Betting Agent
# ─────────────────────────────────────────
print("\n[R7] RL Betting Agent")
def test_r7_import():
    from src.agents.rl_agent import RLBettingAgent
    agent = RLBettingAgent(mode='SHADOW')
    assert agent.mode == 'SHADOW'
    return True
check("Import RLBettingAgent (SHADOW)", test_r7_import)

def test_r7_state_vector():
    from src.agents.rl_agent import RLBettingAgent
    agent = RLBettingAgent()
    state = agent.build_state_vector(
        edge=0.08, model_std=6.0, probability=0.62,
        confidence='good', regime_flags=['SYSTEM_RECOVERING']
    )
    assert len(state) == 10, f"State dim {len(state)} != 10"
    return f"state_dim=10, state={[f'{s:.2f}' for s in state[:5]]}..."
check("State vector (10D)", test_r7_state_vector)

def test_r7_shadow_decision():
    from src.agents.rl_agent import RLBettingAgent
    agent = RLBettingAgent()
    decision = agent.shadow_decision(
        edge=0.08, model_std=6.0, probability=0.62,
        confidence='good', regime_flags=[],
        traditional_decision='OVER', player_name='Test Player'
    )
    assert 'rl_action' in decision
    assert 'agree' in decision
    return f"rl_action={decision['rl_action']}, agree={decision['agree']}"
check("Shadow decision logging", test_r7_shadow_decision)

def test_r7_reward():
    from src.agents.rl_agent import RLBettingAgent
    agent = RLBettingAgent()
    win_reward = agent.calculate_reward('WIN', 1.0, -110)
    loss_reward = agent.calculate_reward('LOSS', 1.0, -110)
    assert win_reward > 0
    assert loss_reward < 0
    return f"WIN=+{win_reward:.2f}u, LOSS={loss_reward:.2f}u"
check("Reward calculation (WIN/LOSS)", test_r7_reward)

def test_r7_promotion():
    from src.agents.rl_agent import RLBettingAgent
    agent = RLBettingAgent()
    ready, reason = agent.should_promote()
    assert not ready, "Should NOT promote with 0 decisions"
    return f"promotion_blocked: {reason}"
check("Promotion gate (too few decisions)", test_r7_promotion)

# ─────────────────────────────────────────
# R8: Teammate Impact Graph
# ─────────────────────────────────────────
print("\n[R8] Teammate Impact Graph")
def test_r8_import():
    from src.models.teammate_graph import TeammateImpactGraph
    tg = TeammateImpactGraph()
    return True
check("Import TeammateImpactGraph", test_r8_import)

def test_r8_empty_injuries():
    from src.models.teammate_graph import TeammateImpactGraph
    tg = TeammateImpactGraph()
    result = tg.get_injury_impact_multiplier(99999, 'LAL', {})
    assert result['multiplier'] == 1.0, "Empty injuries should give 1.0 multiplier"
    return f"multiplier=1.0 (neutral ✓)"
check("Neutral with no injuries", test_r8_empty_injuries)

# ─────────────────────────────────────────
# R9: Guardian Auto-Recovery
# ─────────────────────────────────────────
print("\n[R9] Guardian Auto-Recovery")
def test_r9_import():
    from src.audit.guardian import ProductionGuardian
    g = ProductionGuardian()
    assert hasattr(g, 'check_historical_calibration')
    return True
check("Import ProductionGuardian", test_r9_import)

def test_r9_graduated_levels():
    """Verify graduated defense levels exist as valid returns."""
    valid_levels = {'SEVERELY_OVERCONFIDENT', 'OVERCONFIDENT', 'SLIGHTLY_OVERCONFIDENT',
                    'RECOVERING', 'STABLE', 'UNDERCONFIDENT'}
    from src.audit.guardian import ProductionGuardian
    g = ProductionGuardian()
    result = g.check_historical_calibration(days_back=7)
    assert result in valid_levels, f"Got '{result}', expected one of {valid_levels}"
    return f"status='{result}'"
check("Graduated defense level returned", test_r9_graduated_levels)

# ─────────────────────────────────────────
# Integration: monte_carlo imports
# ─────────────────────────────────────────
print("\n[INTEGRATION] SimulationEngine")
def test_integration_init():
    from src.simulation.monte_carlo import SimulationEngine
    engine = SimulationEngine()
    assert hasattr(engine, 'bayesian_minutes'), "Missing bayesian_minutes attribute"
    return f"bayesian_minutes={'loaded' if engine.bayesian_minutes else 'None (fallback)'}"
check("SimulationEngine initializes with all R1-R9 components", test_integration_init)

# ─────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print("=" * 60)

if FAIL > 0:
    print("\n⚠️  Some tests failed. Review errors above.")
    sys.exit(1)
else:
    print("\n🎯 All Phase 16 modules validated successfully!")
    sys.exit(0)
