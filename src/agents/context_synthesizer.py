
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..utils.database import DatabaseManager
from ..utils.config import get_config
from .data_gatherer import DataGathererAgent
from ..ingestion.news_scraper import NewsScraperAgent

logger = logging.getLogger(__name__)

@dataclass
class NarrativeSignal:
    type: str  # 'rest', 'rivalry', 'streak', 'news'
    score: float # -1.0 to 1.0 (Impact on performance/motivation)
    description: str

class NarrativeHunter:
    """
    Sub-component that hunts for soft signals/narratives.
    """
    def __init__(self, db: DatabaseManager):
        self.db = db

    def hunt(self, player_context: Dict[str, Any], match_context: Dict[str, Any]) -> List[NarrativeSignal]:
        signals = []
        
        # 1. Rest Advantage / Disadvantage (B2B)
        rest_days = player_context.get('rest_days', 1)
        if rest_days == 0:
            signals.append(NarrativeSignal(
                type='rest', 
                score=-0.1, 
                description="Playing on 0 days rest (B2B fatigue risk)"
            ))
        elif rest_days >= 3:
            signals.append(NarrativeSignal(
                type='rest', 
                score=0.05, 
                description=f"Well rested ({rest_days} days off)"
            ))
            
        # 2. Team Motivation (Tanking vs Contending)
        # Placeholder: Check win % or standings from DB if available
        # For now, we assume neutral unless explicit override
            
        # 3. Usage Trend (Context signal)
        signals_dict = player_context.get('inferred_signals', {})
        if signals_dict.get('usage_spike'):
             signals.append(NarrativeSignal(
                type='trend',
                score=0.1,
                description="Usage spiking significantly in last 3 games"
             ))
             
        if signals_dict.get('rotation_tightening'):
             signals.append(NarrativeSignal(
                type='trend',
                score=-0.15,
                description="Rotation tightening, minutes dropping"
             ))

        return signals


@dataclass
class PlayerBrief:
    """The synthesized intelligence briefing for a player."""
    player_id: int
    player_name: str
    team: str
    opponent: str
    
    # 1. Hard Stats (From DataGatherer)
    stats: Dict[str, Any]
    match_context: Dict[str, Any]
    
    # 2. News & Status (From NewsScraper / Injury DB)
    injury_status: str
    news_headline: Optional[str] = None
    news_report: Optional[str] = None
    
    # 3. Narrative (NarrativeHunter)
    narrative_score: float = 0.5  # 0.0=Negative, 0.5=Neutral, 1.0=Positive
    narrative_notes: List[str] = None

class ContextSynthesizerAgent:
    """
    The 'Intelligence Officer' Agent.
    
    Responsibilities:
    1. Orchestrate Data Gathering (Stats)
    2. Orchestrate News Gathering (Status)
    3. Synthesize 'Narrative' (Motivation/Context)
    4. Produce a 'PlayerBrief' for the Model
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.config = get_config()
        
        # Sub-Agents / Tools
        self.data_gatherer = DataGathererAgent(self.db)
        # NewsScraper is usually run *before* pipeline, but we can query its DB artifacts here
        
    def build_player_brief(self, player_id: int, opponent: str, reference_date: Optional[str] = None) -> Optional[PlayerBrief]:
        """
        Construct the full intelligence brief for a player.
        """
        # 1. Gather Hard Stats
        try:
            player_context = self.data_gatherer.gather_player_context(player_id, date_limit=reference_date)
            player_name = self._get_player_name(player_id)
            team = player_context.get('team', 'UNK')
            
            match_context = self.data_gatherer.gather_match_context(team, opponent, game_date=reference_date)
        except Exception as e:
            logger.error(f"ContextSynthesizer stats failed for {player_id}: {e}")
            return None

        # 2. Gather News & Injury Status
        # Check specific injury snapshots (populated by NewsScraper)
        injury_status = "AVAILABLE"
        news_headline = None
        news_report = None
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, reason, report_date 
                FROM injury_snapshots 
                WHERE player_id = ? 
                ORDER BY id DESC LIMIT 1
            """, (player_id,))
            row = cursor.fetchone()
            if row:
                # Check date relevance (e.g., is it from today/yesterday?)
                # For now, we take latest.
                injury_status = row['status']
                news_headline = f"Status: {row['status']}"
                news_report = row['reason']

        # 3. Narrative Analysis
        narrative_hunter = NarrativeHunter(self.db)
        narrative_signals = narrative_hunter.hunt(player_context, match_context)
        
        narrative_score = 0.5 # Neutral baseline
        narrative_notes = []
        
        for sig in narrative_signals:
            narrative_score += sig.score
            narrative_notes.append(sig.description)
            
        # Cap score 0.0 to 1.0
        narrative_score = max(0.0, min(1.0, narrative_score))
            
        return PlayerBrief(
            player_id=player_id,
            player_name=player_name,
            team=team,
            opponent=opponent,
            stats=player_context,
            match_context=match_context,
            injury_status=injury_status,
            news_headline=news_headline,
            news_report=news_report,
            narrative_score=narrative_score,
            narrative_notes=narrative_notes
        )

    def _get_player_name(self, player_id: int) -> str:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT full_name FROM players WHERE player_id = ?", (player_id,))
            row = cursor.fetchone()
            return row['full_name'] if row else f"ID_{player_id}"
