
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from .validator import ValidatorAgent, Vote
from ...utils.database import DatabaseManager
from ...ingestion.news_scraper import NewsScraperAgent # Used for logic reference, data comes from DB

logger = logging.getLogger(__name__)

class NarrativeValidator(ValidatorAgent):
    """
    Validator B: The Narrative Hunter.
    Votes based on News, Injury Status, Motivation, and Rest.
    """
    
    def __init__(self, db: DatabaseManager):
        super().__init__(name="NarrativeValidator")
        self.db = db

    def validate(self, player_id: int, prop_type: str, line: float, reference_date: str = None) -> Vote:
        slate_dt = self._resolve_reference_date(reference_date)
        # 1. Check Injury Status (The Veto)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, reason, report_date, p_play, source_name, fetched_at
                FROM injury_snapshots 
                WHERE player_id = ? 
                ORDER BY
                    CASE
                        WHEN fetched_at IS NULL THEN 1
                        ELSE 0
                    END,
                    fetched_at DESC,
                    id DESC
                LIMIT 1
            """, (player_id,))
            row = cursor.fetchone()
             
        if row:
            status = row['status']
            p_play = row['p_play']
            reason = row['reason'] or "No reason provided"
            fetched_at = self._parse_datetime(row['fetched_at'])
            report_dt = self._parse_datetime(row['report_date'])
            is_stale = self._is_stale_injury_snapshot(status, p_play, fetched_at, report_dt, slate_dt)

            if is_stale:
                return Vote(
                    agent_name=self.name,
                    verdict="ABSTAIN",
                    confidence=0.35,
                    reason=f"Stale injury note ignored: {status} ({reason})",
                    metadata={
                        'status': status,
                        'stale': True,
                        'fetched_at': row['fetched_at'],
                        'report_date': row['report_date'],
                        'source_name': row['source_name'],
                    }
                )
            
            # VETO: If player is OUT or DOUBTFUL
            if p_play < 0.5:
                return Vote(
                    agent_name=self.name,
                    verdict="NO",
                    confidence=1.0, # High confidence veto
                    reason=f"Player Status: {status} ({reason})",
                    metadata={'status': status}
                )
            
            # CAUTION: If GTD
            if status == 'GTD':
                 return Vote(
                    agent_name=self.name,
                    verdict="NO",
                    confidence=0.8,
                    reason=f"Player is GTD (Risk Assessment)",
                    metadata={'status': status}
                )

        # 2. Check Narrative Factors (Rest, B2B)
        # We need player context for this. Ideally passed in, but for now we fetch.
        # (Simplified for Validator MVP - direct DB check for games)
        
        # Check for B2B
        is_b2b = self._check_b2b(player_id, slate_dt)
        if is_b2b:
             return Vote(
                agent_name=self.name,
                verdict="NO", # Generally fade B2B players on overs
                confidence=0.6,
                reason="Player on 0 days rest (B2B)",
                metadata={'rest_days': 0}
            )

        # 3. Default: Optimistic Approval
        return Vote(
            agent_name=self.name,
            verdict="YES",
            confidence=0.5,
            reason="No negative narrative signals found.",
            metadata={}
        )

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _is_stale_injury_snapshot(
        self,
        status: Optional[str],
        p_play: Optional[float],
        fetched_at: Optional[datetime],
        report_dt: Optional[datetime],
        reference_dt: datetime,
    ) -> bool:
        status_text = (status or "").upper()

        freshness_cutoff_days = 2 if (p_play is not None and p_play < 0.5) else 1
        if status_text in {"GTD", "QUESTIONABLE", "PROBABLE", "DAY-TO-DAY", "AVAILABLE"}:
            freshness_cutoff_days = 1

        if fetched_at and (reference_dt - fetched_at) <= timedelta(days=freshness_cutoff_days):
            return False

        if report_dt and (reference_dt.date() - report_dt.date()).days <= freshness_cutoff_days:
            return False

        # If neither timestamp is fresh enough, do not let an old note veto today's market.
        return True

    def _check_b2b(self, player_id: int, reference_dt: datetime) -> bool:
        """Check if player played yesterday."""
        yesterday = (reference_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 1 FROM player_logs 
                WHERE player_id = ? AND game_date = ?
            """, (player_id, yesterday))
            return cursor.fetchone() is not None

    def _resolve_reference_date(self, reference_date: Optional[str]) -> datetime:
        if reference_date:
            parsed = self._parse_datetime(reference_date)
            if parsed:
                return parsed
        return datetime.now()
