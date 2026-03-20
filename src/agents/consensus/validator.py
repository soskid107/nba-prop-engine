
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Vote:
    agent_name: str
    verdict: str  # 'YES' (Over/Pass) or 'NO' (Under/Fail) or 'ABSTAIN'
    confidence: float # 0.0 to 1.0
    reason: str
    metadata: Dict[str, Any]

class ValidatorAgent(ABC):
    """
    Abstract Base Class for GenLayer-style 'Validators'.
    Each Validator observes the same Proposal (Player/Prop) but
    votes based on its own specific domain (Math, Narrative, Market).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def validate(self, player_id: int, prop_type: str, line: float, reference_date: str = None) -> Vote:
        """
        Analyze the prop and cast a vote.
        """
        pass
