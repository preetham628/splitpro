from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Expense:
    """Represents a single expense with category breakdown"""
    description: str
    paid_by: str
    total_amount: float
    veg_amount: float = 0.0
    non_veg_amount: float = 0.0
    veg_participants: List[str] = None
    non_veg_participants: List[str] = None
    
    def __post_init__(self):
        if self.veg_participants is None:
            self.veg_participants = []
        if self.non_veg_participants is None:
            self.non_veg_participants = []
    
    def to_dict(self) -> Dict:
        """Convert expense to dictionary"""
        return {
            "description": self.description,
            "paid_by": self.paid_by,
            "total_amount": self.total_amount,
            "veg": {
                "amount": self.veg_amount,
                "participants": self.veg_participants
            },
            "non_veg": {
                "amount": self.non_veg_amount,
                "participants": self.non_veg_participants
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Expense':
        """Create expense from dictionary"""
        return cls(
            description=data["description"],
            paid_by=data["paid_by"],
            total_amount=data["total_amount"],
            veg_amount=data.get("veg", {}).get("amount", 0.0),
            non_veg_amount=data.get("non_veg", {}).get("amount", 0.0),
            veg_participants=data.get("veg", {}).get("participants", []),
            non_veg_participants=data.get("non_veg", {}).get("participants", [])
        )
