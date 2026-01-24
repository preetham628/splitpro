from typing import Dict, List
from collections import defaultdict
from core.expense import Expense


class SplitEngine:
    """Calculate balances and who owes who"""
    
    def __init__(self, group_users: List[str]):
        self.users = group_users
        self.balances: Dict[str, float] = defaultdict(float)
    
    def process_expense(self, expense: Expense) -> None:
        """Process a single expense and update balances"""
        # Add what the payer paid
        self.balances[expense.paid_by] += expense.total_amount
        
        # Calculate per-person shares for each category
        veg_share_per_person = 0.0
        if expense.veg_participants:
            veg_share_per_person = expense.veg_amount / len(expense.veg_participants)
            for participant in expense.veg_participants:
                self.balances[participant] -= veg_share_per_person
        
        non_veg_share_per_person = 0.0
        if expense.non_veg_participants:
            non_veg_share_per_person = expense.non_veg_amount / len(expense.non_veg_participants)
            for participant in expense.non_veg_participants:
                self.balances[participant] -= non_veg_share_per_person
    
    def get_balances(self) -> Dict[str, float]:
        """Get current balances for all users"""
        return dict(self.balances)
    
    def reset(self) -> None:
        """Reset all balances"""
        self.balances.clear()
