from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Group:
    """Simple group to hold users and expenses"""
    name: str
    users: List[str] = field(default_factory=list)
    expenses: List[Dict] = field(default_factory=list)
    
    def add_user(self, user_name: str) -> None:
        """Add a user to the group"""
        if user_name not in self.users:
            self.users.append(user_name)
    
    def add_expense(self, expense: Dict) -> None:
        """Add an expense to the group"""
        self.expenses.append(expense)
    
    def get_user_list(self) -> List[str]:
        """Get list of all users"""
        return self.users.copy()
