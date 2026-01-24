from typing import List, Optional


def parse_user_list(input_str: str) -> List[str]:
    """Parse comma-separated user names"""
    users = [name.strip() for name in input_str.split(",") if name.strip()]
    return users


def validate_user_names(users: List[str], available_users: List[str]) -> List[str]:
    """Validate that all user names exist in the group"""
    invalid = [u for u in users if u not in available_users]
    return invalid


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:.2f}"
