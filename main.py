#!/usr/bin/env python3
"""
SplitPro - Smart Bill Splitting App
Simple CLI app for splitting expenses with category-based smart splits
"""

from core.group import Group
from core.expense import Expense
from core.split_engine import SplitEngine
from core.settlement import Settlement
from agents.bill_analyzer import BillAnalyzer
from utils.helpers import parse_user_list, validate_user_names


def create_group() -> Group:
    """Create a new expense group"""
    print("\n" + "=" * 50)
    print("CREATE EXPENSE GROUP")
    print("=" * 50)
    name = input("Enter group name: ").strip()
    if not name:
        name = "Untitled Group"
    return Group(name=name)


def add_users(group: Group) -> None:
    """Add users to the group"""
    print("\n" + "=" * 50)
    print("ADD USERS")
    print("=" * 50)
    print("Enter user names (one per line, or 'done' to finish)")
    
    while True:
        user_name = input("> User name (or 'done'): ").strip()
        if user_name.lower() == 'done':
            break
        if user_name:
            group.add_user(user_name)
            print(f"  Added: {user_name}")
    
    if group.users:
        print(f"\nUsers in group: {', '.join(group.users)}")
    else:
        print("\nNo users added!")


def add_expense(group: Group, analyzer: BillAnalyzer) -> None:
    """Add an expense with AI-assisted category splitting"""
    print("\n" + "=" * 50)
    print("ADD EXPENSE")
    print("=" * 50)
    
    # Get expense description
    description = input("Enter expense description or paste bill: ").strip()
    if not description:
        print("Expense description cannot be empty!")
        return
    
    # Analyze bill with AI
    print("\nAnalyzing bill...")
    try:
        suggestion = analyzer.analyze_bill(description)
        print(f"\nAI Analysis:")
        print(f"  Veg amount: ${suggestion['veg_amount']:.2f}")
        print(f"  Non-veg amount: ${suggestion['non_veg_amount']:.2f}")
        print(f"  Reasoning: {suggestion['reasoning']}")
        print(f"  Confidence: {suggestion['confidence']}")
        
        # Ask for confirmation or modification
        print("\nOptions:")
        print("1. Use AI suggestion")
        print("2. Enter custom amounts")
        choice = input("Choose (1 or 2): ").strip()
        
        if choice == "2":
            veg_amount = float(input("Enter veg amount: $"))
            non_veg_amount = float(input("Enter non-veg amount: $"))
        else:
            veg_amount = suggestion['veg_amount']
            non_veg_amount = suggestion['non_veg_amount']
    except Exception as e:
        print(f"Error in analysis: {e}")
        # Fallback to manual input
        total = float(input("Enter total amount: $"))
        veg_amount = float(input("Enter veg amount: $"))
        non_veg_amount = total - veg_amount
    
    total_amount = veg_amount + non_veg_amount
    
    # Get who paid
    print(f"\nAvailable users: {', '.join(group.users)}")
    paid_by = input("Who paid for this expense? ").strip()
    if paid_by not in group.users:
        print(f"Warning: '{paid_by}' not in group. Adding anyway...")
        group.add_user(paid_by)
    
    # Get participants
    print(f"\nEnter participants (comma-separated names)")
    veg_participants_input = input("Veg participants: ").strip()
    veg_participants = parse_user_list(veg_participants_input)
    
    non_veg_participants_input = input("Non-veg participants: ").strip()
    non_veg_participants = parse_user_list(non_veg_participants_input)
    
    # Validate participants
    all_participants = veg_participants + non_veg_participants
    invalid = validate_user_names(all_participants, group.users)
    if invalid:
        print(f"Warning: These users are not in the group: {', '.join(invalid)}")
        response = input("Add them to the group? (y/n): ").strip().lower()
        if response == 'y':
            for user in invalid:
                group.add_user(user)
    
    # Create expense
    expense = Expense(
        description=description,
        paid_by=paid_by,
        total_amount=total_amount,
        veg_amount=veg_amount,
        non_veg_amount=non_veg_amount,
        veg_participants=veg_participants,
        non_veg_participants=non_veg_participants
    )
    
    group.add_expense(expense.to_dict())
    print(f"\n✓ Expense added successfully!")


def generate_report(group: Group) -> None:
    """Generate and display settlement report"""
    if not group.expenses:
        print("\nNo expenses to settle!")
        return
    
    # Calculate balances
    split_engine = SplitEngine(group.users)
    for exp_dict in group.expenses:
        expense = Expense.from_dict(exp_dict)
        split_engine.process_expense(expense)
    
    balances = split_engine.get_balances()
    settlements = Settlement.generate_settlements(balances)
    
    # Display report
    report = Settlement.format_report(balances, settlements)
    print(report)


def main():
    """Main CLI loop"""
    print("\n" + "=" * 50)
    print("SPLITPRO - Smart Bill Splitting")
    print("=" * 50)
    
    # Initialize AI analyzer
    try:
        analyzer = BillAnalyzer()
    except Exception as e:
        print(f"Warning: Could not initialize AI analyzer: {e}")
        print("Continuing without AI assistance...")
        analyzer = None
    
    # Create group
    group = create_group()
    
    # Add users
    add_users(group)
    
    if not group.users:
        print("\nNo users added. Exiting.")
        return
    
    # Main expense loop
    while True:
        print("\n" + "=" * 50)
        print("OPTIONS")
        print("=" * 50)
        print("1. Add expense")
        print("2. Generate settlement report")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            if analyzer:
                add_expense(group, analyzer)
            else:
                print("AI analyzer not available. Manual expense entry coming soon...")
        elif choice == "2":
            generate_report(group)
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
