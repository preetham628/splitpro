from typing import List, Dict, Tuple
from collections import defaultdict


class Settlement:
    """Generate simplified settlement transactions"""
    
    @staticmethod
    def generate_settlements(balances: Dict[str, float]) -> List[Dict[str, any]]:
        """
        Generate simplified transactions to settle all debts.
        Uses a greedy algorithm to minimize number of transactions.
        
        Returns list of transactions: [{"from": "Alice", "to": "Bob", "amount": 30.0}, ...]
        """
        # Separate creditors (positive balance) and debtors (negative balance)
        creditors = []
        debtors = []
        
        for user, balance in balances.items():
            if balance > 0.001:  # Small threshold for floating point
                creditors.append((user, balance))
            elif balance < -0.001:
                debtors.append((user, abs(balance)))
        
        # Sort by amount (largest first)
        creditors.sort(key=lambda x: x[1], reverse=True)
        debtors.sort(key=lambda x: x[1], reverse=True)
        
        settlements = []
        creditor_idx = 0
        debtor_idx = 0
        
        # Greedy matching: match largest creditor with largest debtor
        while creditor_idx < len(creditors) and debtor_idx < len(debtors):
            creditor_name, creditor_amount = creditors[creditor_idx]
            debtor_name, debtor_amount = debtors[debtor_idx]
            
            if creditor_amount > debtor_amount:
                # Creditor gets partial payment, debtor is fully paid
                settlements.append({
                    "from": debtor_name,
                    "to": creditor_name,
                    "amount": round(debtor_amount, 2)
                })
                creditors[creditor_idx] = (creditor_name, creditor_amount - debtor_amount)
                debtor_idx += 1
            elif debtor_amount > creditor_amount:
                # Debtor pays partial, creditor is fully paid
                settlements.append({
                    "from": debtor_name,
                    "to": creditor_name,
                    "amount": round(creditor_amount, 2)
                })
                debtors[debtor_idx] = (debtor_name, debtor_amount - creditor_amount)
                creditor_idx += 1
            else:
                # Exact match
                settlements.append({
                    "from": debtor_name,
                    "to": creditor_name,
                    "amount": round(creditor_amount, 2)
                })
                creditor_idx += 1
                debtor_idx += 1
        
        return settlements
    
    @staticmethod
    def format_report(balances: Dict[str, float], settlements: List[Dict]) -> str:
        """Format a human-readable settlement report"""
        report = "\n" + "=" * 50 + "\n"
        report += "SETTLEMENT REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += "Net Balances:\n"
        report += "-" * 50 + "\n"
        for user, balance in sorted(balances.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(balance) > 0.01:  # Only show significant balances
                status = "owed" if balance > 0 else "owes"
                report += f"- {user}: ${abs(balance):.2f} ({status})\n"
        
        if settlements:
            report += "\n" + "Simplified Transactions:\n"
            report += "-" * 50 + "\n"
            for settlement in settlements:
                report += f"- {settlement['from']} pays {settlement['to']}: ${settlement['amount']:.2f}\n"
        else:
            report += "\nNo settlements needed - all balances are even!\n"
        
        report += "\n" + "=" * 50 + "\n"
        return report
