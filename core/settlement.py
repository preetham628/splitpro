from typing import List, Dict, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from core.session_state import ParsedBill


class Settlement:
    """Generate simplified settlement transactions"""

    @staticmethod
    def compute_balances(
        participants: List[str], bills: List["ParsedBill"]
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Compute each person's net balance across all bills (positive = owed
        money, negative = owes money). Bills with no payer set are skipped.

        The single source of truth for this math — previously duplicated
        between agents/chat_agent.py's calculate_split tool and server.py's
        state-panel preview, and the two copies had drifted: the server.py
        one didn't handle qty_allocations (fractional/per-unit splits) at
        all, so the live preview and the finalized report could disagree.

        Returns (balances, warnings) — warnings flag items that had no
        explicit assignment and were split equally across all participants.
        """
        global_balances: Dict[str, float] = defaultdict(float)
        warnings: List[str] = []

        for bill in bills:
            if not bill.paid_by:
                continue
            person_subtotal: Dict[str, float] = defaultdict(float)

            for item in bill.items:
                if item.qty_allocations:
                    # Qty-based: pay unit_price * units consumed
                    unit_price = item.unit_price
                    allocated_qty = sum(item.qty_allocations.values())
                    for person, pqty in item.qty_allocations.items():
                        person_subtotal[person] += unit_price * pqty

                    # Unallocated remainder goes equally to assigned_to minus those with explicit qtys
                    remaining_qty = item.qty - allocated_qty
                    if remaining_qty > 1e-9:
                        remainder_people = [p for p in item.assigned_to if p not in item.qty_allocations]
                        if not remainder_people:
                            remainder_people = participants
                        if remainder_people:
                            share = unit_price * remaining_qty / len(remainder_people)
                            for person in remainder_people:
                                person_subtotal[person] += share
                else:
                    # Fall back to all participants if still unassigned
                    recipients = item.assigned_to if item.assigned_to else participants
                    if not item.assigned_to:
                        warnings.append(f"'{item.name}' in {bill.bill_id} had no assignment — split equally.")
                    if recipients:
                        share = item.price / len(recipients)
                        for person in recipients:
                            person_subtotal[person] += share

            # Proportionally distribute tax + tip
            bill_subtotal = sum(person_subtotal.values())
            combined_extra = bill.tax + bill.tip
            if combined_extra > 0:
                if bill_subtotal > 0:
                    for person in list(person_subtotal.keys()):
                        proportion = person_subtotal[person] / bill_subtotal
                        person_subtotal[person] += proportion * combined_extra
                elif participants:
                    equal_share = combined_extra / len(participants)
                    for person in participants:
                        person_subtotal[person] += equal_share

            # Accumulate into global balances
            bill_total = sum(person_subtotal.values())
            global_balances[bill.paid_by] += bill_total
            for person, amount in person_subtotal.items():
                global_balances[person] -= amount

        return dict(global_balances), warnings

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
