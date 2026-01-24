# SplitPro

Smart bill splitting app with AI-powered category-based expense splitting.

## Features

- Create expense groups with multiple users
- Add unlimited expenses per group
- AI-powered bill analysis to suggest veg/non-veg splits
- Automatic balance calculation
- Simplified settlement reports (minimizes transactions)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```

2. **Set up OpenAI API key:**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Get your API key from: https://platform.openai.com/api-keys

## Usage

Run the application:
```bash
python main.py
```

### Flow:

1. **Create Group** - Enter a name for your expense group
2. **Add Users** - Add user names (one per line, type 'done' when finished)
3. **Add Expenses** - For each expense:
   - Enter bill description or paste the bill text
   - AI will analyze and suggest veg/non-veg split
   - Confirm or modify the split amounts
   - Enter who paid
   - Enter participants for each category
4. **Generate Report** - View balances and simplified settlement transactions

## Project Structure

```
splitpro/
├── main.py                 # CLI entry point
├── core/                   # Core business logic
│   ├── group.py           # Group and user management
│   ├── expense.py         # Expense data model
│   ├── split_engine.py    # Balance calculation
│   └── settlement.py      # Settlement report generation
├── agents/                 # AI agents
│   └── bill_analyzer.py   # Bill analysis using OpenAI
└── utils/                  # Utility functions
    └── helpers.py
```

## Example

```
> Create expense group
> Enter group name: Weekend Trip

> Add users
> User name: Alice
> User name: Bob
> User name: Charlie
> User name: done

> Add expense
> Enter expense description: Dinner at restaurant - $120
> AI suggests: Veg $60, Non-veg $60
> Who paid: Alice
> Veg participants: Alice, Bob
> Non-veg participants: Charlie

> Generate report
> Shows who owes who and simplified transactions
```
