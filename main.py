#!/usr/bin/env python3
"""
SplitPro - Conversational Bill Splitting Agent
Chat with the AI to split bills fairly across your group.

Usage:
  python3 main.py                          # uses OpenAI (default)
  python3 main.py --provider bedrock       # uses AWS Bedrock
  python3 main.py --provider openai --model gpt-4o
"""

import argparse
from dotenv import load_dotenv
from agents.chat_agent import ChatAgent
from config import ChatAgentConfig, load_config

load_dotenv()

WELCOME = """
================================================
  SplitPro - Conversational Bill Splitting
================================================
Paste your bill(s), tell me who was there, and
I'll figure out who owes what.

Commands:
  summary  - show current session state
  quit     - exit
================================================
"""


def main():
    parser = argparse.ArgumentParser(description="SplitPro CLI")
    parser.add_argument("--provider", default=None, help="LLM provider: openai or bedrock (overrides config)")
    parser.add_argument("--model", default=None, help="Model name/ID override (overrides config)")
    parser.add_argument("--config", default="config/defaults.yaml", help="Path to config YAML")
    args = parser.parse_args()

    app_config = load_config(args.config)

    # CLI flags override config file values
    chat_config = ChatAgentConfig(
        provider=args.provider or app_config.chat_agent.provider,
        model=args.model or app_config.chat_agent.model,
        temperature=app_config.chat_agent.temperature,
        max_iterations=app_config.chat_agent.max_iterations,
    )

    print(WELCOME)
    print(f"Using provider: {chat_config.provider}\n")

    try:
        agent = ChatAgent(config=chat_config)
    except (ValueError, ImportError) as e:
        print(f"Error: {e}")
        return

    # Let the agent open the conversation
    opening = agent.chat("Hello, I'm ready to help split some bills.")
    print(f"SplitPro: {opening}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "summary":
            print("\n--- Current State ---")
            print(agent.state.state_summary())
            print("---------------------\n")
            continue

        response = agent.chat(user_input)
        print(f"\nSplitPro: {response}\n")


if __name__ == "__main__":
    main()
