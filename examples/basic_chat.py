"""
Basic example: start a chat session with headroom tracking.
Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
from headroom import Session, TokenBudget


async def main():
    session = Session(
        model="claude-haiku-4-5-20251001",
        budget=TokenBudget(limit=200_000),
        on_warning=lambda e: print(f"\n  [!] Budget {e.status}: {e.headroom:,} tokens remaining"),
    )

    print(f"headroom | model: {session.model} | limit: {session.budget.limit:,} tokens\n")

    messages = [
        "Hi! Can you explain what a context window is in LLMs?",
        "How does token counting work?",
        "What strategies help manage long conversations?",
    ]

    for msg in messages:
        print(f"You: {msg}")
        response = await session.send(msg)
        text = response.content[0].text
        print(f"Assistant: {text[:200]}{'...' if len(text) > 200 else ''}")
        u = session.token_usage
        print(f"  [tokens: {u.used:,} used | {u.headroom:,} headroom | {u.headroom_pct}% free]\n")

    session.export("basic_chat_session.json")
    print("Session exported to basic_chat_session.json")


if __name__ == "__main__":
    asyncio.run(main())
