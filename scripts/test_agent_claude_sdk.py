import anyio
import os

print(os.getenv('ANTHROPIC_API_KEY'))
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)