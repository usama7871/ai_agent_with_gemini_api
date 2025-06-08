# src/tools.py

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from typing import List
from src.utils import logger

def get_agent_tools() -> List[Tool]:
    """
    Provides a list of tools for the AI agent to use.

    Returns:
        List[Tool]: A list of Langchain Tool objects.
    """
    tools: List[Tool] = []

    # 1. DuckDuckGo Search Tool (Free & Simple)
    try:
        search = DuckDuckGoSearchRun()
        search_tool = Tool(
            name="DuckDuckGoSearch",
            func=search.run,
            description="Useful for when you need to answer questions about current events or look up information on the internet."
        )
        tools.append(search_tool)
        logger.info("Added DuckDuckGoSearch tool.")
    except Exception as e:
        logger.warning(f"Could not initialize DuckDuckGoSearch tool: {e}. Agent will not have web search capabilities.")

    # 2. Custom Dummy Tool (Example for your own functions)
    def get_current_time(timezone: str = "UTC") -> str:
        """Returns the current time in a specified timezone."""
        from datetime import datetime
        import pytz # You'd need to pip install pytz if you truly used this
        try:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
            return f"The current time in {timezone} is {now.strftime('%H:%M:%S')} on {now.strftime('%Y-%m-%d')}."
        except Exception:
            return f"Could not determine time for timezone {timezone}. Please provide a valid timezone."

    current_time_tool = Tool(
        name="CurrentTime",
        func=get_current_time,
        description="Useful for when you need to know the current time in a specific timezone. Input should be a timezone string like 'America/New_York' or 'UTC'."
    )
    # tools.append(current_time_tool) # Uncomment to enable this tool
    logger.info("Custom DummyTool 'CurrentTime' defined (not enabled by default).")

    # Add more custom tools here as needed.
    # Each tool should have a clear `name`, `func` (the function it calls),
    # and a detailed `description` for the LLM to understand its purpose and how to use it.

    return tools
