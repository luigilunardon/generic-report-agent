"Web search utility functions."

import asyncio
import os
import sys

from dotenv import load_dotenv
from tavily import AsyncTavilyClient, TavilyClient

from constants import CONFIG_FILE
from utils.load_data import load_api_key, load_config
from utils.save_file import save_state

load_api_key(['tavily'])
load_dotenv()

config = load_config(CONFIG_FILE)

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
tavily_async_client = AsyncTavilyClient()

tavily_config = config["tavily"]

search_params = {
    "max_results": 3,
    "include_raw_content": True,
    "topic": tavily_config["topic"],
    "search_depth": "advanced",
    "chunks_per_source": 3,
}

if tavily_config["topic"] == "news":
    search_params["days"] = tavily_config["days"]


async def web_search(state, field_name, queries):
    """Search the web for each query and returns a formatted string of sources.

    Args:
        state (StateGraph State): The state of the graph.
        field_name (str): The name of the field in the state that will hold the search results.
        queries (list): Search queries.

    Returns:
        A list of sources, one for each query.

    """
    if not state.load_recovery or (
        getattr(state, field_name) is None or not getattr(state, field_name)
    ):
        state.load_recovery = False

        try:
            search_results = await asyncio.gather(
                *[tavily_async_client.search(query, **search_params) for query in queries]
            )
        except Exception as e:
            print(e)
            state.load_recovery = True
            path = state.recovery_path
            save_state(state, path)
            sys.exit(1)

        sources = [
            entry['content']
            for search_result in search_results
            for entry in search_result['results']
        ]

        unique_sources = list(set(sources))

        return {field_name: "\n\n".join(unique_sources), "load_recovery":False}
    return {}
