"Graph definition."

from dataclasses import dataclass, field
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy

from config import RECOVERY_PATH
from utils.graphs.search_graph import SearchState, search_graph_builder
from utils.llm import llm_t0, query_llm

retry_policy = RetryPolicy(max_attempts=4)


@dataclass
class SmartSearchState:
    """Represents the state of the search process.

    Fields:
        background (str): Background material.
        smart_search_queries (str): AI-generated search queries.
        smart_search_summary (str): Summary of the results.
        retry (bool): Boolean value for hallucination handling.
        load_recovery (bool): Boolean value for loading recovery files.
        recovery_path (Path): Path of the recovery file.
    """

    background: Optional[str] = None
    smart_search_queries: Optional[list] = field(default_factory=list)
    smart_search_summary: Optional[str] = None
    retry: Optional[bool] = False
    load_recovery: Optional[bool] = False
    recovery_path: Optional[str] = str(RECOVERY_PATH / "smart_search.json")


def get_queries(x):
    """Get search results."""
    return query_llm(x, llm_t0, "smart_search_queries", json_output=True)


async def get_summary(x):
    """Summarise search results."""
    sub_graph = search_graph_builder()
    sub_state = SearchState(queries=x.smart_search_queries, load_recovery=False)
    answer = await sub_graph.ainvoke(sub_state, {"max_concurrency": 1})
    return {"smart_search_summary": answer.get("search_summary")}


def smart_search_graph_builder():
    """Builds and compiles a LangGraph StateGraph.

    Returns:
        Compiled StateGraph object.

    """
    graph = StateGraph(SmartSearchState)

    # ----------------------------------
    # Nodes
    # ----------------------------------

    graph.add_node("get_queries", get_queries, retry=retry_policy)
    graph.add_node("get_summary", get_summary)

    # ----------------------------------
    # Edges
    # ----------------------------------

    graph.add_edge(START, "get_queries")
    graph.add_edge("get_queries", "get_summary")
    graph.add_edge("get_summary", END)

    return graph.compile()
