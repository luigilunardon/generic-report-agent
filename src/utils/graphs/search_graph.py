"Graph definition."

import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from constants import RECOVERY_DIR
from utils.llm import check_hallucination, default_rate_limiter, query_llm
from utils.web_search import web_search


@dataclass
class SearchState:
    """Represents the state of the search process.

    Fields:
        queries (list): User queries.
        search_results (str): Query results.
        search_summary (str): Summary of the results.
        retry (bool): Boolean value for hallucination handling.
        max_retry (int): Max number of hallucination retry checks.
        load_recovery (bool): Boolean value for loading recovery files.
        recovery_path (str): Path of the recovery file.
    """

    queries: Optional[list] = field(default_factory=list)
    search_results: Optional[str] = None
    search_summary: Optional[str] = None
    retry: Optional[bool] = False
    max_retry: Optional[int] = 3
    load_recovery: Optional[bool] = False
    recovery_path: Optional[str] = str(RECOVERY_DIR / "search.json")


async def get_search(x):
    """Get search results."""
    print("##########",sep ="", end = "")
    return await web_search(x, "search_results", x.queries)


def get_summary(x):
    """Summarise search results."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
    )

    return query_llm(x, llm, "search_summary")


def check_summary(x):
    """Check summary."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
    )
    human_prompt = f"Sources:\n{x.search_results}\n\n\n\nSummary:\n{x.search_summary}"
    return check_hallucination(x, llm, "search_summary", human_prompt)


def search_graph_builder():
    """Build and compiles a LangGraph StateGraph.

    Returns:
        Compiled StateGraph object.

    """
    graph = StateGraph(SearchState)

    # ----------------------------------
    # Nodes
    # ----------------------------------

    graph.add_node("web_search", get_search)
    graph.add_node("get_summary", get_summary)
    graph.add_node("check_summary", check_summary)

    # ----------------------------------
    # Edges
    # ----------------------------------

    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", "get_summary")
    graph.add_edge("get_summary", "check_summary")

    graph.add_conditional_edges(
        "check_summary",
        lambda s: s.retry,
        {
            "yes": "get_summary",
            "no": END,
        },
    )

    return graph.compile()
