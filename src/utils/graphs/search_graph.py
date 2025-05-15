"Graph definition."

import os
from random import randint

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from utils.graphs.states import SearchState
from utils.llm import check_hallucination, default_rate_limiter, query_llm
from utils.web_search import web_search


async def get_search(x):
    """Get search results."""
    return await web_search(x, "search_results", x.queries)


def get_summary(x):
    """Summarise search results."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )

    return query_llm(x, llm, "search_summary")


def check_summary(x):
    """Check summary."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
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
