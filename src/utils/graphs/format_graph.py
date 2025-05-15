"Graph definition."

import os
from random import randint

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from utils.graphs.states import FormatState
from utils.llm import default_rate_limiter, query_llm


def get_report(x):
    """Get the report."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )

    return query_llm(x, llm, "pre_report")


def format_report(x):
    """Format report."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )

    return query_llm(x, llm, "report")


def format_graph_builder():
    """Build and compiles a LangGraph StateGraph.

    Returns:
        Compiled StateGraph object.

    """
    graph = StateGraph(FormatState)

    # ----------------------------------
    # Nodes
    # ----------------------------------

    graph.add_node("get_report", get_report)
    graph.add_node("format_report", format_report)

    # ----------------------------------
    # Edges
    # ----------------------------------

    graph.add_edge(START, "get_report")
    graph.add_edge("get_report", "format_report")
    graph.add_edge("format_report", END)

    return graph.compile()
