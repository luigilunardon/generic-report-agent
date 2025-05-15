"Graph definition."

import os
from dataclasses import dataclass
from typing import Optional

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from constants import RECOVERY_DIR
from utils.llm import default_rate_limiter, query_llm


@dataclass
class FormatState:
    """Represents the state of the format process.

    Fields:
        background (str): Background information.
        pre_report (str): Preliminary report.
        formatted_report (str): Final report with PANDOC compatibility.
        load_recovery (bool): Boolean value for loading recovery files.
        recovery_path (str): Path of the recovery file.
    """

    background: Optional[str] = None
    pre_report: Optional[str] = None
    report: Optional[str] = None
    load_recovery: Optional[bool] = False
    recovery_path: Optional[str] = str(RECOVERY_DIR / "format.json")


def get_report(x):
    """Get the report."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
    )

    return query_llm(x, llm, "pre_report")


def format_report(x):
    """Format report."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
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
