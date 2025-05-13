"Graph definition."

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, START, StateGraph

from config import RECOVERY_PATH
from utils.llm import llm_t0, query_llm


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
    recovery_path: Optional[str] = str(RECOVERY_PATH / "format.json")


def get_report(x):
    """Get the report."""
    return query_llm(x, llm_t0, "pre_report")


def format_report(x):
    """Format report."""
    return query_llm(x, llm_t0, "report")


def format_graph_builder():
    """Builds and compiles a LangGraph StateGraph.

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
