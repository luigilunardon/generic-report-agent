"Graph definition."

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, START, StateGraph

from constants import RECOVERY_DIR
from utils.llm import check_hallucination, llm_t0, query_llm


@dataclass
class CreateState:
    """Represents the state of the create process.

    Fields:
        query (list): User query.
        background (str): Background information.
        create_output (str): Query results.
        retry (bool): Boolean value for hallucination handling.
        load_recovery (bool): Boolean value for loading recovery files.
        recovery_path (str): Name of the recovery file.
    """

    query: Optional[str] = None
    background: Optional[str] = None
    create_output: Optional[str] = None
    retry: Optional[bool] = False
    load_recovery: Optional[bool] = False
    recovery_path: Optional[str] = str(RECOVERY_DIR / "create.json")


def ask_query(x):
    """Ask query."""
    return query_llm(x, llm_t0, "create_output")


def check_answer(x):
    """Check answer."""
    human_prompt = (
        f"Query:\n{x.query}\n\n\n\n"
        f"AI generated text:\n{x.create_output}\n\n\n\n"
        f"Background:\n{x.background}"
    )
    return check_hallucination(
        x,
        llm_t0,
        "create_output",
        human_prompt,
    )


def create_graph_builder():
    """Build and compiles a LangGraph StateGraph.

    Returns:
        Compiled StateGraph object.

    """
    graph = StateGraph(CreateState)

    # ----------------------------------
    # Nodes
    # ----------------------------------

    graph.add_node("ask_query", ask_query)
    graph.add_node("check_answer", check_answer)

    # ----------------------------------
    # Edges
    # ----------------------------------

    graph.add_edge(START, "ask_query")
    graph.add_edge("ask_query", "check_answer")

    graph.add_conditional_edges(
        "check_answer",
        lambda s: s.retry,
        {
            "yes": "ask_query",
            "no": END,
        },
    )

    return graph.compile()
