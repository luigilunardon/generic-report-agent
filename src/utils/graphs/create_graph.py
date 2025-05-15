"Graph definition."

import os
from random import randint

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from utils.graphs.states import CreateState
from utils.llm import check_hallucination, default_rate_limiter, query_llm


def ask_query(x):
    """Ask query."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )
    return query_llm(x, llm, "create_output")


def check_answer(x):
    """Check answer."""
    human_prompt = (
        f"Query:\n{x.query}\n\n\n\n"
        f"AI generated text:\n{x.create_output}\n\n\n\n"
        f"Background:\n{x.background}"
    )
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )
    return check_hallucination(
        x,
        llm,
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
