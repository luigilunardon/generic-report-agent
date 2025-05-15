"Graph definition."

import os
import shutil
import sys
from pathlib import Path
from random import randint

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy

from constants import CONFIG_FILE, RECOVERY_DIR
from utils.graphs.create_graph import create_graph_builder
from utils.graphs.format_graph import format_graph_builder
from utils.graphs.search_graph import search_graph_builder
from utils.graphs.smart_search_graph import smart_search_graph_builder
from utils.graphs.states import (
    CreateState,
    FormatState,
    SearchState,
    SmartSearchState,
    TaskPlannerState,
)
from utils.llm import default_rate_limiter, human_validation_tasks, query_llm
from utils.load_data import load_config
from utils.save_file import save_state

config = load_config(CONFIG_FILE)

retry_policy = RetryPolicy(max_attempts=4)


async def run_subgraph(builder, state_class, state_args):
    """Run a subgraph."""
    graph = builder()
    state = state_class(**state_args)
    return await graph.ainvoke(state, {"max_concurrency": config["parameters"]["max_concurrency"]})


_task_handler = {
    "search": (
        search_graph_builder,
        SearchState,
        lambda q, _1, _2: {"queries": q},
        "search_summary",
    ),
    "create": (
        create_graph_builder,
        CreateState,
        lambda q, d, a: {"query": q, "background": "\n\n".join([a[i] for i in d])},
        "create_output",
    ),
    "format": (
        format_graph_builder,
        FormatState,
        lambda _, d, a,: {"background": "\n\n".join([a[i] for i in d])},
        "report",
    ),
    "smart_search": (
        smart_search_graph_builder,
        SmartSearchState,
        lambda _, d, a: {"background": "\n\n".join([a[i] for i in d])},
        "smart_search_summary",
    ),
}


def check_recovery(x):
    """Check the presence of the recovery state."""
    match (x.load_recovery, bool(x.tasks)):
        case True, True:
            return {"recovery_task": "execute_tasks"}
        case True, False:
            return {"recovery_task": "get_tasks"}
        case False, _:
            return {"recovery_task": "get_title"}


def get_title(x):
    """Generate a title."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )

    return query_llm(x, llm, "title")


def get_recovery_path(x):
    """Generate the recovery path."""
    directory_path = RECOVERY_DIR / x.title.replace(" ", "_")
    Path.mkdir(directory_path, exist_ok=True, parents=True)
    return {"recovery_path": str(directory_path / "task.json")}


def get_tasks(x):
    """Generate list of tasks."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )

    return query_llm(x, llm, "tasks", json_output=True)


def check_tasks(x):
    """Check list of tasks."""
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
        temperature=0.0,
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        rate_limiter=default_rate_limiter,
        model_kwargs={"seed": randint(0, 2**32)},
    )
    return human_validation_tasks(x, llm)


async def execute_tasks(x):
    """Execute the list of tasks."""
    recovery_file_path = x.recovery_path
    recovery_directory = RECOVERY_DIR / x.title.replace(" ", "_")

    first_task_index = len(x.task_output)
    task_output = x.task_output
    remaining_tasks = x.tasks[first_task_index:]

    for i, (task_type, query, background) in enumerate(remaining_tasks):
        updated_i = i + first_task_index
        builder, state_class, get_args, summary_field = _task_handler[task_type]
        state_args = get_args(query, background, task_output)
        state_args["load_recovery"] = False
        state_args["recovery_path"] = str(recovery_directory / f"{task_type}_{updated_i!s}.json")
        x.task_output = task_output
        save_state(x, recovery_file_path)
        try:
            answer = await run_subgraph(builder, state_class, state_args)
            task_output.append(answer[summary_field])
        except Exception as e:
            print(f"Error in task {updated_i}: {task_type}.\n\n{e}")
            sys.exit(1)
    if config["parameters"]["save_final_state"]:
        save_state(x, recovery_file_path)
    else:
        shutil.rmtree(recovery_directory)

    return {"task_output": task_output}


def task_graph_builder():
    """Build and compiles a LangGraph StateGraph.

    Returns:
        Compiled StateGraph object.

    """
    graph = StateGraph(TaskPlannerState)

    # ----------------------------------
    # Nodes
    # ----------------------------------

    graph.add_node("check_recovery", check_recovery)
    graph.add_node("get_title", get_title)
    graph.add_node("get_recovery_path", get_recovery_path)
    graph.add_node("get_tasks", get_tasks, retry=retry_policy)

    graph.add_node(
        "check_tasks",
        check_tasks,
    )

    graph.add_node("execute_tasks", execute_tasks)

    # ----------------------------------
    # Edges
    # ----------------------------------

    graph.add_edge(START, "check_recovery")
    graph.add_conditional_edges("check_recovery", lambda s: s.recovery_task)
    graph.add_edge("get_title", "get_recovery_path")
    graph.add_edge("get_recovery_path", "get_tasks")
    graph.add_edge("get_tasks", "check_tasks")
    graph.add_conditional_edges(
        "check_tasks",
        lambda s: s.retry,
        {
            "yes": "get_tasks",
            "no": "execute_tasks",
        },
    )
    graph.add_edge("execute_tasks", END)

    return graph.compile()
