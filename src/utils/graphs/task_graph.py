"Graph definition."

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy

from config import RECOVERY_PATH, SAVE_FINAL_STATE
from utils.graphs.create_graph import CreateState, create_graph_builder
from utils.graphs.format_graph import FormatState, format_graph_builder
from utils.graphs.search_graph import SearchState, search_graph_builder
from utils.graphs.smart_search_graph import SmartSearchState, smart_search_graph_builder
from utils.llm import human_validation_llm, llm_t0, query_llm
from utils.save_file import save_state

retry_policy = RetryPolicy(max_attempts=4)


async def run_subgraph(builder, state_class, state_args):
    """Run a subgraph."""
    graph = builder()
    state = state_class(**state_args)
    return await graph.ainvoke(state, {"max_concurrency": 1})


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


@dataclass
class TaskPlannerState:
    """Represents the complete state of the profiling process.

    Fields:
        query (str): User query
        title (str): Name summarising the query
        tasks (list): List of tasks to solve the query
        retry (bool): Boolean value for hallucination handling.
        load_recovery (bool): Boolean value for loading recovery files.
        recovery_path (str): Path of the recovery file.
    """

    query: Optional[str] = None
    title: Optional[str] = None
    tasks: Optional[list] = field(default_factory=list)
    retry: Optional[bool] = False
    load_recovery: Optional[bool] = False
    task_output: Optional[list] = field(default_factory=list)
    recovery_path: Optional[Path] = str(RECOVERY_PATH / "task.json")


def get_title(x):
    """Generate a title."""
    return query_llm(x, llm_t0, "title")


def get_recovery(x):
    """Generate the recovery path."""
    directory_path = RECOVERY_PATH / x.title.replace(" ", "_")
    Path.mkdir(directory_path, exist_ok=True, parents=True)
    return {"recovery_path": str(directory_path / "task.json")}


def get_tasks(x):
    """Generate list of tasks."""
    return query_llm(x, llm_t0, "tasks", json_output=True)


def check_tasks(x):
    """Check list of tasks."""
    return human_validation_llm(x, llm_t0, "tasks")


async def execute_tasks(x):
    """Execute the list of tasks."""
    tasks = x.tasks
    task_output = []
    recovery_file_path = x.recovery_path
    recovery_directory = RECOVERY_PATH / x.title.replace(" ", "_")

    for i, (task_type, query, background) in enumerate(tasks[len(task_output) :]):
        builder, state_class, get_args, summary_field = _task_handler[task_type]
        state_args = get_args(query, background, task_output)
        state_args["load_recovery"] = False
        state_args["recovery_path"] = str(recovery_directory / f"{task_type}_{i!s}.json")
        save_state(x, recovery_file_path)
        try:
            answer = await run_subgraph(builder, state_class, state_args)
            task_output.append(answer[summary_field])
        except Exception as e:
            print(f"Error in task {i}: {task_type}.\n\n{e}")
            x.load_recovery = True
            sys.exit(1)
    if SAVE_FINAL_STATE:
        save_state(x, recovery_file_path)
    else:
        shutil.rmtree(recovery_directory)

    return {"task_output": task_output}


def task_graph_builder():
    """Builds and compiles a LangGraph StateGraph.

    Returns:
        Compiled StateGraph object.

    """
    graph = StateGraph(TaskPlannerState)

    # ----------------------------------
    # Nodes
    # ----------------------------------

    graph.add_node("get_title", get_title)
    graph.add_node("get_recovery", get_recovery)
    graph.add_node("get_tasks", get_tasks, retry=retry_policy)

    graph.add_node(
        "check_tasks",
        check_tasks,
    )

    graph.add_node("execute_tasks", execute_tasks)

    # ----------------------------------
    # Edges
    # ----------------------------------

    graph.add_edge(START, "get_title")
    graph.add_edge("get_title", "get_recovery")
    graph.add_edge("get_recovery", "get_tasks")
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
