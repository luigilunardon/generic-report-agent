"""Graph state definition."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from constants import RECOVERY_DIR


@dataclass
class BaseState:
    """Represents the complete state of the profiling process.

    Fields:
        retry (bool): Boolean value for hallucination handling.
        load_recovery (bool): Boolean value for loading recovery files.
        recovery_path (str): Path of the recovery file.
        max_retry (int): Max number of checks in the creation of the tasks.
    """

    retry: Optional[bool] = False
    load_recovery: Optional[bool] = False
    recovery_path: Optional[Path] = str(RECOVERY_DIR / "base.json")
    max_retry: Optional[int] = 3


@dataclass
class TaskPlannerState(BaseState):
    """Represents the complete state of the profiling process.

    Fields:
        query (str): User query
        title (str): Name summarising the query
        tasks (list): List of tasks to solve the query
        recovery_task (str): First task to execute after the recover file is loaded.
        task_output (list): list of task outputs.
        recovery_path (str): Path of the recovery file.
        max_retry (int): Max number of checks in the creation of the tasks.
    """

    query: Optional[str] = None
    title: Optional[str] = None
    tasks: Optional[list] = field(default_factory=list)
    recovery_task: Optional[str] = None
    task_output: Optional[list] = field(default_factory=list)
    recovery_path: Optional[Path] = str(RECOVERY_DIR / "task.json")
    max_retry: Optional[int] = 5


@dataclass
class CreateState(BaseState):
    """Represents the state of the create process.

    Fields:
        query (list): User query.
        background (str): Background information.
        create_output (str): Query results.
        recovery_path (str): Name of the recovery file.
    """

    query: Optional[str] = None
    background: Optional[str] = None
    create_output: Optional[str] = None
    recovery_path: Optional[str] = str(RECOVERY_DIR / "create.json")


@dataclass
class FormatState(BaseState):
    """Represents the state of the format process.

    Fields:
        background (str): Background information.
        pre_report (str): Preliminary report.
        formatted_report (str): Final report with PANDOC compatibility.
        recovery_path (str): Path of the recovery file.
    """

    background: Optional[str] = None
    pre_report: Optional[str] = None
    report: Optional[str] = None
    recovery_path: Optional[str] = str(RECOVERY_DIR / "format.json")


@dataclass
class SearchState(BaseState):
    """Represents the state of the search process.

    Fields:
        queries (list): User queries.
        search_results (str): Query results.
        search_summary (str): Summary of the results.
    """

    queries: Optional[list] = field(default_factory=list)
    search_results: Optional[str] = None
    search_summary: Optional[str] = None


@dataclass
class SmartSearchState(BaseState):
    """Represents the state of the search process.

    Fields:
        background (str): Background material.
        smart_search_queries (str): AI-generated search queries.
        smart_search_summary (str): Summary of the results.
        recovery_path (Path): Path of the recovery file.
    """

    background: Optional[str] = None
    smart_search_queries: Optional[list] = field(default_factory=list)
    smart_search_summary: Optional[str] = None
    recovery_path: Optional[str] = str(RECOVERY_DIR / "smart_search.json")
