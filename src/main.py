"Main."

import asyncio

from constants import config
from utils.graphs.task_graph import TaskPlannerState, task_graph_builder
from utils.save_file import md_to_docx, mk_output_dir, save_md


async def pipeline(query: str):
    """Process the query.

    This function orchestrates the entire process, which includes:
    1. Generate a list of tasks.
    2. Performs the tasks.

    Args:
        query (str): The query to process.

    """
    graph = task_graph_builder()
    state = TaskPlannerState(query=query, load_recovery=False)

    answer = await graph.ainvoke(state, {"max_concurrency": 1})
    directory_name = answer["title"]
    directory = mk_output_dir(directory_name)
    final_report = answer["task_output"][-1]
    save_md(final_report, directory)
    md_to_docx(directory)


if __name__ == "__main__":
    """
    Entry point for the script. Get the query and run the pipeline.

    If a query is set in the environment variable (QUERY), it will be used.
    Otherwise, it prompts the user to input a company name.
    """
    config_query = config["parameters"]["query"]
    query = config_query if config_query else input("Enter the query: ")

    # Run the pipeline for the specified company
    asyncio.run(pipeline(query))
