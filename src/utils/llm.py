"LLM querying functions."

import json
import sys
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter

from constants import CONFIG_FILE, PROMPT_FILE
from utils.save_file import save_state

with Path.open(PROMPT_FILE) as file:
    PROMPTS = json.load(file)

from dotenv import load_dotenv

from utils.load_data import load_api_key, load_config

config = load_config(CONFIG_FILE)
load_api_key({'groq'})
load_dotenv()


default_rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)


def fix_task_json(tasks):
    """Try to fix the json file if possible.

    Args:
        tasks (list): list of tasks.

    Returns:
        dict: fixed tasks list, if possible.

    """
    checked_tasks = []

    try:
        for i, task in enumerate(tasks):
            new_task = task
            if task[0] not in {"format", "create", "smart_search", "search"}:
                return {}
            if (
                task[0] in {"format", "smart_search"}
                and len(task) == config["parameters"]["task_length"] - 1
            ):
                new_task = [task[0], "", task[1]]
            if len(new_task) != config["parameters"]["task_length"]:
                return {}
            if task[0] == "create" and task[1] == "":
                return {}
            checked_tasks.append([new_task[0], new_task[1], [j for j in new_task[2] if j < i]])
        return {"tasks": checked_tasks}
    except Exception:
        return {}


def human_validation_tasks(state, llm):
    """Use an llm to explain the suggested task workflow in human readable form.

    Args:
        state (dict): Input state containing the field to validate.
        llm (ChatGroq): Language model instance used.

    Returns:
        dict: "no" if no retry is required, "yes" otherwise.

    """
    if state.max_retry < 0:
        message = (
            "Failed to generate the list of tasks."
            "State saved, try manual debugging.\n"
            f"Recovery file: {state.recovery_path}"
        )
        print(message)
        save_state(state, state.recovery_path)
        sys.exit(1)
    prompt_name = "TASKS_VALIDATION_PROMPT"
    field_state = fix_task_json(state.tasks).get("tasks", "")
    if not field_state:
        return {"retry": "yes", "max_retry": state.max_retry - 1}
    prompt = PromptTemplate(template=PROMPTS[prompt_name].get("text"), input_variables=["tasks"])
    prompt_chain = prompt | llm | StrOutputParser()
    llm_answer = prompt_chain.invoke(field_state)
    print(f"\nSUGGESTED WORKFLOW:\n\n{llm_answer}\n\n")
    while True:
        user_answer = input("Proceed? (y/n): ")
        if user_answer:
            if user_answer[0].lower() == "y":
                return {"retry": "no"}
            if user_answer[0].lower() == "n":
                return {"retry": "yes", "max_retry": state.max_retry - 1}


def query_llm(state, llm, field_name, json_output=False):
    """Construct and runs a prompt chain with the LLM based on the given state and prompt.

    Args:
        state (StateGraph State): Input state.
        llm (ChatGroq): Language model instance.
        field_name (str): Key name for returning the result.
        json_output (bool): Whether the output is expected to be a json file.

    Returns:
        dict: Result dictionary with the LLM response under field_name.

    """
    prompt_name = f"{field_name.upper()}_PROMPT"
    if not state.load_recovery or (
        getattr(state, field_name) is None or not getattr(state, field_name)
    ):
        state.load_recovery = False
        keys = PROMPTS[prompt_name].get("keywords", [])
        text = PROMPTS[prompt_name].get("text", "")

        relevant_states = {key: getattr(state, key) for key in keys}

        prompt = PromptTemplate(template=text, input_variables=keys)
        try:
            if not json_output:
                prompt_chain = prompt | llm | StrOutputParser()
                llm_answer = prompt_chain.invoke(relevant_states)
                return {field_name: getattr(llm_answer, "content", llm_answer)}
            prompt_chain = prompt | llm | StrOutputParser()
            llm_answer = json.loads(prompt_chain.invoke(relevant_states))
            answer = dict(llm_answer)
            answer["load_recovery"] = False
            return dict(answer)

        except Exception as e:
            print(e)
            state.load_recovery = True
            path = state.recovery_path
            save_state(state, path)
            sys.exit(1)
    else:
        return {}


def check_hallucination(state, llm, field_name, human_prompt=""):
    """Check a given field in the state for hallucinations using a dedicated grading prompt.

    Args:
        state (dict): Input state containing the field to validate.
        llm (ChatGroq): Language model instance used for the grading.
        field_name (str): Field to check.
        human_prompt (str): Hallucination data.

    Returns:
        dict: "yes" if no hallucination detected, "no" otherwise.

    """
    if not state.load_recovery:
        max_retry = state.max_retry
        if max_retry <= 0:
            hallucination_message = (
                f"WARNING **HALLUCINATION DETECTED**\n\n{getattr(state, field_name)}"
            )
            return {"retry": "no", field_name: hallucination_message}
        system_prompt = PROMPTS[f"{field_name.upper()}_HALLUCINATION"].get("text", "")
        human_prompt = f"{field_name.replace('_', ' ')}: {getattr(state, field_name)}"
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

        hallucination_grader = prompt | llm | StrOutputParser()
        score = ""
        while score not in {"yes", "no"}:
            try:
                if max_retry <= 0:
                    hallucination_message = (
                        f"WARNING **AMBIGUOUS ANSWER**\n\n{getattr(state, field_name)}"
                    )
                    return {
                        "retry": "no",
                        field_name: hallucination_message,
                        "max_retry": max_retry,
                    }
                max_retry = max_retry - 1
                score = hallucination_grader.invoke({})
            except Exception as e:
                print(e)
                state.load_recovery = True
                state.max_retry = 3
                delattr(state, field_name)
                path = state.recovery_path
                save_state(state, path)
                sys.exit(1)
        return {"retry": score, "max_retry": max_retry}
    return {"retry": "no"}
