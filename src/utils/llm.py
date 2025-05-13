"LLM querying functions."

import json
import os
import sys
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from config import PROMPT_PATH
from utils.save_file import save_state

with Path.open(PROMPT_PATH) as file:
    PROMPTS = json.load(file)
from dotenv import load_dotenv

from utils.load_key import load_api_key

load_api_key({'groq'})
load_dotenv()


default_rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

llm_t0 = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
    temperature=0.0,
    max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
    rate_limiter=default_rate_limiter,
)


llm_t9 = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama3-70b-8192"),
    temperature=0.9,
    max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
    rate_limiter=default_rate_limiter,
)


def human_validation_llm(state, llm, field_name):
    """Use an llm to explain the suggested workflow in human readable form.

    Args:
        state (dict): Input state containing the field to validate.
        llm (ChatGroq): Language model instance used.
        field_name (str): Field to check.

    Returns:
        dict: "yes" if approved detected, "no" otherwise.

    """
    prompt_name = f"{field_name.upper()}_VALIDATION_PROMPT"
    field_state = getattr(state, field_name)
    prompt = PromptTemplate(template=PROMPTS[prompt_name].get("text"), input_variables=[field_name])
    prompt_chain = prompt | llm | StrOutputParser()
    llm_answer = prompt_chain.invoke(field_state)
    print(f"\nSUGGESTED WORKFLOW:\n\n{llm_answer}\n\n")
    while True:
        user_answer = input("Proceed? (y/n): ")
        if user_answer:
            if user_answer[0].lower() == "y":
                return {"retry": "no"}
            if user_answer[0].lower() == "n":
                return {"retry": "yes"}


def query_llm(state, llm, field_name, json_output=False):
    """Constructs and runs a prompt chain with the LLM based on the given state and prompt.

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
            return dict(llm_answer)

        except Exception as e:
            print(e)
            state.load_recovery = True
            path = state.recovery_path
            save_state(state, path)
            sys.exit(1)
    else:
        return {}


def check_hallucination(state, llm, field_name, human_prompt=""):
    """Checks a given field in the state for hallucinations using a dedicated grading prompt.

    Args:
        state (dict): Input state containing the field to validate.
        llm (ChatGroq): Language model instance used for the grading.
        field_name (str): Field to check.
        human_prompt (str): Hallucination data.

    Returns:
        dict: "yes" if no hallucination detected, "no" otherwise.
    """
    if not state.load_recovery:
        system_prompt = PROMPTS[f"{field_name.upper()}_HALLUCINATION"].get("text", "")
        human_prompt = f"{field_name.replace('_', ' ')}: {getattr(state, field_name)}"
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )

        hallucination_grader = prompt | llm | StrOutputParser()
        score = ""
        while score not in {"yes", "no"}:
            try:
                score = hallucination_grader.invoke({})
            except Exception as e:
                print(e)
                state.load_recovery = True
                delattr(state, field_name)
                path = state.recovery_path
                save_state(state, path)
                sys.exit(1)
        return {"retry": score}
    return {"retry": "no"}
