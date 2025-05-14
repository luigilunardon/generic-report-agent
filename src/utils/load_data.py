"Load keys in the .env file."

import json
from pathlib import Path

import tomllib
from dotenv import get_key, set_key

from constants import RECOVERY_DIR, SRC_DIR
import shutil

def load_config(filename):
    """Load config from toml file.

    Args:
        filename (Path): Path of the toml file.

    """
    with Path.open(filename, "rb") as f:
        return tomllib.load(f)


def load_api_key(keys):
    """Initialise the .env file with API keys.

    Args:
        keys (list): a list of API keys to initialize.

    """
    env_path = SRC_DIR / ".env"
    env_path.touch()

    for key in keys:
        if get_key(env_path, f"{key.upper()}_API_KEY", encoding='utf-8') is None:
            value = input(f"{key.title()} API key:")
            set_key(dotenv_path=env_path, key_to_set=f"{key.upper()}_API_KEY", value_to_set=value)


def load_tasks_state(query):
    """Load tasks recovery file if available.

    Args:
        query (str): The user query.

    """
    Path.mkdir(RECOVERY_DIR, exist_ok=True, parents=True)
    for directory in RECOVERY_DIR.iterdir():
        recovery_file_path = directory / "task.json"
        if recovery_file_path.exists():
            with Path.open(recovery_file_path) as file:
                recovery_file = json.load(file)
            if query == recovery_file["query"]:
                answer = ""
                while answer not in {"y", "n"}:
                    answer = input(
                        f"Do you want to reuse the recovery file {recovery_file_path}? (y/n): "
                    )
                    if answer:
                        answer = answer[0].lower()
                if answer == "y":
                    recovery_file["load_recovery"] = True
                    return recovery_file
                else:
                    shutil.rmtree(directory)
    return {"query": query, "load_recovery": False}
