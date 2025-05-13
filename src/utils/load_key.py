"Load keys in the .env file."

from dotenv import get_key, set_key

from config import SRC_DIR


def load_api_key(keys):
    """Initialise the .env file with API keys.

    Args.
        keys (list): a list of API keys to initialize.

    """
    env_path = SRC_DIR / ".env"
    env_path.touch()

    for key in keys:
        if get_key(env_path, f"{key.upper()}_API_KEY", encoding='utf-8') is None:
            value = input(f"{key.title()} API key:")
            set_key(dotenv_path=env_path, key_to_set=f"{key.upper()}_API_KEY", value_to_set=value)
