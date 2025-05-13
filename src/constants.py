"System constants."

from pathlib import Path

# Query
QUERY = None

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"
RECOVERY_DIR = SRC_DIR / "json" / "recovery"
PROMPT_FILE = SRC_DIR / "json" / "prompt.json"

# Tavily Configuration
TAVILY_DAYS = 100
TAVILY_TOPIC = "general"

# LLM Configuration
MODEL_NAME = "llama-3.3-70b-versatile"

# Other Configurations
MAX_CONCURRENCY = 1
SAVE_FINAL_STATE = False
