"System configuraitons."

from pathlib import Path

# Query
QUERY = None
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"
PROMPT_PATH = SRC_DIR / "json" / "prompt.json"
RECOVERY_PATH = SRC_DIR / "json" / "recovery"

# Tavily Configuration
TAVILY_DAYS = 100
TAVILY_TOPIC = "general"

# LLM Configuration
MODEL_NAME = "llama-3.3-70b-versatile"

# Other Configurations
MAX_CONCURRENCY = 1
SAVE_FINAL_STATE = False
