"System constants."

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
SRC_DIR = BASE_DIR / "src"
RECOVERY_DIR = SRC_DIR / "json" / "recovery"

PROMPT_FILE = SRC_DIR / "json" / "prompt.json"
CONFIG_FILE = SRC_DIR / "config.toml"
