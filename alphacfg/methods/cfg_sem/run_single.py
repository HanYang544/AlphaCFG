import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacfg.method_entry import run_method


if __name__ == "__main__":
    run_method(__file__, "cfg-sem", forced_reward_mode="single")
