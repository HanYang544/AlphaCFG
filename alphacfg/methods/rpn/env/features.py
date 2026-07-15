from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from alphacfg.env.features import *  # noqa: F401,F403
