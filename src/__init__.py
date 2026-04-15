from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS_DIR = PROJECT_ROOT / ".codex_deps"

if LOCAL_DEPS_DIR.exists() and str(LOCAL_DEPS_DIR) not in sys.path:
    sys.path.append(str(LOCAL_DEPS_DIR))

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("sic_ml")
