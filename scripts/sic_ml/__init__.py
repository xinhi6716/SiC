"""SiC RRAM machine-learning pipeline package."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = PROJECT_ROOT / ".codex_deps"

# 將專案內安裝的 optuna 放在 sys.path 尾端，避免覆蓋全域穩定的 numpy/pandas/scipy。
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.append(str(LOCAL_DEPS))
