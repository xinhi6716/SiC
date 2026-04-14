from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "DATA"
REPORT_DIR = PROJECT_ROOT / "REPORTS"
MODEL_DIR = PROJECT_ROOT / "MODELS"

FEATURE_COLUMNS = ["RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA"]
REGRESSION_TARGETS = [
    "forming_voltage_v",
    "operation_voltage_v",
    "leakage_current_a",
    "on_off_ratio",
    "endurance_cycles",
]
CLASSIFICATION_TARGETS = ["hrs_mechanism", "lrs_mechanism"]

NO_RTA_TEMPERATURE_C = 25.0
LEAKAGE_EPSILON_A = 1e-15
MIN_TRAINING_ROWS = 3
RANDOM_STATE = 42


@dataclass(frozen=True)
class HardConstraints:
    min_on_off_ratio: float = 5.0
    max_operation_voltage_v: float = 3.0


@dataclass(frozen=True)
class ObjectiveWeights:
    endurance: float = 0.60
    leakage: float = 0.40
    uncertainty: float = 0.05
    infeasible_penalty: float = 100.0


CONSTRAINTS = HardConstraints()
OBJECTIVE_WEIGHTS = ObjectiveWeights()
