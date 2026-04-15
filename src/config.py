from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SearchSpaceParameter:
    """Optimization search-space definition for one process feature.

    Args:
        param_type: One of ``float``, ``int``, or ``categorical``.
        low: Lower bound for numeric parameters.
        high: Upper bound for numeric parameters.
        step: Optional step size for numeric parameters.
        choices: Candidate values for categorical parameters.
        description: Human-readable physical meaning.
    """

    param_type: str
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple[float | int | str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class OptimizationConstraint:
    """Hard constraint used by optimizers and recommendation systems.

    Args:
        target: Target column name, e.g. `On_Off_Ratio`.
        operator: Constraint operator. Currently documented as `>=` or `<=`.
        threshold: Numeric threshold for the constraint.
        description: Human-readable physical meaning.
    """

    target: str
    operator: str
    threshold: float
    description: str = ""


@dataclass(frozen=True)
class MaterialConfig:
    """Centralized material-specific configuration.

    Args:
        name: Material label used in folder names and reports.
        feature_columns: Process feature schema used for ML and condition-level aggregation.
        target_columns: Electrical targets extracted from I-V curves.
        default_feature_values: Constants used when raw data lacks process metadata.
        no_rta_temperature_c: Temperature used to encode as-deposited/no-RTA samples.
        leakage_epsilon_a: Lower bound used before log transforms.
        max_reasonable_current_a: Current threshold for flagging likely short/noisy points.
        read_voltage_v: Fixed read voltage. If `None`, DataProcessor may use dynamic read-voltage search.
        dynamic_on_off_ratio: Whether to find the read voltage maximizing HRS/LRS current separation.
        raw_extensions: File extensions considered parseable by the ETL layer.
        search_space: Material-specific optimization search space keyed by feature column.
        constraints: Material-specific hard constraints for downstream optimization.
    """

    name: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    default_feature_values: Mapping[str, float] = field(default_factory=dict)
    no_rta_temperature_c: float = 25.0
    leakage_epsilon_a: float = 1e-15
    max_reasonable_current_a: float = 0.1
    read_voltage_v: float | None = None
    dynamic_on_off_ratio: bool = True
    raw_extensions: tuple[str, ...] = (".csv", ".txt", ".xlsx", ".xls")
    search_space: Mapping[str, SearchSpaceParameter] = field(default_factory=dict)
    constraints: tuple[OptimizationConstraint, ...] = ()


COMMON_TARGETS = (
    "Forming_Voltage_V",
    "Operation_Voltage_V",
    "Leakage_Current_A",
    "On_Off_Ratio",
    "Endurance_Cycles",
)

SIC_CONFIG = MaterialConfig(
    name="SiC",
    feature_columns=("RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA"),
    target_columns=COMMON_TARGETS,
    default_feature_values={},
    read_voltage_v=0.1,
    dynamic_on_off_ratio=False,
    search_space={
        "RF_Power_W": SearchSpaceParameter("float", low=50.0, high=75.0, step=25.0, description="RF sputtering power"),
        "Process_Time_Min": SearchSpaceParameter("float", low=30.0, high=120.0, step=30.0, description="Sputtering time"),
        "RTA_Temperature_C": SearchSpaceParameter(
            "categorical",
            choices=(25.0, 400.0, 500.0),
            description="25 C encodes no-RTA/as-deposited samples",
        ),
    },
    constraints=(
        OptimizationConstraint("On_Off_Ratio", ">=", 5.0, "RRAM memory window lower bound"),
        OptimizationConstraint("Operation_Voltage_V", "<=", 3.0, "Low-voltage operation requirement"),
    ),
)

NIO_CONFIG = MaterialConfig(
    name="NiO",
    feature_columns=("RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA", "Current_Compliance_A"),
    target_columns=COMMON_TARGETS,
    default_feature_values={
        "RF_Power_W": 100.0,
        "Process_Time_Min": 30.0,
    },
    read_voltage_v=None,
    dynamic_on_off_ratio=True,
    search_space={
        "RF_Power_W": SearchSpaceParameter(
            "categorical",
            choices=(100.0,),
            description="Default NiO RF power imputed from experiment metadata",
        ),
        "Process_Time_Min": SearchSpaceParameter(
            "categorical",
            choices=(30.0,),
            description="Default NiO process time imputed from experiment metadata",
        ),
        "RTA_Temperature_C": SearchSpaceParameter(
            "categorical",
            choices=(25.0, 400.0, 500.0),
            description="25 C encodes no-RTA/as-deposited samples",
        ),
        "Current_Compliance_A": SearchSpaceParameter(
            "categorical",
            choices=(0.01, 0.02),
            description="NiO current compliance levels observed in raw folders",
        ),
    },
    constraints=(
        OptimizationConstraint("On_Off_Ratio", ">=", 5.0, "Usable HRS/LRS separation"),
        OptimizationConstraint("Operation_Voltage_V", "<=", 5.0, "NiO exploratory low-voltage target"),
    ),
)

MATERIAL_CONFIGS: dict[str, MaterialConfig] = {
    SIC_CONFIG.name.lower(): SIC_CONFIG,
    NIO_CONFIG.name.lower(): NIO_CONFIG,
}


def get_material_config(material_name: str) -> MaterialConfig:
    """Return the registered material configuration.

    Args:
        material_name: Material name such as `SiC` or `NiO`.

    Returns:
        The matching `MaterialConfig`.

    Raises:
        ValueError: If `material_name` is empty.
    """

    key = str(material_name).strip().lower()
    if not key:
        raise ValueError("material_name must not be empty.")
    return MATERIAL_CONFIGS.get(
        key,
        MaterialConfig(
            name=material_name,
            feature_columns=SIC_CONFIG.feature_columns,
            target_columns=COMMON_TARGETS,
        ),
    )
