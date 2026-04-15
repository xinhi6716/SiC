from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src import paths
from src.config import MATERIAL_CONFIGS, MaterialConfig, get_material_config
from src.model_manager import MaterialModelManager
from src.optuna_optimizer import ConstrainedBayesianOptimizer


logger = logging.getLogger(__name__)

app = FastAPI(
    title="RRAM Multi-Material MLOps API",
    description="FastAPI backend for Electron-based SiC/NiO RRAM prediction and optimization UI.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Prediction request payload.

    The API accepts both shapes below:

    ```json
    {"features": {"RF_Power_W": 75, "Process_Time_Min": 60}}
    ```

    or a flat Electron-friendly object:

    ```json
    {"RF_Power_W": 75, "Process_Time_Min": 60}
    ```
    """

    model_config = ConfigDict(extra="allow")

    features: dict[str, float] | None = Field(default=None, description="Process feature dictionary.")

    def to_features(self) -> dict[str, float]:
        """Return a merged numeric feature dictionary."""

        merged: dict[str, Any] = {}
        if self.features:
            merged.update(self.features)
        if self.model_extra:
            merged.update(self.model_extra)

        features: dict[str, float] = {}
        for key, value in merged.items():
            if key == "features":
                continue
            try:
                features[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {key!r} must be numeric, got {value!r}.") from exc
        return features


class PredictionMetric(BaseModel):
    """Prediction summary for one electrical target."""

    mean: float
    std: float
    ci95_low: float
    ci95_high: float


class PredictionResponse(BaseModel):
    """Prediction response returned to Electron."""

    material: str
    features: dict[str, float]
    predictions: dict[str, PredictionMetric]


class RecipeResponse(BaseModel):
    """Recommended recipe summary."""

    label: str
    strategy: str
    features: dict[str, float]
    leakage_current_a: float | None
    secondary_target: str
    secondary_value: float | None
    score: float | None = None


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Return a lightweight health signal."""

    return {"status": "ok"}


@app.get("/api/config/{material_name}")
def get_material_api_config(material_name: str) -> dict[str, Any]:
    """Return feature schema, search space, targets, and constraints.

    Args:
        material_name: Registered material name such as ``SiC`` or ``NiO``.

    Returns:
        JSON-serializable material configuration.
    """

    config = resolve_material_config(material_name)
    return {
        "material": config.name,
        "feature_columns": list(config.feature_columns),
        "target_columns": list(config.target_columns),
        "default_feature_values": dict(config.default_feature_values),
        "no_rta_temperature_c": config.no_rta_temperature_c,
        "read_voltage_v": config.read_voltage_v,
        "dynamic_on_off_ratio": config.dynamic_on_off_ratio,
        "search_space": {
            name: serialize_dataclass(parameter)
            for name, parameter in config.search_space.items()
        },
        "constraints": [serialize_dataclass(constraint) for constraint in config.constraints],
        "paths": {
            "raw_dir": str(paths.material_raw_dir(config.name)),
            "processed_dir": str(paths.material_processed_dir(config.name)),
            "results_dir": str(paths.material_results_dir(config.name)),
            "models_dir": str(paths.material_models_dir(config.name)),
            "condition_dataset": str(paths.material_condition_dataset_path(config.name)),
        },
    }


@app.post("/api/predict/{material_name}", response_model=PredictionResponse)
def predict_material_properties(material_name: str, payload: PredictionRequest) -> PredictionResponse:
    """Predict all available electrical properties for one recipe.

    Args:
        material_name: Registered material name.
        payload: Process feature dictionary, either nested under ``features``
            or sent as flat JSON.

    Returns:
        Prediction means, standard deviations, and 95% confidence intervals.
    """

    config = resolve_material_config(material_name)
    try:
        features = payload.to_features()
        manager = get_model_manager(config.name)
        predictions = manager.predict(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert backend failure into API error.
        logger.exception("Prediction failed for %s", config.name)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        material=config.name,
        features={key: float(value) for key, value in features.items()},
        predictions={
            target: PredictionMetric(**metric)
            for target, metric in predictions.items()
        },
    )


@app.get("/api/recipes/{material_name}", response_model=list[RecipeResponse])
def get_recommended_recipes(material_name: str) -> list[RecipeResponse]:
    """Return representative Pareto recipes for one material.

    The endpoint returns up to three recipes:
    ultra-low leakage, high secondary objective, and balanced sweet spot.
    The secondary objective is usually endurance, falling back to On/Off ratio
    when endurance is unavailable.
    """

    config = resolve_material_config(material_name)
    table = load_recipe_table(config)
    if table.empty:
        return []

    recipes = select_representative_recipes(table, config)
    return [RecipeResponse(**recipe) for recipe in recipes]


def resolve_material_config(material_name: str) -> MaterialConfig:
    """Resolve material configuration or raise a 404 error."""

    key = material_name.strip().lower()
    if key not in MATERIAL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown material: {material_name}")
    return get_material_config(material_name)


@lru_cache(maxsize=8)
def get_model_manager(material_name: str) -> MaterialModelManager:
    """Return a cached model manager with persisted models loaded."""

    manager = MaterialModelManager(material_name=material_name)
    manager.load_models(strict=False)
    return manager


def load_recipe_table(config: MaterialConfig) -> pd.DataFrame:
    """Load and normalize Pareto recipe data for one material."""

    pareto_path = paths.material_results_dir(config.name) / "part3_optuna_pareto_frontier.csv"
    trials_path = paths.material_results_dir(config.name) / "part3_optuna_trials.csv"

    frames: list[pd.DataFrame] = []
    for csv_path in [pareto_path, trials_path]:
        if csv_path.exists():
            frames.append(normalize_result_table(pd.read_csv(csv_path), config))

    if config.name.lower() == "sic":
        for csv_path in [paths.PARETO_FRONTIER_PATH, paths.OPTUNA_TRIALS_PATH]:
            if csv_path.exists():
                frames.append(normalize_result_table(pd.read_csv(csv_path), config))

    if not frames:
        return empty_recipe_table(config)

    table = pd.concat(frames, ignore_index=True)
    required = {"Leakage", "Secondary"}
    if table.empty or not required.issubset(table.columns):
        return empty_recipe_table(config)

    table = table.dropna(subset=["Leakage", "Secondary"])
    feature_subset = [feature for feature in config.feature_columns if feature in table.columns]
    if feature_subset:
        table = table.drop_duplicates(subset=feature_subset, keep="first")
    return table.reset_index(drop=True)


def normalize_result_table(frame: pd.DataFrame, config: MaterialConfig) -> pd.DataFrame:
    """Normalize several Optuna CSV schemas into a recipe table."""

    if frame.empty:
        return empty_recipe_table(config)

    output = pd.DataFrame(index=frame.index)
    for feature in config.feature_columns:
        alias = legacy_feature_alias(feature)
        source_col = first_existing_column(frame, [feature, f"params_{feature}", alias, f"params_{alias}"])
        if source_col is not None:
            output[feature] = pd.to_numeric(frame[source_col], errors="coerce")

    if "Has_RTA" in config.feature_columns and "Has_RTA" not in output and "RTA_Temperature_C" in output:
        output["Has_RTA"] = np.where(output["RTA_Temperature_C"].eq(config.no_rta_temperature_c), 0.0, 1.0)

    for feature, value in config.default_feature_values.items():
        if feature not in output:
            output[feature] = float(value)

    leakage_col = first_existing_column(frame, leakage_column_candidates())
    secondary_col = first_existing_column(frame, secondary_column_candidates())
    if leakage_col is None or secondary_col is None:
        return empty_recipe_table(config)

    output["Leakage"] = pd.to_numeric(frame[leakage_col], errors="coerce")
    output["Secondary"] = pd.to_numeric(frame[secondary_col], errors="coerce")

    secondary_target_col = first_existing_column(frame, ["Secondary_Objective_Target"])
    if secondary_target_col is not None:
        output["Secondary_Target"] = frame[secondary_target_col].fillna(infer_secondary_target_name(secondary_col))
    else:
        output["Secondary_Target"] = infer_secondary_target_name(secondary_col)

    feasible_col = first_existing_column(frame, ["Is_Feasible", "user_attrs_is_feasible", "user_attrs_Is_Feasible"])
    if feasible_col is not None:
        feasible = frame[feasible_col].astype(str).str.lower().isin(["true", "1", "yes"])
        output = output.loc[feasible]

    return output.dropna(subset=[feature for feature in config.feature_columns if feature in output])


def select_representative_recipes(table: pd.DataFrame, config: MaterialConfig) -> list[dict[str, Any]]:
    """Select low-leakage, high-secondary, and balanced recipes."""

    if table.empty:
        return []

    log_leakage = np.log10(table["Leakage"].clip(lower=1e-15))
    leakage_score = 1.0 - normalize_series(log_leakage)
    secondary_score = normalize_series(table["Secondary"].clip(lower=0.0))
    scored = table.assign(Balanced_Score=0.5 * leakage_score + 0.5 * secondary_score)

    selections = [
        ("ultra_low_leakage", "Minimize leakage current", scored.loc[scored["Leakage"].idxmin()]),
        ("high_secondary_objective", "Maximize endurance or On/Off ratio", scored.loc[scored["Secondary"].idxmax()]),
        ("balanced_sweet_spot", "Balance leakage and secondary objective", scored.loc[scored["Balanced_Score"].idxmax()]),
    ]

    recipes: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for label, strategy, row in selections:
        features = {
            feature: float(row[feature])
            for feature in config.feature_columns
            if feature in row and pd.notna(row[feature])
        }
        signature = tuple(features.get(feature, np.nan) for feature in config.feature_columns)
        if signature in seen:
            continue
        seen.add(signature)
        recipes.append(
            {
                "label": label,
                "strategy": strategy,
                "features": features,
                "leakage_current_a": safe_float(row.get("Leakage")),
                "secondary_target": str(row.get("Secondary_Target", "Secondary")),
                "secondary_value": safe_float(row.get("Secondary")),
                "score": safe_float(row.get("Balanced_Score")),
            }
        )
    return recipes


def empty_recipe_table(config: MaterialConfig) -> pd.DataFrame:
    """Return an empty recipe table with stable columns."""

    return pd.DataFrame(columns=[*config.feature_columns, "Leakage", "Secondary", "Secondary_Target"])


def leakage_column_candidates() -> list[str]:
    """Return supported leakage objective column names."""

    target = "Leakage_Current_A"
    return [
        "Leakage",
        "Leakage_Objective_Minimize",
        target,
        f"{target}_Mean",
        f"{target}_mean",
        f"{target}_median",
        f"{target}_Objective_Minimize",
        f"{target}_Pred",
        f"{target}_Prediction",
        "values_0",
        "value_0",
        "Value_0",
        "objective_0",
        "Objective_0",
    ]


def secondary_column_candidates() -> list[str]:
    """Return supported secondary-objective column names."""

    candidates = ["Secondary", "Secondary_Objective_Maximize", "Endurance_Objective_Maximize"]
    for target in ("Endurance_Cycles", "On_Off_Ratio", "Operation_Voltage_V"):
        candidates.extend(
            [
                target,
                f"{target}_Mean",
                f"{target}_mean",
                f"{target}_median",
                f"{target}_Objective_Maximize",
                f"{target}_Pred",
                f"{target}_Prediction",
            ]
        )
    candidates.extend(["values_1", "value_1", "Value_1", "objective_1", "Objective_1"])
    return candidates


def infer_secondary_target_name(column_name: str) -> str:
    """Infer a readable secondary target name from a CSV column."""

    if "Endurance" in column_name:
        return "Endurance_Cycles"
    if "On_Off" in column_name or "OnOff" in column_name:
        return "On_Off_Ratio"
    if "Operation_Voltage" in column_name:
        return "Operation_Voltage_V"
    return column_name


def legacy_feature_alias(feature_name: str) -> str:
    """Return old aliases for process feature columns."""

    aliases = {
        "RF_Power_W": "RF_Power",
        "Process_Time_Min": "Process_Time",
        "RTA_Temperature_C": "RTA_Temperature",
    }
    return aliases.get(feature_name, feature_name)


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from a candidate list."""

    return next((column for column in candidates if column in frame.columns), None)


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to 0..1 with constant-series protection."""

    minimum = float(series.min())
    maximum = float(series.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or abs(maximum - minimum) < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


def safe_float(value: Any) -> float | None:
    """Convert a scalar to a JSON-safe float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def serialize_dataclass(value: Any) -> dict[str, Any]:
    """Serialize dataclasses with tuple values converted to lists."""

    if not is_dataclass(value):
        return dict(value) if isinstance(value, dict) else {"value": value}
    return to_jsonable(asdict(value))


def to_jsonable(value: Any) -> Any:
    """Recursively convert values into JSON-friendly types."""

    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
