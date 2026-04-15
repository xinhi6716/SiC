from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src import paths
from src.config import MaterialConfig, get_material_config
from src.gpr_model_trainer import GPRModelTrainer


logger = logging.getLogger(__name__)


@dataclass
class MaterialModelManager:
    """Config-driven GPR model manager for one material.

    The manager is the material-aware wrapper around the older
    :class:`GPRModelTrainer`. It reads feature/target schema from
    :mod:`src.config`, persists models under ``outputs/models/{material}/``,
    and exposes a small prediction API suitable for Streamlit, Gradio, or REST.

    Args:
        material_name: Registered material label, e.g. ``SiC`` or ``NiO``.
        dataset: Optional condition-level dataset already loaded in memory.
        dataset_path: Optional path to a condition-level CSV.
        random_state: Reproducibility seed.
        alpha: GPR regularization strength for noisy small-sample experiments.
        kernel_type: Kernel family passed to ``GPRModelTrainer``.
        n_restarts_optimizer: Number of GPR optimizer restarts.
    """

    material_name: str
    dataset: pd.DataFrame | None = None
    dataset_path: str | Path | None = None
    random_state: int = 42
    alpha: float = 1e-2
    kernel_type: str = "matern"
    n_restarts_optimizer: int = 8
    config: MaterialConfig = field(init=False)
    model_dir: Path = field(init=False)
    metrics_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    models_: dict[str, GPRModelTrainer] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.material_name = paths.normalize_material_name(self.material_name)
        self.config = get_material_config(self.material_name)
        paths.ensure_material_directories(self.material_name)
        self.model_dir = paths.material_models_dir(self.material_name)
        self.metrics_path = self.model_dir / "training_metrics.csv"
        self.manifest_path = self.model_dir / "manifest.json"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def feature_columns(self) -> list[str]:
        """Return configured process features for this material."""

        return list(self.config.feature_columns)

    @property
    def target_columns(self) -> list[str]:
        """Return configured electrical targets for this material."""

        return list(self.config.target_columns)

    def load_dataset(self) -> pd.DataFrame:
        """Load the material condition-level dataset.

        Returns:
            Condition-level dataframe.

        Raises:
            FileNotFoundError: If no dataset path can be resolved.
            ValueError: If required feature columns are missing.
        """

        if self.dataset is not None:
            dataset = self.dataset.copy()
        else:
            dataset_path = self._resolve_dataset_path()
            dataset = pd.read_csv(dataset_path, low_memory=False)
            logger.info("Loaded %s condition dataset from %s", self.material_name, dataset_path)

        missing_features = [column for column in self.feature_columns if column not in dataset.columns]
        if missing_features:
            raise ValueError(f"{self.material_name} dataset is missing feature columns: {missing_features}")

        self.dataset = dataset
        return dataset

    def train_all_targets(self, save: bool = True) -> pd.DataFrame:
        """Train one GPR model for every configured target.

        Leakage current and On/Off ratio are trained in log10 space, endurance
        in log1p space, and voltage-like targets in their original scale.

        Args:
            save: Whether to persist ``training_metrics.csv`` and ``manifest.json``.

        Returns:
            Training metric dataframe with one row per attempted target.
        """

        dataset = self.load_dataset()
        metric_rows: list[dict[str, Any]] = []
        self.models_.clear()

        for target in self.target_columns:
            transform = self._target_transform(target)
            trainer = GPRModelTrainer(
                feature_columns=tuple(self.feature_columns),
                target_column=target,
                target_transform=transform,
                kernel_type=self.kernel_type,
                random_state=self.random_state,
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
            )
            try:
                cv_predictions = trainer.cross_validate(dataset)
                cv_summary = trainer.summarize_cv(cv_predictions)
                trainer.fit(dataset)
                model_path = self.model_path_for_target(target)
                if save:
                    joblib.dump(trainer, model_path)
                self.models_[target] = trainer

                row = cv_summary.iloc[0].to_dict()
                row.update(
                    {
                        "Material": self.material_name,
                        "Target": target,
                        "Target_Transform": transform,
                        "Status": "trained",
                        "Model_Path": str(model_path),
                    }
                )
                metric_rows.append(row)
                if save:
                    logger.info("Trained and saved %s/%s model: %s", self.material_name, target, model_path)
                else:
                    logger.info("Trained %s/%s model in memory", self.material_name, target)
            except ValueError as exc:
                logger.warning("Skipped %s/%s: %s", self.material_name, target, exc)
                metric_rows.append(
                    {
                        "Material": self.material_name,
                        "Target": target,
                        "Target_Transform": transform,
                        "Status": f"skipped: {exc}",
                        "Model_Path": pd.NA,
                        "CV_Type": "LOOCV",
                        "N_Predictions": 0,
                        "R2_Model_Space": np.nan,
                        "MSE_Model_Space": np.nan,
                        "R2_Original_Scale": np.nan,
                        "MSE_Original_Scale": np.nan,
                    }
                )

        metrics = pd.DataFrame(metric_rows)
        if save:
            metrics.to_csv(self.metrics_path, index=False, encoding="utf-8-sig")
            self._write_manifest(metrics)
            logger.info("Saved %s training metrics to %s", self.material_name, self.metrics_path)
        return metrics

    def load_models(self, strict: bool = True) -> dict[str, GPRModelTrainer]:
        """Load persisted GPR models from the material model directory.

        Args:
            strict: If ``True``, raise when any configured target model is missing.

        Returns:
            Mapping from target name to loaded trainer.
        """

        loaded: dict[str, GPRModelTrainer] = {}
        missing: list[str] = []
        for target in self.target_columns:
            model_path = self.model_path_for_target(target)
            if not model_path.exists():
                missing.append(target)
                continue
            loaded[target] = joblib.load(model_path)

        if strict and missing:
            raise FileNotFoundError(
                f"Missing {self.material_name} model(s): {missing}. "
                "Run train_all_targets() before prediction."
            )

        self.models_.update(loaded)
        return self.models_

    def predict(self, features: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Predict all trained targets for one process recipe.

        Args:
            features: Mapping of process feature names to values.

        Returns:
            Nested dictionary keyed by target. Each target contains ``mean``,
            ``std``, ``ci95_low``, and ``ci95_high``.

        Raises:
            ValueError: If required features are missing and cannot be imputed.
            FileNotFoundError: If trained model files are missing.
        """

        if not self.models_:
            self.load_models(strict=False)
        if not self.models_:
            raise FileNotFoundError(
                f"No trained {self.material_name} models were found in {self.model_dir}. "
                "Run train_all_targets() before prediction."
            )

        recipe = self._features_to_frame(features)
        predictions: dict[str, dict[str, float]] = {}
        for target, model in self.models_.items():
            pred = model.predict_with_uncertainty(recipe)
            predictions[target] = {
                "mean": float(pred[f"{target}_Mean"].iloc[0]),
                "std": float(pred[f"{target}_Std"].iloc[0]),
                "ci95_low": float(pred[f"{target}_CI95_Low"].iloc[0]),
                "ci95_high": float(pred[f"{target}_CI95_High"].iloc[0]),
            }
        return predictions

    def model_path_for_target(self, target: str) -> Path:
        """Return the persisted model path for one target."""

        safe_target = target.replace("/", "_").replace("\\", "_")
        return self.model_dir / f"{safe_target}.joblib"

    def _resolve_dataset_path(self) -> Path:
        """Resolve the material-specific condition-level dataset path."""

        if self.dataset_path is not None:
            path = Path(self.dataset_path)
            return path if path.is_absolute() else paths.PROJECT_ROOT / path

        material_path = paths.material_condition_dataset_path(self.material_name)
        if material_path.exists():
            return material_path

        if self.material_name.lower() == "sic" and paths.CONDITION_LEVEL_DATA_PATH.exists():
            return paths.CONDITION_LEVEL_DATA_PATH

        raise FileNotFoundError(
            f"Cannot find condition-level dataset for {self.material_name}. "
            f"Expected {material_path}."
        )

    def _features_to_frame(self, features: Mapping[str, float]) -> pd.DataFrame:
        """Normalize user features to a single-row dataframe."""

        row: dict[str, float] = {key: float(value) for key, value in features.items()}
        if "Has_RTA" in self.feature_columns and "Has_RTA" not in row and "RTA_Temperature_C" in row:
            row["Has_RTA"] = 0.0 if row["RTA_Temperature_C"] == self.config.no_rta_temperature_c else 1.0

        for feature, value in self.config.default_feature_values.items():
            row.setdefault(feature, float(value))

        missing = [feature for feature in self.feature_columns if feature not in row]
        if missing:
            raise ValueError(f"Missing required {self.material_name} feature(s): {missing}")

        return pd.DataFrame([{feature: row[feature] for feature in self.feature_columns}])

    def _write_manifest(self, metrics: pd.DataFrame) -> None:
        """Write a JSON manifest describing the model bundle."""

        trained_targets = metrics.loc[metrics["Status"].eq("trained"), "Target"].astype(str).tolist()
        manifest = {
            "material": self.material_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "trained_targets": trained_targets,
            "model_dir": str(self.model_dir),
            "metrics_path": str(self.metrics_path),
            "models": {
                target: str(self.model_path_for_target(target))
                for target in trained_targets
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Saved %s manifest to %s", self.material_name, self.manifest_path)

    @staticmethod
    def _target_transform(target: str) -> str:
        """Return the target-space transform used for GPR training."""

        if target in {"Leakage_Current_A", "On_Off_Ratio"}:
            return "log10"
        if target == "Endurance_Cycles":
            return "log1p"
        return "none"
