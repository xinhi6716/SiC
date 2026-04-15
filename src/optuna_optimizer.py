from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

from src import paths
from src.config import MaterialConfig, OptimizationConstraint, SearchSpaceParameter, get_material_config
from src.model_manager import MaterialModelManager


logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


@dataclass
class ConstrainedBayesianOptimizer:
    """Material-aware constrained multi-objective Bayesian optimizer.

    The optimizer reads search-space and hard constraints from
    :class:`src.config.MaterialConfig`. It uses ``MaterialModelManager`` as the
    GPR surrogate provider, then searches recipes that minimize leakage current
    and maximize endurance while satisfying config-defined constraints.

    Args:
        dataset: Optional condition-level dataframe. Kept as the first argument
            for backward compatibility with the earlier SiC-only optimizer.
        material_name: Registered material label such as ``SiC`` or ``NiO``.
        random_state: Optuna sampler seed.
        penalty_leakage_a: Leakage objective returned for infeasible recipes.
        penalty_endurance_cycles: Endurance objective returned for infeasible recipes.
        manager: Optional preconfigured material model manager.
        save_model_artifacts: Whether surrogate models and metrics are persisted.
    """

    dataset: pd.DataFrame | None = None
    material_name: str = "SiC"
    random_state: int = 42
    penalty_leakage_a: float = 1e3
    penalty_endurance_cycles: float = -1e3
    manager: MaterialModelManager | None = None
    save_model_artifacts: bool = True
    config: MaterialConfig = field(init=False)
    results_dir: Path = field(init=False)
    study_: optuna.study.Study | None = field(default=None, init=False)
    trials_: pd.DataFrame | None = field(default=None, init=False)

    objective_targets: tuple[str, str] = ("Leakage_Current_A", "Endurance_Cycles")
    fallback_secondary_target: str = "On_Off_Ratio"

    def __post_init__(self) -> None:
        self.material_name = paths.normalize_material_name(self.material_name)
        self.config = get_material_config(self.material_name)
        paths.ensure_material_directories(self.material_name)
        self.results_dir = paths.material_results_dir(self.material_name)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.manager is None:
            self.manager = MaterialModelManager(
                material_name=self.material_name,
                dataset=self.dataset,
                random_state=self.random_state,
            )

    def fit_surrogate_models(self) -> "ConstrainedBayesianOptimizer":
        """Train or load the GPR models needed for objective and constraints.

        Returns:
            The optimizer itself for method chaining.

        Raises:
            ValueError: If required target models cannot be trained or loaded.
        """

        if self.manager is None:
            raise RuntimeError("MaterialModelManager is not initialized.")

        metrics = self.manager.train_all_targets(save=self.save_model_artifacts)
        required_targets = {"Leakage_Current_A"} | {constraint.target for constraint in self.config.constraints}
        trained_targets = {
            str(row["Target"])
            for _, row in metrics.iterrows()
            if str(row.get("Status", "")).lower() == "trained"
        }
        missing = sorted(required_targets - trained_targets)
        if missing:
            raise ValueError(
                f"{self.material_name} optimizer requires trained target models {missing}. "
                "Check missing columns or insufficient non-null samples in the condition dataset."
            )
        secondary_target = self._select_secondary_objective_target(trained_targets)
        if secondary_target is None:
            logger.warning(
                "%s optimizer found no trainable secondary objective target. "
                "The second objective will use a safe fallback value of 0.0.",
                self.material_name,
            )
        else:
            logger.info("%s optimizer secondary objective target: %s", self.material_name, secondary_target)
        return self

    def optimize(self, n_trials: int = 100, save: bool = True) -> pd.DataFrame:
        """Run constrained multi-objective optimization.

        Args:
            n_trials: Number of Optuna trials.
            save: Whether to write trials and Pareto frontier CSV files.

        Returns:
            Dataframe containing all Optuna trials.
        """

        if self.manager is None:
            raise RuntimeError("MaterialModelManager is not initialized.")
        if not self.manager.models_:
            self.fit_surrogate_models()

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state,
            multivariate=True,
            constraints_func=self._constraints_func,
        )
        self.study_ = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
        self.study_.optimize(self.objective, n_trials=n_trials, show_progress_bar=False)
        self.trials_ = self._trials_to_frame(self.study_)

        if save:
            self.save_results(self.trials_, self.pareto_frontier())
        return self.trials_

    def objective(self, trial: optuna.Trial) -> tuple[float, float]:
        """Optuna objective function.

        Returns:
            ``(Leakage_Current_A, dynamic_secondary_target)``. The study directions are
            ``["minimize", "maximize"]``.
        """

        features = self._suggest_features(trial)
        predictions = self._predict_features(features)
        constraints = self._evaluate_constraints(predictions)
        is_feasible = all(value <= 0 for value in constraints.values())

        trial.set_user_attr("constraints", tuple(constraints.values()))
        trial.set_user_attr("constraint_names", tuple(constraints.keys()))
        trial.set_user_attr("is_feasible", is_feasible)
        trial.set_user_attr("material_name", self.material_name)
        for feature, value in features.items():
            trial.set_user_attr(feature, float(value))
        for target, metrics in predictions.items():
            for metric_name, value in metrics.items():
                trial.set_user_attr(f"{target}_{metric_name}", float(value))

        if not is_feasible:
            trial.set_user_attr("constraint_penalty_applied", True)
            return self.penalty_leakage_a, self.penalty_endurance_cycles

        trial.set_user_attr("constraint_penalty_applied", False)
        leakage = max(self._prediction_mean(predictions, "Leakage_Current_A", self.penalty_leakage_a), 1e-15)
        secondary_target, secondary_value = self._secondary_objective(predictions)
        trial.set_user_attr("secondary_objective_target", secondary_target)
        trial.set_user_attr("secondary_objective_value", float(secondary_value))
        return float(leakage), float(max(secondary_value, 0.0))

    def pareto_frontier(self) -> pd.DataFrame:
        """Return feasible Optuna Pareto frontier rows.

        Raises:
            RuntimeError: If ``optimize()`` has not been executed.
        """

        if self.study_ is None:
            raise RuntimeError("Call optimize() before pareto_frontier().")

        rows: list[dict[str, Any]] = []
        for trial in self.study_.best_trials:
            if not trial.user_attrs.get("is_feasible", False):
                continue
            rows.append(self._trial_to_row(trial))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(
            ["Leakage_Objective_Minimize", "Secondary_Objective_Maximize"],
            ascending=[True, False],
        )

    def save_results(self, trials: pd.DataFrame, pareto: pd.DataFrame) -> tuple[Any, Any]:
        """Persist optimization outputs under ``data/results/{material}/``.

        Args:
            trials: All trial rows.
            pareto: Feasible Pareto frontier rows.

        Returns:
            Tuple of ``(trials_path, pareto_path)``.
        """

        trials_path = self.results_dir / "part3_optuna_trials.csv"
        pareto_path = self.results_dir / "part3_optuna_pareto_frontier.csv"
        trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
        pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig")
        logger.info("Saved %s Optuna trials to %s", self.material_name, trials_path)
        logger.info("Saved %s Pareto frontier to %s", self.material_name, pareto_path)
        return trials_path, pareto_path

    def _suggest_features(self, trial: optuna.Trial) -> dict[str, float]:
        """Suggest one recipe from ``MaterialConfig.search_space``."""

        if not self.config.search_space:
            raise ValueError(f"{self.material_name} config does not define search_space.")

        features: dict[str, float] = {}
        for feature_name, spec in self.config.search_space.items():
            features[feature_name] = self._suggest_one(trial, feature_name, spec)

        for feature_name, value in self.config.default_feature_values.items():
            features.setdefault(feature_name, float(value))

        if "Has_RTA" in self.config.feature_columns and "Has_RTA" not in features:
            rta_value = features.get("RTA_Temperature_C", self.config.no_rta_temperature_c)
            features["Has_RTA"] = 0.0 if float(rta_value) == self.config.no_rta_temperature_c else 1.0

        missing = [feature for feature in self.config.feature_columns if feature not in features]
        if missing:
            raise ValueError(f"Search space for {self.material_name} is missing feature(s): {missing}")
        return {feature: float(features[feature]) for feature in self.config.feature_columns}

    @staticmethod
    def _suggest_one(trial: optuna.Trial, feature_name: str, spec: SearchSpaceParameter) -> float:
        """Suggest one Optuna parameter based on config spec."""

        param_type = spec.param_type.lower()
        if param_type == "categorical":
            if not spec.choices:
                raise ValueError(f"{feature_name} categorical search space requires choices.")
            return float(trial.suggest_categorical(feature_name, list(spec.choices)))
        if param_type == "int":
            if spec.low is None or spec.high is None:
                raise ValueError(f"{feature_name} int search space requires low/high.")
            step = 1 if spec.step is None else int(spec.step)
            return float(trial.suggest_int(feature_name, int(spec.low), int(spec.high), step=step))
        if param_type == "float":
            if spec.low is None or spec.high is None:
                raise ValueError(f"{feature_name} float search space requires low/high.")
            return float(trial.suggest_float(feature_name, float(spec.low), float(spec.high), step=spec.step))
        raise ValueError(f"Unsupported search-space type for {feature_name}: {spec.param_type}")

    def _predict_features(self, features: Mapping[str, float]) -> dict[str, dict[str, float]]:
        """Predict all configured targets for one recipe."""

        if self.manager is None:
            raise RuntimeError("MaterialModelManager is not initialized.")
        return self.manager.predict(features)

    def _evaluate_constraints(self, predictions: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Convert config constraints into Optuna ``g_i(x) <= 0`` values."""

        values: dict[str, float] = {}
        for constraint in self.config.constraints:
            predicted = predictions.get(constraint.target)
            if predicted is None or "mean" not in predicted:
                logger.warning(
                    "%s constraint target %s is missing from predictions; marking trial infeasible.",
                    self.material_name,
                    constraint.target,
                )
                values[self._constraint_name(constraint)] = float("inf")
                continue
            predicted_value = predicted["mean"]
            values[self._constraint_name(constraint)] = self._constraint_value(constraint, predicted_value)
        return values

    @staticmethod
    def _constraint_value(constraint: OptimizationConstraint, predicted_value: float) -> float:
        """Evaluate one constraint in Optuna's ``<= 0`` convention."""

        operator = constraint.operator.strip()
        if operator in {">=", ">"}:
            return float(constraint.threshold - predicted_value)
        if operator in {"<=", "<"}:
            return float(predicted_value - constraint.threshold)
        if operator == "==":
            return float(abs(predicted_value - constraint.threshold))
        raise ValueError(f"Unsupported constraint operator: {constraint.operator}")

    @staticmethod
    def _constraint_name(constraint: OptimizationConstraint) -> str:
        """Return a stable constraint name for trial tables."""

        safe_operator = constraint.operator.replace(">", "gte").replace("<", "lte").replace("=", "")
        return f"Constraint_{constraint.target}_{safe_operator}_{constraint.threshold:g}"

    @staticmethod
    def _constraints_func(frozen_trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
        """Optuna sampler callback for constrained TPE."""

        return tuple(frozen_trial.user_attrs.get("constraints", (float("inf"),)))

    def _secondary_objective(self, predictions: Mapping[str, Mapping[str, float]]) -> tuple[str, float]:
        """Return the dynamic second objective target and value.

        Priority:
        1. ``Endurance_Cycles`` if the model exists.
        2. ``On_Off_Ratio`` if endurance is unavailable, as in some NiO data.
        3. ``0.0`` safe fallback when neither prediction exists.

        Args:
            predictions: Nested target prediction mapping.

        Returns:
            Tuple of ``(target_name, objective_value)``.
        """

        endurance = predictions.get("Endurance_Cycles", {}).get("mean")
        if endurance is not None and np.isfinite(endurance):
            return "Endurance_Cycles", float(endurance)

        on_off = predictions.get(self.fallback_secondary_target, {}).get("mean")
        if on_off is not None and np.isfinite(on_off):
            return self.fallback_secondary_target, float(on_off)

        return "fallback_0.0", 0.0

    @staticmethod
    def _prediction_mean(
        predictions: Mapping[str, Mapping[str, float]],
        target: str,
        fallback: float,
    ) -> float:
        """Safely return one target mean from nested predictions."""

        value = predictions.get(target, {}).get("mean", fallback)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return value_float if np.isfinite(value_float) else float(fallback)

    def _select_secondary_objective_target(self, trained_targets: set[str]) -> str | None:
        """Select the second objective from available trained models."""

        if "Endurance_Cycles" in trained_targets:
            return "Endurance_Cycles"
        if self.fallback_secondary_target in trained_targets:
            return self.fallback_secondary_target
        return None

    def _trials_to_frame(self, study: optuna.study.Study) -> pd.DataFrame:
        """Convert Optuna trials into a dataframe."""

        rows = [self._trial_to_row(trial) for trial in study.trials]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(
            ["Is_Feasible", "Leakage_Objective_Minimize", "Secondary_Objective_Maximize"],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    def _trial_to_row(self, trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
        """Convert a single frozen trial into a flat report row."""

        values = trial.values if trial.values is not None else [np.nan, np.nan]
        row: dict[str, Any] = {
            "Material": self.material_name,
            "Trial": trial.number,
            "State": str(trial.state),
            "Leakage_Objective_Minimize": values[0],
            "Secondary_Objective_Maximize": values[1],
            "Secondary_Objective_Target": trial.user_attrs.get("secondary_objective_target", "fallback_0.0"),
            "Endurance_Objective_Maximize": values[1],
            "Is_Feasible": trial.user_attrs.get("is_feasible", False),
            "Penalty_Applied": trial.user_attrs.get("constraint_penalty_applied", False),
        }

        for feature in self.config.feature_columns:
            row[feature] = trial.user_attrs.get(feature, trial.params.get(feature))
        row["RF_Power"] = row.get("RF_Power_W")
        row["Process_Time"] = row.get("Process_Time_Min")
        row["RTA_Temperature"] = row.get("RTA_Temperature_C")

        constraint_names = trial.user_attrs.get("constraint_names", ())
        constraint_values = trial.user_attrs.get("constraints", ())
        for name, value in zip(constraint_names, constraint_values):
            row[name] = value

        for key, value in trial.user_attrs.items():
            if key.endswith(("_mean", "_std", "_ci95_low", "_ci95_high")):
                row[key] = value
        return row


if __name__ == "__main__":
    optimizer = ConstrainedBayesianOptimizer(material_name="SiC")
    trials_frame = optimizer.optimize(n_trials=100, save=True)
    pareto_frame = optimizer.pareto_frontier()
    logger.info("Top feasible Optuna trials:\n%s", trials_frame.head(10).to_string(index=False))
    logger.info(
        "Feasible Pareto frontier:\n%s",
        pareto_frame.to_string(index=False) if not pareto_frame.empty else "No feasible Pareto point found.",
    )
