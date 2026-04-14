from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONSTRAINTS, FEATURE_COLUMNS, LEAKAGE_EPSILON_A, OBJECTIVE_WEIGHTS, RANDOM_STATE
from .model_trainer import PredictiveModelSuite

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise ImportError("Optuna is required for constrained Bayesian optimization. Install it into .codex_deps.") from exc

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


RTA_CHOICES = ["No_RTA_25C", "RTA_300C", "RTA_400C", "RTA_500C"]


@dataclass(frozen=True)
class Recipe:
    RF_Power_W: float
    Process_Time_Min: float
    RTA_Temperature_C: float
    Has_RTA: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__], columns=FEATURE_COLUMNS)


def recipe_from_rta_choice(rf_power_w: float, process_time_min: float, rta_choice: str) -> Recipe:
    """將 Optuna 的類別型 RTA 選項轉成模型特徵。"""
    if rta_choice == "No_RTA_25C":
        return Recipe(rf_power_w, process_time_min, 25.0, 0)
    temperature = float(rta_choice.replace("RTA_", "").replace("C", ""))
    return Recipe(rf_power_w, process_time_min, temperature, 1)


def _require_prediction(prediction: pd.DataFrame, target: str) -> float:
    column = f"{target}_mean"
    if column not in prediction:
        raise RuntimeError(f"Missing required prediction target: {target}")
    return float(prediction[column].iloc[0])


def score_prediction(prediction: pd.DataFrame) -> tuple[float, tuple[float, float], bool]:
    """計算 constrained objective。

    硬性約束：
    - g1(X) = 5 - OnOff(X) <= 0
    - g2(X) = OperationVoltage(X) - 3 <= 0

    在可行域內最大化：
    - Endurance 越大越好：log1p(Endurance)
    - Leakage 越小越好：-log10(Leakage)
    同時用 GPR posterior std 作為小樣本不確定性懲罰。
    """
    on_off = _require_prediction(prediction, "on_off_ratio")
    operation_voltage = _require_prediction(prediction, "operation_voltage_v")
    endurance = _require_prediction(prediction, "endurance_cycles")
    leakage = _require_prediction(prediction, "leakage_current_a")

    on_off_violation = CONSTRAINTS.min_on_off_ratio - on_off
    voltage_violation = operation_voltage - CONSTRAINTS.max_operation_voltage_v
    constraints = (on_off_violation, voltage_violation)
    feasible = on_off_violation <= 0 and voltage_violation <= 0

    endurance_score = np.log1p(max(endurance, 0.0))
    leakage_score = -np.log10(max(leakage, LEAKAGE_EPSILON_A))
    uncertainty_penalty = 0.0
    for target in ["endurance_cycles", "leakage_current_a", "on_off_ratio", "operation_voltage_v"]:
        std_column = f"{target}_transformed_std"
        if std_column in prediction:
            uncertainty_penalty += float(prediction[std_column].iloc[0])

    score = (
        OBJECTIVE_WEIGHTS.endurance * endurance_score
        + OBJECTIVE_WEIGHTS.leakage * leakage_score
        - OBJECTIVE_WEIGHTS.uncertainty * uncertainty_penalty
    )
    if not feasible:
        violation_size = sum(max(0.0, value) for value in constraints)
        score -= OBJECTIVE_WEIGHTS.infeasible_penalty * violation_size
    return float(score), constraints, feasible


def build_candidate_grid(
    rf_powers=(50.0, 75.0),
    process_times=(30.0, 60.0, 120.0),
    rta_choices=RTA_CHOICES,
) -> pd.DataFrame:
    """列舉目前論文與資料集中合理的離散製程搜尋空間。"""
    rows = []
    for rf_power, process_time, rta_choice in itertools.product(rf_powers, process_times, rta_choices):
        rows.append(recipe_from_rta_choice(rf_power, process_time, rta_choice).__dict__)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def rank_candidate_grid(model_suite: PredictiveModelSuite, candidate_grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """對所有候選 recipe 做 surrogate prediction 並依 constrained score 排序。"""
    candidate_grid = candidate_grid if candidate_grid is not None else build_candidate_grid()
    predictions = model_suite.predict_properties(candidate_grid)
    scores, feasible_flags, c_onoff, c_voltage = [], [], [], []
    for idx in predictions.index:
        score, constraints, feasible = score_prediction(predictions.loc[[idx]])
        scores.append(score)
        c_onoff.append(constraints[0])
        c_voltage.append(constraints[1])
        feasible_flags.append(feasible)
    predictions["constraint_on_off_g1"] = c_onoff
    predictions["constraint_voltage_g2"] = c_voltage
    predictions["is_feasible"] = feasible_flags
    predictions["constrained_score"] = scores
    return predictions.sort_values(["is_feasible", "constrained_score"], ascending=[False, False]).reset_index(drop=True)


def _constraints_func(frozen_trial):
    return frozen_trial.user_attrs.get("constraints", (float("inf"), float("inf")))


def run_constrained_optuna_search(model_suite: PredictiveModelSuite, n_trials: int = 80) -> tuple[pd.DataFrame, optuna.study.Study]:
    """使用 Optuna TPESampler 進行帶約束的 Bayesian-style sequential search。

    Optuna 的 constraints_func 會將 g_i(X) <= 0 視為可行解，TPE sampler 會偏好可行域。
    objective 本身也加入 penalty，避免早期 trial 在不可行域內被誤判為高分。
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True, constraints_func=_constraints_func)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        recipe = recipe_from_rta_choice(
            rf_power_w=trial.suggest_categorical("RF_Power_W", [50.0, 75.0]),
            process_time_min=trial.suggest_categorical("Process_Time_Min", [30.0, 60.0, 120.0]),
            rta_choice=trial.suggest_categorical("RTA_State", RTA_CHOICES),
        )
        prediction = model_suite.predict_properties(recipe.to_frame())
        score, constraints, feasible = score_prediction(prediction)
        trial.set_user_attr("constraints", constraints)
        trial.set_user_attr("is_feasible", feasible)
        for column, value in prediction.iloc[0].items():
            trial.set_user_attr(column, value)
        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
            "RF_Power_W": trial.params.get("RF_Power_W"),
            "Process_Time_Min": trial.params.get("Process_Time_Min"),
            "RTA_State": trial.params.get("RTA_State"),
            "is_feasible": trial.user_attrs.get("is_feasible", False),
            "constraint_on_off_g1": trial.user_attrs.get("constraints", (np.nan, np.nan))[0],
            "constraint_voltage_g2": trial.user_attrs.get("constraints", (np.nan, np.nan))[1],
        }
        for key, value in trial.user_attrs.items():
            if key not in {"constraints", "is_feasible"}:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["is_feasible", "value"], ascending=[False, False]), study
