from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOCAL_DEPS = PROJECT_ROOT / ".codex_deps"
for candidate_path in [SCRIPTS_DIR, LOCAL_DEPS]:
    if candidate_path.exists() and str(candidate_path) not in sys.path:
        sys.path.append(str(candidate_path))

import optuna

from sic_ml.data_processor import DataProcessor
from sic_ml.gpr_model_trainer import GPRModelTrainer


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


@dataclass
class ConstrainedBayesianOptimizer:
    """Part 3: 以 Optuna 實作 SiC RRAM 約束型多目標反向最佳化。

    優化目標：
    - Objective 1: minimize Leakage Current
    - Objective 2: maximize Endurance

    硬性約束：
    - On/Off Ratio >= 5
    - Operation Voltage <= 3 V

    方法：
    - 先用 Part 2 的 GPRModelTrainer 訓練 surrogate model。
    - Optuna 每提出一組製程條件 X，就呼叫 GPR 預測電性 mean。
    - 若 X 不滿足硬性約束，除了透過 constraints_func 告訴 sampler，也在 objective 中加入大懲罰，
      讓不符合物理條件的 recipe 不會被 Pareto frontier 誤收。
    """

    dataset: pd.DataFrame
    random_state: int = 42
    penalty_leakage_a: float = 1e3
    penalty_endurance_cycles: float = -1e3
    min_on_off_ratio: float = 5.0
    max_operation_voltage_v: float = 3.0
    model_specs: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "Leakage_Current_A": ("log10", "matern"),
            "Endurance_Cycles": ("log1p", "matern"),
            "On_Off_Ratio": ("log10", "matern"),
            "Operation_Voltage_V": ("none", "matern"),
        }
    )

    def __post_init__(self) -> None:
        self.models_: dict[str, GPRModelTrainer] = {}
        self.study_: optuna.study.Study | None = None
        self.trials_: pd.DataFrame | None = None

    def fit_surrogate_models(self) -> "ConstrainedBayesianOptimizer":
        """訓練所有 Optuna objective/constraint 會用到的 GPR surrogate models。"""
        for target, (target_transform, kernel_type) in self.model_specs.items():
            try:
                trainer = GPRModelTrainer(
                    target_column=target,
                    target_transform=target_transform,
                    kernel_type=kernel_type,
                    random_state=self.random_state,
                    alpha=1e-2,
                )
                trainer.fit(self.dataset)
                self.models_[target] = trainer
            except ValueError as exc:
                raise ValueError(
                    f"無法訓練 {target} 的 GPR surrogate。"
                    f"請確認 condition-level dataset 是否有足夠非空樣本。原始錯誤：{exc}"
                ) from exc
        return self

    @staticmethod
    def _has_rta_from_temperature(rta_temperature_c: float) -> int:
        """25 C 在本研究中代表 No RTA，其餘溫度代表實際退火。"""
        return 0 if float(rta_temperature_c) == 25.0 else 1

    def _suggest_recipe(self, trial: optuna.Trial) -> pd.DataFrame:
        """定義製程搜尋空間。

        RF Power 與 Process Time 使用 step-discrete search：
        - RF_Power: 50, 75
        - Process_Time: 30, 60, 90, 120
        RTA_Temperature 使用 categorical：
        - 25 = No RTA
        - 400, 500 = RTA temperature
        """
        rf_power = trial.suggest_float("RF_Power", 50.0, 75.0, step=25.0)
        process_time = trial.suggest_float("Process_Time", 30.0, 120.0, step=30.0)
        rta_temperature = trial.suggest_categorical("RTA_Temperature", [25.0, 400.0, 500.0])
        has_rta = self._has_rta_from_temperature(rta_temperature)
        return pd.DataFrame(
            [
                {
                    "RF_Power_W": rf_power,
                    "Process_Time_Min": process_time,
                    "RTA_Temperature_C": rta_temperature,
                    "Has_RTA": has_rta,
                }
            ]
        )

    def predict_recipe(self, recipe: pd.DataFrame) -> dict[str, float]:
        """用 GPR surrogate 預測單一 recipe 的電性 mean/std。

        回傳 mean 是 objective 使用的主值；std 則用於論文報告中呈現可靠度。
        """
        if not self.models_:
            raise RuntimeError("尚未訓練 GPR surrogate models，請先呼叫 fit_surrogate_models()。")

        outputs: dict[str, float] = {}
        for target, model in self.models_.items():
            pred = model.predict_with_uncertainty(recipe)
            outputs[f"{target}_Mean"] = float(pred[f"{target}_Mean"].iloc[0])
            outputs[f"{target}_Std"] = float(pred[f"{target}_Std"].iloc[0])
            outputs[f"{target}_CI95_Low"] = float(pred[f"{target}_CI95_Low"].iloc[0])
            outputs[f"{target}_CI95_High"] = float(pred[f"{target}_CI95_High"].iloc[0])
        return outputs

    def _constraint_values(self, predictions: dict[str, float]) -> tuple[float, float]:
        """Optuna constraints_func 使用 g_i(X) <= 0 表示可行。

        g1 = 5 - OnOff <= 0  等價於 OnOff >= 5
        g2 = Vop - 3 <= 0   等價於 Vop <= 3 V
        """
        g1_on_off = self.min_on_off_ratio - predictions["On_Off_Ratio_Mean"]
        g2_voltage = predictions["Operation_Voltage_V_Mean"] - self.max_operation_voltage_v
        return float(g1_on_off), float(g2_voltage)

    def objective(self, trial: optuna.Trial) -> tuple[float, float]:
        """Optuna multi-objective function。

        若 recipe 可行：
        - 回傳 (Leakage mean, Endurance mean)
        - Optuna directions = ["minimize", "maximize"]

        若 recipe 不可行：
        - 對 leakage 給極大懲罰值
        - 對 endurance 給極小懲罰值
        這是保守處理，避免硬性物理限制被忽略。
        """
        recipe = self._suggest_recipe(trial)
        predictions = self.predict_recipe(recipe)
        constraints = self._constraint_values(predictions)
        is_feasible = all(value <= 0 for value in constraints)

        trial.set_user_attr("constraints", constraints)
        trial.set_user_attr("is_feasible", is_feasible)
        for column, value in recipe.iloc[0].items():
            trial.set_user_attr(column, float(value))
        for key, value in predictions.items():
            trial.set_user_attr(key, float(value))

        if not is_feasible:
            trial.set_user_attr("constraint_penalty_applied", True)
            return self.penalty_leakage_a, self.penalty_endurance_cycles

        trial.set_user_attr("constraint_penalty_applied", False)
        leakage = max(predictions["Leakage_Current_A_Mean"], 1e-15)
        endurance = max(predictions["Endurance_Cycles_Mean"], 0.0)
        return float(leakage), float(endurance)

    @staticmethod
    def _constraints_func(frozen_trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
        """Optuna sampler 讀取的 constraints function。"""
        return frozen_trial.user_attrs.get("constraints", (float("inf"), float("inf")))

    def optimize(self, n_trials: int = 100) -> pd.DataFrame:
        """執行 constrained multi-objective Bayesian optimization。"""
        if not self.models_:
            self.fit_surrogate_models()

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state,
            multivariate=True,
            constraints_func=self._constraints_func,
        )
        self.study_ = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
        self.study_.optimize(self.objective, n_trials=n_trials, show_progress_bar=False)
        self.trials_ = self._trials_to_frame(self.study_)
        return self.trials_

    def _trials_to_frame(self, study: optuna.study.Study) -> pd.DataFrame:
        """將 Optuna trials 整理成可匯入報告與知識庫的表格。"""
        rows: list[dict[str, float | int | bool | str]] = []
        for trial in study.trials:
            constraints = trial.user_attrs.get("constraints", (np.nan, np.nan))
            values = trial.values if trial.values is not None else [np.nan, np.nan]
            row = {
                "Trial": trial.number,
                "State": str(trial.state),
                "Leakage_Objective_Minimize": values[0],
                "Endurance_Objective_Maximize": values[1],
                "RF_Power": trial.params.get("RF_Power"),
                "Process_Time": trial.params.get("Process_Time"),
                "RTA_Temperature": trial.params.get("RTA_Temperature"),
                "Has_RTA": trial.user_attrs.get("Has_RTA"),
                "Constraint_OnOff_g1": constraints[0],
                "Constraint_Voltage_g2": constraints[1],
                "Is_Feasible": trial.user_attrs.get("is_feasible", False),
                "Penalty_Applied": trial.user_attrs.get("constraint_penalty_applied", False),
            }
            for key, value in trial.user_attrs.items():
                if key not in {"constraints", "is_feasible", "constraint_penalty_applied", "RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA"}:
                    row[key] = value
            rows.append(row)
        frame = pd.DataFrame(rows)
        return frame.sort_values(
            ["Is_Feasible", "Leakage_Objective_Minimize", "Endurance_Objective_Maximize"],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    def pareto_frontier(self) -> pd.DataFrame:
        """輸出 Optuna 的 feasible Pareto frontier。"""
        if self.study_ is None:
            raise RuntimeError("尚未執行 optimize()，無法取得 Pareto frontier。")
        rows: list[dict[str, float | int | bool | str]] = []
        for trial in self.study_.best_trials:
            if not trial.user_attrs.get("is_feasible", False):
                continue
            constraints = trial.user_attrs.get("constraints", (np.nan, np.nan))
            row = {
                "Trial": trial.number,
                "Leakage_Objective_Minimize": trial.values[0],
                "Endurance_Objective_Maximize": trial.values[1],
                "RF_Power": trial.params.get("RF_Power"),
                "Process_Time": trial.params.get("Process_Time"),
                "RTA_Temperature": trial.params.get("RTA_Temperature"),
                "Has_RTA": trial.user_attrs.get("Has_RTA"),
                "Constraint_OnOff_g1": constraints[0],
                "Constraint_Voltage_g2": constraints[1],
            }
            for key, value in trial.user_attrs.items():
                if key.endswith("_Mean") or key.endswith("_Std"):
                    row[key] = value
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["Leakage_Objective_Minimize", "Endurance_Objective_Maximize"], ascending=[True, False])


if __name__ == "__main__":
    processor = DataProcessor()
    condition_dataset = processor.build_condition_level_dataset()

    optimizer = ConstrainedBayesianOptimizer(condition_dataset)
    trials = optimizer.optimize(n_trials=100)
    pareto = optimizer.pareto_frontier()

    output_dir = PROJECT_ROOT / "DATA"
    output_dir.mkdir(exist_ok=True)
    trials.to_csv(output_dir / "part3_optuna_trials.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(output_dir / "part3_optuna_pareto_frontier.csv", index=False, encoding="utf-8-sig")

    print("Top feasible Optuna trials:")
    print(trials.head(10).to_string(index=False))
    print("\nFeasible Pareto frontier:")
    print(pareto.to_string(index=False) if not pareto.empty else "No feasible Pareto point found.")
