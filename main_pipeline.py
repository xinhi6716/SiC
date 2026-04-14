from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class PipelinePaths:
    """集中管理 MLOps 目錄，避免各模組散落硬編碼路徑。"""

    project_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "data" / "results"

    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "outputs" / "figures"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "outputs" / "models"

    @property
    def local_deps_dir(self) -> Path:
        return self.project_root / ".codex_deps"


TARGET_SPECS: dict[str, dict[str, Any]] = {
    "Forming_Voltage_V": {"target_transform": "none", "kernel_type": "matern"},
    "Operation_Voltage_V": {"target_transform": "none", "kernel_type": "matern"},
    # 漏電流跨越多個數量級，必須在 GPR 的 target space 使用 log10。
    "Leakage_Current_A": {"target_transform": "log10", "kernel_type": "matern"},
    "On_Off_Ratio": {"target_transform": "log10", "kernel_type": "matern"},
    "Endurance_Cycles": {"target_transform": "log1p", "kernel_type": "matern"},
}


def bootstrap_import_path(paths: PipelinePaths) -> None:
    """讓根目錄執行的 main_pipeline.py 能乾淨 import src package。"""

    for candidate in (paths.project_root, paths.local_deps_dir):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def import_pipeline_classes(paths: PipelinePaths):
    """優先載入重構後的 src 模組；搬檔前保留 legacy fallback 方便過渡。"""

    bootstrap_import_path(paths)
    try:
        from src.data_processor import DataProcessor
        from src.gpr_model_trainer import GPRModelTrainer
        from src.optuna_optimizer import ConstrainedBayesianOptimizer

        return DataProcessor, GPRModelTrainer, ConstrainedBayesianOptimizer
    except ModuleNotFoundError as src_error:
        legacy_scripts_dir = paths.project_root / "scripts"
        if legacy_scripts_dir.exists() and str(legacy_scripts_dir) not in sys.path:
            sys.path.insert(0, str(legacy_scripts_dir))
        try:
            from sic_ml.data_processor import DataProcessor
            from sic_ml.gpr_model_trainer import GPRModelTrainer
            from sic_ml.optuna_optimizer import ConstrainedBayesianOptimizer

            warnings.warn(
                "目前尚未找到 src package，暫時改用 scripts/sic_ml legacy 模組。"
                "正式重構後請改由 src/ 載入。",
                UserWarning,
            )
            return DataProcessor, GPRModelTrainer, ConstrainedBayesianOptimizer
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "找不到核心模組。請確認 data_processor.py、gpr_model_trainer.py、"
                "optuna_optimizer.py 已移入 src/，且根目錄可 import src package。"
            ) from src_error


def ensure_output_dirs(paths: PipelinePaths) -> None:
    """只建立流程輸出所需目錄，不搬移既有資料。"""

    for output_dir in (paths.processed_dir, paths.results_dir, paths.figures_dir, paths.models_dir):
        output_dir.mkdir(parents=True, exist_ok=True)


def resolve_cleaned_data_path(paths: PipelinePaths, cleaned_data_path: str | Path | None) -> Path:
    """解析清洗後 I-V 資料位置，正式架構以 data/processed 為準。"""

    if cleaned_data_path is not None:
        path = Path(cleaned_data_path)
        return path if path.is_absolute() else paths.project_root / path

    canonical_path = paths.processed_dir / "cleaned_sic_sputtering_data.csv"
    if canonical_path.exists():
        return canonical_path

    legacy_path = paths.project_root / "DATA" / "cleaned_sic_sputtering_data.csv"
    if legacy_path.exists():
        warnings.warn(
            f"找不到新架構資料檔 {canonical_path}，本次暫時使用 legacy 路徑 {legacy_path}。",
            UserWarning,
        )
        return legacy_path

    raise FileNotFoundError(
        "找不到 cleaned_sic_sputtering_data.csv。請放置於 "
        f"{canonical_path}，或用 --cleaned-data 指定檔案。"
    )


def run_preprocessing(paths: PipelinePaths, DataProcessor, cleaned_data_path: Path) -> pd.DataFrame:
    """資料前處理：由 cleaned I-V row-level data 聚合成 condition-level training data。"""

    processor = DataProcessor(data_path=cleaned_data_path)
    condition_dataset = processor.build_condition_level_dataset()
    processor.fit_feature_scaler(condition_dataset)

    output_path = paths.processed_dir / "sic_condition_level_training_data.csv"
    condition_dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[1/3] Preprocessing complete: {output_path}")
    print(f"      Condition-level shape: {condition_dataset.shape}")
    return condition_dataset


def train_gpr_models(paths: PipelinePaths, GPRModelTrainer, condition_dataset: pd.DataFrame) -> pd.DataFrame:
    """訓練每個電性 target 的 GPR，並輸出 LOOCV 指標與模型權重。"""

    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for target, spec in TARGET_SPECS.items():
        trainer = GPRModelTrainer(
            target_column=target,
            target_transform=spec["target_transform"],
            kernel_type=spec["kernel_type"],
            alpha=1e-2,
        )

        try:
            cv_predictions = trainer.cross_validate(condition_dataset)
            cv_summary = trainer.summarize_cv(cv_predictions)
            trainer.fit(condition_dataset)

            model_path = paths.models_dir / f"{target}.joblib"
            joblib.dump(trainer, model_path)

            cv_summary["Status"] = "trained"
            cv_summary["Model_Path"] = str(model_path)
            summary_frames.append(cv_summary)
            prediction_frames.append(cv_predictions.assign(Target=target))
            print(f"      Trained {target}: {model_path}")
        except ValueError as exc:
            summary_frames.append(
                pd.DataFrame(
                    [
                        {
                            "Target": target,
                            "CV_Type": "LOOCV",
                            "N_Predictions": 0,
                            "R2_Model_Space": pd.NA,
                            "MSE_Model_Space": pd.NA,
                            "R2_Original_Scale": pd.NA,
                            "MSE_Original_Scale": pd.NA,
                            "Status": f"skipped: {exc}",
                            "Model_Path": pd.NA,
                        }
                    ]
                )
            )
            warnings.warn(f"{target} 訓練略過：{exc}", UserWarning)

    summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = paths.results_dir / "gpr_loocv_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions_path = paths.results_dir / "gpr_loocv_predictions.csv"
        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print(f"[2/3] GPR training complete: {summary_path}")
    return summary


def run_bayesian_optimization(
    paths: PipelinePaths,
    ConstrainedBayesianOptimizer,
    condition_dataset: pd.DataFrame,
    n_trials: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """執行 constrained multi-objective Bayesian optimization。"""

    optimizer = ConstrainedBayesianOptimizer(condition_dataset, random_state=random_state)
    trials = optimizer.optimize(n_trials=n_trials)
    pareto = optimizer.pareto_frontier()

    trials_path = paths.results_dir / "part3_optuna_trials.csv"
    pareto_path = paths.results_dir / "part3_optuna_pareto_frontier.csv"
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig")

    print(f"[3/3] Optuna optimization complete: {trials_path}")
    print(f"      Pareto frontier: {pareto_path}")
    return trials, pareto


def run_pipeline(cleaned_data_path: str | Path | None, n_trials: int, random_state: int) -> None:
    """單一入口：資料前處理 -> GPR 訓練 -> constrained Bayesian optimization。"""

    paths = PipelinePaths(project_root=Path(__file__).resolve().parent)
    ensure_output_dirs(paths)
    DataProcessor, GPRModelTrainer, ConstrainedBayesianOptimizer = import_pipeline_classes(paths)

    cleaned_path = resolve_cleaned_data_path(paths, cleaned_data_path)
    condition_dataset = run_preprocessing(paths, DataProcessor, cleaned_path)
    train_gpr_models(paths, GPRModelTrainer, condition_dataset)
    run_bayesian_optimization(paths, ConstrainedBayesianOptimizer, condition_dataset, n_trials, random_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SiC RRAM end-to-end MLOps pipeline")
    parser.add_argument(
        "--cleaned-data",
        default=None,
        help="清洗後 I-V CSV 路徑；預設讀取 data/processed/cleaned_sic_sputtering_data.csv。",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna Bayesian optimization trial 數。")
    parser.add_argument("--random-state", type=int, default=42, help="GPR/Optuna 隨機種子。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.cleaned_data, args.n_trials, args.random_state)
