from __future__ import annotations

import json
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from sic_ml.data_processor import DataProcessor
from sic_ml.gpr_model_trainer import GPRModelTrainer


DATA_DIR = PROJECT_ROOT / "DATA"
REPORT_DIR = PROJECT_ROOT / "REPORTS"
MODEL_DIR = PROJECT_ROOT / "MODELS"


@dataclass(frozen=True)
class ConstraintSpec:
    """材料最佳化時的硬性約束。

    範例：
    - On_Off_Ratio >= 5
    - Operation_Voltage_V <= 3

    使用字串 operator 是為了未來方便接 API / Streamlit / Gradio 表單。
    """

    target: str
    operator: str
    threshold: float
    description: str = ""


@dataclass
class MaterialConfig:
    """多材料註冊設定。

    這個 dataclass 是系統擴展到 HfO2、TaOx、Al2O3 等材料時的 metadata 入口。
    每種材料都可以有自己的 feature list、target list、target transform 與硬性約束。
    """

    name: str
    features: list[str]
    targets: list[str]
    constraints: list[ConstraintSpec]
    target_transforms: dict[str, str] = field(default_factory=dict)
    master_dataset_path: Path | None = None
    model_dir: Path | None = None
    raw_data_path: Path | None = None

    def __post_init__(self) -> None:
        safe_name = self.name.replace(" ", "_")
        if self.master_dataset_path is None:
            self.master_dataset_path = DATA_DIR / "materials" / safe_name / "master_dataset.csv"
        if self.model_dir is None:
            self.model_dir = MODEL_DIR / safe_name
        if self.raw_data_path is None and self.name == "SiC":
            self.raw_data_path = DATA_DIR / "cleaned_sic_sputtering_data.csv"

    def transform_for_target(self, target: str) -> str:
        """取得 target 的模型空間轉換方式。"""
        if target in self.target_transforms:
            return self.target_transforms[target]
        if "Leakage" in target or "On_Off" in target:
            return "log10"
        if "Endurance" in target or "Cycle" in target:
            return "log1p"
        return "none"


class MaterialRegistry:
    """材料設定註冊中心。

    未來要新增材料時，只要建立新的 MaterialConfig 並 register，
    PredictionEngine 與 RetrainingManager 就能使用同一套介面。
    """

    def __init__(self) -> None:
        self._configs: dict[str, MaterialConfig] = {}

    def register(self, config: MaterialConfig) -> None:
        key = config.name.lower()
        if key in self._configs:
            raise ValueError(f"材料已註冊，請勿重複註冊：{config.name}")
        self._configs[key] = config

    def get(self, material_name: str) -> MaterialConfig:
        key = material_name.lower()
        if key not in self._configs:
            available = ", ".join(sorted(config.name for config in self._configs.values()))
            raise KeyError(f"找不到材料設定：{material_name}。目前已註冊：{available}")
        return self._configs[key]

    def list_materials(self) -> list[str]:
        return sorted(config.name for config in self._configs.values())

    @classmethod
    def with_default_sic(cls) -> "MaterialRegistry":
        registry = cls()
        registry.register(
            MaterialConfig(
                name="SiC",
                features=["RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA"],
                targets=[
                    "Forming_Voltage_V",
                    "Operation_Voltage_V",
                    "Leakage_Current_A",
                    "On_Off_Ratio",
                    "Endurance_Cycles",
                ],
                target_transforms={
                    "Forming_Voltage_V": "none",
                    "Operation_Voltage_V": "none",
                    "Leakage_Current_A": "log10",
                    "On_Off_Ratio": "log10",
                    "Endurance_Cycles": "log1p",
                },
                constraints=[
                    ConstraintSpec("On_Off_Ratio", ">=", 5.0, "RRAM memory window lower bound"),
                    ConstraintSpec("Operation_Voltage_V", "<=", 3.0, "Low-voltage operation requirement"),
                ],
            )
        )
        return registry


class PredictionEngine:
    """材料電性預測引擎。

    給定 material name 與製程參數 X，系統會：
    1. 載入對應材料設定。
    2. 載入或自動訓練 GPR 模型。
    3. 檢查輸入特徵是否超出歷史資料範圍。
    4. 回傳每個 target 的 mean、std、95% CI。
    """

    def __init__(self, registry: MaterialRegistry | None = None, auto_train: bool = True) -> None:
        self.registry = registry or MaterialRegistry.with_default_sic()
        self.auto_train = auto_train
        self._model_cache: dict[str, dict[str, GPRModelTrainer]] = {}

    def predict(self, material_name: str, process_parameters: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        config = self.registry.get(material_name)
        x_frame = self._normalize_feature_input(config, process_parameters)
        warnings_list = self._check_extrapolation(config, x_frame)
        models = self._load_models(config)

        output = x_frame.copy()
        for target, model in models.items():
            pred = model.predict_with_uncertainty(x_frame)
            output = output.join(pred)

        output["Material"] = config.name
        output["Extrapolation_Warning"] = "; ".join(warnings_list)
        return output

    def _normalize_feature_input(self, config: MaterialConfig, process_parameters: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
        """將 API/dict/DataFrame 輸入統一成 feature DataFrame。"""
        if isinstance(process_parameters, pd.DataFrame):
            frame = process_parameters.copy()
        elif isinstance(process_parameters, dict):
            frame = pd.DataFrame([process_parameters])
        else:
            raise TypeError("process_parameters 必須是 dict 或 pandas DataFrame。")

        missing = [feature for feature in config.features if feature not in frame.columns]
        if missing:
            raise ValueError(f"輸入製程參數缺少特徵欄位：{missing}")

        frame = frame[config.features].copy()
        for feature in config.features:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        return frame

    def _load_models(self, config: MaterialConfig) -> dict[str, GPRModelTrainer]:
        if config.name in self._model_cache:
            return self._model_cache[config.name]

        model_dir = Path(config.model_dir)
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists() and self.auto_train:
            RetrainingManager(self.registry).train_material_models(config.name, reason="auto_train_missing_models")

        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到模型 manifest：{manifest_path}")

        models: dict[str, GPRModelTrainer] = {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for target, model_file in manifest.get("model_files", {}).items():
            path = model_dir / model_file
            if path.exists():
                models[target] = joblib.load(path)
        if not models:
            raise RuntimeError(f"{config.name} 沒有可載入的 GPR 模型。")

        self._model_cache[config.name] = models
        return models

    def _check_extrapolation(self, config: MaterialConfig, x_frame: pd.DataFrame) -> list[str]:
        """檢查輸入是否超出 master dataset 的歷史範圍。

        小樣本 GPR 在外插區域的可靠度會快速下降，因此推論時必須提示使用者。
        """
        dataset = _load_or_bootstrap_master_dataset(config)
        warnings_list: list[str] = []
        for feature in config.features:
            if feature not in dataset.columns:
                continue
            series = pd.to_numeric(dataset[feature], errors="coerce").dropna()
            if series.empty:
                continue
            min_value = float(series.min())
            max_value = float(series.max())
            proposed = pd.to_numeric(x_frame[feature], errors="coerce")
            outside = proposed[(proposed < min_value) | (proposed > max_value)]
            if not outside.empty:
                message = (
                    f"{feature}={outside.iloc[0]} 超出歷史範圍 "
                    f"[{min_value}, {max_value}]，GPR 可能進入外插區。"
                )
                warnings.warn(message, UserWarning)
                warnings_list.append(message)
        return warnings_list


class RetrainingManager:
    """新數據重訓管理器。

    重訓數據流：
    1. 驗證新 CSV 欄位是否符合 MaterialConfig。
    2. 備份舊 master dataset。
    3. append 新資料到 master dataset。
    4. 針對每個 target 重新 fit GPR。
    5. 輸出模型更新報告，比較重訓前後 LOOCV 指標。
    """

    def __init__(self, registry: MaterialRegistry | None = None) -> None:
        self.registry = registry or MaterialRegistry.with_default_sic()

    def retrain_from_new_csv(self, material_name: str, new_csv_path: str | Path) -> Path:
        config = self.registry.get(material_name)
        new_data = self._load_and_validate_new_data(config, Path(new_csv_path))
        old_dataset = _load_or_bootstrap_master_dataset(config)
        old_metrics = self._evaluate_targets(config, old_dataset)

        self._backup_master_dataset(config)
        combined = pd.concat([old_dataset, new_data], ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        _save_master_dataset(config, combined)

        new_metrics = self.train_material_models(material_name, reason="new_data_retraining")
        return self._write_update_report(config, new_data, old_metrics, new_metrics)

    def train_material_models(self, material_name: str, reason: str = "manual_training") -> pd.DataFrame:
        config = self.registry.get(material_name)
        dataset = _load_or_bootstrap_master_dataset(config)
        metrics = self._train_and_save_models(config, dataset, reason=reason)
        return metrics

    def _load_and_validate_new_data(self, config: MaterialConfig, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到新實驗 CSV：{csv_path}")
        new_data = pd.read_csv(csv_path)
        if new_data.empty:
            raise ValueError("新實驗 CSV 是空表，無法重訓。")

        if set(config.features).issubset(new_data.columns):
            condition_data = new_data.copy()
        elif DataProcessor.REQUIRED_COLUMNS.issubset(new_data.columns):
            processor = DataProcessor(data_path=csv_path)
            condition_data = processor.build_condition_level_dataset()
        else:
            raise ValueError(
                "新資料欄位不符合條件層級或 cleaned I-V 格式。"
                f"至少需要 features={config.features}，或 cleaned I-V 必要欄位。"
            )

        missing_features = [feature for feature in config.features if feature not in condition_data.columns]
        missing_targets = [target for target in config.targets if target not in condition_data.columns]
        if missing_features:
            raise ValueError(f"新資料缺少特徵欄位：{missing_features}")
        if len(missing_targets) == len(config.targets):
            raise ValueError(f"新資料至少要包含一個 target；目前缺少全部 targets：{missing_targets}")

        for feature in config.features:
            condition_data[feature] = pd.to_numeric(condition_data[feature], errors="raise")
        return condition_data

    def _train_and_save_models(self, config: MaterialConfig, dataset: pd.DataFrame, reason: str) -> pd.DataFrame:
        model_dir = Path(config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_files: dict[str, str] = {}
        metric_rows: list[dict[str, Any]] = []

        for target in config.targets:
            if target not in dataset.columns:
                continue
            trainable = dataset.dropna(subset=config.features + [target])
            if len(trainable) < 3:
                metric_rows.append({"Target": target, "Status": "skipped_insufficient_rows", "N": len(trainable)})
                continue

            trainer = GPRModelTrainer(
                feature_columns=tuple(config.features),
                target_column=target,
                target_transform=config.transform_for_target(target),
                alpha=1e-2,
            )
            trainer.fit(trainable)
            cv = trainer.cross_validate(trainable)
            summary = trainer.summarize_cv(cv).iloc[0].to_dict()
            summary["Status"] = "trained"
            summary["N"] = len(trainable)
            metric_rows.append(summary)

            model_file = f"{target}.joblib"
            joblib.dump(trainer, model_dir / model_file)
            model_files[target] = model_file

        manifest = {
            "material": config.name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "features": config.features,
            "targets": config.targets,
            "constraints": [asdict(constraint) for constraint in config.constraints],
            "model_files": model_files,
            "feature_ranges": _feature_ranges(dataset, config.features),
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(model_dir / "training_metrics.csv", index=False, encoding="utf-8-sig")
        return metrics

    def _evaluate_targets(self, config: MaterialConfig, dataset: pd.DataFrame) -> pd.DataFrame:
        metric_rows: list[dict[str, Any]] = []
        for target in config.targets:
            if target not in dataset.columns:
                continue
            trainable = dataset.dropna(subset=config.features + [target])
            if len(trainable) < 3:
                metric_rows.append({"Target": target, "Status": "skipped_insufficient_rows", "N": len(trainable)})
                continue
            trainer = GPRModelTrainer(
                feature_columns=tuple(config.features),
                target_column=target,
                target_transform=config.transform_for_target(target),
                alpha=1e-2,
            )
            trainer.fit(trainable)
            summary = trainer.summarize_cv(trainer.cross_validate(trainable)).iloc[0].to_dict()
            summary["Status"] = "evaluated"
            summary["N"] = len(trainable)
            metric_rows.append(summary)
        return pd.DataFrame(metric_rows)

    def _backup_master_dataset(self, config: MaterialConfig) -> Path | None:
        master_path = Path(config.master_dataset_path)
        if not master_path.exists():
            return None
        backup_dir = master_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"master_dataset_{timestamp}.csv"
        shutil.copy2(master_path, backup_path)
        return backup_path

    def _write_update_report(
        self,
        config: MaterialConfig,
        new_data: pd.DataFrame,
        old_metrics: pd.DataFrame,
        new_metrics: pd.DataFrame,
    ) -> Path:
        REPORT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"{config.name}_model_update_report_{timestamp}.md"

        comparison = old_metrics.merge(new_metrics, on="Target", how="outer", suffixes=("_before", "_after"))
        content = [
            f"# {config.name} Continuous Learning Update Report",
            "",
            "## 新增資料摘要",
            "",
            f"- 新增筆數：{len(new_data)}",
            f"- Master dataset：`{Path(config.master_dataset_path).as_posix()}`",
            f"- Model directory：`{Path(config.model_dir).as_posix()}`",
            "",
            "## LOOCV 指標變化",
            "",
            _to_markdown(comparison),
        ]
        report_path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")
        return report_path


def _load_or_bootstrap_master_dataset(config: MaterialConfig) -> pd.DataFrame:
    master_path = Path(config.master_dataset_path)
    if master_path.exists():
        return pd.read_csv(master_path)

    master_path.parent.mkdir(parents=True, exist_ok=True)
    if config.name == "SiC" and config.raw_data_path and Path(config.raw_data_path).exists():
        processor = DataProcessor(data_path=config.raw_data_path)
        dataset = processor.build_condition_level_dataset()
        _save_master_dataset(config, dataset)
        return dataset
    raise FileNotFoundError(f"尚未建立 {config.name} master dataset：{master_path}")


def _save_master_dataset(config: MaterialConfig, dataset: pd.DataFrame) -> None:
    master_path = Path(config.master_dataset_path)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(master_path, index=False, encoding="utf-8-sig")


def _feature_ranges(dataset: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for feature in features:
        if feature not in dataset.columns:
            continue
        values = pd.to_numeric(dataset[feature], errors="coerce").dropna()
        if values.empty:
            continue
        ranges[feature] = {"min": float(values.min()), "max": float(values.max())}
    return ranges


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No metrics available._"
    shown = frame.fillna("")
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    registry = MaterialRegistry.with_default_sic()
    manager = RetrainingManager(registry)
    metrics = manager.train_material_models("SiC", reason="initial_bootstrap")
    print("Training metrics:")
    print(metrics.to_string(index=False))

    engine = PredictionEngine(registry)
    demo_prediction = engine.predict(
        "SiC",
        {
            "RF_Power_W": 75,
            "Process_Time_Min": 30,
            "RTA_Temperature_C": 400,
            "Has_RTA": 1,
        },
    )
    print("\nDemo prediction:")
    print(demo_prediction.to_string(index=False))
