from __future__ import annotations

from pathlib import Path


"""
集中式路徑管理模組。

所有核心模組都應從這裡 import 專案路徑，避免在程式中散落
`./DATA/`, `./FIGURES/`, `./MODELS/` 等 hardcode 路徑。
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Standard MLOps directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DATA_DIR = DATA_DIR / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SRC_DIR = PROJECT_ROOT / "src"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "REPORTS"
LOCAL_DEPS_DIR = PROJECT_ROOT / ".codex_deps"

# Core data files
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_sic_sputtering_data.csv"
CONDITION_LEVEL_DATA_PATH = PROCESSED_DATA_DIR / "sic_condition_level_training_data.csv"

# Modeling and validation outputs
GPR_LOOCV_SUMMARY_PATH = RESULTS_DATA_DIR / "gpr_loocv_summary.csv"
GPR_LOOCV_PREDICTIONS_PATH = RESULTS_DATA_DIR / "gpr_loocv_predictions.csv"

# Optuna optimization outputs
OPTUNA_TRIALS_PATH = RESULTS_DATA_DIR / "part3_optuna_trials.csv"
PARETO_FRONTIER_PATH = RESULTS_DATA_DIR / "part3_optuna_pareto_frontier.csv"

# Thesis figure outputs
PARETO_FIGURE_PATH = FIGURES_DIR / "figure_1_pareto_frontier.png"
SWEET_SPOT_FIGURE_PATH = FIGURES_DIR / "figure_2_sweet_spot_parameters.png"

# Multi-material retraining system paths
MATERIALS_DATA_DIR = PROCESSED_DATA_DIR / "materials"
MATERIAL_MODELS_DIR = MODELS_DIR / "materials"

# Legacy compatibility paths. Keep them here only, not scattered across modules.
LEGACY_DATA_DIR = PROJECT_ROOT / "DATA"
LEGACY_FIGURES_DIR = PROJECT_ROOT / "FIGURES"
LEGACY_MODELS_DIR = PROJECT_ROOT / "MODELS"
LEGACY_CLEANED_DATA_PATH = LEGACY_DATA_DIR / "cleaned_sic_sputtering_data.csv"


DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DATA_DIR,
    CONFIGS_DIR,
    SRC_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
]


def normalize_material_name(material_name: str) -> str:
    """Return a filesystem-safe material name while preserving readable labels."""

    cleaned = str(material_name).strip()
    if not cleaned:
        raise ValueError("material_name must not be empty.")
    return cleaned.replace("/", "_").replace("\\", "_").replace(" ", "_")


def material_raw_dir(material_name: str) -> Path:
    """Raw data directory for one material: data/raw/{material_name}/."""

    return RAW_DATA_DIR / normalize_material_name(material_name)


def material_processed_dir(material_name: str) -> Path:
    """Processed data directory for one material: data/processed/{material_name}/."""

    return PROCESSED_DATA_DIR / normalize_material_name(material_name)


def material_results_dir(material_name: str) -> Path:
    """Optional per-material results directory: data/results/{material_name}/."""

    return RESULTS_DATA_DIR / normalize_material_name(material_name)


def material_models_dir(material_name: str) -> Path:
    """Optional per-material model directory: outputs/models/{material_name}/."""

    return MODELS_DIR / normalize_material_name(material_name)


def material_cleaned_data_path(material_name: str) -> Path:
    """Canonical cleaned data path: data/processed/{material}/cleaned_{material}_data.csv."""

    safe_name = normalize_material_name(material_name)
    return material_processed_dir(safe_name) / f"cleaned_{safe_name.lower()}_data.csv"


def material_condition_dataset_path(material_name: str) -> Path:
    """Canonical condition-level metrics path for one material."""

    safe_name = normalize_material_name(material_name)
    return material_processed_dir(safe_name) / f"{safe_name.lower()}_condition_level_dataset.csv"


def ensure_directories() -> None:
    """建立所有標準輸出資料夾。"""

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_material_directories(material_name: str) -> None:
    """Create raw/processed/results/models folders for a material."""

    for directory in [
        material_raw_dir(material_name),
        material_processed_dir(material_name),
        material_results_dir(material_name),
        material_models_dir(material_name),
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_cleaned_data_path(user_path: str | Path | None = None) -> Path:
    """解析 cleaned I-V data 的實際位置。

    優先順序：
    1. 使用者在 CLI 指定的路徑。
    2. 新 MLOps 架構的 `data/processed/cleaned_sic_sputtering_data.csv`。
    3. 過渡期 legacy 路徑 `DATA/cleaned_sic_sputtering_data.csv`。

    注意：legacy fallback 只集中放在此模組，避免其他模組再次寫死舊路徑。
    """

    if user_path is not None:
        path = Path(user_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    if CLEANED_DATA_PATH.exists():
        return CLEANED_DATA_PATH

    if LEGACY_CLEANED_DATA_PATH.exists():
        return LEGACY_CLEANED_DATA_PATH

    raise FileNotFoundError(
        "找不到 cleaned_sic_sputtering_data.csv。請將資料放在 "
        f"{CLEANED_DATA_PATH}，或使用 --cleaned-data 指定路徑。"
    )


def model_path_for_target(target_name: str) -> Path:
    """回傳單一 target GPR 模型的輸出路徑。"""

    safe_name = target_name.replace("/", "_").replace("\\", "_")
    return MODELS_DIR / f"{safe_name}.joblib"
