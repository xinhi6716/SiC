from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src import paths
from src.config import MATERIAL_CONFIGS
from src.data_processor import DataProcessor
from src.model_manager import MaterialModelManager
from src.optuna_optimizer import ConstrainedBayesianOptimizer


logger = logging.getLogger(__name__)


def run_pipeline(material_name: str, n_trials: int, skip_etl: bool) -> None:
    """Run the material-selectable MLOps workflow.

    Workflow:
    1. Optional ETL from raw/cleaned row-level data to condition-level data.
    2. GPR model training for all configured material targets.
    3. Constrained Optuna optimization using material-specific search space
       and hard constraints from ``src.config``.

    Args:
        material_name: Material label. Supported values are currently ``SiC``
            and ``NiO``.
        n_trials: Number of Optuna optimization trials.
        skip_etl: If ``True``, skip ETL and start from processed condition data.

    Raises:
        Exception: Re-raises stage failures after logging enough context.
    """

    material_name = paths.normalize_material_name(material_name)
    paths.ensure_directories()
    paths.ensure_material_directories(material_name)

    logger.info("============================================================")
    logger.info("=== 開始執行 %s 的 MLOps Pipeline ===", material_name)
    logger.info("Project root: %s", paths.PROJECT_ROOT)
    logger.info("Optuna trials: %s", n_trials)
    logger.info("Skip ETL: %s", skip_etl)
    logger.info("============================================================")

    condition_dataset: pd.DataFrame | None = None

    try:
        if skip_etl:
            logger.info("=== 跳過 %s 的 ETL 流程，直接使用 processed data ===", material_name)
        else:
            logger.info("=== 開始執行 %s 的 ETL 流程 ===", material_name)
            processor = _build_data_processor(material_name)
            cleaned = processor.clean_raw_data(save=True)
            logger.info("%s cleaned row-level data shape: %s", material_name, cleaned.shape)
            condition_dataset = processor.build_condition_level_dataset(save=True)
            logger.info("%s condition-level data shape: %s", material_name, condition_dataset.shape)
            logger.info("=== 完成 %s 的 ETL 流程 ===", material_name)

        logger.info("=== 開始執行 %s 的 GPR 模型訓練 ===", material_name)
        model_manager = MaterialModelManager(material_name=material_name, dataset=condition_dataset)
        training_metrics = model_manager.train_all_targets(save=True)
        logger.info(
            "%s model training status counts: %s",
            material_name,
            training_metrics["Status"].value_counts(dropna=False).to_dict(),
        )
        logger.info("=== 完成 %s 的 GPR 模型訓練 ===", material_name)

        logger.info("=== 開始執行 %s 的 Constrained Bayesian Optimization ===", material_name)
        optimizer = ConstrainedBayesianOptimizer(
            material_name=material_name,
            manager=model_manager,
        )
        trials = optimizer.optimize(n_trials=n_trials, save=True)
        pareto = optimizer.pareto_frontier()
        logger.info("%s Optuna trials shape: %s", material_name, trials.shape)
        logger.info("%s Pareto frontier shape: %s", material_name, pareto.shape)
        logger.info("=== 完成 %s 的 Constrained Bayesian Optimization ===", material_name)

        logger.info("============================================================")
        logger.info("=== %s Pipeline 執行完成 ===", material_name)
        logger.info("Models: %s", paths.material_models_dir(material_name))
        logger.info("Results: %s", paths.material_results_dir(material_name))
        logger.info("============================================================")
    except Exception as exc:
        logger.error("%s pipeline failed: %s", material_name, exc, exc_info=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list for testing. Defaults to ``sys.argv``.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Material-selectable MLOps pipeline for RRAM experiments.",
    )
    parser.add_argument(
        "--material",
        choices=_registered_material_choices(),
        default="SiC",
        help="Material to process. Default: SiC.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna optimization trials. Default: 100.",
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Skip ETL and start from data/processed/{material}/ condition-level data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list for tests or programmatic invocation.

    Returns:
        Process exit code. ``0`` means success and ``1`` means failure.
    """

    args = parse_args(argv)
    if args.n_trials <= 0:
        logger.error("--n-trials must be a positive integer, got %s", args.n_trials)
        return 1

    try:
        run_pipeline(
            material_name=args.material,
            n_trials=args.n_trials,
            skip_etl=args.skip_etl,
        )
    except Exception:
        return 1
    return 0


def _build_data_processor(material_name: str) -> DataProcessor:
    """Create a material-aware DataProcessor.

    SiC currently has historical row-level cleaned data in the legacy fallback
    path. When that file exists, pass it explicitly so the new material-aware
    processor can still rebuild a condition-level dataset without requiring raw
    SiC files.
    """

    if material_name.lower() == "sic":
        try:
            cleaned_path = paths.resolve_cleaned_data_path()
            return DataProcessor(material_name=material_name, data_path=cleaned_path)
        except FileNotFoundError:
            logger.warning("No legacy SiC cleaned data found; falling back to raw SiC ETL.")
    return DataProcessor(material_name=material_name)


def _registered_material_choices() -> list[str]:
    """Return argparse choices from the centralized material registry."""

    return sorted(config.name for config in MATERIAL_CONFIGS.values())


if __name__ == "__main__":
    sys.exit(main())
