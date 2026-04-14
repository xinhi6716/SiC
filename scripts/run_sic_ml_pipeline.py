from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sic_ml.config import DATA_DIR, MODEL_DIR, REPORT_DIR
from sic_ml.data_loader import build_condition_level_dataset
from sic_ml.model_trainer import PredictiveModelSuite, evaluate_classifiers_on_training_data, leave_one_condition_out_cv
from sic_ml.optimizer import rank_candidate_grid, run_constrained_optuna_search


def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        value = float(value)
        if value == 0:
            return "0"
        if abs(value) >= 1e4 or abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def _to_markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No data available._"
    shown = frame.head(max_rows).copy()
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in headers) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_顯示前 {max_rows} 筆，共 {len(frame)} 筆。_")
    return "\n".join(lines)


def write_validation_report(
    dataset: pd.DataFrame,
    cv_summary: pd.DataFrame,
    classifier_summary: pd.DataFrame,
    path: Path,
) -> None:
    content = f"""# SiC RRAM GPR 模型驗證報告

## 設計修正

- `No RTA` 已轉成 `RTA_Temperature_C = 25`，並新增 `Has_RTA = 0`。
- 已退火資料使用實際 RTA 溫度，並設定 `Has_RTA = 1`。
- GPR 預測函數輸出 mean、std、95% confidence interval。

## Condition-Level Dataset

| 項目 | 數值 |
|---|---:|
| 製程條件筆數 | {len(dataset)} |
| 欄位數 | {len(dataset.columns)} |

## Regression LOCO CV Summary

{_to_markdown_table(cv_summary, max_rows=30)}

## Classification Training Sanity Check

{_to_markdown_table(classifier_summary, max_rows=30)}

## 數學邏輯

```text
Leakage target: log10(max(I_leakage, epsilon))
On/Off target: log10(max(OnOff, 1))
Endurance target: log1p(cycles)

GPR posterior:
y(x*) | D ~ Normal(mu(x*), sigma(x*)^2)
CI95 = mu +/- 1.96 * sigma
```
"""
    path.write_text(content, encoding="utf-8")


def write_optimization_report(grid_ranking: pd.DataFrame, optuna_trials: pd.DataFrame, path: Path) -> None:
    key_cols = [
        "RF_Power_W",
        "Process_Time_Min",
        "RTA_Temperature_C",
        "Has_RTA",
        "on_off_ratio_mean",
        "operation_voltage_v_mean",
        "endurance_cycles_mean",
        "leakage_current_a_mean",
        "is_feasible",
        "constrained_score",
    ]
    grid_view = grid_ranking[[col for col in key_cols if col in grid_ranking.columns]]

    trial_cols = [
        "trial_number",
        "RF_Power_W",
        "Process_Time_Min",
        "RTA_State",
        "on_off_ratio_mean",
        "operation_voltage_v_mean",
        "endurance_cycles_mean",
        "leakage_current_a_mean",
        "is_feasible",
        "value",
    ]
    trial_view = optuna_trials[[col for col in trial_cols if col in optuna_trials.columns]]

    content = f"""# SiC RRAM 約束型最佳化報告

## 硬性約束

```text
g1(X) = 5 - OnOff(X) <= 0
g2(X) = OperationVoltage(X) - 3 <= 0
```

## 目標函數

```text
maximize:
0.60 * log1p(Endurance)
+ 0.40 * [-log10(Leakage)]
- 0.05 * predictive_uncertainty
- infeasible_penalty
```

## 候選製程 Grid Ranking

{_to_markdown_table(grid_view, max_rows=30)}

## Optuna Constrained Search Trials

{_to_markdown_table(trial_view, max_rows=30)}
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SiC RRAM GPR + constrained Optuna ML pipeline.")
    parser.add_argument("--n-trials", type=int, default=80, help="Number of Optuna trials.")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    dataset = build_condition_level_dataset()
    dataset.to_csv(DATA_DIR / "ml_condition_level_dataset.csv", index=False, encoding="utf-8-sig")

    model_suite = PredictiveModelSuite().fit(dataset)

    cv_predictions, cv_summary = leave_one_condition_out_cv(dataset)
    classifier_summary = evaluate_classifiers_on_training_data(model_suite, dataset)
    cv_predictions.to_csv(DATA_DIR / "ml_loco_cv_predictions.csv", index=False, encoding="utf-8-sig")
    cv_summary.to_csv(DATA_DIR / "ml_loco_cv_summary.csv", index=False, encoding="utf-8-sig")
    classifier_summary.to_csv(DATA_DIR / "ml_classifier_training_summary.csv", index=False, encoding="utf-8-sig")

    grid_ranking = rank_candidate_grid(model_suite)
    optuna_trials, _study = run_constrained_optuna_search(model_suite, n_trials=args.n_trials)
    grid_ranking.to_csv(DATA_DIR / "ml_candidate_grid_predictions.csv", index=False, encoding="utf-8-sig")
    optuna_trials.to_csv(DATA_DIR / "ml_optuna_trials.csv", index=False, encoding="utf-8-sig")

    write_validation_report(dataset, cv_summary, classifier_summary, REPORT_DIR / "ml_model_validation_report.md")
    write_optimization_report(grid_ranking, optuna_trials, REPORT_DIR / "ml_optimization_report.md")

    print(f"Condition-level rows: {len(dataset)}")
    print(f"Trained regression targets: {sorted(model_suite.regression_models)}")
    print(f"Candidate grid predictions: {len(grid_ranking)}")
    print(f"Optuna trials: {len(optuna_trials)}")
    print(f"Reports written to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
