# SiC RRAM Machine Learning Project

## 專案概述

本專案以碳化矽 (SiC) 薄膜電阻式記憶體 (RRAM) 為研究案例，建立一套從實驗 I-V curve 資料到機器學習製程最佳化的端到端分析流程。

核心目標：

- 讀取並清洗濺射製程與 I-V curve 原始資料。
- 萃取材料科學與半導體元件分析所需的電性指標。
- 使用 Gaussian Process Regression (GPR) 建立小樣本 surrogate model。
- 使用 Optuna 進行 constrained Bayesian optimization，尋找製程 Sweet Spot。
- 產生論文級圖表與 Markdown 分析報告。
- 擴展為可支援多材料、可推論、可重訓的 MLOps 系統。

## 核心製程特徵

| Feature | 說明 |
|---|---|
| `RF_Power_W` | 射頻磁控濺鍍功率 |
| `Process_Time_Min` | 濺鍍時間 |
| `RTA_Temperature_C` | 快速熱退火溫度；未退火以 25°C 表示 |
| `Has_RTA` | 是否退火；未退火為 0，有退火為 1 |

## 核心預測目標

| Target | 說明 |
|---|---|
| `Forming_Voltage_V` | Forming 電壓 |
| `Operation_Voltage_V` | 操作電壓 |
| `Leakage_Current_A` | 漏電流 |
| `On_Off_Ratio` | 高低阻態開關比 |
| `Endurance_Cycles` | 連續操作次數 |
| `Conduction_Mechanism` | 傳導機制，例如 Ohmic、Poole-Frenkel、Hopping、Schottky Emission |

## 目前資料夾結構

```text
SiC/
├── DATA/
│   ├── cleaned_sic_sputtering_data.csv
│   ├── part3_optuna_trials.csv
│   ├── part3_optuna_pareto_frontier.csv
│   └── materials/
│       └── SiC/
│           └── master_dataset.csv
├── FIGURES/
│   ├── figure_1_pareto_frontier.png
│   └── figure_2_sweet_spot_parameters.png
├── MODELS/
│   └── SiC/
│       ├── *.joblib
│       ├── manifest.json
│       └── training_metrics.csv
├── REPORTS/
│   ├── ml_architecture_blueprint.md
│   ├── project_workflow_architecture.md
│   └── *_analysis.md
├── scripts/
│   └── sic_ml/
│       ├── data_processor.py
│       ├── gpr_model_trainer.py
│       ├── optuna_optimizer.py
│       ├── thesis_visualizer.py
│       └── material_prediction_system.py
├── main_pipeline.py
└── READ.MD
```

## 目標 MLOps 架構

後續重構目標：

```text
SiC/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/
│   ├── data_processor.py
│   ├── gpr_model_trainer.py
│   ├── optuna_optimizer.py
│   ├── thesis_visualizer.py
│   └── material_prediction_system.py
├── outputs/
│   ├── figures/
│   └── models/
└── main_pipeline.py
```

## 主要工作流

```text
cleaned I-V data
    -> DataProcessor
    -> condition-level dataset
    -> GPRModelTrainer
    -> GPR surrogate models with uncertainty
    -> ConstrainedBayesianOptimizer
    -> Optuna trials and Pareto frontier
    -> ThesisVisualizer
    -> publication-quality figures
```

## 主要模組

| 模組 | 職責 |
|---|---|
| `data_processor.py` | 讀取 cleaned I-V data，建立 `Has_RTA`，萃取 condition-level 電性指標 |
| `gpr_model_trainer.py` | 訓練 GPR，執行 LOOCV，輸出 mean/std/95% CI |
| `optuna_optimizer.py` | 使用 Optuna 執行 constrained multi-objective Bayesian optimization |
| `thesis_visualizer.py` | 產生 Pareto frontier 與 Sweet Spot 參數分佈圖 |
| `material_prediction_system.py` | 多材料註冊、模型推論、新資料 append 與重訓 |
| `main_pipeline.py` | 根目錄單一入口，串接資料前處理、GPR 訓練與 Optuna 最佳化 |

## 快速執行

目前過渡階段可使用根目錄入口：

```powershell
python .\main_pipeline.py --n-trials 100
```

若已完成 MLOps 目錄搬移，建議指定新資料路徑：

```powershell
python .\main_pipeline.py --cleaned-data .\data\processed\cleaned_sic_sputtering_data.csv --n-trials 100
```

產生論文圖表：

```powershell
python .\scripts\sic_ml\thesis_visualizer.py
```

執行多材料預測與重訓示範：

```powershell
python .\scripts\sic_ml\material_prediction_system.py
```

## 最佳化約束

目前 constrained Bayesian optimization 使用：

```text
Objective 1: minimize Leakage_Current_A
Objective 2: maximize Endurance_Cycles

Constraint 1: On_Off_Ratio >= 5
Constraint 2: Operation_Voltage_V <= 3 V
```

## 重要文件

| 文件 | 說明 |
|---|---|
| `REPORTS/ml_architecture_blueprint.md` | ML 架構藍圖 |
| `REPORTS/project_workflow_architecture.md` | 完整工作流、架構與模組引用關係 |
| `REPORTS/ml_model_validation_report.md` | 模型驗證摘要 |
| `REPORTS/ml_optimization_report.md` | 最佳化結果摘要 |

## 下一步

建議下一階段依序進行：

1. 建立標準 MLOps 目錄：`data/`, `src/`, `outputs/`。
2. 將核心模組從 `scripts/sic_ml/` 搬移到 `src/`。
3. 統一路徑設定，移除舊的 `DATA/`, `FIGURES/`, `MODELS/` 硬編碼。
4. 確認 `main_pipeline.py` 可完整執行。
5. 進入 Phase 6：Explainable AI (XAI) for Material Science。
