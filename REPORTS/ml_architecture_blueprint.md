# ML Architecture Blueprint: SiC RRAM 製程參數預測與逆向最佳化

## Module 1: 問題定義與數學公式化 (Problem Formulation)

### 1.1 輸入特徵 X

本專案的單一製程條件可表示為：

```text
X = [RF Power, Process Time, RTA Temperature, Has_RTA]
```

| Feature | 型態 | 說明 |
|---|---:|---|
| `rf_power_w` | 數值 / 離散 | 射頻濺鍍功率，例如 50 W、75 W；若保留探索資料，也可能含 30 W |
| `process_time_min` | 數值 / 離散 | 濺鍍時間，例如 30、60、120 min |
| `RTA_Temperature_C` | 數值 / 離散 | RTA 溫度；未退火 No RTA 統一視為常溫 25 °C |
| `Has_RTA` | 二元分類 | 未退火 = 0，已退火 = 1，用來避免模型誤解「25 °C No RTA」等同低溫退火 |

### 1.2 預測目標 y

任務拆分為兩條模型路線：

```text
Regression:
f_reg(X) -> [Forming Voltage, Operation Voltage, Leakage Current, On/Off Ratio, Endurance]

Classification:
f_cls(X) -> [HRS Conduction Mechanism, LRS Conduction Mechanism]
```

| Target | 任務 | 資料型態 | 預期分佈 | 建議轉換 |
|---|---|---:|---|---|
| `forming_voltage_v` | 回歸 | positive continuous | 小樣本、可能右偏，受 forming threshold 影響 | 原尺度或 robust-scaled |
| `operation_voltage_v` | 回歸 | positive continuous | 多集中於 1.5-3 V 附近 | 原尺度 |
| `leakage_current_a` | 回歸 | positive continuous | 跨數量級、近似 log-normal | `log10(I_leakage)` |
| `on_off_ratio` | 回歸 | positive continuous | 跨數量級、右偏 | `log10(on_off_ratio)` |
| `endurance_cycles` | 回歸 / count | positive integer | count、右偏、可能 censored | `log1p(cycles)` |
| `hrs_mechanism` | 分類 | categorical | Poole-Frenkel / Ohmic / Hopping / Schottky | class label |
| `lrs_mechanism` | 分類 | categorical | Ohmic / Hopping / Schottky / others | class label |

### 1.3 建模粒度

建議以「製程條件層級」資料建模，而不是直接把每條 I-V curve 都視為獨立樣本。

- 對相同 `RF Power x Process Time x RTA` 的重複曲線做 median 或 robust aggregate。
- 保留曲線層級資料作為不確定性估計與異常追蹤來源。
- 避免同一製程條件下的重複量測同時出現在 train/test 中，造成 data leakage。

## Module 2: 資料前處理管線 (Data Preprocessing Pipeline)

### 2.1 數值轉換

漏電流與 On/Off Ratio 必須使用對數轉換：

```text
y_leakage = log10(max(leakage_current_a, epsilon))
y_onoff = log10(max(on_off_ratio, 1))
y_endurance = log1p(endurance_cycles)
```

理由：

- `leakage_current_a` 常落在 `1e-8` 到 `1e-3 A` 等跨數量級範圍，直接回歸會被大電流樣本主導。
- `on_off_ratio` 的物理意義本身就是數量級差距，`log10` 更貼近材料分析語境。
- `endurance_cycles` 是 count data，`log1p` 可降低 20、50、300、500 cycles 之間的尺度不平衡。

### 2.2 特徵縮放

首選：`StandardScaler`，搭配 Gaussian Process Regression、SVR 或 Bayesian Optimization。

| Scaler | 是否建議 | 理由 |
|---|---|---|
| `StandardScaler` | 首選 | 對 kernel-based model 的距離度量較穩定；特徵只有少數製程參數，解釋性佳 |
| `RobustScaler` | 備選 | 若保留 30 W、300 °C 等探索性條件且被視為 outlier，可用於敏感性分析 |
| `MinMaxScaler` | 不作為主線 | 小樣本下新條件容易落在訓練範圍邊界之外，外插不穩 |

RTA 建議使用混合編碼：

```text
Has_RTA = 0 if No RTA else 1
RTA_Temperature_C = 25 if No RTA else measured_RTA_temperature
RTA_Temperature_C_scaled = StandardScaler(RTA_Temperature_C)
```

### 2.3 驗證策略

小樣本材料實驗不能使用隨機 train/test split 作為唯一驗證。

| Strategy | 建議程度 | 用途 |
|---|---|---|
| Leave-One-Condition-Out CV | 首選 | 每次留下一組完整製程條件，例如 `75W / 60min / 400C`，防止同條件重複曲線造成 leakage |
| LOOCV | 備選 | 若 condition-level 樣本極少，可估計泛化誤差，但方差較高 |
| Bootstrap Confidence Interval | 建議搭配 | 對 MAE、RMSE、R2、F1-score 做不確定性估計 |

## Module 3: 預測模型選型 (Predictive Model Selection)

### 3.1 候選模型比較

| 模型 | 優點 | 缺點 | 適合任務 |
|---|---|---|---|
| Gaussian Process Regression / Classification | 小樣本友善、可輸出不確定性、非常適合接 Bayesian Optimization | 樣本數變大時計算成本高；多類別分類較麻煩 | 首選 regression baseline |
| Random Forest / ExtraTrees | 非線性、抗 outlier、不太需要 scaling | 小樣本下 uncertainty 不夠校準，外插能力弱 | 穩健 sanity check |
| SVR / SVC | 小樣本可用，kernel 可處理非線性 | 超參數敏感，不確定性不如 GPR | 對照模型 |
| XGBoost | 非線性強、表現好 | 小樣本容易過擬合，需嚴格 regularization | 後期資料量增加後再納入 |

### 3.2 首選模型

建議第一階段採用：

```text
Regression baseline:
One Gaussian Process Regression model per numeric target

Classification baseline:
Class-weighted SVC or RandomForestClassifier
```

理由：

- 目前資料屬於小樣本材料實驗，GPR 比大型黑箱模型更合理。
- GPR 的 predictive uncertainty 可以直接進入 Bayesian Optimization 的 acquisition function。
- 每個電性指標的分佈差異很大，one-model-per-target 比單一多輸出模型更容易診斷。

### 3.3 評估指標

| 任務 | 指標 |
|---|---|
| Forming / Operation Voltage | MAE、RMSE、R2 |
| Leakage Current | MAE on `log10(I)`、RMSE on `log10(I)` |
| On/Off Ratio | MAE on `log10(ratio)` |
| Endurance | MAE on `log1p(cycles)`，並回推原尺度誤差 |
| Conduction Mechanism | Macro F1、balanced accuracy、confusion matrix |

## Module 4: 逆向最佳化策略 (Inverse Optimization Strategy)

### 4.1 優化器選型

| 方法 | 優點 | 缺點 | 建議 |
|---|---|---|---|
| Bayesian Optimization | 小樣本效率高；可利用 GPR uncertainty；適合昂貴實驗 | 高維或複雜離散空間時需要小心設計 acquisition function | 首選 |
| Genetic Algorithm | 可處理離散/連續混合與複雜 constraints；不需 surrogate uncertainty | 需要大量 fitness evaluations；小樣本 surrogate 不穩時容易搜尋到假 optimum | 後期作為全域搜尋輔助 |

修正版建議主線：

```text
Primary optimizer: constrained Bayesian Optimization with Optuna TPESampler
Search space: discrete-aware mixed space
Acquisition: Expected Improvement or Upper Confidence Bound
Hard constraints:
g1(X) = 5 - OnOff(X) <= 0
g2(X) = OperationVoltage(X) - 3 <= 0
```

若後續加入更多參數，例如氣體流量、壓力、電極材料、保護氣氛，再考慮 Genetic Algorithm 或 NSGA-II 做多目標 Pareto frontier。

### 4.2 目標函數設計

令模型預測：

```text
L_hat(X) = predicted log10(leakage_current)
R_hat(X) = predicted log10(on_off_ratio)
E_hat(X) = predicted log1p(endurance_cycles)
Vop_hat(X) = predicted operation voltage
U(X) = predictive uncertainty penalty
```

在滿足硬性約束的前提下，單一 scalar objective 可定義為：

```text
Score(X)
= w_E * log1p(Endurance_hat(X))
+ w_L * [-log10(Leakage_hat(X))]
- w_U * uncertainty_penalty(X)
- infeasible_penalty(X)
```

其中：

```text
infeasible_penalty(X)
= lambda * [max(0, 5 - OnOff_hat(X)) + max(0, Vop_hat(X) - 3)]

uncertainty_penalty(X)
= sum predictive_std_target(X)
```

初始權重建議：

```text
w_E = 0.60   # Endurance
w_L = 0.40   # Leakage Current
w_U = 0.05   # GPR uncertainty penalty
lambda = 100 # hard-constraint violation penalty
```

搜尋空間：

```text
RF Power ∈ {50, 75}
Process Time ∈ {30, 60, 120}
RTA Temperature ∈ {as_deposited, 300, 400, 500}
```

可以加入論文先驗作為 soft prior：

```text
prior_bonus(X) > 0 if process_time = 60 min and Has_RTA = 1
```

但此 prior 只能作為 soft bias，不應覆蓋實際資料。

## Module 5: 實作藍圖與虛擬碼 (Implementation Roadmap & Pseudocode)

以下為高度抽象的 Python 虛擬碼架構，不是完整可執行程式碼：

```python
class SiCDataRepository:
    def load_cleaned_data(self):
        pass

    def build_condition_level_dataset(self):
        pass


class SiCPreprocessor:
    def transform_features(self, X):
        pass

    def transform_targets(self, y):
        pass

    def inverse_transform_predictions(self, y_pred):
        pass


class PredictiveModelSuite:
    def train_regression_models(self, X_train, y_train):
        pass

    def train_classification_models(self, X_train, y_train):
        pass

    def predict_properties(self, X_candidate):
        pass

    def estimate_uncertainty(self, X_candidate):
        pass


class ValidationEngine:
    def run_leave_one_condition_out_cv(self, dataset, model_suite):
        pass

    def summarize_metrics(self):
        pass


class SweetSpotObjective:
    def score(self, predicted_properties, uncertainty):
        pass

    def apply_constraints(self, candidate_recipe):
        pass


class BayesianRecipeOptimizer:
    def define_search_space(self):
        pass

    def propose_next_recipe(self):
        pass

    def rank_candidate_recipes(self):
        pass


class SiCMLPipeline:
    def run(self):
        dataset = repository.build_condition_level_dataset()
        processed = preprocessor.fit_transform(dataset)
        validation_report = validator.run_leave_one_condition_out_cv(processed, models)
        models.fit(processed)
        ranked_recipes = optimizer.rank_candidate_recipes(models, objective)
        return validation_report, ranked_recipes
```

### 5.1 預期輸出

| Output | 說明 |
|---|---|
| `model_validation_report.md` | 各 target 的 CV 指標與不確定性 |
| `predicted_property_table.csv` | 每組候選製程的預測電性 |
| `sweet_spot_ranking.csv` | 逆向最佳化排序 |
| `pareto_frontier.md` | 若改用多目標最佳化，輸出 Pareto frontier |
| `next_experiment_recommendations.md` | 建議下一輪最值得實驗驗證的條件 |

### 5.2 第一階段實作建議

建議第一階段採用：

```text
GPR surrogate + Bayesian Optimization + Leave-One-Condition-Out CV
```

這條路線最符合目前的小樣本材料實驗場景，也能自然提供不確定性，方便把模型預測轉換成可執行的下一輪實驗建議。
