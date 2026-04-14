# SiC RRAM GPR 模型驗證報告

## 設計修正

- `No RTA` 已轉成 `RTA_Temperature_C = 25`，並新增 `Has_RTA = 0`。
- 已退火資料使用實際 RTA 溫度，並設定 `Has_RTA = 1`。
- GPR 預測函數輸出 mean、std、95% confidence interval。

## Condition-Level Dataset

| 項目 | 數值 |
|---|---:|
| 製程條件筆數 | 17 |
| 欄位數 | 15 |

## Regression LOCO CV Summary

| target | n_predictions | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| endurance_cycles | 5 | 13.92 | 14.41 | -1.089 |
| forming_voltage_v | 17 | 1.023 | 1.199 | -0.52 |
| leakage_current_a | 15 | 1.778e-04 | 2.222e-04 | -0.9327 |
| on_off_ratio | 15 | 8173 | 1.486e+04 | -0.8505 |
| operation_voltage_v | 15 | 0.04718 | 0.07137 | 0.1911 |

## Classification Training Sanity Check

| target | n_samples | macro_f1 | balanced_accuracy |
| --- | --- | --- | --- |
| hrs_mechanism | 15 | 0.5114 | 0.6321 |
| lrs_mechanism | 15 | 1 | 1 |

## 數學邏輯

```text
Leakage target: log10(max(I_leakage, epsilon))
On/Off target: log10(max(OnOff, 1))
Endurance target: log1p(cycles)

GPR posterior:
y(x*) | D ~ Normal(mu(x*), sigma(x*)^2)
CI95 = mu +/- 1.96 * sigma
```
