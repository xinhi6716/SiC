# Endurance 分析

資料來源：`RAW_DATA/` machine-readable CSV/XLSX/TXT；論文背景：`M1003101碩論.pdf`。


## 重點結論

- PDF baseline 顯示 RTA 對 Endurance 有明顯提升，尤其 75 W / 500 C 約 500 cycles。
- Machine-readable 分析以 Endurance CSV 中的 sweep block 數量作為 cycle count。
- 統計顯著性檢定結果：Kruskal-Wallis statistic=0.158, p=0.691
- 若某些 RTA 條件缺少至少兩個獨立 Endurance 檔案，顯著性檢定會標記為不適用。

## Machine-Readable Endurance 摘要

| rf_power_w | process_time_min | rta_condition | files | endurance_cycles_max | endurance_cycles_median | on_off_log10_median | leakage_median_a |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 60 | as_deposited | 2 | 51 | 35.5 | 0.0137 | 7.417e-04 |
| 75 | 30 | 400C | 1 | 50 | 50 | 0.0157 | 2.210e-05 |
| 75 | 120 | as_deposited | 1 | 50 | 50 | 0.0062 | 6.566e-04 |
| 50 | 120 | 400C | 1 | 30 | 30 | 0.0129 | 5.706e-04 |
| 75 | 60 | as_deposited | 1 | 30 | 30 | 0.0129 | 5.706e-04 |
| 50 |  | as_deposited | 1 | 20 | 20 | 0.0419 | 3.778e-05 |

## PDF Baseline Endurance

| rf_power_w | rta_condition | pdf_cycles |
| --- | --- | --- |
| 50 | as_deposited | 20 |
| 50 | 400C | 100 |
| 50 | 500C | 300 |
| 75 | as_deposited | 30 |
| 75 | 400C | 300 |
| 75 | 500C | 500 |

## Sweet Spot 參考

| rf_power_w | process_time_min | rta_condition | endurance_cycles | endurance_data_available | on_off_log10_median | leakage_median_a | sweet_spot_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | 30 | 400C | 50 | True | 0.016 | 2.210e-05 | 0.354 |
| 50 | 60 | as_deposited | 51 | True | 0.0075 | 6.965e-04 | 0.25 |
| 75 | 120 | as_deposited | 50 | True | 0.00839 | 6.550e-04 | 0.24 |
| 75 | 30 | 300C |  | False | 4.6 | 1.965e-08 | 0.188 |
| 75 | 60 | 300C |  | False | 4.27 | 5.300e-08 | 0.172 |
| 50 | 60 | 300C |  | False | 3.9 | 1.717e-07 | 0.154 |
| 50 | 30 | 300C |  | False | 1.34 | 4.391e-05 | 0.0522 |
| 50 | 120 | as_deposited |  | False | 0.878 | 2.652e-05 | 0.0463 |
| 50 | 30 | as_deposited |  | False | 0.666 | 1.743e-04 | 0.0259 |
| 75 | 120 | 400C |  | False | 0.438 | 2.090e-04 | 0.0194 |
| 50 | 30 | 400C |  | False | 0.607 | 3.671e-04 | 0.0184 |
| 75 | 60 | as_deposited | 30 | True | 0.0162 | 5.529e-04 | 0.00847 |
| 75 | 30 | as_deposited |  | False | 0.175 | 3.963e-04 | 0.00836 |
| 50 | 120 | 400C | 30 | True | 0.0138 | 5.687e-04 | 0.00732 |
| 75 | 120 | 300C |  | False | 0.0772 | 3.865e-04 | 0.00644 |

## 統計方法

```python
if each_RTA_group_has_at_least_two_independent_files:
    scipy.stats.kruskal(*cycle_count_groups)
else:
    significance = "not_applicable"
```
