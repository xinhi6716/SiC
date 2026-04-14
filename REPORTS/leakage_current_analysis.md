# 漏電流分析

資料來源：`RAW_DATA/` machine-readable CSV/XLSX/TXT；論文背景：`M1003101碩論.pdf`。


## 重點結論

- PDF baseline 指出未退火漏電流約 1e-4 A，RTA 後 1 H 條件可降到約 1e-7 A 等級。
- 本報告以低讀取電壓下較低電流分支作為 leakage proxy。
- 若低電壓區中位電流超過 1e-3 A，曲線會警示為可能短路或預先形成。

## 漏電流排序

| rf_power_w | process_time_min | rta_condition | curves | leakage_median_a | leakage_min_a | leakage_max_a | on_off_log10_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | 30 | 300C | 2 | 1.965e-08 | 1.667e-08 | 2.264e-08 | 4.6 |
| 75 | 60 | 300C | 2 | 5.300e-08 | 5.300e-08 | 5.300e-08 | 4.27 |
| 50 | 60 | 300C | 4 | 1.717e-07 | 3.145e-08 | 2.719e-04 | 3.9 |
| 75 | 30 | 400C | 52 | 2.210e-05 | 1.615e-05 | 1.975e-04 | 0.016 |
| 30 |  | as_deposited | 2 | 2.547e-05 | 4.189e-07 | 5.052e-05 | 1.85 |
| 50 | 120 | as_deposited | 113 | 2.652e-05 | 4.047e-07 | 3.797e-05 | 0.878 |
| 50 |  | as_deposited | 20 | 3.778e-05 | 9.743e-08 | 4.080e-05 | 0.0419 |
| 50 | 30 | 300C | 3 | 4.391e-05 | 1.398e-08 | 1.694e-04 | 1.34 |
| 50 | 30 | as_deposited | 4 | 1.743e-04 | 1.277e-04 | 2.630e-04 | 0.666 |
| 75 | 120 | 400C | 1 | 2.090e-04 | 2.090e-04 | 2.090e-04 | 0.438 |
| 50 | 30 | 400C | 4 | 3.671e-04 | 9.223e-08 | 8.010e-04 | 0.607 |
| 75 | 120 | 300C | 2 | 3.865e-04 | 7.195e-05 | 7.010e-04 | 0.0772 |
| 75 | 30 | as_deposited | 3 | 3.963e-04 | 3.745e-05 | 5.502e-04 | 0.175 |
| 75 | 60 | as_deposited | 35 | 5.529e-04 | 2.735e-08 | 6.100e-04 | 0.0162 |
| 50 | 120 | 400C | 31 | 5.687e-04 | 3.186e-04 | 6.100e-04 | 0.0138 |
| 75 | 120 | as_deposited | 55 | 6.550e-04 | 1.936e-04 | 7.926e-04 | 0.00839 |
| 50 | 60 | as_deposited | 79 | 6.965e-04 | 4.013e-07 | 0.00104 | 0.0075 |

## Sweet Spot 參考

| rf_power_w | process_time_min | rta_condition | curve_count | leakage_median_a | on_off_log10_median | endurance_cycles | endurance_data_available | sweet_spot_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | 30 | 400C | 52 | 2.210e-05 | 0.016 | 50 | True | 0.354 |
| 50 | 60 | as_deposited | 79 | 6.965e-04 | 0.0075 | 51 | True | 0.25 |
| 75 | 120 | as_deposited | 55 | 6.550e-04 | 0.00839 | 50 | True | 0.24 |
| 75 | 30 | 300C | 2 | 1.965e-08 | 4.6 |  | False | 0.188 |
| 75 | 60 | 300C | 2 | 5.300e-08 | 4.27 |  | False | 0.172 |
| 50 | 60 | 300C | 4 | 1.717e-07 | 3.9 |  | False | 0.154 |
| 50 | 30 | 300C | 3 | 4.391e-05 | 1.34 |  | False | 0.0522 |
| 50 | 120 | as_deposited | 113 | 2.652e-05 | 0.878 |  | False | 0.0463 |
| 50 | 30 | as_deposited | 4 | 1.743e-04 | 0.666 |  | False | 0.0259 |
| 75 | 120 | 400C | 1 | 2.090e-04 | 0.438 |  | False | 0.0194 |
| 50 | 30 | 400C | 4 | 3.671e-04 | 0.607 |  | False | 0.0184 |
| 75 | 60 | as_deposited | 35 | 5.529e-04 | 0.0162 | 30 | True | 0.00847 |
| 75 | 30 | as_deposited | 3 | 3.963e-04 | 0.175 |  | False | 0.00836 |
| 50 | 120 | 400C | 31 | 5.687e-04 | 0.0138 | 30 | True | 0.00732 |
| 75 | 120 | 300C | 2 | 3.865e-04 | 0.0772 |  | False | 0.00644 |

## 計算公式

```text
Leakage proxy = min(|I_read|) at the selected low read voltage
Sweet spot score = 0.40 * normalized log10(On/Off) + 0.35 * normalized low leakage + 0.25 * normalized endurance
```
