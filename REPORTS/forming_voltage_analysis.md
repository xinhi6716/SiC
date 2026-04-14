# Forming 電壓分析

資料來源：`RAW_DATA/` machine-readable CSV/XLSX/TXT；論文背景：`M1003101碩論.pdf`。


## 重點結論

- PDF baseline 指出 Forming 電壓約 2.5 V，並以 10 mA compliance 避免元件崩潰。
- 本報告的 Forming 電壓由正偏壓 sweep 中的電流躍升點或半 compliance 門檻估計。
- 若曲線在低電壓已呈現高電流，會標記為 `low_voltage_short_or_preformed`。

## 統計摘要

| rf_power_w | process_time_min | rta_condition | curves | forming_voltage_median_v | forming_voltage_min_v | forming_voltage_max_v | low_voltage_median_current_a |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 30 | 300C | 4 | 0.55 | 0.1 | 2.3 | 3.852e-04 |
| 50 | 60 | 300C | 3 | 2.2 | 0.1 | 2.7 | 9.165e-09 |
| 75 | 30 | 300C | 2 | 1.4 | 0.1 | 2.7 | 3.153e-04 |
| 75 | 60 | 300C | 2 | 2.6 | 2.5 | 2.7 | 1.191e-08 |
| 75 | 120 | 300C | 3 | 0.1 | 0.1 | 3.6 | 5.312e-04 |
| 50 | 30 | 400C | 3 | 0.1 | 0.1 | 2.3 | 9.603e-04 |
| 50 | 60 | 400C | 2 | 0.1 | 0.1 | 0.1 | 7.534e-04 |
| 50 | 120 | 400C | 1 | 2.2 | 2.2 | 2.2 | 1.734e-08 |
| 75 | 30 | 400C | 2 | 1.5 | 0.1 | 2.9 | 4.218e-04 |
| 75 | 60 | 400C | 1 | 1.8 | 1.8 | 1.8 | 2.274e-04 |
| 75 | 120 | 400C | 1 | 0.1 | 0.1 | 0.1 | 2.857e-04 |
| 30 |  | as_deposited | 4 | 0.65 | 0.1 | 1.5 | 5.573e-07 |
| 50 |  | as_deposited | 2 | 0.1 | 0.1 | 0.1 | 1.149e-05 |
| 50 | 30 | as_deposited | 4 | 0.1 | 0.1 | 2.6 | 7.850e-05 |
| 50 | 60 | as_deposited | 6 | 0.1 | 0.1 | 5 | 3.989e-05 |
| 50 | 120 | as_deposited | 1 | 0.1 | 0.1 | 0.1 | 1.577e-06 |
| 75 | 30 | as_deposited | 3 | 0.9 | 0.1 | 1.3 | 1.023e-06 |
| 75 | 60 | as_deposited | 3 | 2.7 | 0.1 | 2.8 | 3.382e-08 |
| 75 | 120 | as_deposited | 4 | 0.1 | 0.1 | 0.1 | 2.088e-04 |

## 曲線層級估計

| source_file | rf_power_w | process_time_min | rta_condition | forming_voltage_v | forming_detection_method | curve_quality_flag | curve_invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAW_DATA/20230414 -金覺/75W-30min-Forming.xlsx | 75 | 30 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230818-K/RTA300-50w-30min-forming-2V.xlsx | 50 | 30 | 300C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230818-K/RTA300-50w-1H-forming-2V.xlsx | 50 | 60 | 300C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230421-金覺/75W-1H-Forming.xlsx | 75 | 60 | as_deposited | 0.1 | log_current_jump | invalid | extreme_current;low_voltage_short_or_preformed |
| RAW_DATA/20230421-金覺/75W-1H-Forming-1.xlsx | 75 | 60 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230224 -金覺/forming-50W-30min.xlsx | 50 | 30 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230224 -金覺/forming-50W-30min-2.xlsx | 50 | 30 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230224 -金覺/forming-50W-30min-1.xlsx | 50 | 30 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230818-K/RTA300-50w-30min-forming-1V.xlsx | 50 | 30 | 300C | 0.1 | log_current_jump | warning | low_voltage_short_or_preformed |
| RAW_DATA/20221119/SiC 75W-2h-Ar10 Forming-2.xlsx | 75 | 120 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221119/SiC 75W-2h-Ar10 Forming-1.xlsx | 75 | 120 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221119/SiC 50W-1h-Ar10 Forming.xlsx | 50 | 60 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221119/SiC 75W-2h-Ar10 Forming.xlsx | 75 | 120 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221119/SiC 50W-1h-Ar10 Forming-4.xlsx | 50 | 60 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221119/SiC 50W-1h-Ar10 Forming-2.xlsx | 50 | 60 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221119/SiC 50W-1h-Ar10 Forming-1.xlsx | 50 | 60 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221112/SiC 75W-2h forming.xlsx | 75 | 120 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221112/SiC 50W-2h forming.xlsx | 50 | 120 | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221105/SiC 30W  forming-3.xlsx | 30 |  | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221105/SiC 50W  forming.xlsx | 50 |  | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20221105/SiC 30W  forming.xlsx | 30 |  | as_deposited | 0.1 | log_current_jump | warning | low_voltage_short_or_preformed |
| RAW_DATA/20221105/SiC 50W  forming-1.xlsx | 50 |  | as_deposited | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230908-R400/50W-30m-Forming1.csv | 50 | 30 | 400C | 0.1 | log_current_jump | warning | low_voltage_short_or_preformed |
| RAW_DATA/20230901-K/RTA300-75W-2H-Forming15.csv | 75 | 120 | 300C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230901-K/RTA300-75W-2H-Forming2.csv | 75 | 120 | 300C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230901-K/RTA300-75W-30min-Forming2V.csv | 75 | 30 | 300C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230908-R400/50W-30m-Forming105.csv | 50 | 30 | 400C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230908-R400/75W-2H-Forming3.csv | 75 | 120 | 400C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230908-R400/50W-1H-Forming105.csv | 50 | 60 | 400C | 0.1 | log_current_jump | ok |  |
| RAW_DATA/20230908-R400/50W-1H-Forming2.csv | 50 | 60 | 400C | 0.1 | log_current_jump | ok |  |

_顯示前 30 筆，共 52 筆。_

## 方法

```python
if max_delta_log10_current >= 1.0:
    forming_voltage = voltage_at_largest_log_current_jump
else:
    forming_voltage = first_voltage_where_current_exceeds_half_compliance_proxy
```
