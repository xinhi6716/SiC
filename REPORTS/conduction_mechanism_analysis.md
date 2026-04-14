# 傳導機制分析

資料來源：`RAW_DATA/` machine-readable CSV/XLSX/TXT；論文背景：`M1003101碩論.pdf`。


## 重點結論

- PDF baseline：未退火樣品多為 HRS Poole-Frenkel、LRS Ohmic；RTA 後會出現 Hopping 或 Schottky Emission。
- 本資料集以斜率與線性擬合的 R-squared 自動選擇 HRS/LRS 的候選傳導模型。
- Ohmic 另加上 log(I)-log(V) slope 接近 1 的規則，避免只靠 R-squared 過度判讀。

## Machine-Readable 模型摘要

| rf_power_w | process_time_min | rta_condition | hrs_best_model | lrs_best_model | curves | hrs_r2_median | lrs_r2_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 |  | as_deposited | Hopping | Poole-Frenkel | 1 | 0.982 | 0.981 |
| 30 |  | as_deposited | Ohmic | Ohmic | 1 | 0.989 | 0.969 |
| 50 |  | as_deposited | Ohmic | Ohmic | 19 | 0.995 | 0.995 |
| 50 |  | as_deposited | Schottky Emission | Ohmic | 1 | 0.911 | 0.988 |
| 50 | 30 | 300C | Schottky Emission | Ohmic | 2 | 0.992 | 0.99 |
| 50 | 30 | 300C | Hopping | Ohmic | 1 | 0.831 | 0.991 |
| 50 | 30 | 400C | Ohmic | Ohmic | 2 | 0.993 | 0.988 |
| 50 | 30 | 400C | Schottky Emission | Ohmic | 2 | 0.98 | 0.995 |
| 50 | 30 | as_deposited | Schottky Emission | Ohmic | 3 | 0.987 | 0.991 |
| 50 | 30 | as_deposited | Ohmic | Ohmic | 1 | 0.98 | 0.991 |
| 50 | 60 | 300C | Hopping | Ohmic | 3 | 0.974 | 1 |
| 50 | 60 | 300C | Ohmic | Ohmic | 1 | 0.997 | 1 |
| 50 | 60 | as_deposited | Ohmic | Ohmic | 75 | 1 | 1 |
| 50 | 60 | as_deposited | Schottky Emission | Ohmic | 3 | 0.987 | 0.997 |
| 50 | 60 | as_deposited | Poole-Frenkel | Hopping | 1 | 0.4 | 0.79 |
| 50 | 120 | 400C | Ohmic | Ohmic | 31 | 1 | 1 |
| 50 | 120 | as_deposited | Schottky Emission | Ohmic | 80 | 0.953 | 0.961 |
| 50 | 120 | as_deposited | Hopping | Ohmic | 29 | 0.935 | 0.952 |
| 50 | 120 | as_deposited | Ohmic | Ohmic | 4 | 0.936 | 0.885 |
| 75 | 30 | 300C | Poole-Frenkel | Ohmic | 2 | 0.829 | 1 |
| 75 | 30 | 400C | Ohmic | Ohmic | 51 | 0.997 | 0.997 |
| 75 | 30 | 400C | Schottky Emission | Ohmic | 1 | 0.988 | 0.999 |
| 75 | 30 | as_deposited | Ohmic | Ohmic | 2 | 1 | 1 |
| 75 | 30 | as_deposited | Schottky Emission | Ohmic | 1 | 0.988 | 1 |
| 75 | 60 | 300C | Hopping | Ohmic | 2 | 0.972 | 0.997 |
| 75 | 60 | as_deposited | Ohmic | Ohmic | 30 | 1 | 1 |
| 75 | 60 | as_deposited | Schottky Emission | Ohmic | 3 | 0.986 | 0.998 |
| 75 | 60 | as_deposited | Hopping | Ohmic | 1 | 0.993 | 0.999 |
| 75 | 60 | as_deposited | Schottky Emission | Schottky Emission | 1 | 0.807 | 0.994 |
| 75 | 120 | 300C | Ohmic | Ohmic | 1 | 0.998 | 0.998 |
| 75 | 120 | 300C | Schottky Emission | Ohmic | 1 | 0.988 | 0.995 |
| 75 | 120 | 400C | Ohmic | Ohmic | 1 | 0.995 | 0.999 |
| 75 | 120 | as_deposited | Ohmic | Ohmic | 54 | 1 | 0.999 |
| 75 | 120 | as_deposited | Schottky Emission | Ohmic | 1 | 0.983 | 0.997 |

## 曲線層級模型

| source_file | rf_power_w | process_time_min | rta_condition | hrs_best_model | hrs_best_r2 | hrs_ohmic_slope | lrs_best_model | lrs_best_r2 | lrs_ohmic_slope | curve_quality_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW_DATA/20221105/SiC 30W  I-V Curve -1.xlsx | 30 |  | as_deposited | Hopping | 0.982 | 0.852 | Poole-Frenkel | 0.981 | 0.213 | ok |
| RAW_DATA/20221105/SiC 30W  I-V Curve.xlsx | 30 |  | as_deposited | Ohmic | 0.989 | 1.3 | Ohmic | 0.969 | 0.907 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.996 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.995 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.995 | 1.15 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.17 | Ohmic | 0.994 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.996 | 1.16 | Ohmic | 0.994 | 1.17 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.994 | 1.2 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.17 | Ohmic | 0.995 | 1.14 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.13 | Ohmic | 0.995 | 1.14 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.991 | 1.18 | Ohmic | 0.993 | 1.17 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.998 | 1.29 | Ohmic | 0.993 | 1.19 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.983 | 1.58 | Ohmic | 0.932 | 1.22 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Schottky Emission | 0.911 | 2.03 | Ohmic | 0.988 | 1.27 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.996 | 1.17 | Ohmic | 0.995 | 1.15 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.17 | Ohmic | 0.995 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.995 | 1.15 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.994 | 1.17 | Ohmic | 0.995 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.17 | Ohmic | 0.995 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.995 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.16 | Ohmic | 0.995 | 1.16 | ok |
| RAW_DATA/20221105/sic 50W endurance 1.3.7 total 20次.csv | 50 |  | as_deposited | Ohmic | 0.995 | 1.17 | Ohmic | 0.995 | 1.15 | ok |
| RAW_DATA/20230818-K/RTA300-50w-30min-IV-2.5V-1.xlsx | 50 | 30 | 300C | Schottky Emission | 0.993 | 1.41 | Ohmic | 0.981 | 0.912 | ok |
| RAW_DATA/20230818-K/RTA300-50w-30min-IV-2.5V.xlsx | 50 | 30 | 300C | Hopping | 0.831 | 0.267 | Ohmic | 0.991 | 0.943 | ok |
| RAW_DATA/20230818-K/RTA300-50w-30min-IV-2V.xlsx | 50 | 30 | 300C | Schottky Emission | 0.991 | 1.32 | Ohmic | 1 | 1.01 | ok |
| RAW_DATA/20230908-R400/50W-30m-IV-1.csv | 50 | 30 | 400C | Ohmic | 0.99 | 0.947 | Ohmic | 0.99 | 0.931 | ok |
| RAW_DATA/20230908-R400/50W-30m-IV-105-1.csv | 50 | 30 | 400C | Schottky Emission | 0.964 | 1.92 | Ohmic | 0.996 | 0.956 | ok |
| RAW_DATA/20230908-R400/50W-30m-IV-105.csv | 50 | 30 | 400C | Ohmic | 0.995 | 0.998 | Ohmic | 0.986 | 0.917 | ok |
| RAW_DATA/20230908-R400/50W-30m-IV-205.csv | 50 | 30 | 400C | Schottky Emission | 0.997 | 1.59 | Ohmic | 0.995 | 0.956 | ok |
| RAW_DATA/20230224 -金覺/IV-50W-30min-1.xlsx | 50 | 30 | as_deposited | Schottky Emission | 0.987 | 1.43 | Ohmic | 0.989 | 0.935 | ok |
| RAW_DATA/20230224 -金覺/IV-50W-30min-2.xlsx | 50 | 30 | as_deposited | Schottky Emission | 0.992 | 1.39 | Ohmic | 1 | 0.996 | ok |
| RAW_DATA/20230224 -金覺/IV-50W-30min-3.xlsx | 50 | 30 | as_deposited | Schottky Emission | 0.967 | 1.38 | Ohmic | 0.991 | 0.937 | ok |
| RAW_DATA/20230224 -金覺/IV-50W-30min.xlsx | 50 | 30 | as_deposited | Ohmic | 0.98 | 1.65 | Ohmic | 0.991 | 1.05 | ok |
| RAW_DATA/20230818-K/RTA300-50w-1H-IV-2.5V.xlsx | 50 | 60 | 300C | Ohmic | 0.997 | 1.13 | Ohmic | 1 | 1.03 | ok |
| RAW_DATA/20230818-K/RTA300-50w-1H-IV-2V.xlsx | 50 | 60 | 300C | Hopping | 0.974 | 0.813 | Ohmic | 1 | 0.982 | ok |
| RAW_DATA/20230818-K/RTA300-50w-1H-IV-3V.xlsx | 50 | 60 | 300C | Hopping | 0.993 | 0.415 | Ohmic | 1 | 0.985 | ok |
| RAW_DATA/Origin DATA/RTA300-50w-1H-IV-2V.xlsx | 50 | 60 | 300C | Hopping | 0.974 | 0.813 | Ohmic | 1 | 0.982 | ok |
| RAW_DATA/20230908-R400/50W-1H-IV105-NG.csv | 50 | 60 | 400C | Ohmic | 1 | 1.02 | Ohmic | 0.999 | 0.995 | invalid |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 1 | 0.98 | Ohmic | 1 | 0.98 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 1 | 0.98 | Ohmic | 1 | 0.977 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 1 | 0.973 | Ohmic | 0.998 | 0.972 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.992 | 1.02 | Ohmic | 1 | 0.984 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.994 | 1.06 | Ohmic | 0.999 | 0.978 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.999 | 1.01 | Ohmic | 1 | 0.983 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.998 | 1.02 | Ohmic | 0.999 | 0.974 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.996 | 1.03 | Ohmic | 0.998 | 0.973 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.996 | 0.972 | Ohmic | 0.996 | 0.964 | warning |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.995 | 0.96 | Ohmic | 0.994 | 0.952 | warning |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.996 | 1.01 | Ohmic | 0.999 | 0.978 | ok |
| RAW_DATA/20221119/I_V Sweep Sic 50W 1h(20) Endurance.csv | 50 | 60 | as_deposited | Ohmic | 0.998 | 1 | Ohmic | 0.998 | 0.968 | ok |

_顯示前 50 筆，共 415 筆。_

## PDF Baseline 機制

| condition | hrs | lrs |
| --- | --- | --- |
| As-deposited, 50 W and 75 W | Poole-Frenkel | Ohmic |
| 50 W / 30 min / 400 C | Poole-Frenkel | Hopping |
| 50 W / 1 H / 400 C | Poole-Frenkel | Hopping |
| 50 W / 30 min or 1 H / 500 C | Ohmic | Ohmic |
| 75 W / 30 min / 400 C | Poole-Frenkel | Hopping |
| 75 W / 1 H / 400 C | Schottky Emission | Schottky Emission |
| 75 W / 1 H / 500 C | Hopping | Schottky Emission |
| 75 W / 2 H / 400 C | Ohmic | Hopping |
| 75 W / 2 H / 500 C | Ohmic | Ohmic |

## 擬合座標

```text
Ohmic:              log10(I) vs log10(V), slope close to 1
Poole-Frenkel:      ln(I / V) vs sqrt(V)
Schottky Emission:  ln(I) vs sqrt(V)
Hopping:            ln(I) vs V
```
