from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURE_COLUMNS, NO_RTA_TEMPERATURE_C


def _majority_label(values: pd.Series) -> str | float:
    """取分類欄位的眾數；若該條件沒有足夠資料，回傳 NaN。"""
    values = values.dropna()
    values = values[values.astype(str).str.len() > 0]
    if values.empty:
        return np.nan
    return values.value_counts().idxmax()


def _add_ml_rta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """將退火狀態轉成 ML 特徵：未退火視為 25 C，並額外加入 Has_RTA。

    數學意義：
    - RTA_Temperature_C 保留溫度大小資訊。
    - Has_RTA 是結構性 indicator，用來告訴模型 25 C 的 No RTA 並非低溫退火製程。
    """
    frame = frame.copy()
    rta_condition = frame["rta_condition"].fillna("as_deposited").astype(str)
    rta_temp = pd.to_numeric(frame["rta_temp_c"], errors="coerce")
    frame["Has_RTA"] = np.where(rta_condition.eq("as_deposited") | rta_temp.isna(), 0, 1)
    frame["RTA_Temperature_C"] = np.where(frame["Has_RTA"].eq(1), rta_temp, NO_RTA_TEMPERATURE_C)
    frame["RF_Power_W"] = pd.to_numeric(frame["rf_power_w"], errors="coerce")
    frame["Process_Time_Min"] = pd.to_numeric(frame["process_time_min"], errors="coerce")
    return frame


def load_curve_metrics(path=None) -> pd.DataFrame:
    """讀取 ETL 階段產生的 curve-level metrics。"""
    path = path or DATA_DIR / "curve_metrics_sic_sputtering_data.csv"
    return pd.read_csv(path)


def load_endurance_summary(path=None) -> pd.DataFrame:
    """讀取 Endurance sweep block 摘要。"""
    path = path or DATA_DIR / "endurance_summary.csv"
    return pd.read_csv(path)


def build_condition_level_dataset(
    curve_metrics_path=None,
    endurance_summary_path=None,
    drop_missing_features: bool = True,
) -> pd.DataFrame:
    """建立製程條件層級資料集，作為 GPR 與分類模型的輸入。

    重要策略：
    - 相同 RF Power / Process Time / RTA 狀態的曲線以 median 聚合。
    - 電性預測使用 condition-level target，避免同製程重複曲線造成 validation leakage。
    - Endurance 以同條件中可讀取到的最大有效 cycle count 作為可靠度上限 proxy。
    """
    metrics = load_curve_metrics(curve_metrics_path)
    endurance = load_endurance_summary(endurance_summary_path)

    usable = metrics[metrics["curve_quality_flag"].isin(["ok", "warning"])].copy()
    usable = _add_ml_rta_features(usable)

    feature_group = FEATURE_COLUMNS
    iv_like = usable[usable["measurement_type"].eq("iv")].copy()
    forming = usable[usable["measurement_type"].eq("forming")].copy()
    endurance = _add_ml_rta_features(endurance)

    iv_targets = (
        iv_like.groupby(feature_group, dropna=False)
        .agg(
            operation_voltage_v=("operation_voltage_est_v", "median"),
            leakage_current_a=("read_low_current_a", "median"),
            on_off_ratio=("on_off_ratio", "median"),
            hrs_mechanism=("hrs_best_model", _majority_label),
            lrs_mechanism=("lrs_best_model", _majority_label),
            iv_curve_count=("curve_id", "nunique"),
        )
        .reset_index()
        if not iv_like.empty
        else pd.DataFrame(columns=feature_group)
    )

    forming_targets = (
        forming.groupby(feature_group, dropna=False)
        .agg(
            forming_voltage_v=("forming_voltage_v", "median"),
            forming_curve_count=("curve_id", "nunique"),
        )
        .reset_index()
        if not forming.empty
        else pd.DataFrame(columns=feature_group)
    )

    endurance_targets = (
        endurance.groupby(feature_group, dropna=False)
        .agg(
            endurance_cycles=("cycle_count_valid_or_warning", "max"),
            endurance_file_count=("source_file", "nunique"),
        )
        .reset_index()
        if not endurance.empty
        else pd.DataFrame(columns=feature_group)
    )

    dataset = iv_targets.merge(forming_targets, on=feature_group, how="outer")
    dataset = dataset.merge(endurance_targets, on=feature_group, how="outer")
    dataset["condition_id"] = dataset.apply(
        lambda row: (
            f"{row['RF_Power_W']:.0f}W_"
            f"{row['Process_Time_Min']:.0f}min_"
            f"{'RTA' + str(int(row['RTA_Temperature_C'])) + 'C' if row['Has_RTA'] else 'NoRTA25C'}"
        )
        if pd.notna(row["RF_Power_W"]) and pd.notna(row["Process_Time_Min"])
        else np.nan,
        axis=1,
    )

    if drop_missing_features:
        dataset = dataset.dropna(subset=FEATURE_COLUMNS).copy()

    return dataset.sort_values(FEATURE_COLUMNS).reset_index(drop=True)
