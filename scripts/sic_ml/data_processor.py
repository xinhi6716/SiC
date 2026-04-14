from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Part 1: SiC RRAM 資料前處理模組。

    這個模組直接讀取 ETL 後的 row-level I-V 資料：
    DATA/cleaned_sic_sputtering_data.csv

    核心物理與數學設計：
    1. No RTA 不是 0 度退火，而是「沒有經過熱退火相變/缺陷修復」的狀態。
       因此將溫度設成常溫 25 C，並新增 Has_RTA indicator 讓模型區分製程路徑。
    2. Leakage Current 跨越多個數量級，使用 log10 轉換，避免大電流樣本支配回歸模型。
    3. GPR/SVR 等 kernel-based model 對尺度敏感，因此提供 StandardScaler 特徵縮放。
    """

    FEATURE_COLUMNS = ["RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA"]
    REQUIRED_COLUMNS = {
        "curve_id",
        "measurement_type",
        "rf_power_w",
        "process_time_min",
        "rta_temp_c",
        "rta_condition",
        "voltage_v",
        "current_a",
        "abs_current_a",
        "is_valid_point",
    }

    def __init__(
        self,
        data_path: str | Path = "DATA/cleaned_sic_sputtering_data.csv",
        no_rta_temperature_c: float = 25.0,
        leakage_epsilon_a: float = 1e-15,
    ) -> None:
        self.data_path = Path(data_path)
        self.no_rta_temperature_c = no_rta_temperature_c
        self.leakage_epsilon_a = leakage_epsilon_a
        self.scaler = StandardScaler()
        self.raw_data_: pd.DataFrame | None = None
        self.condition_dataset_: pd.DataFrame | None = None

    def load_cleaned_data(self) -> pd.DataFrame:
        """讀取 cleaned I-V 資料，並檢查必要欄位是否存在。"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"找不到資料檔：{self.data_path.resolve()}")

        frame = pd.read_csv(self.data_path, low_memory=False)
        missing = sorted(self.REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"cleaned data 缺少必要欄位：{missing}")

        if frame.empty:
            raise ValueError("cleaned data 是空表，無法建立 ML dataset。")

        self.raw_data_ = frame
        return frame

    def add_rta_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """新增 Has_RTA 與 RTA_Temperature_C。

        物理意義：
        - RTA_Temperature_C = 25：代表未退火樣品處於常溫製程路徑，而不是 0 C。
        - Has_RTA = 0/1：讓模型知道「未退火」與「低溫退火」是不同製程機制。
        """
        output = frame.copy()
        rta_condition = output["rta_condition"].fillna("as_deposited").astype(str)
        rta_temp = pd.to_numeric(output["rta_temp_c"], errors="coerce")

        output["Has_RTA"] = np.where(rta_condition.eq("as_deposited") | rta_temp.isna(), 0, 1)
        output["RTA_Temperature_C"] = np.where(output["Has_RTA"].eq(1), rta_temp, self.no_rta_temperature_c)
        output["RF_Power_W"] = pd.to_numeric(output["rf_power_w"], errors="coerce")
        output["Process_Time_Min"] = pd.to_numeric(output["process_time_min"], errors="coerce")
        return output

    def _estimate_curve_metrics(self, group: pd.DataFrame) -> dict[str, float | str]:
        """從單條 I-V curve 估計 ML target。

        說明：
        - Leakage proxy：低讀取電壓附近的較小電流，近似 HRS leakage。
        - On/Off Ratio：同一讀取電壓附近 max(|I|) / min(|I|)。
        - Operation Voltage：首次達到顯著電流門檻的最低 |V|，作為 set/reset 操作電壓 proxy。
        - Forming Voltage：forming curve 中 log current 躍升最大的電壓。
        """
        valid = group[group["is_valid_point"].astype(bool)].copy()
        valid = valid[np.isfinite(valid["voltage_v"]) & np.isfinite(valid["current_a"])]
        if len(valid) < 5:
            return {}

        abs_current = valid["abs_current_a"].clip(lower=self.leakage_epsilon_a)
        low_voltage = valid[(valid["voltage_v"].abs() > 0) & (valid["voltage_v"].abs() <= 0.5)].copy()
        if low_voltage.empty:
            read_low_current = np.nan
            read_high_current = np.nan
            on_off_ratio = np.nan
        else:
            # 選擇最接近 0.1 V 的低電壓點，對應 RRAM 常見 read voltage 的低擾動量測。
            low_voltage["distance_to_read"] = (low_voltage["voltage_v"].abs() - 0.1).abs()
            read_slice = low_voltage[low_voltage["distance_to_read"].eq(low_voltage["distance_to_read"].min())]
            read_currents = read_slice["abs_current_a"].clip(lower=self.leakage_epsilon_a)
            read_low_current = float(read_currents.min())
            read_high_current = float(read_currents.max())
            on_off_ratio = float(read_high_current / read_low_current) if read_low_current > 0 else np.nan

        max_current = float(abs_current.max())
        current_threshold = max(min(1e-3, 0.1 * max_current), 1e-6)
        switching_points = valid[(valid["voltage_v"].abs() > 0) & (valid["abs_current_a"] >= current_threshold)]
        operation_voltage = float(switching_points["voltage_v"].abs().min()) if not switching_points.empty else np.nan

        forming_voltage = np.nan
        if str(valid["measurement_type"].iloc[0]) == "forming":
            positive = valid[(valid["voltage_v"] >= 0) & (valid["abs_current_a"] > self.leakage_epsilon_a)]
            positive = positive.sort_values(["voltage_v", "point_index"])
            if len(positive) >= 5:
                log_i = np.log10(positive["abs_current_a"].clip(lower=self.leakage_epsilon_a).to_numpy())
                delta_log_i = np.diff(log_i)
                if len(delta_log_i) and np.nanmax(delta_log_i) >= 1.0:
                    forming_voltage = float(positive.iloc[int(np.nanargmax(delta_log_i)) + 1]["voltage_v"])

        return {
            "Measurement_Type": str(valid["measurement_type"].iloc[0]),
            "Leakage_Current_A": read_low_current,
            "On_Off_Ratio": on_off_ratio,
            "Operation_Voltage_V": operation_voltage,
            "Forming_Voltage_V": forming_voltage,
            "Max_Abs_Current_A": max_current,
            "N_Points": int(len(valid)),
        }

    def build_condition_level_dataset(self, drop_missing_features: bool = True) -> pd.DataFrame:
        """由 row-level I-V 資料建立製程條件層級 ML dataset。"""
        if self.raw_data_ is None:
            frame = self.load_cleaned_data()
        else:
            frame = self.raw_data_.copy()

        frame = self.add_rta_features(frame)
        if drop_missing_features:
            frame = frame.dropna(subset=self.FEATURE_COLUMNS).copy()
        if frame.empty:
            raise ValueError("加入 RTA/製程特徵後沒有可訓練資料，請檢查 rf_power/process_time 欄位。")

        curve_rows: list[dict[str, float | str]] = []
        for curve_id, group in frame.groupby("curve_id", sort=False):
            base = group.iloc[0][self.FEATURE_COLUMNS].to_dict()
            source_file = group.iloc[0].get("source_file", "")
            metrics = self._estimate_curve_metrics(group)
            if metrics:
                curve_rows.append({"Curve_ID": curve_id, "Source_File": source_file, **base, **metrics})

        curve_metrics = pd.DataFrame(curve_rows)
        if curve_metrics.empty:
            raise ValueError("無法從 cleaned I-V 資料萃取 curve-level metrics。")

        feature_cols = self.FEATURE_COLUMNS
        iv_like = curve_metrics[curve_metrics["Measurement_Type"].eq("iv")]
        forming = curve_metrics[curve_metrics["Measurement_Type"].eq("forming")]
        endurance = curve_metrics[curve_metrics["Measurement_Type"].eq("endurance")]

        iv_targets = (
            iv_like.groupby(feature_cols, dropna=False)
            .agg(
                Leakage_Current_A=("Leakage_Current_A", "median"),
                On_Off_Ratio=("On_Off_Ratio", "median"),
                Operation_Voltage_V=("Operation_Voltage_V", "median"),
                IV_Curve_Count=("Curve_ID", "nunique"),
            )
            .reset_index()
        )

        forming_targets = (
            forming.groupby(feature_cols, dropna=False)
            .agg(Forming_Voltage_V=("Forming_Voltage_V", "median"), Forming_Curve_Count=("Curve_ID", "nunique"))
            .reset_index()
        )

        if endurance.empty:
            endurance_targets = pd.DataFrame(columns=feature_cols)
        else:
            # Endurance 的物理意義是單一元件/檔案可穩定切換的 cycle count。
            # 若同一製程條件有多個檔案，不能直接相加，否則會高估可靠度；
            # 這裡先計算每個檔案的 cycle count，再取同條件下的最大可觀測 cycle count。
            endurance_by_file = (
                endurance.groupby(feature_cols + ["Source_File"], dropna=False)
                .agg(File_Endurance_Cycles=("Curve_ID", "nunique"))
                .reset_index()
            )
            endurance_targets = (
                endurance_by_file.groupby(feature_cols, dropna=False)
                .agg(
                    Endurance_Cycles=("File_Endurance_Cycles", "max"),
                    Endurance_File_Count=("Source_File", "nunique"),
                )
                .reset_index()
            )

        dataset = iv_targets.merge(forming_targets, on=feature_cols, how="outer")
        dataset = dataset.merge(endurance_targets, on=feature_cols, how="outer")
        dataset["Leakage_Current_Log10"] = np.log10(dataset["Leakage_Current_A"].clip(lower=self.leakage_epsilon_a))
        dataset["On_Off_Log10"] = np.log10(dataset["On_Off_Ratio"].clip(lower=1.0))

        self.condition_dataset_ = dataset.sort_values(feature_cols).reset_index(drop=True)
        return self.condition_dataset_

    def fit_feature_scaler(self, dataset: pd.DataFrame | None = None) -> StandardScaler:
        """對製程特徵 fit StandardScaler。"""
        dataset = dataset if dataset is not None else self.condition_dataset_
        if dataset is None:
            raise ValueError("尚未建立 condition-level dataset，請先呼叫 build_condition_level_dataset()。")
        clean = dataset.dropna(subset=self.FEATURE_COLUMNS)
        if clean.empty:
            raise ValueError("沒有完整特徵列可用來 fit StandardScaler。")
        self.scaler.fit(clean[self.FEATURE_COLUMNS])
        return self.scaler

    def transform_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """將 RF Power、Process Time、RTA Temperature、Has_RTA 轉成標準化特徵。"""
        scaled = self.scaler.transform(dataset[self.FEATURE_COLUMNS])
        return pd.DataFrame(scaled, columns=[f"{col}_Scaled" for col in self.FEATURE_COLUMNS], index=dataset.index)

    def fit_transform_features(self, dataset: pd.DataFrame | None = None) -> pd.DataFrame:
        """fit StandardScaler 後回傳標準化特徵矩陣。"""
        dataset = dataset if dataset is not None else self.condition_dataset_
        if dataset is None:
            raise ValueError("尚未建立 condition-level dataset。")
        self.fit_feature_scaler(dataset)
        return self.transform_features(dataset.dropna(subset=self.FEATURE_COLUMNS))


if __name__ == "__main__":
    processor = DataProcessor()
    condition_dataset = processor.build_condition_level_dataset()
    scaled_features = processor.fit_transform_features(condition_dataset)
    print("Condition-level dataset shape:", condition_dataset.shape)
    print(condition_dataset.head().to_string(index=False))
    print("Scaled feature preview:")
    print(scaled_features.head().to_string(index=False))
