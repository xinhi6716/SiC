from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import paths
from src.config import MaterialConfig, NIO_CONFIG, SIC_CONFIG, get_material_config
from src.origin_parser import OriginExtractor


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawFileReport:
    """ETL status for one raw source file.

    Args:
        source_file: File path handled by the parser.
        status: One of ``parsed``, ``skipped``, ``error`` or ``origin_*``.
        rows: Number of exported or parsed rows.
        message: Human-readable detail for reports and logs.
    """

    source_file: str
    status: str
    rows: int = 0
    message: str = ""


class DataProcessor:
    """Material-aware ETL processor for RRAM I-V curves.

    Material-specific schema, read-voltage strategy, constraints, and default
    process-feature imputations are loaded from :mod:`src.config`. The core
    physics logic remains unchanged: parse I-V tables, flag nonphysical points,
    extract leakage/current-window metrics, then aggregate by process condition.

    Args:
        material_name: Registered material label such as ``SiC`` or ``NiO``.
        data_path: Optional cleaned row-level CSV path.
        raw_dir: Optional raw-data folder override.
        processed_dir: Optional processed-data folder override.
        no_rta_temperature_c: Optional no-RTA temperature override.
        leakage_epsilon_a: Optional lower bound for log-current calculations.
        max_reasonable_current_a: Optional current ceiling used to flag shorts.
        default_rf_power_w: Optional RF power imputation override.
        default_process_time_min: Optional process time imputation override.
        read_voltage_v: Optional fixed read voltage.
        dynamic_on_off_ratio: Optional dynamic On/Off extraction override.
        enable_origin_extraction: Whether to extract ``.opju`` files first.
        auto_install_originpro: Whether Origin extraction may install
            ``originpro`` automatically.
    """

    FEATURE_COLUMNS: list[str] = list(SIC_CONFIG.feature_columns)
    NIO_FEATURE_COLUMNS: list[str] = list(NIO_CONFIG.feature_columns)
    REQUIRED_COLUMNS: set[str] = {
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
        material_name: str = "SiC",
        data_path: str | Path | None = None,
        raw_dir: str | Path | None = None,
        processed_dir: str | Path | None = None,
        no_rta_temperature_c: float | None = None,
        leakage_epsilon_a: float | None = None,
        max_reasonable_current_a: float | None = None,
        default_rf_power_w: float | None = None,
        default_process_time_min: float | None = None,
        read_voltage_v: float | None = None,
        dynamic_on_off_ratio: bool | None = None,
        enable_origin_extraction: bool = True,
        auto_install_originpro: bool = True,
    ) -> None:
        self.material_name = paths.normalize_material_name(material_name)
        self.config: MaterialConfig = get_material_config(self.material_name)

        paths.ensure_material_directories(self.material_name)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else paths.material_raw_dir(self.material_name)
        self.processed_dir = Path(processed_dir) if processed_dir is not None else paths.material_processed_dir(self.material_name)
        self.cleaned_data_path = Path(data_path) if data_path is not None else paths.material_cleaned_data_path(self.material_name)
        self.condition_dataset_path = paths.material_condition_dataset_path(self.material_name)

        self.no_rta_temperature_c = (
            self.config.no_rta_temperature_c if no_rta_temperature_c is None else float(no_rta_temperature_c)
        )
        self.leakage_epsilon_a = self.config.leakage_epsilon_a if leakage_epsilon_a is None else float(leakage_epsilon_a)
        self.max_reasonable_current_a = (
            self.config.max_reasonable_current_a
            if max_reasonable_current_a is None
            else float(max_reasonable_current_a)
        )
        self.read_voltage_v = self.config.read_voltage_v if read_voltage_v is None else float(read_voltage_v)
        self.dynamic_on_off_ratio = (
            self.config.dynamic_on_off_ratio if dynamic_on_off_ratio is None else bool(dynamic_on_off_ratio)
        )

        default_values = dict(self.config.default_feature_values)
        if default_rf_power_w is not None:
            default_values["RF_Power_W"] = float(default_rf_power_w)
        if default_process_time_min is not None:
            default_values["Process_Time_Min"] = float(default_process_time_min)
        self.default_feature_values: dict[str, float] = default_values

        self.raw_extensions = set(self.config.raw_extensions)
        self.enable_origin_extraction = enable_origin_extraction
        self.auto_install_originpro = auto_install_originpro
        self.scaler = StandardScaler()
        self.raw_data_: pd.DataFrame | None = None
        self.condition_dataset_: pd.DataFrame | None = None
        self.file_reports_: list[RawFileReport] = []
        logger.info("Initialized DataProcessor for %s", self.material_name)

    @property
    def feature_columns(self) -> list[str]:
        """Return the configured ML feature schema."""

        return list(self.config.feature_columns)

    def load_cleaned_data(self) -> pd.DataFrame:
        """Load standardized cleaned row-level I-V data.

        Returns:
            Cleaned row-level dataframe.

        Raises:
            FileNotFoundError: If the configured cleaned CSV does not exist.
            ValueError: If required columns are missing or the file is empty.
        """

        if not self.cleaned_data_path.exists():
            raise FileNotFoundError(f"Cannot find cleaned data: {self.cleaned_data_path.resolve()}")
        frame = pd.read_csv(self.cleaned_data_path, low_memory=False)
        missing = sorted(self.REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"cleaned data is missing required columns: {missing}")
        if frame.empty:
            raise ValueError("cleaned data is empty; cannot build an ML dataset.")
        self.raw_data_ = frame
        logger.info("Loaded cleaned %s data: shape=%s", self.material_name, frame.shape)
        return frame

    def clean_raw_data(self, save: bool = True) -> pd.DataFrame:
        """Parse raw files and write standardized row-level I-V data."""

        if self.material_name.lower() == "sic" and self.cleaned_data_path.exists():
            logger.info("Using existing SiC cleaned data at %s", self.cleaned_data_path)
            return self.load_cleaned_data()

        self.file_reports_ = []
        self._extract_origin_projects_if_available()
        raw_files = sorted(path for path in self.raw_dir.rglob("*") if path.is_file())
        if not raw_files:
            raise FileNotFoundError(f"No raw data files found under {self.raw_dir}.")

        frames: list[pd.DataFrame] = []
        for file_path in raw_files:
            suffix = file_path.suffix.lower()
            if suffix not in self.raw_extensions:
                self.file_reports_.append(RawFileReport(str(file_path), "skipped", 0, f"Unsupported extension: {suffix}"))
                continue
            try:
                parsed = self._parse_excel_file(file_path) if suffix in {".xlsx", ".xls"} else self._parse_text_table(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse %s: %s", file_path, exc)
                self.file_reports_.append(RawFileReport(str(file_path), "error", 0, str(exc)))
                continue
            if parsed.empty:
                self.file_reports_.append(RawFileReport(str(file_path), "skipped", 0, "No voltage/current rows found"))
                continue
            frames.append(parsed)
            self.file_reports_.append(RawFileReport(str(file_path), "parsed", len(parsed), ""))

        if not frames:
            raise ValueError(f"No parseable I-V raw data found under {self.raw_dir}.")

        cleaned = self._postprocess_standardized_rows(pd.concat(frames, ignore_index=True))
        self.raw_data_ = cleaned
        if save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(self.cleaned_data_path, index=False, encoding="utf-8-sig")
            logger.info("Saved cleaned data to %s", self.cleaned_data_path)
            logger.info("Saved ETL report to %s", self.write_file_report())
        return cleaned

    def write_file_report(self) -> Path:
        """Write a file-level ETL status report."""

        report_path = self.processed_dir / f"{self.material_name.lower()}_etl_file_report.csv"
        pd.DataFrame([report.__dict__ for report in self.file_reports_]).to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )
        return report_path

    def add_rta_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create RTA-aware ML features.

        No-RTA samples are encoded as room temperature plus an explicit
        ``Has_RTA`` bit. This helps the model separate thermal history from
        numeric temperature alone.
        """

        output = frame.copy()
        rta_condition = self._string_series(output, "rta_condition", "as_deposited").fillna("as_deposited")
        rta_temp = self._numeric_series(output, "rta_temp_c")
        has_rta = ~(rta_condition.astype(str).str.lower().eq("as_deposited") | rta_temp.isna())

        output["Has_RTA"] = np.where(has_rta, 1, 0)
        output["RTA_Temperature_C"] = np.where(has_rta, rta_temp, self.no_rta_temperature_c)
        output["RF_Power_W"] = self._numeric_series(output, "rf_power_w").combine_first(
            self._numeric_series(output, "RF_Power_W")
        )
        output["Process_Time_Min"] = self._numeric_series(output, "process_time_min").combine_first(
            self._numeric_series(output, "Process_Time_Min")
        )
        return self._impute_constant_features(output)

    def build_condition_level_dataset(self, drop_missing_features: bool = True, save: bool = False) -> pd.DataFrame:
        """Aggregate row-level I-V curves into condition-level ML records.

        Args:
            drop_missing_features: Whether to drop rows with missing configured features.
            save: Whether to write the condition-level dataset to disk.

        Returns:
            Condition-level dataframe with process features and electrical targets.
        """

        if self.raw_data_ is None:
            frame = self.load_cleaned_data() if self.cleaned_data_path.exists() else self.clean_raw_data(save=True)
        else:
            frame = self.raw_data_.copy()

        frame = self.add_rta_features(frame)
        feature_cols = self.feature_columns
        if drop_missing_features:
            frame = frame.dropna(subset=feature_cols).copy()
        if frame.empty:
            raise ValueError(f"{self.material_name} has no feature-complete rows for condition-level aggregation.")

        curve_rows: list[dict[str, Any]] = []
        for curve_id, group in frame.groupby("curve_id", sort=False):
            metrics = self._estimate_curve_metrics(group)
            if metrics:
                curve_rows.append(
                    {
                        "Curve_ID": curve_id,
                        "Source_File": group.iloc[0].get("source_file", ""),
                        **group.iloc[0][feature_cols].to_dict(),
                        **metrics,
                    }
                )

        curve_metrics = pd.DataFrame(curve_rows)
        if curve_metrics.empty:
            raise ValueError(f"{self.material_name} has no valid curve-level metrics.")

        dataset = self._aggregate_curve_metrics(curve_metrics, feature_cols)
        if "Leakage_Current_A" in dataset:
            dataset["Leakage_Current_Log10"] = np.log10(dataset["Leakage_Current_A"].clip(lower=self.leakage_epsilon_a))
        if "On_Off_Ratio" in dataset:
            dataset["On_Off_Log10"] = np.log10(dataset["On_Off_Ratio"].clip(lower=1.0))

        self.condition_dataset_ = self._safe_sort_values(dataset, feature_cols).reset_index(drop=True)
        if save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self.condition_dataset_.to_csv(self.condition_dataset_path, index=False, encoding="utf-8-sig")
            logger.info("Saved condition-level data to %s", self.condition_dataset_path)
        return self.condition_dataset_

    def fit_feature_scaler(self, dataset: pd.DataFrame | None = None) -> StandardScaler:
        """Fit ``StandardScaler`` on configured process features."""

        dataset = dataset if dataset is not None else self.condition_dataset_
        if dataset is None:
            raise ValueError("Build a condition-level dataset before fitting the feature scaler.")
        clean = dataset.dropna(subset=self.feature_columns)
        if clean.empty:
            raise ValueError("No feature-complete rows are available for StandardScaler fitting.")
        self.scaler.fit(clean[self.feature_columns])
        return self.scaler

    def transform_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Scale configured features with the fitted scaler."""

        scaled = self.scaler.transform(dataset[self.feature_columns])
        return pd.DataFrame(scaled, columns=[f"{col}_Scaled" for col in self.feature_columns], index=dataset.index)

    def fit_transform_features(self, dataset: pd.DataFrame | None = None) -> pd.DataFrame:
        """Fit the feature scaler and return scaled features."""

        dataset = dataset if dataset is not None else self.condition_dataset_
        if dataset is None:
            raise ValueError("Build a condition-level dataset before scaling features.")
        self.fit_feature_scaler(dataset)
        return self.transform_features(dataset.dropna(subset=self.feature_columns))

    def _extract_origin_projects_if_available(self) -> None:
        """Extract Origin ``.opju`` projects before CSV/Excel scanning."""

        if not self.enable_origin_extraction:
            return
        opju_files = list(self.raw_dir.rglob("*.opju"))
        if not opju_files:
            return

        logger.info("Found %d Origin project(s) under %s", len(opju_files), self.raw_dir)
        extractor = OriginExtractor(
            self.raw_dir,
            auto_install=self.auto_install_originpro,
            visible=False,
            overwrite=False,
        )
        for result in extractor.extract_all():
            rows = len(result.exported_files)
            message = result.message
            if result.exported_files:
                message = "; ".join(str(path.relative_to(paths.PROJECT_ROOT)) for path in result.exported_files)
            self.file_reports_.append(RawFileReport(str(result.opju_path), f"origin_{result.status}", rows, message))
            if result.status == "error":
                logger.warning("Origin extraction failed for %s: %s", result.opju_path, result.message)

    def _parse_text_table(self, file_path: Path) -> pd.DataFrame:
        """Parse one CSV/TXT table and map it to the canonical schema."""

        try:
            frame = pd.read_csv(file_path)
        except Exception:
            frame = pd.read_csv(file_path, sep=None, engine="python")
        return self._standardize_curve_frame(frame, file_path, sheet_name=file_path.stem)

    def _parse_excel_file(self, file_path: Path) -> pd.DataFrame:
        """Parse all voltage/current worksheets from one Excel file."""

        excel = pd.ExcelFile(file_path)
        sheet_frames: list[pd.DataFrame] = []
        for sheet_name in excel.sheet_names:
            try:
                frame = pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping unreadable sheet %s::%s: %s", file_path, sheet_name, exc)
                continue
            standardized = self._standardize_curve_frame(frame, file_path, sheet_name=sheet_name)
            if not standardized.empty:
                sheet_frames.append(standardized)
        return pd.concat(sheet_frames, ignore_index=True) if sheet_frames else pd.DataFrame()

    def _standardize_curve_frame(self, frame: pd.DataFrame, file_path: Path, sheet_name: str) -> pd.DataFrame:
        """Map one raw table into the canonical row-level I-V schema."""

        if frame.empty:
            return pd.DataFrame()
        frame = frame.copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        voltage_col = self._find_column(list(frame.columns), ["voltage", "volt"])
        current_col = self._find_column(list(frame.columns), ["current", "curr"])
        point_col = self._find_column(list(frame.columns), ["point", "index"])
        repeat_col = self._find_column(list(frame.columns), ["repeat"])
        if voltage_col is None or current_col is None:
            return pd.DataFrame()

        output = pd.DataFrame(
            {
                "material_name": self.material_name,
                "source_file": str(file_path.relative_to(paths.PROJECT_ROOT)),
                "sheet_name": str(sheet_name),
                "curve_id": self._build_curve_id(file_path, sheet_name),
                "measurement_type": self._infer_measurement_type(file_path, sheet_name),
                "point_index": (
                    pd.to_numeric(frame[point_col], errors="coerce")
                    if point_col is not None
                    else pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
                ),
                "repeat_index": (
                    pd.to_numeric(frame[repeat_col], errors="coerce")
                    if repeat_col is not None
                    else pd.Series(1, index=frame.index)
                ),
                "voltage_v": pd.to_numeric(frame[voltage_col], errors="coerce"),
                "current_a": pd.to_numeric(frame[current_col], errors="coerce"),
            }
        )
        for key, value in self._extract_metadata_from_path(file_path, sheet_name).items():
            output[key] = value
        return output

    def _postprocess_standardized_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Flag invalid points and attach engineered process features."""

        output = frame.copy()
        output["voltage_v"] = pd.to_numeric(output["voltage_v"], errors="coerce")
        output["current_a"] = pd.to_numeric(output["current_a"], errors="coerce")
        output["abs_current_a"] = output["current_a"].abs()
        output["point_index"] = pd.to_numeric(output["point_index"], errors="coerce")

        finite = np.isfinite(output["voltage_v"]) & np.isfinite(output["current_a"])
        current_reasonable = output["abs_current_a"].le(self.max_reasonable_current_a)
        not_duplicate = ~output.duplicated(subset=["curve_id", "point_index", "voltage_v", "current_a"])
        output["is_missing_or_nonfinite"] = ~finite
        output["is_extreme_current"] = finite & ~current_reasonable
        output["is_duplicate_point"] = ~not_duplicate
        output["is_valid_point"] = finite & current_reasonable & not_duplicate

        output["Current_Compliance_A"] = self._numeric_series(output, "current_compliance_a")
        output["RF_Power_W"] = self._numeric_series(output, "rf_power_w")
        output["Process_Time_Min"] = self._numeric_series(output, "process_time_min")
        output = self.add_rta_features(output)
        sort_cols = ["source_file", "sheet_name", "repeat_index", "point_index"]
        return self._safe_sort_values(output, sort_cols).reset_index(drop=True)

    def _impute_constant_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill configured missing process features with material constants."""

        output = frame.copy()
        for feature_name, value in self.default_feature_values.items():
            if feature_name not in output.columns:
                output[feature_name] = np.nan
            missing = pd.to_numeric(output[feature_name], errors="coerce").isna()
            output[f"{feature_name}_Was_Imputed"] = missing
            output.loc[missing, feature_name] = value

        for feature_name in self.config.feature_columns:
            flag_name = f"{feature_name}_Was_Imputed"
            if flag_name not in output.columns:
                output[flag_name] = False
        return output

    def _estimate_curve_metrics(self, group: pd.DataFrame) -> dict[str, float | str]:
        """Extract electrical metrics from one curve."""

        valid = group[group["is_valid_point"].astype(bool)].copy()
        valid = valid[np.isfinite(valid["voltage_v"]) & np.isfinite(valid["current_a"])]
        if len(valid) < 5:
            return {}

        valid = self._safe_sort_values(valid, ["repeat_index", "point_index", "voltage_v"])
        abs_current = valid["abs_current_a"].clip(lower=self.leakage_epsilon_a)
        on_off_metrics = self._estimate_on_off_ratio(valid)
        max_current = float(abs_current.max())
        compliance = self._numeric_series(valid, "Current_Compliance_A").dropna()
        compliance_value = float(compliance.median()) if not compliance.empty else np.nan
        current_threshold = max(min(1e-3, 0.1 * max_current), 1e-6)
        switching_points = valid[(valid["voltage_v"].abs() > 0) & (valid["abs_current_a"] >= current_threshold)]
        operation_voltage = float(switching_points["voltage_v"].abs().min()) if not switching_points.empty else np.nan
        measurement_type = str(valid["measurement_type"].iloc[0])
        forming_voltage = self._estimate_forming_voltage(valid) if measurement_type == "forming" else np.nan

        return {
            "Measurement_Type": measurement_type,
            "Leakage_Current_A": on_off_metrics["read_low_current"],
            "On_Off_Ratio": on_off_metrics["on_off_ratio"],
            "Read_Voltage_V": on_off_metrics["read_voltage_v"],
            "Operation_Voltage_V": operation_voltage,
            "Forming_Voltage_V": forming_voltage,
            "Max_Abs_Current_A": max_current,
            "Current_Compliance_A": compliance_value,
            "N_Points": int(len(valid)),
        }

    def _estimate_on_off_ratio(self, valid: pd.DataFrame) -> dict[str, float]:
        """Estimate HRS/LRS current ratio at fixed or dynamic read voltage.

        Physical logic:
        ``ratio(Vread) = max(|I(Vread)|) / min(|I(Vread)|)``.
        For NiO-like unipolar behavior, dynamic mode bins ``|V|`` and selects
        the voltage with the largest log-current separation:
        ``V* = argmax_V(log10(I_high) - log10(I_low))``.
        """

        candidate = valid[(valid["voltage_v"].abs() > 0) & np.isfinite(valid["abs_current_a"])].copy()
        if candidate.empty:
            return {"read_low_current": np.nan, "read_high_current": np.nan, "on_off_ratio": np.nan, "read_voltage_v": np.nan}

        if self.read_voltage_v is not None or not self.dynamic_on_off_ratio:
            target_voltage = 0.1 if self.read_voltage_v is None else abs(float(self.read_voltage_v))
            candidate["distance_to_read"] = (candidate["voltage_v"].abs() - target_voltage).abs()
            read_slice = candidate[candidate["distance_to_read"].eq(candidate["distance_to_read"].min())]
            currents = read_slice["abs_current_a"].clip(lower=self.leakage_epsilon_a)
            low = float(currents.min())
            high = float(currents.max())
            ratio = float(high / low) if low > 0 else np.nan
            return {
                "read_low_current": low,
                "read_high_current": high,
                "on_off_ratio": ratio,
                "read_voltage_v": float(read_slice["voltage_v"].abs().median()),
            }

        candidate["abs_voltage_bin"] = (candidate["voltage_v"].abs() / 0.02).round() * 0.02
        rows: list[dict[str, float]] = []
        for voltage_bin, voltage_group in candidate.groupby("abs_voltage_bin"):
            if len(voltage_group) < 2 or voltage_bin <= 0:
                continue
            currents = voltage_group["abs_current_a"].clip(lower=self.leakage_epsilon_a)
            low = float(currents.min())
            high = float(currents.max())
            ratio = float(high / low) if low > 0 else np.nan
            if np.isfinite(ratio) and ratio >= 1:
                rows.append(
                    {
                        "read_voltage_v": float(voltage_bin),
                        "read_low_current": low,
                        "read_high_current": high,
                        "on_off_ratio": ratio,
                        "log_spread": float(np.log10(high) - np.log10(low)),
                    }
                )

        if not rows:
            return {"read_low_current": np.nan, "read_high_current": np.nan, "on_off_ratio": np.nan, "read_voltage_v": np.nan}

        best = max(rows, key=lambda item: (item["log_spread"], item["on_off_ratio"]))
        return {
            "read_low_current": best["read_low_current"],
            "read_high_current": best["read_high_current"],
            "on_off_ratio": best["on_off_ratio"],
            "read_voltage_v": best["read_voltage_v"],
        }

    def _estimate_forming_voltage(self, valid: pd.DataFrame) -> float:
        """Estimate forming voltage from the largest positive log-current jump."""

        positive = valid[(valid["voltage_v"] >= 0) & (valid["abs_current_a"] > self.leakage_epsilon_a)]
        positive = self._safe_sort_values(positive, ["voltage_v", "point_index"])
        if len(positive) < 5:
            return np.nan

        log_i = np.log10(positive["abs_current_a"].clip(lower=self.leakage_epsilon_a).to_numpy())
        delta_log_i = np.diff(log_i)
        if len(delta_log_i) and np.nanmax(delta_log_i) >= 1.0:
            return float(positive.iloc[int(np.nanargmax(delta_log_i)) + 1]["voltage_v"])
        return np.nan

    def _aggregate_curve_metrics(self, curve_metrics: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """Aggregate curve-level metrics into one row per process condition."""

        iv_like = curve_metrics[curve_metrics["Measurement_Type"].eq("iv")]
        forming = curve_metrics[curve_metrics["Measurement_Type"].eq("forming")]
        endurance = curve_metrics[curve_metrics["Measurement_Type"].eq("endurance")]

        iv_targets = (
            iv_like.groupby(feature_cols, dropna=False)
            .agg(
                Leakage_Current_A=("Leakage_Current_A", "median"),
                On_Off_Ratio=("On_Off_Ratio", "median"),
                Read_Voltage_V=("Read_Voltage_V", "median"),
                Operation_Voltage_V=("Operation_Voltage_V", "median"),
                Max_Abs_Current_A=("Max_Abs_Current_A", "median"),
                IV_Curve_Count=("Curve_ID", "nunique"),
            )
            .reset_index()
            if not iv_like.empty
            else pd.DataFrame(columns=feature_cols)
        )
        forming_targets = (
            forming.groupby(feature_cols, dropna=False)
            .agg(Forming_Voltage_V=("Forming_Voltage_V", "median"), Forming_Curve_Count=("Curve_ID", "nunique"))
            .reset_index()
            if not forming.empty
            else pd.DataFrame(columns=feature_cols)
        )
        endurance_targets = (
            endurance.groupby(feature_cols, dropna=False)
            .agg(Endurance_Cycles=("Curve_ID", "nunique"), Endurance_File_Count=("Source_File", "nunique"))
            .reset_index()
            if not endurance.empty
            else pd.DataFrame(columns=feature_cols)
        )

        dataset = iv_targets.merge(forming_targets, on=feature_cols, how="outer")
        return dataset.merge(endurance_targets, on=feature_cols, how="outer")

    @staticmethod
    def _find_column(columns: list[str], keywords: list[str]) -> str | None:
        """Find a column whose normalized name contains one of the keywords."""

        normalized = [(column, re.sub(r"[^a-z0-9]+", "", column.lower())) for column in columns]
        for column, compact in normalized:
            if any(keyword in compact for keyword in keywords):
                if "time" in compact and "current" not in compact:
                    continue
                return column
        return None

    def _build_curve_id(self, file_path: Path, sheet_name: str) -> str:
        """Build a stable curve identifier from path and sheet name."""

        relative = file_path.relative_to(self.raw_dir)
        safe_path = "__".join(relative.with_suffix("").parts)
        safe_sheet = re.sub(r"[^A-Za-z0-9_.-]+", "_", sheet_name.strip())
        return f"{self.material_name}__{safe_path}__{safe_sheet}"

    @staticmethod
    def _infer_measurement_type(file_path: Path, sheet_name: str) -> str:
        """Infer forming, endurance, or normal I-V from filename metadata."""

        text = f"{file_path.name} {sheet_name}".lower()
        if "forming" in text or re.search(r"(^|[^a-z])f([^a-z]|$)", text):
            return "forming"
        if "endurance" in text or "cycle" in text:
            return "endurance"
        return "iv"

    def _extract_metadata_from_path(self, file_path: Path, sheet_name: str) -> dict[str, Any]:
        """Infer process metadata from material folder naming conventions."""

        relative_parts = list(file_path.relative_to(self.raw_dir).parts)
        text = "/".join(relative_parts + [sheet_name])

        rta_temp = np.nan
        for part in relative_parts:
            match = re.fullmatch(r"(\d{1,4})", part.strip())
            if match:
                value = float(match.group(1))
                if 0 <= value <= 1000:
                    rta_temp = self.no_rta_temperature_c if value == 0 else value
                    break
        if np.isnan(rta_temp):
            match = re.search(r"(?<!\d)([3456]\d{2})(?!\d)", text)
            if match:
                rta_temp = float(match.group(1))
        if np.isnan(rta_temp):
            rta_temp = self.no_rta_temperature_c

        compliance = np.nan
        for part in reversed(relative_parts):
            match = re.search(r"(?<!\d)(0\.\d+)(?!\d)", part)
            if match:
                compliance = float(match.group(1))
                break

        return {
            "rf_power_w": np.nan,
            "process_time_min": np.nan,
            "rta_temp_c": np.nan if rta_temp == self.no_rta_temperature_c else rta_temp,
            "rta_condition": "as_deposited" if rta_temp == self.no_rta_temperature_c else f"RTA_{int(rta_temp)}C",
            "current_compliance_a": compliance,
        }

    def _numeric_series(self, frame: pd.DataFrame, column_name: str, default: float = np.nan) -> pd.Series:
        """Return one numeric column or a same-index default series."""

        if column_name in frame.columns:
            return pd.to_numeric(frame[column_name], errors="coerce")
        return pd.Series(default, index=frame.index, dtype="float64")

    @staticmethod
    def _safe_sort_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Sort a dataframe using only columns that are actually present.

        Some legacy SiC datasets do not contain newer parser columns such as
        ``repeat_index``. This helper keeps chronological ordering when the
        columns exist and becomes a no-op when none of the requested columns are
        available.

        Args:
            frame: Dataframe to sort.
            columns: Preferred sort columns in priority order.

        Returns:
            Sorted dataframe if at least one requested column exists; otherwise
            a copy preserving the original row order.
        """

        sort_cols = [column for column in columns if column in frame.columns]
        if not sort_cols:
            return frame.copy()
        return frame.sort_values(sort_cols)

    @staticmethod
    def _string_series(frame: pd.DataFrame, column_name: str, default: str) -> pd.Series:
        """Return one string-like column or a same-index default series."""

        if column_name in frame.columns:
            return frame[column_name].astype("object")
        return pd.Series(default, index=frame.index, dtype="object")
