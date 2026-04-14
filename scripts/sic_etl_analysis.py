from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "RAW_DATA"
DATA_DIR = PROJECT_ROOT / "DATA"
REPORT_DIR = PROJECT_ROOT / "REPORTS"

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls"}
CURRENT_FLOOR_A = 1e-15
EXTREME_CURRENT_A = 0.1
DEFAULT_READ_VOLTAGE_V = 0.1

PDF_BASELINE = {
    "process": [
        "RF power: 50 W and 75 W; process time: 30, 60, and 120 min.",
        "RTA: 400 C and 500 C for 10 min; raw files also include 300 C.",
        "Forming baseline: about 2.5 V with 10 mA current compliance.",
    ],
    "electrical": [
        "As-deposited: on/off ratio about 1, operation voltage about 1.5 V, leakage about 1e-4 A.",
        "The thesis uses 1 H sputtering as the main baseline for endurance testing.",
        "RTA improves endurance from about 20/30 cycles to 100, 300, or 500 cycles depending on power and temperature.",
    ],
}

PDF_MECHANISM_BASELINE = [
    {"condition": "As-deposited, 50 W and 75 W", "hrs": "Poole-Frenkel", "lrs": "Ohmic"},
    {"condition": "50 W / 30 min / 400 C", "hrs": "Poole-Frenkel", "lrs": "Hopping"},
    {"condition": "50 W / 1 H / 400 C", "hrs": "Poole-Frenkel", "lrs": "Hopping"},
    {"condition": "50 W / 30 min or 1 H / 500 C", "hrs": "Ohmic", "lrs": "Ohmic"},
    {"condition": "75 W / 30 min / 400 C", "hrs": "Poole-Frenkel", "lrs": "Hopping"},
    {"condition": "75 W / 1 H / 400 C", "hrs": "Schottky Emission", "lrs": "Schottky Emission"},
    {"condition": "75 W / 1 H / 500 C", "hrs": "Hopping", "lrs": "Schottky Emission"},
    {"condition": "75 W / 2 H / 400 C", "hrs": "Ohmic", "lrs": "Hopping"},
    {"condition": "75 W / 2 H / 500 C", "hrs": "Ohmic", "lrs": "Ohmic"},
]


@dataclass(frozen=True)
class SourceMetadata:
    rf_power_w: float | None
    process_time_min: float | None
    rta_temp_c: float | None
    rta_condition: str
    measurement_type: str
    source_has_ng_flag: bool
    source_has_fake_flag: bool
    source_date: str | None


def norm(value: object) -> str:
    return str(value).strip().replace("\ufeff", "")


def norm_path(path: Path) -> str:
    return re.sub(r"\s+", " ", str(path).replace("\\", "/").lower())


def parse_float(value: object) -> float:
    try:
        if value is None:
            return math.nan
        if isinstance(value, str) and not value.strip():
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def extract_source_date(text: str) -> str | None:
    match = re.search(r"(20\d{6})", text)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def extract_rta_condition(text: str) -> tuple[float | None, str]:
    if "未退火" in text or "as-deposited" in text or "as_deposited" in text:
        return None, "as_deposited"
    for pattern in [
        r"rta\s*([345]\d{2})",
        r"r([345]\d{2})(?=[^0-9]|$)",
        r"(?<!\d)([345]\d{2})\s*(?:c|℃|度)",
        r"(?<!\d)([345]\d{2})(?=[_-])",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            temp = float(match.group(1))
            return temp, f"{int(temp)}C"
    return None, "as_deposited"


def extract_time_min(text: str) -> float | None:
    hour = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", text, re.I)
    if hour:
        return float(hour.group(1)) * 60.0
    minute = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b", text, re.I)
    if minute:
        return float(minute.group(1))
    compact = re.search(r"(?<!\d)(30|60|120)\s*m(?![a-z])", text, re.I)
    return float(compact.group(1)) if compact else None


def extract_rf_power_w(text: str) -> float | None:
    matches = re.findall(r"(?<!\d)(\d{2,3})\s*w(?![a-z])", text, re.I)
    return float(matches[-1]) if matches else None


def classify_measurement_type(path: Path) -> str:
    name = norm_path(Path(path.name))
    full = norm_path(path)
    if "forming" in name:
        return "forming"
    if "endurance" in name or re.search(r"(^|[-_\s])end($|[-_\s.])", name, re.I):
        return "endurance"
    if "retention" in name:
        return "retention"
    if "iv" in name or "i-v" in name or "i_v" in name or "i/v" in full:
        return "iv"
    return "unknown"


def extract_metadata(path: Path) -> SourceMetadata:
    text = norm_path(path)
    rta_temp_c, rta_condition = extract_rta_condition(text)
    return SourceMetadata(
        rf_power_w=extract_rf_power_w(text),
        process_time_min=extract_time_min(text),
        rta_temp_c=rta_temp_c,
        rta_condition=rta_condition,
        measurement_type=classify_measurement_type(path),
        source_has_ng_flag=bool(re.search(r"(^|[-_\s])ng($|[-_\s.])", text)),
        source_has_fake_flag="fake" in text,
        source_date=extract_source_date(text),
    )


def discover_raw_files(raw_dir: Path) -> list[Path]:
    return sorted(
        [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


def make_record(
    path: Path,
    metadata: SourceMetadata,
    sheet: str | None,
    cycle_index: int,
    point_index: int,
    voltage_v: float,
    current_a: float,
) -> dict[str, object]:
    return {
        "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
        "source_name": path.name,
        "source_extension": path.suffix.lower(),
        "source_date": metadata.source_date,
        "sheet": sheet or "",
        "measurement_type": metadata.measurement_type,
        "rf_power_w": metadata.rf_power_w,
        "process_time_min": metadata.process_time_min,
        "rta_temp_c": metadata.rta_temp_c,
        "rta_condition": metadata.rta_condition,
        "source_has_ng_flag": metadata.source_has_ng_flag,
        "source_has_fake_flag": metadata.source_has_fake_flag,
        "cycle_index": cycle_index,
        "point_index": point_index,
        "voltage_v": voltage_v,
        "current_a": current_a,
    }


def parse_dataname_csv(path: Path, metadata: SourceMetadata) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cycle_index, point_index = -1, 0
    header: list[str] | None = None
    v_pos = i_pos = None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            label = norm(row[0])
            if label == "DataName":
                header = [norm(cell) for cell in row[1:]]
                v_pos = next((idx for idx, name in enumerate(header) if name.upper().startswith("V")), None)
                i_pos = next((idx for idx, name in enumerate(header) if name.upper().startswith("I")), None)
                cycle_index += 1
                point_index = 0
            elif label == "DataValue" and header is not None and v_pos is not None and i_pos is not None:
                values = row[1:]
                if len(values) <= max(v_pos, i_pos):
                    continue
                voltage, current = parse_float(values[v_pos]), parse_float(values[i_pos])
                if np.isfinite(voltage) and np.isfinite(current):
                    records.append(make_record(path, metadata, None, cycle_index, point_index, voltage, current))
                    point_index += 1
    return records


def parse_repeating_header_csv(path: Path, metadata: SourceMetadata) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cycle_index, point_index = -1, 0
    header: list[str] | None = None
    v_pos = i_pos = None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            cells = [norm(cell) for cell in row]
            upper = [cell.upper() for cell in cells]
            if "V1" in upper and "I1" in upper:
                header = upper
                v_pos, i_pos = header.index("V1"), header.index("I1")
                cycle_index += 1
                point_index = 0
                continue
            if header is None or v_pos is None or i_pos is None or len(cells) <= max(v_pos, i_pos):
                continue
            voltage, current = parse_float(cells[v_pos]), parse_float(cells[i_pos])
            if np.isfinite(voltage) and np.isfinite(current):
                records.append(make_record(path, metadata, None, cycle_index, point_index, voltage, current))
                point_index += 1
    return records


def parse_csv_or_txt(path: Path) -> list[dict[str, object]]:
    metadata = extract_metadata(path)
    sample = path.read_text(encoding="utf-8-sig", errors="replace")
    if "DataName" in sample and "DataValue" in sample:
        return parse_dataname_csv(path, metadata)
    if "V1" in sample and "I1" in sample:
        return parse_repeating_header_csv(path, metadata)
    return []


def find_v_i_header_rows(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    header_rows: list[tuple[int, int, int]] = []
    for row_idx in range(len(df)):
        row = [norm(value).upper() for value in df.iloc[row_idx].tolist()]
        for col_idx in range(max(0, len(row) - 1)):
            if row[col_idx] == "V1" and row[col_idx + 1] == "I1":
                header_rows.append((row_idx, col_idx, col_idx + 1))
    return header_rows


def parse_excel(path: Path) -> list[dict[str, object]]:
    metadata = extract_metadata(path)
    records: list[dict[str, object]] = []
    cycle_index = -1
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:
        print(f"[WARN] Could not open Excel file {path}: {exc}")
        return records
    for sheet_name in workbook.sheet_names:
        if not sheet_name.lower().startswith("list"):
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        except Exception as exc:
            print(f"[WARN] Could not parse sheet {sheet_name} in {path}: {exc}")
            continue
        headers = find_v_i_header_rows(df)
        for header_idx, v_col, i_col in headers:
            cycle_index += 1
            next_headers = [row for row, _, _ in headers if row > header_idx]
            end_idx = min(next_headers) if next_headers else len(df)
            point_index = 0
            for row_idx in range(header_idx + 1, end_idx):
                row = df.iloc[row_idx]
                voltage = parse_float(row.iloc[v_col] if v_col < len(row) else np.nan)
                current = parse_float(row.iloc[i_col] if i_col < len(row) else np.nan)
                if np.isfinite(voltage) and np.isfinite(current):
                    records.append(make_record(path, metadata, sheet_name, cycle_index, point_index, voltage, current))
                    point_index += 1
    return records


def load_raw_iv_data(raw_files: Iterable[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for path in raw_files:
        try:
            parsed = parse_csv_or_txt(path) if path.suffix.lower() in {".csv", ".txt"} else parse_excel(path)
            status, message = ("parsed" if parsed else "skipped_no_iv_table"), ""
        except Exception as exc:
            parsed, status, message = [], "error", str(exc)
        records.extend(parsed)
        source_rows.append(
            {
                "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_extension": path.suffix.lower(),
                "parse_status": status,
                "records": len(parsed),
                "message": message,
            }
        )
    return pd.DataFrame(records), pd.DataFrame(source_rows)


def add_curve_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    key = df["source_file"].astype(str) + "::" + df["sheet"].fillna("").astype(str) + "::" + df["cycle_index"].astype(str)
    codes, uniques = pd.factorize(key, sort=True)
    df["curve_id"] = ["curve_" + str(code + 1).zfill(5) for code in codes]
    curve_map = {"curve_" + str(idx + 1).zfill(5): curve_key for idx, curve_key in enumerate(uniques)}
    df["curve_key"] = df["curve_id"].map(curve_map)
    return df


def clean_iv_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = add_curve_ids(df)
    cleaned["voltage_v"] = pd.to_numeric(cleaned["voltage_v"], errors="coerce")
    cleaned["current_a"] = pd.to_numeric(cleaned["current_a"], errors="coerce")
    cleaned["abs_current_a"] = cleaned["current_a"].abs()
    cleaned["log_abs_current_a"] = np.where(
        cleaned["abs_current_a"] > 0,
        np.log10(cleaned["abs_current_a"].clip(lower=CURRENT_FLOOR_A)),
        np.nan,
    )
    cleaned["point_quality_flag"] = "ok"
    cleaned["invalid_reason"] = ""

    invalid = ~np.isfinite(cleaned["voltage_v"]) | ~np.isfinite(cleaned["current_a"])
    cleaned.loc[invalid, ["point_quality_flag", "invalid_reason"]] = ["invalid", "non_finite_voltage_or_current"]

    extreme = cleaned["abs_current_a"] > EXTREME_CURRENT_A
    cleaned.loc[extreme, "point_quality_flag"] = "warning"
    cleaned.loc[extreme & cleaned["invalid_reason"].eq(""), "invalid_reason"] = "extreme_current_over_0.1A"

    huge_voltage = cleaned["voltage_v"].abs() > 20
    cleaned.loc[huge_voltage, "point_quality_flag"] = "warning"
    cleaned.loc[huge_voltage & cleaned["invalid_reason"].eq(""), "invalid_reason"] = "voltage_outside_expected_window"

    cleaned["is_valid_point"] = cleaned["point_quality_flag"].isin(["ok", "warning"])
    cleaned = cleaned.sort_values(["curve_id", "point_index"]).reset_index(drop=True)
    cleaned["sweep_direction"] = "unknown"
    for _, idx in cleaned.groupby("curve_id").groups.items():
        idx_list = list(idx)
        voltage = cleaned.loc[idx_list, "voltage_v"].to_numpy()
        if len(voltage) < 2:
            continue
        diffs = np.diff(voltage, prepend=voltage[0])
        cleaned.loc[idx_list, "sweep_direction"] = np.where(diffs > 0, "forward", np.where(diffs < 0, "reverse", "hold"))
    return cleaned


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return math.nan, math.nan, math.nan
    fit = stats.linregress(x, y)
    return float(fit.slope), float(fit.intercept), float(fit.rvalue**2)


def fit_conduction_models(voltage: np.ndarray, current: np.ndarray) -> dict[str, float | str]:
    voltage, current = np.asarray(voltage, dtype=float), np.asarray(current, dtype=float)
    mask = np.isfinite(voltage) & np.isfinite(current) & (voltage > 0) & (current > CURRENT_FLOOR_A)
    voltage, current = voltage[mask], np.abs(current[mask])
    if len(voltage) < 5:
        return {"best_model": "insufficient_data", "best_r2": math.nan}

    log_i = np.log10(np.clip(current, CURRENT_FLOOR_A, None))
    ln_i = np.log(np.clip(current, CURRENT_FLOOR_A, None))
    log_v = np.log10(np.clip(voltage, 1e-12, None))
    sqrt_v = np.sqrt(voltage)
    models = {
        "Ohmic": linear_fit(log_v, log_i),
        "Poole-Frenkel": linear_fit(sqrt_v, np.log(np.clip(current / voltage, CURRENT_FLOOR_A, None))),
        "Schottky Emission": linear_fit(sqrt_v, ln_i),
        "Hopping": linear_fit(voltage, ln_i),
    }

    output: dict[str, float | str] = {}
    for model, (slope, _, r2) in models.items():
        slug = model.lower().replace(" ", "_").replace("-", "_")
        output[f"{slug}_slope"] = slope
        output[f"{slug}_r2"] = r2

    ohmic_slope, _, ohmic_r2 = models["Ohmic"]
    valid_r2 = {model: vals[2] for model, vals in models.items() if np.isfinite(vals[2])}
    if np.isfinite(ohmic_slope) and np.isfinite(ohmic_r2) and 0.75 <= ohmic_slope <= 1.25 and ohmic_r2 >= 0.85:
        best_model = "Ohmic"
    elif valid_r2:
        best_model = max(valid_r2, key=valid_r2.get)
    else:
        best_model = "insufficient_data"
    output["best_model"] = best_model
    output["best_r2"] = valid_r2.get(best_model, math.nan)
    return output


def branch_by_voltage(df: pd.DataFrame, branch: str) -> pd.DataFrame:
    working = df.copy()
    voltage_limit = max(1.5, working["voltage_v"].quantile(0.75))
    working = working[(working["voltage_v"] > 0) & (working["voltage_v"] <= voltage_limit)]
    working = working[working["abs_current_a"] > CURRENT_FLOOR_A]
    if working.empty:
        return working
    working["voltage_round"] = working["voltage_v"].round(4)
    agg_func = "min" if branch == "hrs" else "max"
    return (
        working.groupby("voltage_round", as_index=False)
        .agg(voltage_v=("voltage_v", "median"), abs_current_a=("abs_current_a", agg_func))
        .sort_values("voltage_v")
    )


def estimate_forming_voltage(df: pd.DataFrame) -> tuple[float, str]:
    positive = df[(df["voltage_v"] >= 0) & (df["abs_current_a"] > CURRENT_FLOOR_A)].copy()
    positive = positive.sort_values(["voltage_v", "point_index"])
    if len(positive) < 5:
        return math.nan, "insufficient_data"
    voltage = positive["voltage_v"].to_numpy(dtype=float)
    current = positive["abs_current_a"].to_numpy(dtype=float)
    delta_log = np.diff(np.log10(np.clip(current, CURRENT_FLOOR_A, None)))
    if len(delta_log) and np.nanmax(delta_log) >= 1.0:
        idx = int(np.nanargmax(delta_log)) + 1
        return float(voltage[idx]), "log_current_jump"
    threshold = max(min(0.005, 0.5 * float(np.nanmax(current))), 1e-6)
    hits = positive[positive["abs_current_a"] >= threshold]
    if not hits.empty:
        return float(hits.iloc[0]["voltage_v"]), "half_compliance_threshold"
    return math.nan, "no_threshold_crossing"


def estimate_read_metrics(df: pd.DataFrame) -> dict[str, float | str]:
    usable = df[(df["abs_current_a"] > CURRENT_FLOOR_A) & np.isfinite(df["voltage_v"])].copy()
    positive = usable[(usable["voltage_v"] > 0) & (usable["voltage_v"] <= 0.5)].copy()
    if positive.empty:
        positive = usable[(usable["voltage_v"] != 0) & (usable["voltage_v"].abs() <= 0.5)].copy()
    if positive.empty:
        return {
            "read_voltage_v": math.nan,
            "read_low_current_a": math.nan,
            "read_high_current_a": math.nan,
            "on_off_ratio": math.nan,
            "on_off_log10": math.nan,
            "read_metric_method": "insufficient_low_voltage_points",
        }
    positive["voltage_round"] = positive["voltage_v"].round(4)
    counts = positive.groupby("voltage_round").size()
    duplicate_voltages = counts[counts >= 2].index.to_numpy(dtype=float)
    if len(duplicate_voltages):
        chosen = float(duplicate_voltages[np.argmin(np.abs(duplicate_voltages - DEFAULT_READ_VOLTAGE_V))])
        subset = positive[positive["voltage_round"] == chosen]
        method = "duplicate_voltage_at_read"
    else:
        positive["distance_to_read"] = (positive["voltage_v"].abs() - DEFAULT_READ_VOLTAGE_V).abs()
        subset = positive[positive["distance_to_read"] == positive["distance_to_read"].min()]
        chosen = float(subset["voltage_v"].abs().median())
        method = "nearest_single_pass_low_voltage"
    currents = subset["abs_current_a"].to_numpy(dtype=float)
    low, high = float(np.nanmin(currents)), float(np.nanmax(currents))
    ratio = high / low if np.isfinite(low) and low > 0 and np.isfinite(high) else math.nan
    return {
        "read_voltage_v": chosen,
        "read_low_current_a": low,
        "read_high_current_a": high,
        "on_off_ratio": ratio,
        "on_off_log10": math.log10(ratio) if ratio and ratio > 0 else math.nan,
        "read_metric_method": method,
    }


def estimate_operation_voltage(df: pd.DataFrame) -> float:
    usable = df[(df["abs_current_a"] > CURRENT_FLOOR_A) & (df["voltage_v"].abs() > 0)].copy()
    if usable.empty:
        return math.nan
    threshold = max(min(0.001, 0.1 * float(usable["abs_current_a"].max())), 1e-6)
    hits = usable[usable["abs_current_a"] >= threshold]
    return float(hits["voltage_v"].abs().min()) if not hits.empty else math.nan


def classify_curve_quality(metric: pd.Series) -> tuple[str, str]:
    reasons: list[str] = []
    if metric.get("source_has_ng_flag", False):
        reasons.append("filename_ng_flag")
    if metric.get("source_has_fake_flag", False):
        reasons.append("filename_fake_flag")
    if metric.get("n_points", 0) < 10:
        reasons.append("too_few_points")
    if metric.get("max_abs_current_a", 0) > EXTREME_CURRENT_A:
        reasons.append("extreme_current")
    if metric.get("low_voltage_median_current_a", 0) > 1e-3:
        reasons.append("low_voltage_short_or_preformed")
    if metric.get("measurement_type") == "forming" and not np.isfinite(metric.get("forming_voltage_v", math.nan)):
        reasons.append("forming_not_detected")
    if not reasons:
        return "ok", ""
    if any(reason in reasons for reason in ["filename_ng_flag", "filename_fake_flag", "too_few_points", "extreme_current"]):
        return "invalid", ";".join(reasons)
    return "warning", ";".join(reasons)


def extract_curve_metrics(cleaned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if cleaned.empty:
        return pd.DataFrame(rows)
    group_cols = [
        "curve_id",
        "source_file",
        "source_name",
        "source_extension",
        "source_date",
        "sheet",
        "measurement_type",
        "rf_power_w",
        "process_time_min",
        "rta_temp_c",
        "rta_condition",
        "source_has_ng_flag",
        "source_has_fake_flag",
        "cycle_index",
    ]
    for key, group in cleaned.groupby(group_cols, dropna=False, sort=True):
        base = dict(zip(group_cols, key))
        valid = group[group["is_valid_point"]].copy()
        if valid.empty:
            continue
        read = estimate_read_metrics(valid)
        forming_voltage_v, forming_method = (
            estimate_forming_voltage(valid) if base["measurement_type"] == "forming" else (math.nan, "not_forming_curve")
        )
        hrs = fit_conduction_models(
            branch_by_voltage(valid, "hrs")["voltage_v"].to_numpy(),
            branch_by_voltage(valid, "hrs")["abs_current_a"].to_numpy(),
        )
        lrs = fit_conduction_models(
            branch_by_voltage(valid, "lrs")["voltage_v"].to_numpy(),
            branch_by_voltage(valid, "lrs")["abs_current_a"].to_numpy(),
        )
        low_v = valid[valid["voltage_v"].abs() <= 0.2]
        row = {
            **base,
            "n_points": int(len(valid)),
            "voltage_min_v": float(valid["voltage_v"].min()),
            "voltage_max_v": float(valid["voltage_v"].max()),
            "current_min_a": float(valid["current_a"].min()),
            "current_max_a": float(valid["current_a"].max()),
            "max_abs_current_a": float(valid["abs_current_a"].max()),
            "median_abs_current_a": float(valid["abs_current_a"].median()),
            "low_voltage_median_current_a": float(low_v["abs_current_a"].median()) if not low_v.empty else math.nan,
            "forming_voltage_v": forming_voltage_v,
            "forming_detection_method": forming_method,
            "operation_voltage_est_v": estimate_operation_voltage(valid),
            **read,
            "hrs_best_model": hrs.get("best_model", "insufficient_data"),
            "hrs_best_r2": hrs.get("best_r2", math.nan),
            "hrs_ohmic_slope": hrs.get("ohmic_slope", math.nan),
            "hrs_ohmic_r2": hrs.get("ohmic_r2", math.nan),
            "hrs_poole_frenkel_r2": hrs.get("poole_frenkel_r2", math.nan),
            "hrs_schottky_emission_r2": hrs.get("schottky_emission_r2", math.nan),
            "hrs_hopping_r2": hrs.get("hopping_r2", math.nan),
            "lrs_best_model": lrs.get("best_model", "insufficient_data"),
            "lrs_best_r2": lrs.get("best_r2", math.nan),
            "lrs_ohmic_slope": lrs.get("ohmic_slope", math.nan),
            "lrs_ohmic_r2": lrs.get("ohmic_r2", math.nan),
            "lrs_poole_frenkel_r2": lrs.get("poole_frenkel_r2", math.nan),
            "lrs_schottky_emission_r2": lrs.get("schottky_emission_r2", math.nan),
            "lrs_hopping_r2": lrs.get("hopping_r2", math.nan),
        }
        quality, reason = classify_curve_quality(pd.Series(row))
        row["curve_quality_flag"] = quality
        row["curve_invalid_reason"] = reason
        rows.append(row)
    metrics = pd.DataFrame(rows)
    for col in ["rf_power_w", "process_time_min", "rta_temp_c"]:
        if col in metrics:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    return metrics


def aggregate_condition_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    valid = metrics[metrics["curve_quality_flag"].isin(["ok", "warning"])].copy()
    valid = valid[valid["measurement_type"].isin(["iv", "endurance", "forming"])]
    if valid.empty:
        return pd.DataFrame()
    group_cols = ["rf_power_w", "process_time_min", "rta_condition", "rta_temp_c", "measurement_type"]
    return (
        valid.groupby(group_cols, dropna=False)
        .agg(
            curve_count=("curve_id", "nunique"),
            on_off_median=("on_off_ratio", "median"),
            on_off_log10_median=("on_off_log10", "median"),
            leakage_median_a=("read_low_current_a", "median"),
            operation_voltage_median_v=("operation_voltage_est_v", "median"),
            forming_voltage_median_v=("forming_voltage_v", "median"),
            max_abs_current_median_a=("max_abs_current_a", "median"),
        )
        .reset_index()
        .sort_values(["measurement_type", "rf_power_w", "process_time_min", "rta_temp_c"], na_position="first")
    )


def summarize_endurance(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    endurance = metrics[metrics["measurement_type"] == "endurance"].copy()
    if endurance.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["source_file", "rf_power_w", "process_time_min", "rta_condition", "rta_temp_c"]
    for key, group in endurance.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        usable = group[group["curve_quality_flag"].isin(["ok", "warning"])]
        row.update(
            {
                "cycle_count_total": int(group["cycle_index"].nunique()),
                "cycle_count_valid_or_warning": int(usable["cycle_index"].nunique()),
                "on_off_log10_median": float(usable["on_off_log10"].median()) if not usable.empty else math.nan,
                "on_off_ratio_median": float(usable["on_off_ratio"].median()) if not usable.empty else math.nan,
                "leakage_median_a": float(usable["read_low_current_a"].median()) if not usable.empty else math.nan,
                "operation_voltage_median_v": float(usable["operation_voltage_est_v"].median()) if not usable.empty else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rta_temp_c", "rf_power_w", "process_time_min"], na_position="first")


def compute_sweet_spots(metrics: pd.DataFrame, endurance_summary: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    usable = metrics[
        metrics["curve_quality_flag"].isin(["ok", "warning"])
        & metrics["measurement_type"].isin(["iv", "endurance"])
        & metrics["rf_power_w"].notna()
        & metrics["process_time_min"].notna()
    ].copy()
    if usable.empty:
        return pd.DataFrame()
    condition_cols = ["rf_power_w", "process_time_min", "rta_condition", "rta_temp_c"]
    condition = (
        usable.groupby(condition_cols, dropna=False)
        .agg(
            curve_count=("curve_id", "nunique"),
            on_off_log10_median=("on_off_log10", "median"),
            leakage_median_a=("read_low_current_a", "median"),
            operation_voltage_median_v=("operation_voltage_est_v", "median"),
        )
        .reset_index()
    )
    if not endurance_summary.empty:
        end_cond = (
            endurance_summary.groupby(condition_cols, dropna=False)
            .agg(endurance_cycles=("cycle_count_valid_or_warning", "max"))
            .reset_index()
        )
        condition = condition.merge(end_cond, on=condition_cols, how="left")
    else:
        condition["endurance_cycles"] = np.nan

    def norm_high(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        if series.notna().sum() == 0 or series.max() == series.min():
            return pd.Series(np.where(series.notna(), 0.5, np.nan), index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    def norm_low(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        if series.notna().sum() == 0 or series.max() == series.min():
            return pd.Series(np.where(series.notna(), 0.5, np.nan), index=series.index)
        return 1 - (series - series.min()) / (series.max() - series.min())

    condition["score_on_off"] = norm_high(condition["on_off_log10_median"]).fillna(0)
    condition["score_leakage"] = norm_low(np.log10(condition["leakage_median_a"].clip(lower=CURRENT_FLOOR_A))).fillna(0)
    condition["score_endurance"] = norm_high(condition["endurance_cycles"]).fillna(0)
    condition["endurance_data_available"] = condition["endurance_cycles"].notna()
    base_score = 0.4 * condition["score_on_off"] + 0.35 * condition["score_leakage"] + 0.25 * condition["score_endurance"]
    condition["sweet_spot_score"] = base_score * np.where(condition["endurance_data_available"], 1.0, 0.25)
    return condition.sort_values("sweet_spot_score", ascending=False)


def significance_endurance_by_rta(endurance_summary: pd.DataFrame) -> dict[str, object]:
    if endurance_summary.empty:
        return {"test": "not_applicable", "reason": "No machine-readable endurance files were parsed.", "p_value": math.nan, "statistic": math.nan}
    grouped = [
        pd.to_numeric(group["cycle_count_valid_or_warning"], errors="coerce").dropna().to_numpy()
        for _, group in endurance_summary.groupby("rta_condition", dropna=False)
    ]
    grouped = [values for values in grouped if len(values) >= 2]
    if len(grouped) >= 2:
        stat, p_value = stats.kruskal(*grouped)
        return {"test": "Kruskal-Wallis", "reason": "", "p_value": float(p_value), "statistic": float(stat)}
    return {
        "test": "not_applicable",
        "reason": "At least two RTA groups with two or more independent endurance files are required.",
        "p_value": math.nan,
        "statistic": math.nan,
    }


def format_number(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return ""
    if value_float == 0:
        return "0"
    if abs(value_float) >= 1e4 or abs(value_float) < 1e-3:
        return f"{value_float:.{digits}e}"
    return f"{value_float:.{digits}g}"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data available._"
    shown = df.head(max_rows).copy().fillna("")
    headers = list(shown.columns)
    rows = []
    for _, row in shown.iterrows():
        rows.append(
            [
                format_number(row[col]) if isinstance(row[col], (int, float, np.number)) else str(row[col])
                for col in headers
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if len(df) > max_rows:
        lines.append(f"\n_顯示前 {max_rows} 筆，共 {len(df)} 筆。_")
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def report_header(title: str) -> str:
    return f"# {title}\n\n資料來源：`RAW_DATA/` machine-readable CSV/XLSX/TXT；論文背景：`M1003101碩論.pdf`。\n"


def generate_forming_report(metrics: pd.DataFrame) -> str:
    forming = metrics[metrics["measurement_type"] == "forming"].copy()
    if not forming.empty:
        summary = (
            forming[forming["curve_quality_flag"].isin(["ok", "warning"])]
            .groupby(["rf_power_w", "process_time_min", "rta_condition"], dropna=False)
            .agg(
                curves=("curve_id", "nunique"),
                forming_voltage_median_v=("forming_voltage_v", "median"),
                forming_voltage_min_v=("forming_voltage_v", "min"),
                forming_voltage_max_v=("forming_voltage_v", "max"),
                low_voltage_median_current_a=("low_voltage_median_current_a", "median"),
            )
            .reset_index()
            .sort_values(["rta_condition", "rf_power_w", "process_time_min"], na_position="first")
        )
        top = forming.sort_values("forming_voltage_v")[
            [
                "source_file",
                "rf_power_w",
                "process_time_min",
                "rta_condition",
                "forming_voltage_v",
                "forming_detection_method",
                "curve_quality_flag",
                "curve_invalid_reason",
            ]
        ]
    else:
        summary, top = pd.DataFrame(), pd.DataFrame()
    return f"""{report_header("Forming 電壓分析")}

## 重點結論

- PDF baseline 指出 Forming 電壓約 2.5 V，並以 10 mA compliance 避免元件崩潰。
- 本報告的 Forming 電壓由正偏壓 sweep 中的電流躍升點或半 compliance 門檻估計。
- 若曲線在低電壓已呈現高電流，會標記為 `low_voltage_short_or_preformed`。

## 統計摘要

{dataframe_to_markdown(summary, max_rows=30)}

## 曲線層級估計

{dataframe_to_markdown(top, max_rows=30)}

## 方法

```python
if max_delta_log10_current >= 1.0:
    forming_voltage = voltage_at_largest_log_current_jump
else:
    forming_voltage = first_voltage_where_current_exceeds_half_compliance_proxy
```
"""


def generate_on_off_report(metrics: pd.DataFrame, condition_summary: pd.DataFrame) -> str:
    iv = metrics[metrics["measurement_type"].isin(["iv", "endurance"])].copy()
    if not iv.empty:
        rf_time = (
            iv[iv["curve_quality_flag"].isin(["ok", "warning"])]
            .groupby(["rf_power_w", "process_time_min", "rta_condition"], dropna=False)
            .agg(
                curves=("curve_id", "nunique"),
                on_off_median=("on_off_ratio", "median"),
                on_off_log10_median=("on_off_log10", "median"),
                leakage_median_a=("read_low_current_a", "median"),
                read_voltage_median_v=("read_voltage_v", "median"),
            )
            .reset_index()
            .sort_values("on_off_log10_median", ascending=False)
        )
    else:
        rf_time = pd.DataFrame()
    condition_view = condition_summary[condition_summary["measurement_type"].isin(["iv", "endurance"])].copy() if not condition_summary.empty else pd.DataFrame()
    return f"""{report_header("On/Off Ratio 分析")}

## 重點結論

- PDF 指出未退火樣品 on/off ratio 約為 1；RTA 後在 1 H 條件下通常有較明顯記憶窗口。
- 本資料集優先用同一低讀取電壓在雙向 sweep 中的重複點計算高/低電流比。
- 趨勢比較建議看 `on_off_log10_median`，因為比值跨多個數量級。

## RF Power × Time × RTA 趨勢

{dataframe_to_markdown(rf_time, max_rows=40)}

## 全條件摘要

{dataframe_to_markdown(condition_view, max_rows=40)}

## 計算公式

```text
On/Off Ratio = max(|I_read|) / min(|I_read|)
log10(On/Off Ratio) = log10(max(|I_read|)) - log10(min(|I_read|))
```
"""


def generate_leakage_report(metrics: pd.DataFrame, sweet_spots: pd.DataFrame) -> str:
    iv = metrics[metrics["measurement_type"].isin(["iv", "endurance"])].copy()
    if not iv.empty:
        leakage = (
            iv[iv["curve_quality_flag"].isin(["ok", "warning"])]
            .groupby(["rf_power_w", "process_time_min", "rta_condition"], dropna=False)
            .agg(
                curves=("curve_id", "nunique"),
                leakage_median_a=("read_low_current_a", "median"),
                leakage_min_a=("read_low_current_a", "min"),
                leakage_max_a=("read_low_current_a", "max"),
                on_off_log10_median=("on_off_log10", "median"),
            )
            .reset_index()
            .sort_values("leakage_median_a", ascending=True)
        )
    else:
        leakage = pd.DataFrame()
    sweet_cols = ["rf_power_w", "process_time_min", "rta_condition", "curve_count", "leakage_median_a", "on_off_log10_median", "endurance_cycles", "endurance_data_available", "sweet_spot_score"]
    sweet_view = sweet_spots[sweet_cols] if not sweet_spots.empty else pd.DataFrame()
    return f"""{report_header("漏電流分析")}

## 重點結論

- PDF baseline 指出未退火漏電流約 1e-4 A，RTA 後 1 H 條件可降到約 1e-7 A 等級。
- 本報告以低讀取電壓下較低電流分支作為 leakage proxy。
- 若低電壓區中位電流超過 1e-3 A，曲線會警示為可能短路或預先形成。

## 漏電流排序

{dataframe_to_markdown(leakage, max_rows=40)}

## Sweet Spot 參考

{dataframe_to_markdown(sweet_view, max_rows=20)}

## 計算公式

```text
Leakage proxy = min(|I_read|) at the selected low read voltage
Sweet spot score = 0.40 * normalized log10(On/Off) + 0.35 * normalized low leakage + 0.25 * normalized endurance
```
"""


def generate_endurance_report(endurance_summary: pd.DataFrame, significance: dict[str, object], sweet_spots: pd.DataFrame) -> str:
    if not endurance_summary.empty:
        condition = (
            endurance_summary.groupby(["rf_power_w", "process_time_min", "rta_condition"], dropna=False)
            .agg(
                files=("source_file", "nunique"),
                endurance_cycles_max=("cycle_count_valid_or_warning", "max"),
                endurance_cycles_median=("cycle_count_valid_or_warning", "median"),
                on_off_log10_median=("on_off_log10_median", "median"),
                leakage_median_a=("leakage_median_a", "median"),
            )
            .reset_index()
            .sort_values("endurance_cycles_max", ascending=False)
        )
    else:
        condition = pd.DataFrame()
    sig_text = (
        f"{significance['test']} statistic={format_number(significance.get('statistic'))}, p={format_number(significance.get('p_value'))}"
        if significance.get("test") != "not_applicable"
        else f"Not applicable: {significance.get('reason')}"
    )
    sweet_cols = ["rf_power_w", "process_time_min", "rta_condition", "endurance_cycles", "endurance_data_available", "on_off_log10_median", "leakage_median_a", "sweet_spot_score"]
    sweet_view = sweet_spots[sweet_cols] if not sweet_spots.empty else pd.DataFrame()
    pdf_endurance = pd.DataFrame(
        [
            {"rf_power_w": 50, "rta_condition": "as_deposited", "pdf_cycles": 20},
            {"rf_power_w": 50, "rta_condition": "400C", "pdf_cycles": 100},
            {"rf_power_w": 50, "rta_condition": "500C", "pdf_cycles": 300},
            {"rf_power_w": 75, "rta_condition": "as_deposited", "pdf_cycles": 30},
            {"rf_power_w": 75, "rta_condition": "400C", "pdf_cycles": 300},
            {"rf_power_w": 75, "rta_condition": "500C", "pdf_cycles": 500},
        ]
    )
    return f"""{report_header("Endurance 分析")}

## 重點結論

- PDF baseline 顯示 RTA 對 Endurance 有明顯提升，尤其 75 W / 500 C 約 500 cycles。
- Machine-readable 分析以 Endurance CSV 中的 sweep block 數量作為 cycle count。
- 統計顯著性檢定結果：{sig_text}
- 若某些 RTA 條件缺少至少兩個獨立 Endurance 檔案，顯著性檢定會標記為不適用。

## Machine-Readable Endurance 摘要

{dataframe_to_markdown(condition, max_rows=40)}

## PDF Baseline Endurance

{dataframe_to_markdown(pdf_endurance, max_rows=20)}

## Sweet Spot 參考

{dataframe_to_markdown(sweet_view, max_rows=20)}

## 統計方法

```python
if each_RTA_group_has_at_least_two_independent_files:
    scipy.stats.kruskal(*cycle_count_groups)
else:
    significance = "not_applicable"
```
"""


def generate_conduction_report(metrics: pd.DataFrame) -> str:
    iv = metrics[metrics["measurement_type"].isin(["iv", "endurance"])].copy()
    if not iv.empty:
        model_summary = (
            iv[iv["curve_quality_flag"].isin(["ok", "warning"])]
            .groupby(["rf_power_w", "process_time_min", "rta_condition", "hrs_best_model", "lrs_best_model"], dropna=False)
            .agg(curves=("curve_id", "nunique"), hrs_r2_median=("hrs_best_r2", "median"), lrs_r2_median=("lrs_best_r2", "median"))
            .reset_index()
            .sort_values(["rf_power_w", "process_time_min", "rta_condition", "curves"], ascending=[True, True, True, False], na_position="first")
        )
        curve_view = iv[
            [
                "source_file",
                "rf_power_w",
                "process_time_min",
                "rta_condition",
                "hrs_best_model",
                "hrs_best_r2",
                "hrs_ohmic_slope",
                "lrs_best_model",
                "lrs_best_r2",
                "lrs_ohmic_slope",
                "curve_quality_flag",
            ]
        ].sort_values(["rf_power_w", "process_time_min", "rta_condition"], na_position="first")
    else:
        model_summary, curve_view = pd.DataFrame(), pd.DataFrame()
    pdf_table = pd.DataFrame(PDF_MECHANISM_BASELINE)
    return f"""{report_header("傳導機制分析")}

## 重點結論

- PDF baseline：未退火樣品多為 HRS Poole-Frenkel、LRS Ohmic；RTA 後會出現 Hopping 或 Schottky Emission。
- 本資料集以斜率與線性擬合的 R-squared 自動選擇 HRS/LRS 的候選傳導模型。
- Ohmic 另加上 log(I)-log(V) slope 接近 1 的規則，避免只靠 R-squared 過度判讀。

## Machine-Readable 模型摘要

{dataframe_to_markdown(model_summary, max_rows=50)}

## 曲線層級模型

{dataframe_to_markdown(curve_view, max_rows=50)}

## PDF Baseline 機制

{dataframe_to_markdown(pdf_table, max_rows=20)}

## 擬合座標

```text
Ohmic:              log10(I) vs log10(V), slope close to 1
Poole-Frenkel:      ln(I / V) vs sqrt(V)
Schottky Emission:  ln(I) vs sqrt(V)
Hopping:            ln(I) vs V
```
"""


def generate_reports(metrics: pd.DataFrame, condition_summary: pd.DataFrame, endurance_summary: pd.DataFrame, sweet_spots: pd.DataFrame) -> None:
    significance = significance_endurance_by_rta(endurance_summary)
    write_markdown(REPORT_DIR / "forming_voltage_analysis.md", generate_forming_report(metrics))
    write_markdown(REPORT_DIR / "on_off_ratio_analysis.md", generate_on_off_report(metrics, condition_summary))
    write_markdown(REPORT_DIR / "leakage_current_analysis.md", generate_leakage_report(metrics, sweet_spots))
    write_markdown(REPORT_DIR / "endurance_analysis.md", generate_endurance_report(endurance_summary, significance, sweet_spots))
    write_markdown(REPORT_DIR / "conduction_mechanism_analysis.md", generate_conduction_report(metrics))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    raw_files = discover_raw_files(RAW_DIR)
    raw_df, source_summary = load_raw_iv_data(raw_files)
    cleaned = clean_iv_data(raw_df)
    metrics = extract_curve_metrics(cleaned)
    condition_summary = aggregate_condition_metrics(metrics)
    endurance_summary = summarize_endurance(metrics)
    sweet_spots = compute_sweet_spots(metrics, endurance_summary)

    cleaned.to_csv(DATA_DIR / "cleaned_sic_sputtering_data.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(DATA_DIR / "curve_metrics_sic_sputtering_data.csv", index=False, encoding="utf-8-sig")
    source_summary.to_csv(DATA_DIR / "source_file_parse_summary.csv", index=False, encoding="utf-8-sig")
    condition_summary.to_csv(DATA_DIR / "condition_metric_summary.csv", index=False, encoding="utf-8-sig")
    endurance_summary.to_csv(DATA_DIR / "endurance_summary.csv", index=False, encoding="utf-8-sig")
    sweet_spots.to_csv(DATA_DIR / "sweet_spot_ranking.csv", index=False, encoding="utf-8-sig")
    generate_reports(metrics, condition_summary, endurance_summary, sweet_spots)

    print(f"Parsed files: {len(raw_files)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Curve metrics: {len(metrics)}")
    print(f"Reports written to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
