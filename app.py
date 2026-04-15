from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import paths
from src.config import MATERIAL_CONFIGS, MaterialConfig, SearchSpaceParameter, get_material_config
from src.model_manager import MaterialModelManager


TARGET_ORDER = [
    "Leakage_Current_A",
    "Endurance_Cycles",
    "On_Off_Ratio",
    "Operation_Voltage_V",
    "Forming_Voltage_V",
]

DISPLAY_NAMES = {
    "SiC": "SiC",
    "NiO": "NiO",
    "RF_Power_W": "RF Power",
    "Process_Time_Min": "Process Time",
    "RTA_Temperature_C": "RTA Temperature",
    "Has_RTA": "RTA Enabled",
    "Current_Compliance_A": "Current Compliance",
    "Leakage_Current_A": "Leakage Current",
    "Endurance_Cycles": "Endurance",
    "On_Off_Ratio": "On/Off Ratio",
    "Operation_Voltage_V": "Operation Voltage",
    "Forming_Voltage_V": "Forming Voltage",
}

UNITS = {
    "RF_Power_W": "W",
    "Process_Time_Min": "min",
    "RTA_Temperature_C": "°C",
    "Current_Compliance_A": "A",
    "Leakage_Current_A": "A",
    "Endurance_Cycles": "cycles",
    "Operation_Voltage_V": "V",
    "Forming_Voltage_V": "V",
}


def configure_page() -> None:
    """Configure Streamlit page metadata."""

    st.set_page_config(
        page_title="RRAM Material Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_apple_style_css() -> None:
    """Inject Apple/macOS-inspired CSS."""

    st.markdown(
        """
        <style>
        :root {
            --app-bg: #F5F5F7;
            --sidebar-bg: #ECECF1;
            --surface: rgba(255, 255, 255, 0.86);
            --surface-solid: #FFFFFF;
            --ink: #1d1d1f;
            --muted: #6e6e73;
            --line: rgba(0, 0, 0, 0.08);
            --accent: #007aff;
            --accent-soft: rgba(0, 122, 255, 0.10);
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            --shadow-hover: 0 10px 28px rgba(0, 0, 0, 0.08);
            --radius: 16px;
        }

        html, body, [class*="css"], .stApp {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
                Roboto, Helvetica, Arial, sans-serif;
            color: var(--ink);
        }

        .stApp {
            background: var(--app-bg);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F0F0F4 0%, var(--sidebar-bg) 100%);
            border-right: 1px solid var(--line);
            box-shadow: 8px 0 28px rgba(0, 0, 0, 0.035);
        }

        div[data-testid="stSidebarContent"] {
            padding: 1.4rem 1.05rem 2.2rem 1.05rem;
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        h1 {
            font-weight: 780;
        }

        p, label, span {
            letter-spacing: 0;
        }

        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stMetric"]) {
            gap: 1rem;
        }

        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stImage"],
        .stAlert,
        .stExpander,
        .recipe-card,
        .hero-panel,
        .glass-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        div[data-testid="stMetric"] {
            padding: 1.15rem 1.2rem;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        div[data-testid="stMetric"]:hover,
        .recipe-card:hover,
        .glass-panel:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
            border-color: rgba(0, 122, 255, 0.18);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 700;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(0, 122, 255, 0.20);
            background: linear-gradient(180deg, #FFFFFF 0%, #EEF6FF 100%);
            box-shadow: 0 3px 12px rgba(0, 122, 255, 0.10);
            color: #0057D9;
            font-weight: 700;
            padding: 0.66rem 1.15rem;
            transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }

        .stButton > button:hover {
            transform: scale(1.018);
            border-color: rgba(0, 122, 255, 0.45);
            background: linear-gradient(180deg, #F9FCFF 0%, #E5F1FF 100%);
            box-shadow: 0 8px 20px rgba(0, 122, 255, 0.16);
            color: #004FC4;
        }

        .stButton > button:active {
            transform: scale(0.992);
        }

        .stSelectbox, .stSlider, .stNumberInput {
            padding-bottom: 0.4rem;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div {
            height: 4px;
        }

        div[data-testid="stSlider"] [role="slider"] {
            width: 18px;
            height: 18px;
            border: 2px solid #FFFFFF;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.86);
            border-color: rgba(0, 0, 0, 0.08);
        }

        .hero-panel {
            padding: 1.65rem 1.7rem;
            margin-bottom: 1.2rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 249, 255, 0.84) 100%);
        }

        .hero-title {
            font-size: 2.25rem;
            line-height: 1.12;
            font-weight: 790;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.5rem;
            max-width: 880px;
        }

        .recipe-card {
            padding: 1.1rem;
            min-height: 248px;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .recipe-eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .recipe-title {
            font-size: 1.04rem;
            font-weight: 740;
            margin-bottom: 0.35rem;
        }

        .recipe-text {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
            min-height: 3.8rem;
        }

        .recipe-kv {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.35rem 0.8rem;
            margin-top: 0.75rem;
            font-size: 0.86rem;
        }

        .recipe-kv strong {
            color: var(--ink);
        }

        .glass-panel {
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
        }

        .soft-label {
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def available_materials() -> list[str]:
    """Return registered material names."""

    return sorted(config.name for config in MATERIAL_CONFIGS.values())


def feature_key(material_name: str, feature_name: str) -> str:
    """Return a material-scoped Streamlit key for feature widgets."""

    return f"feature_control__{material_name}__{feature_name}"


def reset_state_on_material_change(material_name: str) -> None:
    """Reset material-scoped widget state when switching materials.

    Streamlit persists widget values by key. If SiC values are reused for NiO,
    stale values can be outside the new search space and cause widget errors.
    We avoid this by deleting all feature-control keys whenever the selected
    material changes, then rendering fresh defaults from MaterialConfig.
    """

    previous_material = st.session_state.get("active_material")
    if previous_material == material_name:
        return

    for key in list(st.session_state.keys()):
        if key.startswith("feature_control__"):
            del st.session_state[key]
    st.session_state["active_material"] = material_name


@st.cache_resource(show_spinner=False)
def load_model_manager(material_name: str) -> MaterialModelManager:
    """Load a material model manager and any available persisted models."""

    manager = MaterialModelManager(material_name=material_name)
    manager.load_models(strict=False)
    return manager


@st.cache_data(show_spinner=False)
def load_results_csv(material_name: str, filename: str) -> pd.DataFrame:
    """Load material-specific optimization result CSV with SiC legacy fallback."""

    material_path = paths.material_results_dir(material_name) / filename
    if material_path.exists():
        return pd.read_csv(material_path)

    if material_name.lower() == "sic":
        fallback = {
            "part3_optuna_trials.csv": paths.OPTUNA_TRIALS_PATH,
            "part3_optuna_pareto_frontier.csv": paths.PARETO_FRONTIER_PATH,
        }.get(filename)
        if fallback is not None and fallback.exists():
            return pd.read_csv(fallback)
    return pd.DataFrame()


def render_header(material_name: str) -> None:
    """Render the top dashboard header."""

    st.markdown(
        f"""
        <div class="hero-panel">
            <p class="soft-label">RRAM Material Intelligence</p>
            <h1 class="hero-title">{material_name} Process Explorer</h1>
            <p class="hero-subtitle">
                Gaussian Process prediction, constrained optimization, and uncertainty-aware recipe review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, MaterialConfig, dict[str, float]]:
    """Render material switcher and config-driven feature controls."""

    st.sidebar.markdown("### Material")
    material_name = st.sidebar.selectbox(
        "Material",
        available_materials(),
        index=available_materials().index("SiC") if "SiC" in available_materials() else 0,
        key="selected_material",
    )
    reset_state_on_material_change(material_name)

    config = get_material_config(material_name)
    st.sidebar.markdown("### Recipe")
    features = build_dynamic_feature_controls(material_name, config)
    st.sidebar.divider()
    st.sidebar.markdown("### Feature Vector")
    st.sidebar.dataframe(pd.DataFrame([features]), hide_index=True, width="stretch")
    return material_name, config, features


def build_dynamic_feature_controls(material_name: str, config: MaterialConfig) -> dict[str, float]:
    """Build controls from ``MaterialConfig.search_space``."""

    values: dict[str, float] = {}
    for feature_name, spec in config.search_space.items():
        values[feature_name] = render_feature_widget(material_name, feature_name, spec, config)

    for feature_name, default_value in config.default_feature_values.items():
        values.setdefault(feature_name, float(default_value))

    if "Has_RTA" in config.feature_columns:
        rta_value = values.get("RTA_Temperature_C", config.no_rta_temperature_c)
        values["Has_RTA"] = 0.0 if float(rta_value) == config.no_rta_temperature_c else 1.0

    for feature_name in config.feature_columns:
        if feature_name in values:
            continue
        if feature_name == "Has_RTA":
            values[feature_name] = 0.0
            continue
        values[feature_name] = st.sidebar.number_input(
            feature_label(feature_name),
            value=float(config.default_feature_values.get(feature_name, 0.0)),
            key=feature_key(material_name, feature_name),
        )

    return {feature_name: float(values[feature_name]) for feature_name in config.feature_columns}


def render_feature_widget(
    material_name: str,
    feature_name: str,
    spec: SearchSpaceParameter,
    config: MaterialConfig,
) -> float:
    """Render one feature widget based on its search-space specification."""

    key = feature_key(material_name, feature_name)
    label = feature_label(feature_name)
    param_type = spec.param_type.lower()

    if param_type == "categorical":
        choices = list(spec.choices)
        if not choices:
            choices = [config.default_feature_values.get(feature_name, 0.0)]
        display = [format_choice(feature_name, choice, config) for choice in choices]
        selected_label = st.sidebar.selectbox(label, display, key=key)
        return float(choices[display.index(selected_label)])

    if param_type in {"float", "int"}:
        low = float(0.0 if spec.low is None else spec.low)
        high = float(low if spec.high is None else spec.high)
        step = float(1.0 if spec.step is None else spec.step)
        default = float(config.default_feature_values.get(feature_name, low))
        default = min(max(default, low), high)
        value = st.sidebar.slider(
            label,
            min_value=low,
            max_value=high,
            value=default,
            step=step,
            key=key,
        )
        return float(value)

    return float(
        st.sidebar.number_input(
            label,
            value=float(config.default_feature_values.get(feature_name, 0.0)),
            key=key,
        )
    )


def feature_label(feature_name: str) -> str:
    """Return a human-friendly feature label."""

    unit = UNITS.get(feature_name)
    label = DISPLAY_NAMES.get(feature_name, feature_name)
    return f"{label} ({unit})" if unit else label


def format_choice(feature_name: str, choice: Any, config: MaterialConfig) -> str:
    """Format categorical choices for sidebar widgets."""

    if feature_name == "RTA_Temperature_C":
        value = float(choice)
        if value == config.no_rta_temperature_c:
            return "No RTA"
        return f"{value:g} °C"
    if feature_name == "Current_Compliance_A":
        return f"{float(choice):.3g} A"
    return f"{choice:g}" if isinstance(choice, (int, float)) else str(choice)


def predict_properties(material_name: str, features: Mapping[str, float]) -> tuple[dict[str, dict[str, float]], str | None]:
    """Predict material properties with a cached MaterialModelManager."""

    manager = load_model_manager(material_name)
    try:
        predictions = manager.predict(features)
    except FileNotFoundError as exc:
        return {}, str(exc)
    except ValueError as exc:
        return {}, str(exc)
    return predictions, None


def render_metrics(predictions: Mapping[str, Mapping[str, float]]) -> None:
    """Render prediction metrics."""

    visible_targets = [target for target in TARGET_ORDER if target in predictions]
    if not visible_targets:
        st.info("No trained model bundle is available for this material yet.")
        return

    columns = st.columns(min(len(visible_targets), 4))
    for index, target in enumerate(visible_targets[:4]):
        with columns[index]:
            prediction = predictions[target]
            st.metric(
                DISPLAY_NAMES.get(target, target),
                format_value(prediction["mean"], UNITS.get(target, "")),
                delta=f"95% CI {format_value(prediction['ci95_low'], UNITS.get(target, ''))} ~ "
                f"{format_value(prediction['ci95_high'], UNITS.get(target, ''))}",
            )


def render_recommended_recipes(material_name: str, config: MaterialConfig) -> None:
    """Render material-specific recommended recipe cards."""

    recipes = select_recommended_recipes(material_name, config)
    st.subheader("Recommended Recipes")

    if not recipes:
        st.info("Run the material pipeline to generate Pareto recipes.")
        return

    columns = st.columns(3)
    for index, recipe in enumerate(recipes):
        with columns[index]:
            st.markdown(recipe_card_html(recipe, config), unsafe_allow_html=True)
            if st.button("Apply Recipe", key=f"apply_recipe__{material_name}__{index}", width="stretch"):
                apply_recipe_to_session(material_name, config, recipe)
                st.rerun()


def select_recommended_recipes(material_name: str, config: MaterialConfig) -> list[dict[str, Any]]:
    """Select low-leakage, high-secondary, and balanced recipes."""

    pareto = normalize_result_table(load_results_csv(material_name, "part3_optuna_pareto_frontier.csv"), config)
    trials = normalize_result_table(load_results_csv(material_name, "part3_optuna_trials.csv"), config)
    table = pd.concat([pareto, trials], ignore_index=True).drop_duplicates()

    required_objective_cols = {"Leakage", "Secondary"}
    if table.empty or not required_objective_cols.issubset(table.columns):
        return []

    table = table.dropna(subset=["Leakage", "Secondary"])
    if table.empty:
        return []

    feature_subset = [feature for feature in config.feature_columns if feature in table.columns]
    if feature_subset:
        table = table.drop_duplicates(subset=feature_subset, keep="first")

    log_leakage = np.log10(table["Leakage"].clip(lower=1e-15))
    leakage_score = 1.0 - normalize_series(log_leakage)
    secondary_score = normalize_series(table["Secondary"].clip(lower=0.0))
    table = table.assign(Balanced_Score=0.5 * leakage_score + 0.5 * secondary_score)

    picks = [
        ("Ultra-Low Leakage", "Retention-first", table.loc[table["Leakage"].idxmin()]),
        ("High Output Window", "Maximize endurance or On/Off", table.loc[table["Secondary"].idxmax()]),
        ("Balanced Sweet Spot", "Stable compromise", table.loc[table["Balanced_Score"].idxmax()]),
    ]

    recipes: list[dict[str, Any]] = []
    for title, subtitle, row in picks:
        recipe = {feature: float(row[feature]) for feature in config.feature_columns if feature in row}
        recipe.update(
            {
                "title": title,
                "subtitle": subtitle,
                "Leakage": float(row["Leakage"]),
                "Secondary": float(row["Secondary"]),
                "Secondary_Target": str(row.get("Secondary_Target", "Secondary")),
            }
        )
        recipes.append(recipe)
    return recipes


def normalize_result_table(frame: pd.DataFrame, config: MaterialConfig) -> pd.DataFrame:
    """Normalize Optuna trials or Pareto CSV into a recipe table.

    Optuna CSV schemas have changed across phases. This parser accepts explicit
    objective names, real target names such as ``Leakage_Current_A``, GPR mean
    columns, and Optuna dataframe fallbacks such as ``values_0`` / ``values_1``.
    """

    if frame.empty:
        return empty_recipe_table(config)

    output = pd.DataFrame(index=frame.index)
    for feature in config.feature_columns:
        source_col = first_existing_column(frame, [feature, legacy_feature_alias(feature)])
        if source_col is not None:
            output[feature] = pd.to_numeric(frame[source_col], errors="coerce")

    if "Has_RTA" in config.feature_columns and "Has_RTA" not in output and "RTA_Temperature_C" in output:
        output["Has_RTA"] = np.where(output["RTA_Temperature_C"].eq(config.no_rta_temperature_c), 0.0, 1.0)

    for feature, value in config.default_feature_values.items():
        if feature not in output:
            output[feature] = float(value)

    leakage_col = first_existing_column(frame, leakage_column_candidates(config))
    secondary_col = first_existing_column(frame, secondary_column_candidates(config))
    secondary_target_col = first_existing_column(frame, ["Secondary_Objective_Target"])
    if leakage_col is None or secondary_col is None:
        return empty_recipe_table(config)

    output["Leakage"] = pd.to_numeric(frame[leakage_col], errors="coerce")
    output["Secondary"] = pd.to_numeric(frame[secondary_col], errors="coerce")
    if secondary_target_col:
        output["Secondary_Target"] = frame[secondary_target_col].fillna(infer_secondary_target_name(secondary_col))
    else:
        output["Secondary_Target"] = infer_secondary_target_name(secondary_col)

    if "Is_Feasible" in frame.columns:
        feasible = frame["Is_Feasible"].astype(str).str.lower().isin(["true", "1", "yes"])
        output = output.loc[feasible]

    return output.dropna(subset=[feature for feature in config.feature_columns if feature in output])


def empty_recipe_table(config: MaterialConfig) -> pd.DataFrame:
    """Return an empty normalized recipe table with stable columns."""

    return pd.DataFrame(columns=[*config.feature_columns, "Leakage", "Secondary", "Secondary_Target"])


def leakage_column_candidates(config: MaterialConfig) -> list[str]:
    """Return robust leakage objective column candidates."""

    target = "Leakage_Current_A"
    candidates = [
        "Leakage",
        "Leakage_Objective_Minimize",
        target,
        f"{target}_Mean",
        f"{target}_mean",
        f"{target}_median",
        f"{target}_Objective_Minimize",
        f"{target}_Pred",
        f"{target}_Prediction",
        "values_0",
        "value_0",
        "Value_0",
        "objective_0",
        "Objective_0",
    ]
    if target not in config.target_columns:
        candidates.append(target)
    return candidates


def secondary_column_candidates(config: MaterialConfig) -> list[str]:
    """Return robust secondary objective column candidates."""

    candidates = ["Secondary", "Secondary_Objective_Maximize", "Endurance_Objective_Maximize"]
    for target in ("Endurance_Cycles", "On_Off_Ratio", "Operation_Voltage_V"):
        candidates.extend(
            [
                target,
                f"{target}_Mean",
                f"{target}_mean",
                f"{target}_median",
                f"{target}_Objective_Maximize",
                f"{target}_Pred",
                f"{target}_Prediction",
            ]
        )
    candidates.extend(["values_1", "value_1", "Value_1", "objective_1", "Objective_1"])
    return candidates


def infer_secondary_target_name(column_name: str) -> str:
    """Infer a readable secondary target label from a CSV column name."""

    if "Endurance" in column_name:
        return "Endurance_Cycles"
    if "On_Off" in column_name or "OnOff" in column_name:
        return "On_Off_Ratio"
    if "Operation_Voltage" in column_name:
        return "Operation_Voltage_V"
    if column_name in {"values_1", "value_1", "Value_1", "objective_1", "Objective_1"}:
        return "Objective_1"
    return column_name


def legacy_feature_alias(feature_name: str) -> str:
    """Return old CSV aliases for process feature columns."""

    aliases = {
        "RF_Power_W": "RF_Power",
        "Process_Time_Min": "Process_Time",
        "RTA_Temperature_C": "RTA_Temperature",
    }
    return aliases.get(feature_name, feature_name)


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first existing column from candidates."""

    return next((column for column in candidates if column in frame.columns), None)


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to 0..1 with constant-series protection."""

    minimum = float(series.min())
    maximum = float(series.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or abs(maximum - minimum) < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - minimum) / (maximum - minimum)


def recipe_card_html(recipe: Mapping[str, Any], config: MaterialConfig) -> str:
    """Return one recipe card as HTML."""

    rows = []
    for feature in config.feature_columns:
        if feature == "Has_RTA" or feature not in recipe:
            continue
        rows.append(
            f"<span>{DISPLAY_NAMES.get(feature, feature)}</span>"
            f"<strong>{format_value(float(recipe[feature]), UNITS.get(feature, ''))}</strong>"
        )
    rows.append("<span>Leakage</span><strong>" + format_value(float(recipe["Leakage"]), "A") + "</strong>")
    secondary_label = DISPLAY_NAMES.get(str(recipe.get("Secondary_Target", "Secondary")), str(recipe.get("Secondary_Target", "Secondary")))
    rows.append(
        f"<span>{secondary_label}</span>"
        f"<strong>{format_value(float(recipe['Secondary']), '')}</strong>"
    )
    return f"""
    <div class="recipe-card">
        <div class="recipe-eyebrow">{recipe['subtitle']}</div>
        <div class="recipe-title">{recipe['title']}</div>
        <div class="recipe-text">Review this recipe against the model uncertainty before committing a wafer run.</div>
        <div class="recipe-kv">{''.join(rows)}</div>
    </div>
    """


def apply_recipe_to_session(material_name: str, config: MaterialConfig, recipe: Mapping[str, Any]) -> None:
    """Apply a recipe to sidebar widgets using material-scoped keys."""

    for feature in config.search_space:
        if feature not in recipe:
            continue
        key = feature_key(material_name, feature)
        st.session_state[key] = coerce_recipe_widget_value(recipe[feature], config.search_space[feature], feature, config)


def coerce_recipe_widget_value(
    value: Any,
    spec: SearchSpaceParameter,
    feature_name: str,
    config: MaterialConfig,
) -> Any:
    """Convert recipe values to the exact widget representation."""

    if spec.param_type.lower() == "categorical":
        choices = list(spec.choices)
        if not choices:
            return value
        nearest = min(choices, key=lambda choice: abs(float(choice) - float(value)))
        return format_choice(feature_name, nearest, config)

    numeric_value = float(value)
    if spec.low is not None:
        numeric_value = max(numeric_value, float(spec.low))
    if spec.high is not None:
        numeric_value = min(numeric_value, float(spec.high))
    if spec.step:
        low = float(0.0 if spec.low is None else spec.low)
        step = float(spec.step)
        numeric_value = round((numeric_value - low) / step) * step + low
    return float(numeric_value)


def render_pareto_plot(material_name: str, predictions: Mapping[str, Mapping[str, float]]) -> None:
    """Render trials, Pareto frontier, and current prediction with dashed trend line."""

    trials = normalize_result_table(load_results_csv(material_name, "part3_optuna_trials.csv"), get_material_config(material_name))
    pareto = normalize_result_table(load_results_csv(material_name, "part3_optuna_pareto_frontier.csv"), get_material_config(material_name))

    if trials.empty and pareto.empty:
        st.info("No optimization results are available for this material yet.")
        return

    secondary_target, secondary_value = current_secondary_objective(predictions)
    current_leakage = predictions.get("Leakage_Current_A", {}).get("mean", np.nan)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    if not trials.empty:
        ax.scatter(trials["Secondary"], trials["Leakage"], color="#8e8e93", alpha=0.28, label="Trials", s=34)
    if not pareto.empty:
        pareto_sorted = pareto.sort_values("Secondary")
        ax.scatter(pareto_sorted["Secondary"], pareto_sorted["Leakage"], color="#ff3b30", alpha=0.92, label="Pareto", s=52)
        ax.plot(
            pareto_sorted["Secondary"],
            pareto_sorted["Leakage"],
            color="#ff3b30",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label="Pareto trend",
        )

    if np.isfinite(current_leakage) and np.isfinite(secondary_value):
        ax.scatter(
            [secondary_value],
            [current_leakage],
            color="#007aff",
            edgecolor="#ffffff",
            linewidth=1.2,
            s=110,
            zorder=5,
            label="Current recipe",
        )

    if (pd.concat([trials.get("Leakage", pd.Series(dtype=float)), pareto.get("Leakage", pd.Series(dtype=float))]) > 0).any():
        ax.set_yscale("log")
    ax.set_xlabel(secondary_target)
    ax.set_ylabel("Leakage Current (A)")
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def current_secondary_objective(predictions: Mapping[str, Mapping[str, float]]) -> tuple[str, float]:
    """Return the secondary objective used for current recipe plotting."""

    if "Endurance_Cycles" in predictions:
        return "Endurance Cycles", predictions["Endurance_Cycles"]["mean"]
    if "On_Off_Ratio" in predictions:
        return "On/Off Ratio", predictions["On_Off_Ratio"]["mean"]
    return "Secondary Objective", np.nan


def render_expert_agent(material_name: str, features: Mapping[str, float], predictions: Mapping[str, Mapping[str, float]]) -> None:
    """Render LLM expert-agent placeholder diagnosis."""

    st.subheader("Semiconductor Expert Agent")
    if not predictions:
        st.info("Train material models before requesting a diagnosis.")
        return
    st.markdown(generate_expert_diagnosis(material_name, features, predictions))


def generate_expert_diagnosis(
    material_name: str,
    features: Mapping[str, float],
    predictions: Mapping[str, Mapping[str, float]],
) -> str:
    """Build a compact mock expert diagnosis and API-ready prompt preview."""

    leakage = predictions.get("Leakage_Current_A", {}).get("mean", np.nan)
    on_off = predictions.get("On_Off_Ratio", {}).get("mean", np.nan)
    endurance = predictions.get("Endurance_Cycles", {}).get("mean", np.nan)
    rta = features.get("RTA_Temperature_C", np.nan)

    mock = (
        f"For {material_name}, the selected recipe is predicted to operate with leakage near "
        f"{format_value(leakage, 'A')} and an On/Off ratio near {format_value(on_off, '')}. "
        f"RTA at {format_value(rta, '°C')} should be interpreted together with the uncertainty interval; "
        "run a confirmation sweep around the closest Pareto recipe before locking the process window."
    )
    if np.isfinite(endurance):
        mock += f" The endurance estimate is {format_value(endurance, 'cycles')}, so cycling validation remains essential."

    system_prompt = (
        "You are a semiconductor materials expert specializing in RRAM thin films. "
        "Explain the physical implication of the recipe and recommend the next experiment in about 100 Chinese words."
    )
    user_prompt = f"Material: {material_name}\nFeatures: {dict(features)}\nPredictions: {dict(predictions)}"
    return f"{mock}\n\n```text\nSYSTEM: {system_prompt}\nUSER: {user_prompt}\n```"


def format_value(value: float, unit: str = "") -> str:
    """Format values for cards and metrics."""

    if value is None or not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        text = f"{value:.3e}"
    else:
        text = f"{value:.4g}"
    return f"{text} {unit}".strip()


def main() -> None:
    """Run the Streamlit app."""

    configure_page()
    inject_apple_style_css()

    material_name, config, features = render_sidebar()
    render_header(material_name)

    predictions, error_message = predict_properties(material_name, features)
    if error_message:
        st.warning(error_message)

    render_recommended_recipes(material_name, config)
    st.divider()

    st.subheader("Predicted Electrical Metrics")
    render_metrics(predictions)
    st.divider()

    st.subheader("Optimization Trade-off")
    render_pareto_plot(material_name, predictions)
    st.divider()

    render_expert_agent(material_name, features, predictions)

    with st.expander("Run Context"):
        st.dataframe(pd.DataFrame([features]), hide_index=True, width="stretch")
        st.code(str(paths.material_models_dir(material_name)))
        st.code(str(paths.material_results_dir(material_name)))


if __name__ == "__main__":
    main()
