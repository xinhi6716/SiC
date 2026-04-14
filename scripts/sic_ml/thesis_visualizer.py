from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "DATA"
FIGURE_DIR = PROJECT_ROOT / "FIGURES"

TRIALS_PATH = DATA_DIR / "part3_optuna_trials.csv"
PARETO_PATH = DATA_DIR / "part3_optuna_pareto_frontier.csv"


class ThesisVisualizer:
    """論文級圖表產生器。

    此模組專門將 Part 3 Optuna 最佳化結果轉成論文可用圖表。
    圖片輸出解析度固定為 300 DPI，字體大小依 A4 論文版面調整。
    """

    def __init__(
        self,
        trials_path: Path = TRIALS_PATH,
        pareto_path: Path = PARETO_PATH,
        output_dir: Path = FIGURE_DIR,
    ) -> None:
        self.trials_path = Path(trials_path)
        self.pareto_path = Path(pareto_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._configure_style()

    @staticmethod
    def _configure_style() -> None:
        """設定論文圖表風格。

        字體大小：
        - Title: 14 pt
        - Axis label: 12 pt
        - Tick / legend: 10 pt
        """
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "font.size": 10,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "axes.linewidth": 1.0,
            }
        )

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, required_columns: list[str], frame_name: str) -> None:
        missing = [col for col in required_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"{frame_name} 缺少必要欄位：{missing}")

    @staticmethod
    def _to_bool(series: pd.Series) -> pd.Series:
        """兼容 CSV 中 bool 可能被存成 True/False 字串的情況。"""
        if series.dtype == bool:
            return series
        return series.astype(str).str.lower().isin(["true", "1", "yes"])

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """讀取 Optuna trials 與 Pareto frontier 結果。"""
        if not self.trials_path.exists():
            raise FileNotFoundError(f"找不到 Optuna trials 檔案：{self.trials_path}")
        if not self.pareto_path.exists():
            raise FileNotFoundError(f"找不到 Pareto frontier 檔案：{self.pareto_path}")

        trials = pd.read_csv(self.trials_path)
        pareto = pd.read_csv(self.pareto_path)

        required = [
            "Endurance_Cycles_Mean",
            "Leakage_Current_A_Mean",
            "Constraint_OnOff_g1",
            "Constraint_Voltage_g2",
            "Is_Feasible",
            "RF_Power",
            "Process_Time",
            "RTA_Temperature",
        ]
        self._ensure_columns(trials, required, "part3_optuna_trials.csv")
        self._ensure_columns(pareto, [col for col in required if col != "Is_Feasible"], "part3_optuna_pareto_frontier.csv")

        trials = trials.copy()
        pareto = pareto.copy()
        trials["Is_Feasible"] = self._to_bool(trials["Is_Feasible"])
        for frame in [trials, pareto]:
            frame["Endurance_Cycles_Mean"] = pd.to_numeric(frame["Endurance_Cycles_Mean"], errors="coerce")
            frame["Leakage_Current_A_Mean"] = pd.to_numeric(frame["Leakage_Current_A_Mean"], errors="coerce")
            frame["Constraint_OnOff_g1"] = pd.to_numeric(frame["Constraint_OnOff_g1"], errors="coerce")
            frame["Constraint_Voltage_g2"] = pd.to_numeric(frame["Constraint_Voltage_g2"], errors="coerce")

        return trials, pareto

    @staticmethod
    def _filter_plot_domain(frame: pd.DataFrame) -> pd.DataFrame:
        """移除無法進入 log axis 的點。

        Optuna penalty 可能讓某些 objective 欄位變成負值；這裡採用 GPR mean 欄位，
        並只保留 Endurance > 0、Leakage > 0 的物理可畫點。
        """
        clean = frame.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["Endurance_Cycles_Mean", "Leakage_Current_A_Mean"]
        )
        clean = clean[(clean["Endurance_Cycles_Mean"] > 0) & (clean["Leakage_Current_A_Mean"] > 0)].copy()
        return clean

    @staticmethod
    def _should_use_log_x(frame: pd.DataFrame) -> bool:
        x = frame["Endurance_Cycles_Mean"].dropna()
        x = x[x > 0]
        if x.empty:
            return False
        return (x.max() / x.min()) >= 10

    def plot_pareto_frontier(self) -> Path:
        """Figure 1: 帕雷托前緣分析圖。

        X 軸：Endurance，越大越好。
        Y 軸：Leakage Current，使用 log scale，越小越好。
        灰色點：所有 Optuna trials。
        橘色叉號：被硬性約束剃除的點。
        紅色點與虛線：Pareto frontier / sweet spot。
        """
        trials, pareto = self.load_data()
        trials = self._filter_plot_domain(trials)
        pareto = self._filter_plot_domain(pareto)
        if trials.empty:
            raise ValueError("沒有可繪製的 Optuna trial 點。")

        feasible = trials[trials["Is_Feasible"]].copy()
        infeasible_onoff = trials[trials["Constraint_OnOff_g1"] > 0].copy()
        infeasible_voltage = trials[trials["Constraint_Voltage_g2"] > 0].copy()

        pareto_unique = (
            pareto.drop_duplicates(
                subset=["Endurance_Cycles_Mean", "Leakage_Current_A_Mean", "RF_Power", "Process_Time", "RTA_Temperature"]
            )
            .sort_values("Endurance_Cycles_Mean")
            .copy()
        )

        fig, ax = plt.subplots(figsize=(6.8, 4.8))

        ax.scatter(
            trials["Endurance_Cycles_Mean"],
            trials["Leakage_Current_A_Mean"],
            color="0.65",
            alpha=0.30,
            s=42,
            label="All Optuna trials",
            edgecolors="none",
        )

        if not feasible.empty:
            ax.scatter(
                feasible["Endurance_Cycles_Mean"],
                feasible["Leakage_Current_A_Mean"],
                color="#4C78A8",
                alpha=0.45,
                s=46,
                label="Feasible trials",
                edgecolors="white",
                linewidths=0.3,
            )

        if not infeasible_onoff.empty:
            ax.scatter(
                infeasible_onoff["Endurance_Cycles_Mean"],
                infeasible_onoff["Leakage_Current_A_Mean"],
                marker="x",
                color="#F58518",
                alpha=0.90,
                s=70,
                linewidths=1.4,
                label="Pruned: On/Off < 5",
            )

        if not infeasible_voltage.empty:
            ax.scatter(
                infeasible_voltage["Endurance_Cycles_Mean"],
                infeasible_voltage["Leakage_Current_A_Mean"],
                marker="^",
                color="#9467BD",
                alpha=0.80,
                s=60,
                label="Pruned: Vop > 3 V",
                edgecolors="white",
                linewidths=0.4,
            )

        if not pareto_unique.empty:
            ax.plot(
                pareto_unique["Endurance_Cycles_Mean"],
                pareto_unique["Leakage_Current_A_Mean"],
                linestyle="--",
                linewidth=1.5,
                color="#D62728",
                alpha=0.90,
                label="Pareto frontier",
            )
            ax.scatter(
                pareto_unique["Endurance_Cycles_Mean"],
                pareto_unique["Leakage_Current_A_Mean"],
                color="#D62728",
                s=90,
                marker="D",
                edgecolors="white",
                linewidths=0.7,
                label="Sweet spot candidates",
                zorder=5,
            )

            best = pareto_unique.sort_values(["Leakage_Current_A_Mean", "Endurance_Cycles_Mean"], ascending=[True, False]).iloc[0]
            ax.annotate(
                "Low-leakage\nPareto point",
                xy=(best["Endurance_Cycles_Mean"], best["Leakage_Current_A_Mean"]),
                xytext=(12, 18),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 1.0},
                fontsize=10,
            )

        ax.set_yscale("log")
        if self._should_use_log_x(trials):
            ax.set_xscale("log")

        ax.set_title("Pareto Frontier: Endurance vs. Leakage Current")
        ax.set_xlabel("Predicted Endurance (cycles, higher is better)")
        ax.set_ylabel("Predicted Leakage Current (A, lower is better)")
        ax.legend(frameon=True, loc="best")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
        fig.tight_layout()

        output_path = self.output_dir / "figure_1_pareto_frontier.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_sweet_spot_parallel_coordinates(self) -> Path:
        """Figure 2: Sweet Spot 參數分佈圖。

        使用 Pareto frontier 解的 RF Power、Process Time、RTA Temperature。
        以 min-max normalization 放在同一張平行座標圖中，便於觀察 AI 推薦參數是否集中。
        """
        _, pareto = self.load_data()
        pareto = self._filter_plot_domain(pareto)
        if pareto.empty:
            raise ValueError("Pareto frontier 沒有可繪製的點。")

        param_cols = ["RF_Power", "Process_Time", "RTA_Temperature"]
        pareto_params = pareto[param_cols + ["Leakage_Current_A_Mean", "Endurance_Cycles_Mean"]].copy()
        pareto_params = pareto_params.dropna(subset=param_cols).reset_index(drop=True)
        pareto_params["Recipe_Count"] = 1
        recipe_summary = (
            pareto_params.groupby(param_cols, dropna=False)
            .agg(
                Recipe_Count=("Recipe_Count", "sum"),
                Leakage_Current_A_Mean=("Leakage_Current_A_Mean", "median"),
                Endurance_Cycles_Mean=("Endurance_Cycles_Mean", "median"),
            )
            .reset_index()
            .sort_values("Recipe_Count", ascending=False)
        )

        normalized = recipe_summary.copy()
        for col in param_cols:
            col_min = normalized[col].min()
            col_max = normalized[col].max()
            if col_max == col_min:
                normalized[f"{col}_Norm"] = 0.5
            else:
                normalized[f"{col}_Norm"] = (normalized[col] - col_min) / (col_max - col_min)

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), gridspec_kw={"width_ratios": [1.4, 1.0]})
        ax_parallel, ax_count = axes

        x_positions = np.arange(len(param_cols))
        cmap = sns.color_palette("crest", n_colors=max(len(normalized), 3))

        for idx, row in normalized.reset_index(drop=True).iterrows():
            y_values = [row[f"{col}_Norm"] for col in param_cols]
            label = f"{int(row['RF_Power'])}W / {int(row['Process_Time'])}min / {int(row['RTA_Temperature'])}C"
            ax_parallel.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=1.8,
                alpha=0.85,
                color=cmap[idx % len(cmap)],
                label=label,
            )

        ax_parallel.set_title("Pareto Sweet Spot Parameter Pattern")
        ax_parallel.set_xticks(x_positions)
        ax_parallel.set_xticklabels(["RF Power", "Process Time", "RTA Temp."])
        ax_parallel.set_ylabel("Normalized parameter value")
        ax_parallel.set_ylim(-0.05, 1.05)
        ax_parallel.grid(True, axis="y", linestyle=":", alpha=0.7)
        ax_parallel.legend(frameon=True, fontsize=8, loc="best")

        recipe_summary["Recipe_Label"] = recipe_summary.apply(
            lambda row: f"{int(row['RF_Power'])}W\n{int(row['Process_Time'])}min\n{int(row['RTA_Temperature'])}C",
            axis=1,
        )
        sns.barplot(
            data=recipe_summary,
            x="Recipe_Count",
            y="Recipe_Label",
            ax=ax_count,
            color="#4C78A8",
        )
        ax_count.set_title("Pareto Recipe Frequency")
        ax_count.set_xlabel("Count in Pareto set")
        ax_count.set_ylabel("")
        ax_count.grid(True, axis="x", linestyle=":", alpha=0.7)

        fig.tight_layout()
        output_path = self.output_dir / "figure_2_sweet_spot_parameters.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def generate_all_figures(self) -> list[Path]:
        """一次產生所有 thesis figures。"""
        return [
            self.plot_pareto_frontier(),
            self.plot_sweet_spot_parallel_coordinates(),
        ]


if __name__ == "__main__":
    visualizer = ThesisVisualizer()
    outputs = visualizer.generate_all_figures()
    print("Generated thesis figures:")
    for path in outputs:
        print(path)
