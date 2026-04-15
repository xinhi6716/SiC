from __future__ import annotations

import itertools
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import paths


logger = logging.getLogger(__name__)
matplotlib.use("Agg")
from matplotlib import pyplot as plt


FEATURE_COLUMNS = ["RF_Power_W", "Process_Time_Min", "RTA_Temperature_C", "Has_RTA"]
CORE_TARGETS = ["Leakage_Current_A", "Endurance_Cycles"]


@dataclass
class PredictorBundle:
    """統一包裝 XAI 需要的預測函數與模型來源資訊。"""

    target: str
    mode: str
    predict_transformed: Callable[[pd.DataFrame], np.ndarray]
    model_object: object


@dataclass
class ModelExplainer:
    """針對 SiC RRAM GPR surrogate model 的 XAI 分析器。

    本類別的設計重點：
    - 嚴格透過 `src.paths` 取得資料與輸出路徑，避免回到舊版 `DATA/` hardcode。
    - 優先讀取 `outputs/models/` 中已訓練的 GPR joblib 模型，直接解釋主模型。
    - GPR 不支援 TreeSHAP；本研究只有 4 個製程特徵，因此採用「精確列舉式 KernelSHAP」。
      對 4 維特徵而言，所有 coalition 只需 2^4 組，計算量低且不依賴外部 `shap` 套件。
    - 若 GPR 模型尚未輸出，則訓練 RandomForestRegressor 作為 surrogate explainer。
      這個 fallback 僅供架構驗證，正式論文圖建議先跑 `main_pipeline.py` 產生 GPR 模型。
    """

    dataset_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: paths.FIGURES_DIR)
    feature_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
    targets: list[str] = field(default_factory=lambda: CORE_TARGETS.copy())
    random_state: int = 42
    background_size: int = 24
    explanation_size: int = 64
    dpi: int = 300

    def __post_init__(self) -> None:
        paths.ensure_directories()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_: pd.DataFrame | None = None

    def load_dataset(self) -> pd.DataFrame:
        """讀取 condition-level dataset。

        半導體物理意義：
        - 每一列代表一組製程 recipe，例如 50 W / 60 min / No RTA。
        - XAI 解釋的是這些 recipe feature 如何影響 leakage 與 endurance。
        """

        candidates = [
            self.dataset_path,
            paths.CONDITION_LEVEL_DATA_PATH,
            paths.PROCESSED_DATA_DIR / "ml_condition_level_dataset.csv",
            paths.MATERIALS_DATA_DIR / "SiC" / "master_dataset.csv",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate = Path(candidate)
            if candidate.exists():
                dataset = pd.read_csv(candidate)
                self._validate_dataset(dataset, candidate)
                self.dataset_ = dataset
                logger.info("Loaded XAI dataset: %s", candidate)
                return dataset

        raise FileNotFoundError(
            "找不到可用的 condition-level dataset。請先執行 main_pipeline.py，"
            f"產生 {paths.CONDITION_LEVEL_DATA_PATH}。"
        )

    def _validate_dataset(self, dataset: pd.DataFrame, source_path: Path) -> None:
        """確認資料集含有必要 feature 與 target 欄位。"""

        missing_features = [column for column in self.feature_columns if column not in dataset.columns]
        missing_targets = [target for target in self.targets if target not in dataset.columns]

        if missing_features:
            raise ValueError(f"{source_path} 缺少 XAI feature 欄位：{missing_features}")
        if missing_targets:
            raise ValueError(f"{source_path} 缺少 XAI target 欄位：{missing_targets}")

    @staticmethod
    def _transform_target(target: str, values: np.ndarray | pd.Series) -> np.ndarray:
        """將 target 轉到較符合模型學習的尺度。

        - Leakage current 橫跨多個數量級，使用 log10 後才容易比較 RF/RTA 對漏電流的影響。
        - Endurance cycles 是非負計數，使用 log1p 可降低極端循環次數對 XAI 圖的主導。
        """

        array = np.asarray(values, dtype=float)
        if target == "Leakage_Current_A":
            return np.log10(np.clip(array, 1e-15, None))
        if target == "Endurance_Cycles":
            return np.log1p(np.clip(array, 0.0, None))
        return array

    @staticmethod
    def _target_axis_label(target: str) -> str:
        if target == "Leakage_Current_A":
            return "Model output: log10(Leakage Current / A)"
        if target == "Endurance_Cycles":
            return "Model output: log1p(Endurance Cycles)"
        return f"Model output: {target}"

    def _training_frame(self, target: str) -> pd.DataFrame:
        """取得某 target 可用的訓練資料列。"""

        dataset = self.dataset_ if self.dataset_ is not None else self.load_dataset()
        frame = dataset.dropna(subset=self.feature_columns + [target]).copy()
        if len(frame) < 3:
            raise ValueError(f"{target} 可用資料少於 3 筆，無法進行可靠 XAI 分析。")
        return frame

    def _load_gpr_predictor(self, target: str) -> PredictorBundle | None:
        """嘗試讀取 outputs/models/ 中的 GPR 模型。

        GPRModelTrainer.predict_with_uncertainty() 會回傳 transformed mean/std。
        XAI 在 transformed target space 上分析，較符合 leakage/endurance 的物理尺度。
        """

        model_path = paths.model_path_for_target(target)
        if not model_path.exists():
            return None

        model = joblib.load(model_path)

        def predict_transformed(x_frame: pd.DataFrame) -> np.ndarray:
            prediction = model.predict_with_uncertainty(x_frame[self.feature_columns])
            transformed_column = f"{target}_Transformed_Mean"
            original_column = f"{target}_Mean"
            if transformed_column in prediction.columns:
                return prediction[transformed_column].to_numpy(dtype=float)
            return self._transform_target(target, prediction[original_column].to_numpy(dtype=float))

        return PredictorBundle(
            target=target,
            mode="gpr",
            predict_transformed=predict_transformed,
            model_object=model,
        )

    def _train_rf_surrogate_predictor(self, target: str) -> PredictorBundle:
        """訓練 RandomForest surrogate explainer。

        選擇理由：
        - 正式解釋優先使用 GPR。
        - 若 GPR joblib 尚未存在，RandomForest 可快速建立非線性代理模型，讓 XAI pipeline 可先驗證。
        - RF 對小樣本的尺度敏感度較低，不需要額外 StandardScaler。
        """

        frame = self._training_frame(target)
        x_train = frame[self.feature_columns].copy()
        y_train = self._transform_target(target, frame[target])

        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=self.random_state,
            bootstrap=True,
        )
        rf_model.fit(x_train, y_train)

        def predict_transformed(x_frame: pd.DataFrame) -> np.ndarray:
            return rf_model.predict(x_frame[self.feature_columns])

        warnings.warn(
            f"{target} 的 GPR 模型不存在，已改用 RandomForest surrogate 進行 XAI。"
            "正式論文圖建議先執行 main_pipeline.py 產生 GPR 模型。",
            UserWarning,
        )
        return PredictorBundle(
            target=target,
            mode="random_forest_surrogate",
            predict_transformed=predict_transformed,
            model_object=rf_model,
        )

    def _get_predictor(self, target: str) -> PredictorBundle:
        """取得 target 的可解釋預測器。"""

        gpr_bundle = self._load_gpr_predictor(target)
        if gpr_bundle is not None:
            return gpr_bundle
        return self._train_rf_surrogate_predictor(target)

    def _sample_background(self, frame: pd.DataFrame) -> pd.DataFrame:
        """選擇 SHAP marginalization 背景資料。

        背景資料代表「實驗歷史分佈」。當某個 feature 被遮蔽時，KernelSHAP 會用背景資料
        模擬該 feature 在真實實驗中的可能取值，而不是填入任意常數。
        """

        x_frame = frame[self.feature_columns].reset_index(drop=True)
        if len(x_frame) <= self.background_size:
            return x_frame
        return x_frame.sample(n=self.background_size, random_state=self.random_state).reset_index(drop=True)

    def _sample_explanation_points(self, frame: pd.DataFrame) -> pd.DataFrame:
        """選擇要計算 SHAP 的 recipe 點。"""

        x_frame = frame[self.feature_columns].reset_index(drop=True)
        if len(x_frame) <= self.explanation_size:
            return x_frame
        return x_frame.sample(n=self.explanation_size, random_state=self.random_state).reset_index(drop=True)

    def _value_function(
        self,
        predictor: Callable[[pd.DataFrame], np.ndarray],
        x_row: pd.Series,
        present_feature_indices: tuple[int, ...],
        background: pd.DataFrame,
    ) -> float:
        """KernelSHAP value function v(S)。

        S 表示被保留的 feature set：
        - 在 S 內的 feature 使用欲解釋 recipe 的值。
        - 不在 S 內的 feature 使用背景資料分佈平均。

        這對應材料實驗中的反事實問題：
        「若固定 RF power，但讓其他製程條件回到歷史分佈，模型平均會預測什麼？」
        """

        synthetic = background.copy()
        for feature_index in present_feature_indices:
            feature_name = self.feature_columns[feature_index]
            synthetic[feature_name] = x_row[feature_name]
        return float(np.mean(predictor(synthetic[self.feature_columns])))

    def compute_exact_kernel_shap(self, target: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float, str]:
        """計算小樣本精確 KernelSHAP。

        傳統 KernelSHAP 會抽樣 coalition 並以加權線性模型估計 SHAP value。
        本研究只有四個製程特徵，所以可以完整列舉所有 coalition，直接用 Shapley 公式：

        phi_j = sum_S [ |S|! (M-|S|-1)! / M! ] * [ v(S union {j}) - v(S) ]

        這讓圖表解釋具備可重現性，適合論文結果與討論章節。
        """

        frame = self._training_frame(target)
        background = self._sample_background(frame)
        explain_x = self._sample_explanation_points(frame)
        bundle = self._get_predictor(target)

        n_features = len(self.feature_columns)
        factorial_m = math.factorial(n_features)
        shap_values = np.zeros((len(explain_x), n_features), dtype=float)
        base_values: list[float] = []

        feature_indices = tuple(range(n_features))
        coalition_cache: dict[tuple[int, tuple[int, ...]], float] = {}

        for row_id, (_, x_row) in enumerate(explain_x.iterrows()):
            base = self._value_function(bundle.predict_transformed, x_row, tuple(), background)
            base_values.append(base)

            for feature_index in feature_indices:
                remaining = [idx for idx in feature_indices if idx != feature_index]
                phi = 0.0

                for subset_size in range(n_features):
                    for subset in itertools.combinations(remaining, subset_size):
                        subset = tuple(sorted(subset))
                        subset_with_feature = tuple(sorted(subset + (feature_index,)))
                        weight = (
                            math.factorial(len(subset))
                            * math.factorial(n_features - len(subset) - 1)
                            / factorial_m
                        )

                        cache_key_without = (row_id, subset)
                        cache_key_with = (row_id, subset_with_feature)

                        if cache_key_without not in coalition_cache:
                            coalition_cache[cache_key_without] = self._value_function(
                                bundle.predict_transformed,
                                x_row,
                                subset,
                                background,
                            )
                        if cache_key_with not in coalition_cache:
                            coalition_cache[cache_key_with] = self._value_function(
                                bundle.predict_transformed,
                                x_row,
                                subset_with_feature,
                                background,
                            )

                        phi += weight * (coalition_cache[cache_key_with] - coalition_cache[cache_key_without])

                shap_values[row_id, feature_index] = phi

        return explain_x, background, shap_values, float(np.mean(base_values)), bundle.mode

    def plot_shap_summary(self, target: str) -> Path:
        """繪製 SHAP summary plot。

        圖的讀法：
        - bar plot 表示全局重要性 mean(|SHAP|)。
        - beeswarm-like scatter 表示每一筆 recipe 在該 feature 上對模型輸出的正負貢獻。
        - 顏色代表該 feature 的高低值，有助判斷例如高 RTA 是否推高 endurance。
        """

        explain_x, _, shap_values, base_value, model_mode = self.compute_exact_kernel_shap(target)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs)
        ordered_features = [self.feature_columns[index] for index in order]

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=self.dpi)
        fig.suptitle(f"SHAP Summary for {target} ({model_mode})", fontsize=14)

        axes[0].barh(range(len(order)), mean_abs[order], color="#2f6f9f", alpha=0.88)
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels(ordered_features, fontsize=10)
        axes[0].set_xlabel("mean(|SHAP value|)", fontsize=12)
        axes[0].set_title("Global Feature Importance", fontsize=12)
        axes[0].grid(axis="x", linestyle="--", alpha=0.3)

        rng = np.random.default_rng(self.random_state)
        scatter_handle = None
        for y_position, feature_index in enumerate(order):
            feature_name = self.feature_columns[feature_index]
            values = explain_x[feature_name].to_numpy(dtype=float)
            value_min = np.nanmin(values)
            value_max = np.nanmax(values)
            if np.isclose(value_min, value_max):
                colors = np.full_like(values, 0.5, dtype=float)
            else:
                colors = (values - value_min) / (value_max - value_min)
            jitter = rng.normal(loc=0.0, scale=0.055, size=len(values))
            scatter_handle = axes[1].scatter(
                shap_values[:, feature_index],
                np.full(len(values), y_position) + jitter,
                c=colors,
                cmap="coolwarm",
                s=30,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.86,
            )

        axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[1].set_yticks(range(len(order)))
        axes[1].set_yticklabels(ordered_features, fontsize=10)
        axes[1].set_xlabel(f"SHAP value on {self._target_axis_label(target)}", fontsize=12)
        axes[1].set_title("Recipe-Level Contribution", fontsize=12)
        axes[1].grid(axis="x", linestyle="--", alpha=0.3)

        if scatter_handle is not None:
            colorbar = fig.colorbar(scatter_handle, ax=axes[1], fraction=0.046, pad=0.04)
            colorbar.set_label("Feature value: low -> high", fontsize=10)

        fig.text(
            0.01,
            0.01,
            f"Baseline E[f(X)] = {base_value:.4g}; output shown in transformed target space.",
            fontsize=9,
        )
        plt.tight_layout(rect=(0, 0.04, 1, 0.94))

        output_path = self.output_dir / f"xai_shap_summary_{target}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP summary: %s", output_path)
        return output_path

    def _pdp_grid_for_feature(self, frame: pd.DataFrame, feature_name: str) -> np.ndarray:
        """依據製程特徵的物理意義建立 PDP 掃描網格。"""

        if feature_name == "RF_Power_W":
            return np.linspace(50.0, 75.0, 26)
        if feature_name == "Process_Time_Min":
            return np.linspace(30.0, 120.0, 31)
        if feature_name == "RTA_Temperature_C":
            return np.array([25.0, 400.0, 500.0])
        if feature_name == "Has_RTA":
            return np.array([0.0, 1.0])

        values = pd.to_numeric(frame[feature_name], errors="coerce").dropna()
        return np.linspace(float(values.min()), float(values.max()), 25)

    def _apply_physical_feature_setting(self, x_frame: pd.DataFrame, feature_name: str, value: float) -> pd.DataFrame:
        """在 PDP 掃描時維持 RTA_Temperature_C 與 Has_RTA 的物理一致性。"""

        adjusted = x_frame.copy()
        adjusted[feature_name] = value

        if feature_name == "RTA_Temperature_C":
            adjusted["Has_RTA"] = 0.0 if float(value) == 25.0 else 1.0
        elif feature_name == "Has_RTA":
            if float(value) == 0.0:
                adjusted["RTA_Temperature_C"] = 25.0
            else:
                rta_values = adjusted["RTA_Temperature_C"]
                non_room = rta_values[rta_values > 25.0]
                adjusted["RTA_Temperature_C"] = float(non_room.median()) if not non_room.empty else 400.0

        return adjusted

    def compute_partial_dependence(self, target: str, feature_name: str) -> pd.DataFrame:
        """計算單一 feature 的 partial dependence。

        PDP 的物理意義：
        - 固定某一個製程參數為掃描值。
        - 其他參數維持在歷史實驗 recipes。
        - 取平均預測，得到該參數對 leakage/endurance 的平均邊際影響。
        """

        frame = self._training_frame(target)
        x_base = frame[self.feature_columns].reset_index(drop=True)
        bundle = self._get_predictor(target)
        grid = self._pdp_grid_for_feature(frame, feature_name)

        rows: list[dict[str, float | str]] = []
        for value in grid:
            x_eval = self._apply_physical_feature_setting(x_base, feature_name, float(value))
            predictions = bundle.predict_transformed(x_eval[self.feature_columns])
            rows.append(
                {
                    "Target": target,
                    "Feature": feature_name,
                    "Feature_Value": float(value),
                    "Mean_Prediction": float(np.mean(predictions)),
                    "Std_Prediction": float(np.std(predictions)),
                    "Model_Mode": bundle.mode,
                }
            )
        return pd.DataFrame(rows)

    def plot_partial_dependence(self, target: str) -> Path:
        """針對單一 target 繪製 2x2 PDP 圖。"""

        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), dpi=self.dpi)
        axes = axes.ravel()
        fig.suptitle(f"Partial Dependence Analysis for {target}", fontsize=14)

        all_pdp: list[pd.DataFrame] = []
        for axis, feature_name in zip(axes, self.feature_columns):
            pdp = self.compute_partial_dependence(target, feature_name)
            all_pdp.append(pdp)

            axis.plot(
                pdp["Feature_Value"],
                pdp["Mean_Prediction"],
                color="#b33c2e",
                marker="o",
                linewidth=2.0,
                markersize=4.5,
            )
            axis.fill_between(
                pdp["Feature_Value"].to_numpy(dtype=float),
                (pdp["Mean_Prediction"] - pdp["Std_Prediction"]).to_numpy(dtype=float),
                (pdp["Mean_Prediction"] + pdp["Std_Prediction"]).to_numpy(dtype=float),
                color="#b33c2e",
                alpha=0.15,
                linewidth=0,
            )
            axis.set_title(feature_name, fontsize=12)
            axis.set_xlabel(feature_name, fontsize=11)
            axis.set_ylabel(self._target_axis_label(target), fontsize=11)
            axis.grid(True, linestyle="--", alpha=0.3)

            if feature_name == "RTA_Temperature_C":
                axis.set_xticks([25.0, 400.0, 500.0])
                axis.set_xticklabels(["No RTA\n25 C", "400 C", "500 C"], fontsize=9)
            elif feature_name == "Has_RTA":
                axis.set_xticks([0.0, 1.0])
                axis.set_xticklabels(["No RTA", "RTA"], fontsize=9)

        plt.tight_layout(rect=(0, 0, 1, 0.95))

        output_path = self.output_dir / f"xai_pdp_{target}.png"
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        pdp_table = pd.concat(all_pdp, ignore_index=True)
        pdp_table.to_csv(paths.RESULTS_DATA_DIR / f"xai_pdp_values_{target}.csv", index=False, encoding="utf-8-sig")
        logger.info("Saved PDP figure: %s", output_path)
        return output_path

    def generate_all(self) -> list[Path]:
        """一次產生 Leakage 與 Endurance 的 SHAP summary plot 與 PDP 圖。"""

        self.load_dataset()
        outputs: list[Path] = []
        for target in self.targets:
            outputs.append(self.plot_shap_summary(target))
            outputs.append(self.plot_partial_dependence(target))
        return outputs


def main() -> None:
    """CLI 入口：可直接執行 `python -m src.model_explainer`。"""

    explainer = ModelExplainer()
    outputs = explainer.generate_all()
    logger.info("Generated XAI figures:")
    for output in outputs:
        logger.info("  %s", output)


if __name__ == "__main__":
    main()
