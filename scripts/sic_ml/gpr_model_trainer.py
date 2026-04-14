from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from sic_ml.data_processor import DataProcessor


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class GPRModelTrainer:
    """Part 2: 高斯過程迴歸訓練模組。

    GPR 的材料科學意義：
    - 小樣本實驗下，我們不只需要一個預測值，還需要知道模型有多不確定。
    - Gaussian Process 會將 y(x*) 視為 posterior distribution：
      y(x*) | D ~ Normal(mean(x*), std(x*)^2)
    - 因此 predict_with_uncertainty 可以回傳 mean、std、confidence interval。
    """

    feature_columns: tuple[str, ...] = tuple(DataProcessor.FEATURE_COLUMNS)
    target_column: str = "Leakage_Current_A"
    target_transform: str = "log10"
    kernel_type: str = "matern"
    random_state: int = 42
    n_restarts_optimizer: int = 8
    alpha: float = 1e-2

    def __post_init__(self) -> None:
        if self.target_column == "Leakage_Current_A":
            # 漏電流跨越多個數量級，強制使用 log10 空間訓練，避免原尺度極端值拉扯 GPR。
            self.target_transform = "log10"
        self.model_: Pipeline | None = None

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        """將物理量轉成 GPR 較容易學習的 target 空間。"""
        y = np.asarray(y, dtype=float)
        if self.target_transform == "log10":
            return np.log10(np.clip(y, 1e-15, None))
        if self.target_transform == "log1p":
            return np.log1p(np.clip(y, 0.0, None))
        if self.target_transform == "none":
            return y
        raise ValueError(f"不支援的 target_transform：{self.target_transform}")

    def _inverse_transform_y(self, y_transformed: np.ndarray) -> np.ndarray:
        """將 GPR 模型空間預測值轉回原始物理尺度。"""
        y_transformed = np.asarray(y_transformed, dtype=float)
        if self.target_transform == "log10":
            return np.power(10.0, y_transformed)
        if self.target_transform == "log1p":
            return np.expm1(y_transformed)
        if self.target_transform == "none":
            return y_transformed
        raise ValueError(f"不支援的 target_transform：{self.target_transform}")

    def _build_kernel(self, n_features: int):
        """建立 GPR kernel。

        Matern kernel 比 RBF 更適合材料實驗：
        - RBF 假設函數非常平滑。
        - Matern 允許較不平滑的局部變化，較符合製程條件造成缺陷態突變的情境。
        """
        if self.kernel_type == "matern":
            base_kernel = Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        elif self.kernel_type == "rbf":
            base_kernel = RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
        else:
            raise ValueError(f"不支援的 kernel_type：{self.kernel_type}")

        return ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(
            noise_level=1e-2,
            noise_level_bounds=(1e-5, 1e1),
        )

    def _build_pipeline(self, n_features: int) -> Pipeline:
        """建立包含 StandardScaler 與 GPR 的 Pipeline。

        注意：
        Cross-validation 時 scaler 必須只在 training fold fit，
        否則 test fold 的尺度資訊會洩漏進訓練流程。
        """
        gpr = GaussianProcessRegressor(
            kernel=self._build_kernel(n_features),
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            alpha=self.alpha,
        )
        return Pipeline(steps=[("scaler", StandardScaler()), ("gpr", gpr)])

    def _validate_training_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """檢查訓練資料是否具備完整特徵與目標欄位。"""
        missing_features = [col for col in self.feature_columns if col not in dataset.columns]
        if missing_features:
            raise ValueError(f"dataset 缺少特徵欄位：{missing_features}")
        if self.target_column not in dataset.columns:
            raise ValueError(f"dataset 缺少 target 欄位：{self.target_column}")

        frame = dataset.dropna(subset=list(self.feature_columns) + [self.target_column]).copy()
        if len(frame) < 3:
            raise ValueError(f"{self.target_column} 可訓練樣本少於 3 筆，GPR 無法穩定訓練。")
        return frame

    def fit(self, dataset: pd.DataFrame) -> "GPRModelTrainer":
        """訓練 GPR 模型。"""
        frame = self._validate_training_frame(dataset)
        X = frame[list(self.feature_columns)]
        y = self._transform_y(frame[self.target_column].to_numpy(dtype=float))

        self.model_ = self._build_pipeline(n_features=X.shape[1])
        self.model_.fit(X, y)
        return self

    def predict_with_uncertainty(self, X: pd.DataFrame, ci_z: float = 1.96) -> pd.DataFrame:
        """回傳 mean、std 與 95% confidence interval。

        對 log target 而言，CI 先在 log space 計算，再轉回原尺度。
        這符合 leakage current / on-off ratio 常呈 log-normal 分佈的假設。
        """
        if self.model_ is None:
            raise RuntimeError("模型尚未 fit，請先呼叫 fit(dataset)。")

        missing_features = [col for col in self.feature_columns if col not in X.columns]
        if missing_features:
            raise ValueError(f"X 缺少特徵欄位：{missing_features}")

        X_eval = X[list(self.feature_columns)].copy()
        scaler = self.model_.named_steps["scaler"]
        gpr = self.model_.named_steps["gpr"]
        X_scaled = scaler.transform(X_eval)
        mean_t, std_t = gpr.predict(X_scaled, return_std=True)

        low_t = mean_t - ci_z * std_t
        high_t = mean_t + ci_z * std_t
        mean_original = self._inverse_transform_y(mean_t)
        low_original = self._inverse_transform_y(low_t)
        high_original = self._inverse_transform_y(high_t)
        std_original = (high_original - low_original) / (2.0 * ci_z)

        return pd.DataFrame(
            {
                f"{self.target_column}_Mean": mean_original,
                f"{self.target_column}_Std": std_original,
                f"{self.target_column}_CI95_Low": low_original,
                f"{self.target_column}_CI95_High": high_original,
                f"{self.target_column}_Transformed_Mean": mean_t,
                f"{self.target_column}_Transformed_Std": std_t,
            },
            index=X.index,
        )

    def cross_validate(self, dataset: pd.DataFrame, n_splits: int | None = None) -> pd.DataFrame:
        """以 CV 評估 GPR，輸出 R2 與 MSE。

        小樣本材料資料建議：
        - 若樣本數 <= 8，使用 LOOCV 最大化訓練資料使用率。
        - 若樣本數較多，使用 KFold 並 shuffle，降低 fold 偶然性。
        """
        frame = self._validate_training_frame(dataset)
        X = frame[list(self.feature_columns)].reset_index(drop=True)
        y_original = frame[self.target_column].to_numpy(dtype=float)
        y_transformed = self._transform_y(y_original)

        # 本研究的 condition-level 樣本目前只有十幾筆。固定使用 LOOCV，
        # 讓每一回合都保留 N-1 筆作為訓練集，最大化小樣本資料利用率。
        splitter = LeaveOneOut()
        cv_name = "LOOCV"

        rows: list[dict[str, float | str]] = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            model = self._build_pipeline(n_features=X.shape[1])
            model.fit(X.iloc[train_idx], y_transformed[train_idx])

            scaler = model.named_steps["scaler"]
            gpr = model.named_steps["gpr"]
            X_test_scaled = scaler.transform(X.iloc[test_idx])
            pred_t, std_t = gpr.predict(X_test_scaled, return_std=True)
            pred_original = self._inverse_transform_y(pred_t)

            for local_idx, pred_value, pred_value_t, pred_std_t in zip(test_idx, pred_original, pred_t, std_t):
                rows.append(
                    {
                        "CV_Type": cv_name,
                        "Fold": fold_id,
                        "Target": self.target_column,
                        "Y_True": y_original[local_idx],
                        "Y_Pred": float(pred_value),
                        "Y_True_Transformed": y_transformed[local_idx],
                        "Y_Pred_Transformed": float(pred_value_t),
                        "Predictive_Std_Transformed": float(pred_std_t),
                    }
                )

        predictions = pd.DataFrame(rows)
        y_true = predictions["Y_True"].to_numpy(dtype=float)
        y_pred = predictions["Y_Pred"].to_numpy(dtype=float)
        y_true_t = predictions["Y_True_Transformed"].to_numpy(dtype=float)
        y_pred_t = predictions["Y_Pred_Transformed"].to_numpy(dtype=float)
        predictions.attrs["R2_Original_Scale"] = r2_score(y_true, y_pred) if len(predictions) >= 2 else np.nan
        predictions.attrs["MSE_Original_Scale"] = mean_squared_error(y_true, y_pred)
        predictions.attrs["R2_Model_Space"] = r2_score(y_true_t, y_pred_t) if len(predictions) >= 2 else np.nan
        predictions.attrs["MSE_Model_Space"] = mean_squared_error(y_true_t, y_pred_t)
        predictions.attrs["R2"] = predictions.attrs["R2_Model_Space"]
        predictions.attrs["MSE"] = predictions.attrs["MSE_Model_Space"]
        return predictions

    @staticmethod
    def summarize_cv(cv_predictions: pd.DataFrame) -> pd.DataFrame:
        """將 cross_validate 結果整理成研究人員易讀的指標表。"""
        if cv_predictions.empty:
            raise ValueError("CV predictions 是空表，無法計算指標。")
        return pd.DataFrame(
            [
                {
                    "Target": cv_predictions["Target"].iloc[0],
                    "CV_Type": cv_predictions["CV_Type"].iloc[0],
                    "N_Predictions": len(cv_predictions),
                    "R2_Model_Space": cv_predictions.attrs.get("R2_Model_Space", np.nan),
                    "MSE_Model_Space": cv_predictions.attrs.get("MSE_Model_Space", np.nan),
                    "R2_Original_Scale": cv_predictions.attrs.get("R2_Original_Scale", np.nan),
                    "MSE_Original_Scale": cv_predictions.attrs.get("MSE_Original_Scale", np.nan),
                }
            ]
        )


if __name__ == "__main__":
    processor = DataProcessor()
    dataset = processor.build_condition_level_dataset()

    # 範例：訓練 Leakage Current 的 GPR。Leakage 跨數量級，因此 target_transform 使用 log10。
    trainer = GPRModelTrainer(target_column="Leakage_Current_A", target_transform="log10", kernel_type="matern")
    trainer.fit(dataset)

    prediction = trainer.predict_with_uncertainty(dataset[DataProcessor.FEATURE_COLUMNS].head())
    cv_predictions = trainer.cross_validate(dataset)
    cv_summary = trainer.summarize_cv(cv_predictions)

    print("Prediction with uncertainty preview:")
    print(prediction.to_string(index=False))
    print("\nCV summary:")
    print(cv_summary.to_string(index=False))
