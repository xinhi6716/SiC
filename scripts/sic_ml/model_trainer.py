from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CLASSIFICATION_TARGETS, FEATURE_COLUMNS, LEAKAGE_EPSILON_A, MIN_TRAINING_ROWS, RANDOM_STATE, REGRESSION_TARGETS


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _transform_target(target: str, values: np.ndarray) -> np.ndarray:
    """將原始物理量轉到較接近高斯的模型空間。

    - Leakage Current 與 On/Off Ratio 跨數量級，使用 log10。
    - Endurance 是 count-like target，使用 log1p。
    - 電壓類 target 保留原尺度，避免過度扭曲物理可解釋性。
    """
    values = np.asarray(values, dtype=float)
    if target == "leakage_current_a":
        return np.log10(np.clip(values, LEAKAGE_EPSILON_A, None))
    if target == "on_off_ratio":
        return np.log10(np.clip(values, 1.0, None))
    if target == "endurance_cycles":
        return np.log1p(np.clip(values, 0.0, None))
    return values


def _inverse_transform_target(target: str, values: np.ndarray) -> np.ndarray:
    """將 GPR 模型空間的預測值回推到原始物理尺度。"""
    values = np.asarray(values, dtype=float)
    if target in {"leakage_current_a", "on_off_ratio"}:
        return np.power(10.0, values)
    if target == "endurance_cycles":
        return np.expm1(values)
    return values


@dataclass
class GPRTargetModel:
    """單一 target 的 Gaussian Process surrogate。

    GPR 會輸出 posterior mean 與 std：
    y(x*) | D ~ Normal(mu(x*), sigma(x*)^2)
    因此可自然產生 95% confidence interval: mu +/- 1.96 * sigma。
    """

    target: str
    feature_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
    random_state: int = RANDOM_STATE
    scaler: StandardScaler = field(default_factory=StandardScaler)
    model: GaussianProcessRegressor | None = None

    def _build_model(self, n_features: int) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GPRTargetModel":
        X_values = X[self.feature_columns].to_numpy(dtype=float)
        y_values = _transform_target(self.target, y.to_numpy(dtype=float))
        X_scaled = self.scaler.fit_transform(X_values)
        self.model = self._build_model(X_scaled.shape[1])
        self.model.fit(X_scaled, y_values)
        return self

    def predict(self, X: pd.DataFrame, ci_z: float = 1.96) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError(f"GPR target model for {self.target} has not been fitted.")
        X_values = X[self.feature_columns].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X_values)
        mean_t, std_t = self.model.predict(X_scaled, return_std=True)

        lower_t = mean_t - ci_z * std_t
        upper_t = mean_t + ci_z * std_t
        mean = _inverse_transform_target(self.target, mean_t)
        lower = _inverse_transform_target(self.target, lower_t)
        upper = _inverse_transform_target(self.target, upper_t)

        # log-space target 的 std 回到原尺度時不是對稱分佈，這裡用 95% CI 半寬近似原尺度 std。
        std_original = (upper - lower) / (2.0 * ci_z)
        return pd.DataFrame(
            {
                f"{self.target}_mean": mean,
                f"{self.target}_std": std_original,
                f"{self.target}_ci95_low": lower,
                f"{self.target}_ci95_high": upper,
                f"{self.target}_transformed_mean": mean_t,
                f"{self.target}_transformed_std": std_t,
            },
            index=X.index,
        )


@dataclass
class PredictiveModelSuite:
    """管理所有 regression 與 classification surrogate models。"""

    feature_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
    regression_models: dict[str, GPRTargetModel] = field(default_factory=dict)
    classification_models: dict[str, object] = field(default_factory=dict)
    classification_constants: dict[str, str] = field(default_factory=dict)

    def fit(self, dataset: pd.DataFrame) -> "PredictiveModelSuite":
        self.fit_regression_models(dataset)
        self.fit_classification_models(dataset)
        return self

    def fit_regression_models(self, dataset: pd.DataFrame) -> None:
        for target in REGRESSION_TARGETS:
            frame = dataset.dropna(subset=self.feature_columns + [target]).copy()
            if len(frame) < MIN_TRAINING_ROWS:
                continue
            model = GPRTargetModel(target=target, feature_columns=self.feature_columns)
            model.fit(frame[self.feature_columns], frame[target])
            self.regression_models[target] = model

    def fit_classification_models(self, dataset: pd.DataFrame) -> None:
        for target in CLASSIFICATION_TARGETS:
            if target not in dataset:
                continue
            frame = dataset.dropna(subset=self.feature_columns + [target]).copy()
            labels = frame[target].astype(str)
            if len(frame) < MIN_TRAINING_ROWS or labels.nunique() == 0:
                continue
            if labels.nunique() == 1:
                self.classification_constants[target] = labels.iloc[0]
                continue
            classifier = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
                ]
            )
            classifier.fit(frame[self.feature_columns], labels)
            self.classification_models[target] = classifier

    def predict_properties(self, X: pd.DataFrame) -> pd.DataFrame:
        output = X.copy()
        for target, model in self.regression_models.items():
            output = output.join(model.predict(X))
        for target, classifier in self.classification_models.items():
            output[f"{target}_pred"] = classifier.predict(X[self.feature_columns])
        for target, label in self.classification_constants.items():
            output[f"{target}_pred"] = label
        return output


def leave_one_condition_out_cv(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以製程條件為單位做 LOCO CV，避免同條件重複量測洩漏到測試集。"""
    prediction_rows: list[dict[str, float | str]] = []
    for target in REGRESSION_TARGETS:
        frame = dataset.dropna(subset=FEATURE_COLUMNS + [target]).copy()
        if len(frame) < MIN_TRAINING_ROWS + 1:
            continue
        for test_index, test_row in frame.iterrows():
            train = frame.drop(index=test_index)
            if len(train) < MIN_TRAINING_ROWS:
                continue
            model = GPRTargetModel(target=target)
            model.fit(train[FEATURE_COLUMNS], train[target])
            pred = model.predict(pd.DataFrame([test_row[FEATURE_COLUMNS]], index=[test_index]))
            prediction_rows.append(
                {
                    "target": target,
                    "condition_id": test_row.get("condition_id", str(test_index)),
                    "y_true": float(test_row[target]),
                    "y_pred": float(pred[f"{target}_mean"].iloc[0]),
                    "y_std": float(pred[f"{target}_std"].iloc[0]),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    summary_rows: list[dict[str, float | str]] = []
    for target, group in predictions.groupby("target"):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "target": target,
                "n_predictions": len(group),
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred) if len(group) >= 2 else np.nan,
            }
        )
    return predictions, pd.DataFrame(summary_rows)


def evaluate_classifiers_on_training_data(model_suite: PredictiveModelSuite, dataset: pd.DataFrame) -> pd.DataFrame:
    """小樣本下先提供訓練集分類 sanity check；正式報告仍應以 LOCO CV 擴充。"""
    rows: list[dict[str, float | str]] = []
    for target in CLASSIFICATION_TARGETS:
        frame = dataset.dropna(subset=FEATURE_COLUMNS + [target]).copy()
        if frame.empty:
            continue
        y_true = frame[target].astype(str)
        if target in model_suite.classification_models:
            y_pred = model_suite.classification_models[target].predict(frame[FEATURE_COLUMNS])
        elif target in model_suite.classification_constants:
            y_pred = np.repeat(model_suite.classification_constants[target], len(frame))
        else:
            continue
        rows.append(
            {
                "target": target,
                "n_samples": len(frame),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)
