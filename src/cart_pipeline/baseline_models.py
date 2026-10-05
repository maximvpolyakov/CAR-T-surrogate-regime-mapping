from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from .regimes import classify_arrays
from .validation import regression_metrics

@dataclass
class RegressionBaselines:
    linear: Any
    quadratic: Any

    @classmethod
    def fit(cls, X: np.ndarray, y_end: np.ndarray, y_min: np.ndarray) -> 'RegressionBaselines':
        X = np.asarray(X, dtype=float)
        Y = np.c_[np.asarray(y_end, dtype=float), np.asarray(y_min, dtype=float)]
        linear = LinearRegression().fit(X, Y)
        quadratic = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)), ('ols', LinearRegression())]).fit(X, Y)
        return cls(linear=linear, quadratic=quadratic)

    def predict(self, name: str, X: np.ndarray) -> Dict[str, np.ndarray]:
        model = self.linear if name == 'linear' else self.quadratic
        pred = np.asarray(model.predict(np.asarray(X, dtype=float)), dtype=float)
        return {'y_end_mean': pred[:, 0], 'y_min_mean': pred[:, 1]}

def evaluate_point_predictions(y_end_true, y_min_true, y_end_pred, y_min_pred, regime_cfg=None):
    regime_cfg = regime_cfg or {}
    y_end_true = np.asarray(y_end_true, dtype=float)
    y_min_true = np.asarray(y_min_true, dtype=float)
    y_end_pred = np.asarray(y_end_pred, dtype=float)
    y_min_pred = np.asarray(y_min_pred, dtype=float)
    true_reg = classify_arrays(y_end_true, y_min_true, c_ctrl=regime_cfg.get('c_ctrl', 0.7), c_fail=regime_cfg.get('c_fail', 1.0), c_resp=regime_cfg.get('c_resp', 0.95))
    pred_reg = classify_arrays(y_end_pred, y_min_pred, c_ctrl=regime_cfg.get('c_ctrl', 0.7), c_fail=regime_cfg.get('c_fail', 1.0), c_resp=regime_cfg.get('c_resp', 0.95))
    labels = ['DC', 'PR', 'RL', 'Failure']
    return {'Y_end': regression_metrics(y_end_true, y_end_pred), 'Y_min': regression_metrics(y_min_true, y_min_pred), 'regime_accuracy': float(np.mean(true_reg == pred_reg)), 'regime_balanced_accuracy': float(balanced_accuracy_score(true_reg, pred_reg)), 'confusion_matrix': confusion_matrix(true_reg, pred_reg, labels=labels).tolist(), 'regime_labels': labels, 'pred_regime': pred_reg, 'true_regime': true_reg}
