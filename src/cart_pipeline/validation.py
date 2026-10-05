from __future__ import annotations
import numpy as np
from sklearn.metrics import r2_score, balanced_accuracy_score, confusion_matrix
from .regimes import classify_arrays

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_abs_error = float(np.max(np.abs(err))) if len(err) else float('nan')
    denom = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 1.0
    nrmse = rmse / max(denom, 1e-12)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')
    return {'rmse': rmse, 'mae': mae, 'max_abs_error': max_abs_error, 'nrmse': nrmse, 'r2': r2}

def validate_surrogate(surrogate, X, y_end, y_min, regime_cfg=None):
    regime_cfg = regime_cfg or {}
    pred = surrogate.predict(X)
    m_end = regression_metrics(y_end, pred['y_end_mean'])
    m_min = regression_metrics(y_min, pred['y_min_mean'])
    true_reg = classify_arrays(y_end, y_min, c_ctrl=regime_cfg.get('c_ctrl', 0.7), c_fail=regime_cfg.get('c_fail', 1.0), c_resp=regime_cfg.get('c_resp', 0.95))
    pred_reg = classify_arrays(pred['y_end_mean'], pred['y_min_mean'], c_ctrl=regime_cfg.get('c_ctrl', 0.7), c_fail=regime_cfg.get('c_fail', 1.0), c_resp=regime_cfg.get('c_resp', 0.95))
    labels = ['DC', 'PR', 'RL', 'Failure']
    return {'end': m_end, 'min': m_min, 'regime_accuracy': float(np.mean(true_reg == pred_reg)), 'regime_balanced_accuracy': float(balanced_accuracy_score(true_reg, pred_reg)), 'confusion_matrix': confusion_matrix(true_reg, pred_reg, labels=labels).tolist(), 'regime_labels': labels, 'true_regime': true_reg, 'pred_regime': pred_reg, 'pred': pred}
