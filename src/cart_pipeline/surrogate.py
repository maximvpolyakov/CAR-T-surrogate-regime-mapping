from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

@dataclass
class _OutputScale:
    mean: float
    std: float

class GPSurrogate:

    def __init__(self, dim: int=6, random_state: int=123, n_restarts: int=3, alpha: float=1e-08):
        self.dim = dim
        self.random_state = random_state
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.models = {}
        self.scales = {}

    def _new_gp(self):
        kernel = ConstantKernel(1.0, (0.001, 1000.0)) * Matern(length_scale=np.ones(self.dim), length_scale_bounds=(0.01, 100.0), nu=2.5)
        return GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, normalize_y=False, n_restarts_optimizer=self.n_restarts, random_state=self.random_state)

    def fit(self, X: np.ndarray, y_end: np.ndarray, y_min: np.ndarray):
        X = np.asarray(X, dtype=float)
        for name, y in [('end', y_end), ('min', y_min)]:
            y = np.asarray(y, dtype=float)
            mean = float(np.mean(y))
            std = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
            std = max(std, 1e-12)
            ys = (y - mean) / std
            gp = self._new_gp()
            gp.fit(X, ys)
            self.models[name] = gp
            self.scales[name] = _OutputScale(mean, std)
        return self

    def predict_one(self, name: str, X: np.ndarray, standardized: bool=False):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mu, sd = self.models[name].predict(X, return_std=True)
        if standardized:
            return (mu, sd)
        sc = self.scales[name]
        return (mu * sc.std + sc.mean, sd * sc.std)

    def predict(self, X: np.ndarray):
        me, se = self.predict_one('end', X)
        mm, sm = self.predict_one('min', X)
        return {'y_end_mean': me, 'y_end_std': se, 'y_min_mean': mm, 'y_min_std': sm, 'y_reb_mean': me - mm, 'y_reb_std': np.sqrt(se ** 2 + sm ** 2)}
