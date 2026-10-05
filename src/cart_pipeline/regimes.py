from __future__ import annotations
from itertools import combinations
import numpy as np
REGIME_TO_INT = {'DC': 0, 'PR': 1, 'RL': 2, 'Failure': 3}
INT_TO_REGIME = {v: k for k, v in REGIME_TO_INT.items()}
REGIME_LABELS = {'DC': 'durable control', 'PR': 'partial response', 'RL': 'relapse', 'Failure': 'failure / no meaningful response'}

def classify_arrays(y_end, y_min, c_ctrl=0.7, c_fail=1.0, c_resp=0.95, **_ignored):
    y_end = np.asarray(y_end, dtype=float)
    y_min = np.asarray(y_min, dtype=float)
    out = np.full(y_end.shape, 'Failure', dtype='<U7')
    dc = y_end <= c_ctrl
    pr = ~dc & (y_end < c_fail)
    rl = (y_end >= c_fail) & (y_min <= c_resp)
    out[dc] = 'DC'
    out[pr] = 'PR'
    out[rl] = 'RL'
    return out

def build_regime_maps(surrogate, mechanism_names, total_indices, grid_n=41, lower=0.6, upper=1.4, c_ctrl=0.7, c_fail=1.0, c_resp=0.95, **_ignored):
    top = np.argsort(total_indices)[::-1][:3]
    values = np.linspace(lower, upper, grid_n)
    maps = []
    for i, j in combinations(top, 2):
        s1, s2 = np.meshgrid(values, values, indexing='xy')
        phys = np.ones((grid_n * grid_n, len(mechanism_names)))
        phys[:, i] = s1.ravel()
        phys[:, j] = s2.ravel()
        z = (phys - lower) / (upper - lower)
        pred = surrogate.predict(z)
        reg = classify_arrays(pred['y_end_mean'], pred['y_min_mean'], c_ctrl=c_ctrl, c_fail=c_fail, c_resp=c_resp)
        maps.append({'i': int(i), 'j': int(j), 'name_i': mechanism_names[i], 'name_j': mechanism_names[j], 'x': values, 'y': values, 'regime': reg.reshape(grid_n, grid_n), 'y_end': pred['y_end_mean'].reshape(grid_n, grid_n), 'y_min': pred['y_min_mean'].reshape(grid_n, grid_n), 'y_end_std': pred['y_end_std'].reshape(grid_n, grid_n), 'y_min_std': pred['y_min_std'].reshape(grid_n, grid_n)})
    return maps

def boundary_candidates(regime_map):
    R = regime_map['regime']
    mask = np.zeros(R.shape, dtype=bool)
    mask[:-1, :] |= R[:-1, :] != R[1:, :]
    mask[1:, :] |= R[:-1, :] != R[1:, :]
    mask[:, :-1] |= R[:, :-1] != R[:, 1:]
    mask[:, 1:] |= R[:, :-1] != R[:, 1:]
    iy, ix = np.where(mask)
    xs = regime_map['x'][ix]
    ys = regime_map['y'][iy]
    return np.c_[xs, ys]
