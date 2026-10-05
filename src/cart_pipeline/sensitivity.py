from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import qmc

@dataclass
class SobolResult:
    first: np.ndarray
    total: np.ndarray
    variance: float
    n_base: int

def jansen_sobol_from_predictor(predict_fn, d: int=6, n_base: int=8192, seed: int=123) -> SobolResult:
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    if n_base > 0 and n_base & n_base - 1 == 0:
        AB = sampler.random_base2(int(np.log2(n_base)))
    else:
        AB = sampler.random(n_base)
    A, B = (AB[:, :d], AB[:, d:])
    fA = np.asarray(predict_fn(A), dtype=float).ravel()
    fB = np.asarray(predict_fn(B), dtype=float).ravel()
    V = float(np.var(np.concatenate([fA, fB]), ddof=1))
    if V <= 1e-15:
        return SobolResult(np.zeros(d), np.zeros(d), V, n_base)
    first = np.empty(d)
    total = np.empty(d)
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB = np.asarray(predict_fn(ABi), dtype=float).ravel()
        first[i] = 1.0 - np.mean((fB - fAB) ** 2) / (2.0 * V)
        total[i] = np.mean((fA - fAB) ** 2) / (2.0 * V)
    return SobolResult(first=first, total=total, variance=V, n_base=n_base)

def bootstrap_jansen(predict_fn, d: int=6, n_base: int=8192, n_boot: int=200, seed: int=123):
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    if n_base > 0 and n_base & n_base - 1 == 0:
        X = sampler.random_base2(int(np.log2(n_base)))
    else:
        X = sampler.random(n_base)
    A, B = (X[:, :d], X[:, d:])
    fA = np.asarray(predict_fn(A), dtype=float).ravel()
    fB = np.asarray(predict_fn(B), dtype=float).ravel()
    fAB = []
    for i in range(d):
        M = A.copy()
        M[:, i] = B[:, i]
        fAB.append(np.asarray(predict_fn(M), dtype=float).ravel())
    fAB = np.asarray(fAB)
    rng = np.random.default_rng(seed + 991)
    Sb = np.empty((n_boot, d))
    STb = np.empty((n_boot, d))
    for b in range(n_boot):
        idx = rng.integers(0, n_base, size=n_base)
        va = np.var(np.concatenate([fA[idx], fB[idx]]), ddof=1)
        if va <= 1e-15:
            Sb[b] = 0
            STb[b] = 0
            continue
        for i in range(d):
            Sb[b, i] = 1 - np.mean((fB[idx] - fAB[i, idx]) ** 2) / (2 * va)
            STb[b, i] = np.mean((fA[idx] - fAB[i, idx]) ** 2) / (2 * va)
    return {'first_low': np.quantile(Sb, 0.025, axis=0), 'first_high': np.quantile(Sb, 0.975, axis=0), 'total_low': np.quantile(STb, 0.025, axis=0), 'total_high': np.quantile(STb, 0.975, axis=0)}
