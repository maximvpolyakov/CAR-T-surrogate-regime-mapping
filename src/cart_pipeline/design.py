from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc

def maximin_lhs(n: int, d: int, seed: int=123, n_candidates: int=32) -> np.ndarray:
    best = None
    best_min = -np.inf
    for j in range(n_candidates):
        sampler = qmc.LatinHypercube(d=d, seed=seed + j)
        x = sampler.random(n=n)
        md = float(np.min(pdist(x))) if n > 1 else np.inf
        if md > best_min:
            best, best_min = (x, md)
    return np.asarray(best)

def sobol_points(n: int, d: int, seed: int=123) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    if n > 0 and n & n - 1 == 0:
        return sampler.random_base2(int(np.log2(n)))
    return sampler.random(n)

def farthest_subset(points: np.ndarray, n: int, seed: int=123) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= n:
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(points)))]
    dmin = np.linalg.norm(points - points[selected[0]], axis=1)
    for _ in range(1, n):
        idx = int(np.argmax(dmin))
        selected.append(idx)
        dmin = np.minimum(dmin, np.linalg.norm(points - points[idx], axis=1))
    return np.asarray(selected, dtype=int)
