from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np
import pandas as pd
from .design import sobol_points
from .surrogate import GPSurrogate
from .sensitivity import jansen_sobol_from_predictor
from .validation import validate_surrogate

def _farthest_extension(existing: np.ndarray, candidates: np.ndarray, n: int) -> np.ndarray:
    existing = np.asarray(existing, dtype=float)
    cand = np.asarray(candidates, dtype=float)
    if n <= 0:
        return np.empty((0, cand.shape[1]), dtype=float)
    dmin = np.full(len(cand), np.inf)
    for x in existing:
        dmin = np.minimum(dmin, np.linalg.norm(cand - x, axis=1))
    selected = []
    mask = np.ones(len(cand), dtype=bool)
    for _ in range(min(n, len(cand))):
        score = np.where(mask, dmin, -np.inf)
        idx = int(np.argmax(score))
        selected.append(cand[idx].copy())
        mask[idx] = False
        dmin = np.minimum(dmin, np.linalg.norm(cand - cand[idx], axis=1))
    return np.asarray(selected, dtype=float)

def _dataset_frame(space, X, y_end, y_min, y_reb, regime, source):
    phys = space.from_unit(np.asarray(X, dtype=float))
    df = pd.DataFrame(phys, columns=space.names)
    for i, name in enumerate(space.names):
        df[f'z_{name}'] = np.asarray(X)[:, i]
    df['Y_end'] = np.asarray(y_end, dtype=float)
    df['Y_min'] = np.asarray(y_min, dtype=float)
    df['Y_reb'] = np.asarray(y_reb, dtype=float)
    df['regime'] = np.asarray(regime, dtype=str)
    df['source'] = source
    return df

def run_space_filling_comparator(model, space, cfg: Dict[str, Any], out_dir: str | Path, initial_X: np.ndarray, initial_y_end: np.ndarray, initial_y_min: np.ndarray, initial_y_reb: np.ndarray, initial_regime: np.ndarray, validation: Dict[str, Any]):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_add = int(cfg.get('n_comparator_additional', 72))
    pool_n = int(cfg.get('comparator_candidate_pool', 32768))
    seed = int(cfg.get('seed', 123)) + 5100
    check_every = int(cfg.get('check_every', 12))
    n_base = int(cfg.get('sobol_n_base', 16384))
    pool = sobol_points(pool_n, space.dim, seed=seed)
    Xadd = _farthest_extension(initial_X, pool, n_add)
    y_end_add, y_min_add, y_reb_add, reg_add = ([], [], [], [])
    rows = []
    checkpoints = []
    gp0 = GPSurrogate(space.dim, random_state=int(cfg.get('seed', 123)) + 800, n_restarts=int(cfg.get('gp_restarts', 3))).fit(initial_X, initial_y_end, initial_y_min)
    val0 = validate_surrogate(gp0, validation['X'], validation['y_end'], validation['y_min'], cfg.get('regimes', {}))
    sob0 = jansen_sobol_from_predictor(lambda Z: gp0.predict_one('end', Z)[0], d=space.dim, n_base=n_base, seed=seed)
    checkpoints.append({'method': 'Space-filling', 'n_train': int(len(initial_X)), 'n_additional': 0, 'nrmse_end': val0['end']['nrmse'], 'nrmse_min': val0['min']['nrmse'], 'r2_end': val0['end']['r2'], 'r2_min': val0['min']['r2'], 'regime_accuracy': val0['regime_accuracy'], 'S': sob0.first.tolist(), 'ST': sob0.total.tolist(), 'kernel_end': str(gp0.models['end'].kernel_), 'kernel_min': str(gp0.models['min'].kernel_)})
    with (out / 'comparator_checkpoints.json').open('w', encoding='utf-8') as f:
        json.dump(checkpoints, f, indent=2, ensure_ascii=False)
    for k, z in enumerate(Xadd, 1):
        s = space.from_unit(z)
        r = model.simulate(s)
        y_end_add.append(r.y_end)
        y_min_add.append(r.y_min)
        y_reb_add.append(r.y_reb)
        reg_add.append(r.regime)
        row = {name: float(v) for name, v in zip(space.names, s)}
        for i, name in enumerate(space.names):
            row[f'z_{name}'] = float(z[i])
        row.update({'Y_end': r.y_end, 'Y_min': r.y_min, 'Y_reb': r.y_reb, 'regime': r.regime, 'sequence': k, 'cache_key': r.cache_key})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'comparator_additional_data.csv', index=False)
        print(f'[comparator] {k}/{n_add}: Y_end={r.y_end:.4g}, regime={r.regime}')
        if k % check_every == 0 or k == n_add:
            X = np.vstack([initial_X, np.asarray(Xadd[:k])])
            ye = np.r_[initial_y_end, np.asarray(y_end_add)]
            ym = np.r_[initial_y_min, np.asarray(y_min_add)]
            gp = GPSurrogate(space.dim, random_state=int(cfg.get('seed', 123)) + 800, n_restarts=int(cfg.get('gp_restarts', 3)))
            gp.fit(X, ye, ym)
            val = validate_surrogate(gp, validation['X'], validation['y_end'], validation['y_min'], cfg.get('regimes', {}))
            sob = jansen_sobol_from_predictor(lambda Z: gp.predict_one('end', Z)[0], d=space.dim, n_base=n_base, seed=seed + k)
            rec = {'method': 'Space-filling', 'n_train': int(len(X)), 'n_additional': int(k), 'nrmse_end': val['end']['nrmse'], 'nrmse_min': val['min']['nrmse'], 'r2_end': val['end']['r2'], 'r2_min': val['min']['r2'], 'regime_accuracy': val['regime_accuracy'], 'S': sob.first.tolist(), 'ST': sob.total.tolist(), 'kernel_end': str(gp.models['end'].kernel_), 'kernel_min': str(gp.models['min'].kernel_)}
            checkpoints.append(rec)
            with (out / 'comparator_checkpoints.json').open('w', encoding='utf-8') as f:
                json.dump(checkpoints, f, indent=2, ensure_ascii=False)
    Xfinal = np.vstack([initial_X, Xadd])
    yef = np.r_[initial_y_end, np.asarray(y_end_add)]
    ymf = np.r_[initial_y_min, np.asarray(y_min_add)]
    yrf = np.r_[initial_y_reb, np.asarray(y_reb_add)]
    rgf = np.r_[initial_regime, np.asarray(reg_add)]
    final_gp = GPSurrogate(space.dim, random_state=int(cfg.get('seed', 123)) + 801, n_restarts=int(cfg.get('gp_restarts', 3))).fit(Xfinal, yef, ymf)
    final_val = validate_surrogate(final_gp, validation['X'], validation['y_end'], validation['y_min'], cfg.get('regimes', {}))
    final_sob = jansen_sobol_from_predictor(lambda Z: final_gp.predict_one('end', Z)[0], d=space.dim, n_base=n_base, seed=seed + 999)
    full_df = _dataset_frame(space, Xfinal, yef, ymf, yrf, rgf, 'space_filling')
    full_df.to_csv(out / 'comparator_training_data.csv', index=False)
    return {'X': Xfinal, 'y_end': yef, 'y_min': ymf, 'y_reb': yrf, 'regime': rgf, 'surrogate': final_gp, 'validation': final_val, 'sobol': final_sob, 'checkpoints': checkpoints, 'X_additional': Xadd}
