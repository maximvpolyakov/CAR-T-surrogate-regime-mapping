from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
import joblib
import numpy as np
import pandas as pd
from .baseline_models import RegressionBaselines, evaluate_point_predictions
from .design import maximin_lhs
from .regimes import classify_arrays

def _json_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in m.items():
        if k in {'pred_regime', 'true_regime'}:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out

def _collides(X: np.ndarray, excluded: np.ndarray, tol: float) -> bool:
    if excluded is None or len(excluded) == 0:
        return False
    X = np.asarray(X, float)
    E = np.asarray(excluded, float)
    for x in X:
        if np.any(np.max(np.abs(E - x), axis=1) <= tol):
            return True
    return False

def independent_blind_design(space, n: int, seed: int, excluded_unit=None, tol: float=1e-10, lhs_candidates: int=64) -> np.ndarray:
    excluded = np.asarray(excluded_unit, float) if excluded_unit is not None and len(excluded_unit) else np.empty((0, space.dim))
    for attempt in range(200):
        X = maximin_lhs(n, space.dim, seed=seed + 1009 * attempt, n_candidates=lhs_candidates)
        if not _collides(X, excluded, tol):
            return X
    raise RuntimeError('Could not generate a collision-free blind design after 200 attempts.')

def run_one_factor_analysis(model, space, cfg: Dict[str, Any], out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels = cfg.get('levels', [0.6, 1.4])
    levels = [float(x) for x in levels]
    rows = []
    scenarios = [('reference', None, 1.0, np.ones(space.dim))]
    for i, name in enumerate(space.names):
        for level in levels:
            s = np.ones(space.dim)
            s[i] = level
            scenarios.append((f'{name}_{level:g}', name, level, s))
    for k, (scenario, mech, mult, s) in enumerate(scenarios, 1):
        r = model.simulate(s)
        row = {name: float(v) for name, v in zip(space.names, s)}
        row.update({'scenario': scenario, 'mechanism': mech if mech is not None else 'reference', 'multiplier': float(mult), 'Y_end': r.y_end, 'Y_min': r.y_min, 'Y_reb': r.y_reb, 'regime': r.regime, 'cache_key': r.cache_key})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'one_factor_effects.csv', index=False)
        print(f'[one-factor] {k}/{len(scenarios)}: {scenario}, Y_end={r.y_end:.5g}, regime={r.regime}')
    return pd.DataFrame(rows)

def run_blind_test(model, space, cfg: Dict[str, Any], surrogate, X_train: np.ndarray, y_end_train: np.ndarray, y_min_train: np.ndarray, out_dir: str | Path, model_dir: str | Path, excluded_unit=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg.get('n', 30))
    seed = int(cfg.get('seed', 17123))
    tol = float(cfg.get('collision_tolerance', 1e-10))
    lhs_candidates = int(cfg.get('lhs_candidates', 64))
    Xb = independent_blind_design(space, n, seed, excluded_unit, tol, lhs_candidates)
    phys = space.from_unit(Xb)
    design = pd.DataFrame(phys, columns=space.names)
    for i, name in enumerate(space.names):
        design[f'z_{name}'] = Xb[:, i]
    design.insert(0, 'blind_index', np.arange(1, n + 1))
    design.to_csv(out / 'blind_design_reserved.csv', index=False)
    baselines = RegressionBaselines.fit(X_train, y_end_train, y_min_train)
    joblib.dump(baselines.linear, model_dir / 'linear_regression.joblib')
    joblib.dump(baselines.quadratic, model_dir / 'quadratic_regression.joblib')
    rows = []
    yend = []
    ymin = []
    regimes = []
    for k, (z, s) in enumerate(zip(Xb, phys), 1):
        r = model.simulate(s)
        yend.append(r.y_end)
        ymin.append(r.y_min)
        regimes.append(r.regime)
        row = {'blind_index': k, **{name: float(v) for name, v in zip(space.names, s)}, **{f'z_{name}': float(z[i]) for i, name in enumerate(space.names)}, 'Y_end_PDE': r.y_end, 'Y_min_PDE': r.y_min, 'Y_reb_PDE': r.y_reb, 'regime_PDE': r.regime, 'cache_key': r.cache_key}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'blind_pde_partial.csv', index=False)
        print(f'[blind] {k}/{n}: Y_end={r.y_end:.5g}, Y_min={r.y_min:.5g}, regime={r.regime}')
    yend = np.asarray(yend, float)
    ymin = np.asarray(ymin, float)
    gp = surrogate.predict(Xb)
    lin = baselines.predict('linear', Xb)
    quad = baselines.predict('quadratic', Xb)
    regime_cfg = cfg.get('regimes', {})
    gp_eval = evaluate_point_predictions(yend, ymin, gp['y_end_mean'], gp['y_min_mean'], regime_cfg)
    lin_eval = evaluate_point_predictions(yend, ymin, lin['y_end_mean'], lin['y_min_mean'], regime_cfg)
    quad_eval = evaluate_point_predictions(yend, ymin, quad['y_end_mean'], quad['y_min_mean'], regime_cfg)
    gp_reg = gp_eval['pred_regime']
    lin_reg = lin_eval['pred_regime']
    quad_reg = quad_eval['pred_regime']
    cov_end = np.mean(np.abs(yend - gp['y_end_mean']) <= 1.96 * np.maximum(gp['y_end_std'], 1e-15))
    cov_min = np.mean(np.abs(ymin - gp['y_min_mean']) <= 1.96 * np.maximum(gp['y_min_std'], 1e-15))
    gp_eval['Y_end_coverage_95'] = float(cov_end)
    gp_eval['Y_min_coverage_95'] = float(cov_min)
    df = design.copy()
    df['Y_end_PDE'] = yend
    df['Y_min_PDE'] = ymin
    df['regime_PDE'] = regimes
    df['Y_end_GP'] = gp['y_end_mean']
    df['Y_end_GP_sd'] = gp['y_end_std']
    df['Y_min_GP'] = gp['y_min_mean']
    df['Y_min_GP_sd'] = gp['y_min_std']
    df['regime_GP'] = gp_reg
    df['Y_end_linear'] = lin['y_end_mean']
    df['Y_min_linear'] = lin['y_min_mean']
    df['regime_linear'] = lin_reg
    df['Y_end_quadratic'] = quad['y_end_mean']
    df['Y_min_quadratic'] = quad['y_min_mean']
    df['regime_quadratic'] = quad_reg
    df['cache_key'] = [r['cache_key'] for r in rows]
    df.to_csv(out / 'blind_test_predictions.csv', index=False)
    metrics = {'gp': _json_metrics(gp_eval), 'linear': _json_metrics(lin_eval), 'quadratic': _json_metrics(quad_eval), 'n_blind': n, 'design': 'independent maximin Latin hypercube; no response- or surrogate-based selection'}
    (out / 'blind_test_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    comp_rows = []
    for model_name in ['gp', 'quadratic', 'linear']:
        for output_name in ['Y_end', 'Y_min']:
            m = metrics[model_name][output_name]
            comp_rows.append({'model': model_name, 'output': output_name, **m, 'regime_accuracy': metrics[model_name]['regime_accuracy'], 'coverage_95': metrics[model_name].get(f'{output_name}_coverage_95', np.nan)})
    pd.DataFrame(comp_rows).to_csv(out / 'blind_model_comparison.csv', index=False)
    try:
        (out / 'blind_pde_partial.csv').unlink()
    except FileNotFoundError:
        pass
    return {'X': Xb, 'data': df, 'metrics': metrics, 'linear': baselines.linear, 'quadratic': baselines.quadratic}
