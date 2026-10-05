from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd
from .design import maximin_lhs, sobol_points
from .surrogate import GPSurrogate
from .sensitivity import jansen_sobol_from_predictor
from .validation import validate_surrogate

@dataclass
class AdaptiveHistory:
    X: list = field(default_factory=list)
    y_end: list = field(default_factory=list)
    y_min: list = field(default_factory=list)
    y_reb: list = field(default_factory=list)
    regime: list = field(default_factory=list)
    cache_key: list = field(default_factory=list)
    stage: list = field(default_factory=list)
    checkpoints: list = field(default_factory=list)
    acquisitions: list = field(default_factory=list)

class AdaptiveStudy:

    def __init__(self, model, space, cfg: Dict[str, Any], output_dir: str | Path):
        self.model = model
        self.space = space
        self.cfg = cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.history = AdaptiveHistory()
        self.surrogate = GPSurrogate(dim=space.dim, random_state=int(cfg.get('seed', 123)), n_restarts=int(cfg.get('gp_restarts', 3)))
        self.candidates = sobol_points(int(cfg.get('candidate_pool', 16384)), space.dim, int(cfg.get('seed', 123)) + 700)
        self.validation = None
        self.recommended_stop_n_train = None

    def _eval_z(self, z):
        s = self.space.from_unit(np.asarray(z, dtype=float))
        return self.model.simulate(s)

    def _append(self, z, r, stage: str):
        self.history.X.append(np.asarray(z, dtype=float))
        self.history.y_end.append(float(r.y_end))
        self.history.y_min.append(float(r.y_min))
        self.history.y_reb.append(float(r.y_reb))
        self.history.regime.append(r.regime)
        self.history.cache_key.append(r.cache_key or '')
        self.history.stage.append(stage)

    def _arrays(self):
        return (np.asarray(self.history.X, dtype=float), np.asarray(self.history.y_end, dtype=float), np.asarray(self.history.y_min, dtype=float))

    def _fit(self):
        X, ye, ym = self._arrays()
        self.surrogate.fit(X, ye, ym)

    @staticmethod
    def _norm(v):
        v = np.asarray(v, dtype=float)
        lo, hi = (np.nanmin(v), np.nanmax(v))
        if hi - lo < 1e-15:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    def acquisition(self):
        pred = self.surrogate.predict(self.candidates)
        _, se_std = self.surrogate.predict_one('end', self.candidates, standardized=True)
        _, sm_std = self.surrogate.predict_one('min', self.candidates, standardized=True)
        U = np.maximum(se_std, sm_std)
        rcfg = self.cfg.get('regimes', {})
        c_ctrl = float(rcfg.get('c_ctrl', 0.7))
        c_fail = float(rcfg.get('c_fail', 1.0))
        c_resp = float(rcfg.get('c_resp', 0.95))
        eps = 1e-12
        d_ctrl = np.abs(pred['y_end_mean'] - c_ctrl) / (pred['y_end_std'] + eps)
        d_fail = np.abs(pred['y_end_mean'] - c_fail) / (pred['y_end_std'] + eps)
        d_resp = np.abs(pred['y_min_mean'] - c_resp) / (pred['y_min_std'] + eps)
        b_ctrl = np.exp(-0.5 * d_ctrl * d_ctrl)
        b_fail = np.exp(-0.5 * d_fail * d_fail)
        z_fail = np.clip((pred['y_end_mean'] - c_fail) / (pred['y_end_std'] + eps), -30.0, 30.0)
        p_failure_side = 1.0 / (1.0 + np.exp(-z_fail))
        b_resp = p_failure_side * np.exp(-0.5 * d_resp * d_resp)
        B = np.maximum.reduce([b_ctrl, b_fail, b_resp])
        w_unc = float(self.cfg.get('acquisition_uncertainty_weight', 0.5))
        w_bnd = float(self.cfg.get('acquisition_boundary_weight', 0.5))
        A = w_unc * self._norm(U) + w_bnd * self._norm(B)
        return (A, U, B, pred)

    def _predict_end(self, X):
        return self.surrogate.predict_one('end', X)[0]

    def compute_sobol(self):
        return jansen_sobol_from_predictor(self._predict_end, d=self.space.dim, n_base=int(self.cfg.get('sobol_n_base', 16384)), seed=int(self.cfg.get('seed', 123)) + 1400)

    def _checkpoint(self, adaptive_count: int, previous_sobol=None):
        n_train = len(self.history.X)
        if self.history.checkpoints and self.history.checkpoints[-1]['n_train'] == n_train:
            rec = self.history.checkpoints[-1]
            sob = self.compute_sobol()
            val = validate_surrogate(self.surrogate, self.validation['X'], self.validation['y_end'], self.validation['y_min'], self.cfg.get('regimes', {}))
            return (sob, val, rec)
        sob = self.compute_sobol()
        val = validate_surrogate(self.surrogate, self.validation['X'], self.validation['y_end'], self.validation['y_min'], self.cfg.get('regimes', {}))
        delta = None
        if previous_sobol is not None:
            delta = float(max(np.max(np.abs(sob.first - previous_sobol.first)), np.max(np.abs(sob.total - previous_sobol.total))))
        rec = {'method': 'Adaptive', 'n_train': n_train, 'n_adaptive': adaptive_count, 'delta_sobol': delta, 'nrmse_end': val['end']['nrmse'], 'nrmse_min': val['min']['nrmse'], 'r2_end': val['end']['r2'], 'r2_min': val['min']['r2'], 'regime_accuracy': val['regime_accuracy'], 'S': sob.first.tolist(), 'ST': sob.total.tolist(), 'kernel_end': str(self.surrogate.models['end'].kernel_), 'kernel_min': str(self.surrogate.models['min'].kernel_)}
        self.history.checkpoints.append(rec)
        self._save_progress()
        return (sob, val, rec)

    def _save_progress(self):
        X, ye, ym = self._arrays()
        phys = self.space.from_unit(X) if len(X) else np.empty((0, self.space.dim))
        df = pd.DataFrame(phys, columns=self.space.names)
        if len(X):
            for i, name in enumerate(self.space.names):
                df[f'z_{name}'] = X[:, i]
            df['Y_end'] = ye
            df['Y_min'] = ym
            df['Y_reb'] = np.asarray(self.history.y_reb)
            df['regime'] = self.history.regime
            df['stage'] = self.history.stage
            df['cache_key'] = self.history.cache_key
        df.to_csv(self.out / 'training_data.csv', index=False)
        cpdf = pd.DataFrame([{k: v for k, v in r.items() if k not in {'S', 'ST'}} for r in self.history.checkpoints])
        cpdf.to_csv(self.out / 'adaptive_checkpoints.csv', index=False)
        with (self.out / 'adaptive_checkpoints.json').open('w', encoding='utf-8') as f:
            json.dump(self.history.checkpoints, f, ensure_ascii=False, indent=2)
        pd.DataFrame(self.history.acquisitions).to_csv(self.out / 'acquisition_history.csv', index=False)

    def prepare_validation(self):
        n_val = int(self.cfg.get('n_validation', 30))
        Xv = maximin_lhs(n_val, self.space.dim, int(self.cfg.get('seed', 123)) + 300)
        y_end, y_min, y_reb, regime, cache_keys = ([], [], [], [], [])
        print(f'[validation] Running {n_val} independent PDE simulations...')
        for k, z in enumerate(Xv, 1):
            r = self._eval_z(z)
            y_end.append(r.y_end)
            y_min.append(r.y_min)
            y_reb.append(r.y_reb)
            regime.append(r.regime)
            cache_keys.append(r.cache_key)
            print(f'  validation {k}/{n_val}: Y_end={r.y_end:.4g}, regime={r.regime}')
        self.validation = {'X': Xv, 'y_end': np.asarray(y_end), 'y_min': np.asarray(y_min), 'y_reb': np.asarray(y_reb), 'regime': np.asarray(regime), 'cache_key': np.asarray(cache_keys)}
        phys = self.space.from_unit(Xv)
        df = pd.DataFrame(phys, columns=self.space.names)
        for i, name in enumerate(self.space.names):
            df[f'z_{name}'] = Xv[:, i]
        df['Y_end'] = y_end
        df['Y_min'] = y_min
        df['Y_reb'] = y_reb
        df['regime'] = regime
        df['cache_key'] = cache_keys
        df.to_csv(self.out / 'validation_data.csv', index=False)

    def run(self):
        seed = int(self.cfg.get('seed', 123))
        n0 = int(self.cfg.get('n_initial', 48))
        nmax = int(self.cfg.get('n_adaptive_max', 72))
        check_every = int(self.cfg.get('check_every', 12))
        sobol_tol = float(self.cfg.get('sobol_tolerance', 0.02))
        nrmse_tol = float(self.cfg.get('nrmse_tolerance', 0.12))
        regime_acc_tol = float(self.cfg.get('regime_accuracy_tolerance', 0.9))
        stable_needed = int(self.cfg.get('stable_checks', 2))
        min_before_stop = int(self.cfg.get('min_adaptive_before_stop', 24))
        continue_after_stop = bool(self.cfg.get('continue_after_stop', True))
        self.prepare_validation()
        X0 = maximin_lhs(n0, self.space.dim, seed)
        print(f'[initial] Running {n0} initial PDE simulations...')
        for k, z in enumerate(X0, 1):
            r = self._eval_z(z)
            self._append(z, r, 'initial')
            self._save_progress()
            print(f'  initial {k}/{n0}: Y_end={r.y_end:.4g}, regime={r.regime}')
        self._fit()
        prev_sobol, _, _ = self._checkpoint(0, None)
        stable = 0
        for a in range(1, nmax + 1):
            acq, U, B, pred = self.acquisition()
            idx = int(np.nanargmax(acq))
            z = self.candidates[idx].copy()
            s = self.space.from_unit(z)
            r = self._eval_z(z)
            acq_row = {'iteration': a, 'n_train_after': len(self.history.X) + 1, 'A': float(acq[idx]), 'uncertainty_score': float(U[idx]), 'boundary_score': float(B[idx]), 'pred_Y_end': float(pred['y_end_mean'][idx]), 'pred_Y_min': float(pred['y_min_mean'][idx]), 'obs_Y_end': r.y_end, 'obs_Y_min': r.y_min, 'regime': r.regime, 'cache_key': r.cache_key}
            for name, v in zip(self.space.names, s):
                acq_row[name] = float(v)
            self.history.acquisitions.append(acq_row)
            self._append(z, r, 'adaptive')
            self.candidates = np.delete(self.candidates, idx, axis=0)
            self._fit()
            self._save_progress()
            print(f'[adaptive] {a}/{nmax}: A={acq[idx]:.3f}, Y_end={r.y_end:.4g}, regime={r.regime}')
            if a % check_every == 0:
                sob, val, rec = self._checkpoint(a, prev_sobol)
                delta = rec['delta_sobol']
                accurate = max(rec['nrmse_end'], rec['nrmse_min']) < nrmse_tol and rec['regime_accuracy'] >= regime_acc_tol
                if a >= min_before_stop and delta is not None and (delta < sobol_tol) and accurate:
                    stable += 1
                else:
                    stable = 0
                print(f"  checkpoint: ΔSobol={delta}, NRMSE max={max(rec['nrmse_end'], rec['nrmse_min']):.3f}, regime acc={rec['regime_accuracy']:.3f}, stable={stable}/{stable_needed}")
                prev_sobol = sob
                if stable >= stable_needed and self.recommended_stop_n_train is None:
                    self.recommended_stop_n_train = len(self.history.X)
                    print(f'[adaptive] Stopping criterion first satisfied at n_train={self.recommended_stop_n_train}.')
                    if not continue_after_stop:
                        break
                    print('[adaptive] Continuing to the configured maximum.')
        final_sobol, final_val, _ = self._checkpoint(len(self.history.X) - n0, prev_sobol)
        with (self.out / 'stopping_recommendation.json').open('w', encoding='utf-8') as f:
            json.dump({'recommended_stop_n_train': self.recommended_stop_n_train, 'configured_max_n_train': n0 + nmax, 'continued_after_stop': continue_after_stop}, f, indent=2)
        return {'surrogate': self.surrogate, 'sobol': final_sobol, 'validation': final_val, 'history': self.history, 'recommended_stop_n_train': self.recommended_stop_n_train}
