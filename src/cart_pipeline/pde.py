from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import csv
import hashlib
import json
import os
import subprocess
import numpy as np
from .build_solver import build_solver, default_binary_path

@dataclass
class SimulationResult:
    time: np.ndarray
    tumour_burden: np.ndarray
    y_end: float
    y_min: float
    y_reb: float
    regime: str
    t_min: float | None = None
    trajectory: Dict[str, np.ndarray] | None = None
    final_fields: Dict[str, np.ndarray] | None = None
    diagnostics: Dict[str, float] | None = None
    cache_key: str | None = None
    run_dir: str | None = None

def classify_regime(y_end: float, y_min: float, c_ctrl: float=0.7, c_fail: float=1.0, c_resp: float=0.95, **_ignored) -> str:
    if y_end <= c_ctrl:
        return 'DC'
    if y_end < c_fail:
        return 'PR'
    if y_min <= c_resp:
        return 'RL'
    return 'Failure'

class ReactionDiffusionCARTModel:
    PARAM_KEYS = ('D_C', 'D_T', 'D_E', 'D_S', 'D_A', 'D_H', 'rho_C', 'r_T', 'delta_CT', 'gamma', 'd_C', 'k_e0', 'K_C', 'sigma_S', 'lambda_S', 'sigma_A', 'sigma_M', 'lambda_A', 'eta', 'lambda_H', 'chi_0', 'p_r', 'd_E', 'd_TA', 'd_TB', 'mu_0', 'mu_1', 'nu', 'xi', 'alpha_A', 'beta_A', 'beta_H', 'K_H')

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.grid = dict(cfg['grid'])
        self.time_cfg = dict(cfg['time'])
        self.num_cfg = dict(cfg.get('numerics', {}))
        self.base = {k: float(v) for k, v in dict(cfg['parameters']).items()}
        self.init_cfg = dict(cfg['initial_conditions'])
        self.regime_cfg = dict(cfg.get('regimes', {}))
        self.backend_cfg = dict(cfg.get('backend', {}))
        self.formulation = str(cfg.get('formulation', 'normalised')).lower()
        if self.formulation not in {'normalised', 'normalized'}:
            raise ValueError('pde.formulation must be normalised. The raw-density formulation is not supported by this solver. because it mixes dimensional cell counts with effective interaction coefficients.')
        self.cache_dir = Path(cfg.get('cache_dir', '.cart_pde_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        missing = [k for k in self.PARAM_KEYS if k not in self.base]
        if missing:
            raise ValueError(f'Missing normalised PDE parameters: {missing}')
        binary = self.backend_cfg.get('binary')
        self.binary = Path(binary).expanduser().resolve() if binary else default_binary_path()
        if not self.binary.exists():
            if not bool(self.backend_cfg.get('auto_build', True)):
                raise FileNotFoundError(f'C++ PDE solver not found: {self.binary}')
            self.binary = build_solver(verbose=True)

    def scaled_parameters(self, s: np.ndarray, overrides: Dict[str, float] | None=None) -> Dict[str, float]:
        s = np.asarray(s, dtype=float).ravel()
        if len(s) != 6:
            raise ValueError('Expected six mechanism multipliers.')
        smig, scyt, sexh, sesc, senv, shyp = map(float, s)
        p = dict(self.base)
        p['chi_0'] *= smig
        p['gamma'] *= scyt
        p['k_e0'] *= sexh
        p['mu_0'] *= sesc
        p['nu'] *= sesc
        p['sigma_A'] *= senv
        p['alpha_A'] *= senv
        p['beta_A'] *= senv
        p['eta'] *= shyp
        p['beta_H'] *= shyp
        if overrides:
            for k, v in overrides.items():
                if k not in p:
                    raise KeyError(f'Unknown PDE parameter override: {k}')
                p[k] = float(v)
        return p

    def _fingerprint(self, s: np.ndarray, parameter_overrides=None, initial_overrides=None, cache_tag: str | None=None) -> str:
        payload = {'s': [round(float(x), 12) for x in np.asarray(s).ravel()], 'grid': self.grid, 'time': self.time_cfg, 'numerics': self.num_cfg, 'parameters': self.base, 'initial_conditions': self.init_cfg, 'parameter_overrides': parameter_overrides or {}, 'initial_overrides': initial_overrides or {}, 'formulation': self.formulation, 'cache_tag': cache_tag, 'solver_schema': 6}
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()[:24]

    def _command(self, p: Dict[str, float], summary_path: Path, trajectory_path: Path | None=None, fields_path: Path | None=None, initial_overrides: Dict[str, float] | None=None) -> list[str]:
        g, t, n = (self.grid, self.time_cfg, self.num_cfg)
        ic = dict(self.init_cfg)
        if initial_overrides:
            ic.update({k: float(v) for k, v in initial_overrides.items()})
        cmd = [str(self.binary)]
        args = {'nx': int(g.get('nx', 40)), 'ny': int(g.get('ny', 40)), 'nz': int(g.get('nz', 40)), 'Lx': float(g.get('Lx', 1.0)), 'Ly': float(g.get('Ly', 1.0)), 'Lz': float(g.get('Lz', 1.0)), 'dt': float(t.get('dt', 0.01)), 'T': float(t.get('horizon', 30.0)), 'exp_substeps': int(n.get('exp_substeps', 1)), 'diff_iters': int(n.get('diff_iters', 60)), 'diff_tol': float(n.get('diff_tol', 1e-08)), 'trajectory_every': int(t.get('trajectory_every', 10)), 'threads': int(self.backend_cfg.get('threads', 0)), 'tumor_radius': float(ic.get('tumor_radius', 0.2)), 'tumor_value': float(ic.get('tumor_value', 0.1)), 'cart_layer': float(ic.get('cart_layer', 0.05)), 'cart_radius': float(ic.get('cart_radius', 0.2)), 'cart_value': float(ic.get('cart_value', 0.5)), 'summary_path': str(summary_path)}
        if trajectory_path is not None:
            args['trajectory_path'] = str(trajectory_path)
        if fields_path is not None:
            args['final_fields_path'] = str(fields_path)
        args.update(p)
        for k, v in args.items():
            cmd.append(f'--{k}={v}')
        return cmd

    @staticmethod
    def _read_summary(path: Path) -> Dict[str, float]:
        with path.open('r', encoding='utf-8', newline='') as f:
            row = next(csv.DictReader(f))
        return {k: float(v) for k, v in row.items()}

    @staticmethod
    def _read_trajectory(path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            return {}
        data = np.genfromtxt(path, delimiter=',', names=True)
        if data.size == 0 or data.dtype.names is None:
            return {}
        return {name: np.atleast_1d(data[name]).astype(float) for name in data.dtype.names}

    def simulate(self, s: np.ndarray, keep_fields: bool=False, parameter_overrides: Dict[str, float] | None=None, initial_overrides: Dict[str, float] | None=None, cache_tag: str | None=None) -> SimulationResult:
        s = np.asarray(s, dtype=float).ravel()
        p = self.scaled_parameters(s, parameter_overrides)
        key = self._fingerprint(s, parameter_overrides, initial_overrides, cache_tag)
        run_dir = self.cache_dir / key
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / 'summary.csv'
        trajectory_path = run_dir / 'trajectory.csv' if bool(self.backend_cfg.get('store_trajectories', False)) else None
        fields_path = run_dir / 'final_fields.csv' if keep_fields else None
        input_meta = {'cache_key': key, 'mechanism_multipliers': [float(x) for x in s], 'scaled_parameters': p, 'grid': self.grid, 'time': self.time_cfg, 'numerics': self.num_cfg, 'initial_conditions': self.init_cfg, 'parameter_overrides': parameter_overrides or {}, 'initial_overrides': initial_overrides or {}, 'cache_tag': cache_tag, 'formulation': self.formulation}
        with (run_dir / 'input.json').open('w', encoding='utf-8') as f:
            json.dump(input_meta, f, indent=2, ensure_ascii=False)
        need_run = not summary_path.exists() or (keep_fields and (not fields_path.exists()))
        if trajectory_path is not None and (not trajectory_path.exists()):
            need_run = True
        if need_run:
            cmd = self._command(p, summary_path, trajectory_path, fields_path, initial_overrides)
            log_path = run_dir / 'solver.log'
            env = os.environ.copy()
            with log_path.open('w', encoding='utf-8') as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0 or not summary_path.exists():
                tail = ''
                try:
                    tail = log_path.read_text(encoding='utf-8')[-4000:]
                except Exception:
                    pass
                raise RuntimeError(f'C++ PDE solver failed (exit={proc.returncode}).\n{tail}')
        q = self._read_summary(summary_path)
        if trajectory_path is not None:
            traj = self._read_trajectory(trajectory_path)
            tt = traj.get('t', np.empty(0))
            rr = traj.get('R', np.empty(0))
        else:
            traj = {}
            tt, rr = (np.empty(0), np.empty(0))
        y_end, y_min = (float(q['Y_end']), float(q['Y_min']))
        y_reb = float(q['Y_reb'])
        t_min = float(q['t_min']) if 't_min' in q else None
        regime = classify_regime(y_end, y_min, c_ctrl=float(self.regime_cfg.get('c_ctrl', 0.7)), c_fail=float(self.regime_cfg.get('c_fail', 1.0)), c_resp=float(self.regime_cfg.get('c_resp', 0.95)))
        diag_keys = ['C_peak_mean', 't_C_peak', 'C_final_mean', 'E_final_mean', 'TA_final_mean', 'TB_final_mean', 'A_peak_mean', 'H_peak_mean', 'antigen_low_final', 'tumour_peak_local_final', 'C_peak_local', 't_C_peak_local', 'E_peak_mean', 't_E_peak', 'initial_C_mean', 'initial_C_to_tumour_ratio', 'runtime_seconds']
        diagnostics = {k: float(q[k]) for k in diag_keys if k in q}
        result_meta = {'cache_key': key, 'Y_end': y_end, 'Y_min': y_min, 'Y_reb': y_reb, 't_min': t_min, 'regime': regime, 'diagnostics': diagnostics, 'trajectory_file': 'trajectory.csv' if trajectory_path is not None else None, 'final_fields_file': 'final_fields.csv' if keep_fields else None}
        with (run_dir / 'result.json').open('w', encoding='utf-8') as f:
            json.dump(result_meta, f, indent=2, ensure_ascii=False)
        return SimulationResult(time=tt, tumour_burden=rr, y_end=y_end, y_min=y_min, y_reb=y_reb, regime=regime, t_min=t_min, trajectory=traj, final_fields=None, diagnostics=diagnostics, cache_key=key, run_dir=str(run_dir))
