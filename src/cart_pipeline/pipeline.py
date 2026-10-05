from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import sys
import joblib
import numpy as np
import pandas as pd
from .adaptive import AdaptiveStudy
from .blind import run_blind_test, run_one_factor_analysis
from .comparison import run_space_filling_comparator
from .config import MechanismSpace, load_yaml, save_yaml
from .design import farthest_subset
from .pde import ReactionDiffusionCARTModel, classify_regime
from .plots import save_blind_validation_panel, save_model_baseline_comparison, save_one_factor_effects, save_performance_comparison, save_regime_map, save_representative_trajectories, save_sampling_design, save_sobol_plot
from .regimes import boundary_candidates, build_regime_maps
from .sensitivity import bootstrap_jansen

def _validation_summary(value):
    return {'Y_end': value['end'], 'Y_min': value['min'], 'regime_accuracy': float(value['regime_accuracy']), 'regime_balanced_accuracy': float(value.get('regime_balanced_accuracy', np.nan)), 'confusion_matrix': value.get('confusion_matrix'), 'regime_labels': value.get('regime_labels')}

def _environment_snapshot():
    packages = {}
    for package in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'PyYAML', 'joblib']:
        try:
            packages[package] = metadata.version(package)
        except Exception:
            packages[package] = 'unknown'
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(), 'packages': packages}

def _save_map_table(regime_map, path):
    xx, yy = np.meshgrid(regime_map['x'], regime_map['y'], indexing='xy')
    frame = pd.DataFrame({regime_map['name_i']: xx.ravel(), regime_map['name_j']: yy.ravel(), 'Y_end_mean': np.asarray(regime_map['y_end']).ravel(), 'Y_min_mean': np.asarray(regime_map['y_min']).ravel(), 'Y_end_std': np.asarray(regime_map['y_end_std']).ravel(), 'Y_min_std': np.asarray(regime_map['y_min_std']).ravel(), 'regime': np.asarray(regime_map['regime']).ravel()})
    frame.to_csv(path, index=False)

def _checkpoint_table(checkpoints, names):
    rows = []
    for record in checkpoints:
        for index, name in enumerate(names):
            rows.append({'method': record.get('method', ''), 'n_train': record['n_train'], 'mechanism': name, 'S_i': record['S'][index], 'ST_i': record['ST'][index], 'nrmse_end': record.get('nrmse_end'), 'nrmse_min': record.get('nrmse_min'), 'regime_accuracy': record.get('regime_accuracy')})
    return pd.DataFrame(rows)

def _scan_run_manifest(root):
    rows = []
    for input_path in root.rglob('input.json'):
        try:
            input_data = json.loads(input_path.read_text(encoding='utf-8'))
            result_path = input_path.with_name('result.json')
            result_data = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
            multipliers = input_data.get('mechanism_multipliers', [])
            row = {'cache_key': input_data.get('cache_key'), 'run_dir': str(input_path.parent.relative_to(root)), 'nx': input_data.get('grid', {}).get('nx'), 'ny': input_data.get('grid', {}).get('ny'), 'nz': input_data.get('grid', {}).get('nz'), 'dt': input_data.get('time', {}).get('dt'), 'T': input_data.get('time', {}).get('horizon'), 'Y_end': result_data.get('Y_end'), 'Y_min': result_data.get('Y_min'), 'Y_reb': result_data.get('Y_reb'), 'regime': result_data.get('regime')}
            for index, name in enumerate(MechanismSpace().names):
                row[name] = multipliers[index] if index < len(multipliers) else None
            rows.append(row)
        except Exception:
            continue
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.drop_duplicates(subset=['run_dir']).sort_values(['nx', 'dt', 'run_dir'])
    frame.to_csv(root / 'data' / 'pde_run_manifest.csv', index=False)
    return frame

def _artifact_manifest(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            rows.append({'path': str(path.relative_to(root)), 'size_bytes': path.stat().st_size})
    pd.DataFrame(rows).to_csv(root / 'artifact_manifest.csv', index=False)

def _representative_trajectories(root, manifest):
    if manifest.empty:
        return pd.DataFrame()
    frame = manifest.copy()
    frame = frame[frame['regime'].isin(['DC', 'PR', 'RL', 'Failure'])]
    selected = []
    reference = frame.copy()
    for name in MechanismSpace().names:
        reference = reference[np.isclose(reference[name].astype(float), 1.0)]
    if len(reference):
        selected.append(('PR', reference.iloc[0]))
    targets = {'DC': (0.5, 0.5), 'RL': (1.2, 0.93), 'Failure': (1.5, 1.0)}
    for regime, target in targets.items():
        candidates = frame[frame['regime'] == regime].copy()
        if not len(candidates):
            continue
        score = (candidates['Y_end'].astype(float) - target[0]) ** 2 + (candidates['Y_min'].astype(float) - target[1]) ** 2
        selected.append((regime, candidates.loc[score.idxmin()]))
    trajectories = []
    for regime, row in selected:
        path = root / row['run_dir'] / 'trajectory.csv'
        if not path.exists():
            continue
        trajectory = pd.read_csv(path)
        trajectory.insert(0, 'scenario', regime)
        trajectories.append(trajectory)
    if not trajectories:
        return pd.DataFrame()
    result = pd.concat(trajectories, ignore_index=True)
    result.to_csv(root / 'data' / 'representative_trajectories.csv', index=False)
    return result

def run_pipeline(config_path: str | Path, output_dir: str | Path):
    cfg = load_yaml(config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / 'data'
    figure_dir = root / 'figures'
    model_dir = root / 'models'
    report_dir = root / 'reports'
    for path in [data_dir, figure_dir, model_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, root / 'config_used.yaml')
    (root / 'environment.json').write_text(json.dumps(_environment_snapshot(), indent=2, ensure_ascii=False), encoding='utf-8')
    study_cfg = dict(cfg['study'])
    study_cfg['regimes'] = cfg['study'].get('regimes', {})
    space = MechanismSpace(lower=float(study_cfg.get('lower', 0.6)), upper=float(study_cfg.get('upper', 1.4)))
    pde_cfg = dict(cfg['pde'])
    pde_cfg['regimes'] = study_cfg['regimes']
    pde_cfg['cache_dir'] = str(root / 'pde_cache')
    model = ReactionDiffusionCARTModel(pde_cfg)
    adaptive_dir = data_dir / 'adaptive'
    adaptive_dir.mkdir(exist_ok=True)
    study = AdaptiveStudy(model, space, study_cfg, adaptive_dir)
    adaptive = study.run()
    surrogate = adaptive['surrogate']
    sobol = adaptive['sobol']
    validation = adaptive['validation']
    joblib.dump(surrogate, model_dir / 'gp_adaptive.joblib')
    n_initial = int(study_cfg.get('n_initial', 48))
    X = np.asarray(adaptive['history'].X)
    y_end = np.asarray(adaptive['history'].y_end)
    y_min = np.asarray(adaptive['history'].y_min)
    y_reb = np.asarray(adaptive['history'].y_reb)
    regimes = np.asarray(adaptive['history'].regime)
    comparator = run_space_filling_comparator(model, space, study_cfg, data_dir / 'comparator', X[:n_initial], y_end[:n_initial], y_min[:n_initial], y_reb[:n_initial], regimes[:n_initial], study.validation)
    joblib.dump(comparator['surrogate'], model_dir / 'gp_space_filling.joblib')
    intervals = bootstrap_jansen(lambda values: surrogate.predict_one('end', values)[0], d=space.dim, n_base=int(study_cfg.get('sobol_n_base', 16384)), n_boot=int(study_cfg.get('sobol_bootstrap', 500)), seed=int(study_cfg.get('seed', 123)) + 2200)
    sobol_frame = pd.DataFrame({'mechanism': space.names, 'S_i': sobol.first, 'S_i_low': intervals['first_low'], 'S_i_high': intervals['first_high'], 'ST_i': sobol.total, 'ST_i_low': intervals['total_low'], 'ST_i_high': intervals['total_high']}).sort_values('ST_i', ascending=False)
    sobol_frame.to_csv(data_dir / 'sobol_indices_final.csv', index=False)
    save_sobol_plot(space.names, sobol, figure_dir / 'Fig_Sobol_final', intervals)
    adaptive_prediction = validation['pred']
    comparator_prediction = comparator['validation']['pred']
    validation_frame = pd.DataFrame(space.from_unit(study.validation['X']), columns=space.names)
    validation_frame['Y_end_PDE'] = study.validation['y_end']
    validation_frame['Y_min_PDE'] = study.validation['y_min']
    validation_frame['regime_PDE'] = study.validation['regime']
    validation_frame['Y_end_adaptive'] = adaptive_prediction['y_end_mean']
    validation_frame['Y_min_adaptive'] = adaptive_prediction['y_min_mean']
    validation_frame['regime_adaptive'] = validation['pred_regime']
    validation_frame['Y_end_space_filling'] = comparator_prediction['y_end_mean']
    validation_frame['Y_min_space_filling'] = comparator_prediction['y_min_mean']
    validation_frame['regime_space_filling'] = comparator['validation']['pred_regime']
    validation_frame.to_csv(data_dir / 'validation_predictions.csv', index=False)
    validation_metrics = {'adaptive': _validation_summary(validation), 'space_filling': _validation_summary(comparator['validation'])}
    (data_dir / 'validation_metrics.json').write_text(json.dumps(validation_metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    adaptive_checkpoints = adaptive['history'].checkpoints
    comparator_checkpoints = comparator['checkpoints']
    save_performance_comparison(adaptive_checkpoints, comparator_checkpoints, figure_dir / 'Fig_adaptive_vs_space_filling')
    checkpoint_frame = pd.concat([_checkpoint_table(adaptive_checkpoints, space.names), _checkpoint_table(comparator_checkpoints, space.names)], ignore_index=True)
    checkpoint_frame.to_csv(data_dir / 'checkpoint_metrics_long.csv', index=False)
    maps = build_regime_maps(surrogate, space.names, sobol.total, grid_n=int(study_cfg.get('map_grid_n', 61)), lower=space.lower, upper=space.upper, **study_cfg.get('regimes', {}))
    map_data_dir = data_dir / 'regime_maps'
    map_figure_dir = figure_dir / 'regime_maps'
    map_data_dir.mkdir(exist_ok=True)
    map_figure_dir.mkdir(exist_ok=True)
    for regime_map in maps:
        stem = f"{regime_map['name_i']}__{regime_map['name_j']}"
        _save_map_table(regime_map, map_data_dir / f'{stem}.csv')
    check_budget = int(study_cfg.get('n_map_checks', 12))
    check_rows = []
    checked_by_map = {}
    remaining = check_budget
    for map_index, regime_map in enumerate(maps):
        candidates = boundary_candidates(regime_map)
        target = int(np.ceil(remaining / max(1, len(maps) - map_index)))
        count = max(0, min(len(candidates), target))
        chosen = candidates[farthest_subset(candidates, count, int(study_cfg.get('seed', 123)) + map_index)] if count else np.empty((0, 2))
        if len(chosen) < target:
            xx, yy = np.meshgrid(regime_map['x'], regime_map['y'], indexing='xy')
            flat = np.c_[xx.ravel(), yy.ravel()]
            order = np.argsort(np.asarray(regime_map['y_end_std']).ravel())[::-1]
            extra = []
            for item in order:
                point = flat[item]
                if len(chosen) and np.min(np.linalg.norm(chosen - point, axis=1)) < 1e-08:
                    continue
                extra.append(point)
                if len(chosen) + len(extra) >= target:
                    break
            if extra:
                chosen = np.vstack([chosen, np.asarray(extra)]) if len(chosen) else np.asarray(extra)
        checked = []
        for xy in chosen[:remaining]:
            s = np.ones(space.dim)
            s[regime_map['i']] = xy[0]
            s[regime_map['j']] = xy[1]
            z = space.to_unit(s)
            prediction = surrogate.predict(z[None, :])
            predicted_regime = classify_regime(float(prediction['y_end_mean'][0]), float(prediction['y_min_mean'][0]), **study_cfg.get('regimes', {}))
            result = model.simulate(s)
            row = {name: float(value) for name, value in zip(space.names, s)}
            row.update({'map_pair': f"{regime_map['name_i']}__{regime_map['name_j']}", 'predicted_regime': predicted_regime, 'pde_regime': result.regime, 'pred_Y_end': float(prediction['y_end_mean'][0]), 'pde_Y_end': result.y_end, 'pred_Y_min': float(prediction['y_min_mean'][0]), 'pde_Y_min': result.y_min, 'correct': int(predicted_regime == result.regime), 'cache_key': result.cache_key})
            check_rows.append(row)
            checked.append(xy)
            remaining -= 1
            if remaining <= 0:
                break
        checked_by_map[f"{regime_map['name_i']}__{regime_map['name_j']}"] = np.asarray(checked)
        if remaining <= 0:
            break
    checks = pd.DataFrame(check_rows)
    checks.to_csv(data_dir / 'regime_boundary_checks.csv', index=False)
    for regime_map in maps:
        stem = f"{regime_map['name_i']}__{regime_map['name_j']}"
        save_regime_map(regime_map, map_figure_dir / f'{stem}_regimes', checked_by_map.get(stem))
    if maps:
        save_sampling_design(space, X[:n_initial], X[n_initial:], maps[0], figure_dir / 'Fig_adaptive_sampling_top_pair')
    one_factor = pd.DataFrame()
    if cfg.get('one_factor', {}).get('enabled', True):
        one_factor = run_one_factor_analysis(model, space, dict(cfg.get('one_factor', {})), data_dir / 'one_factor')
        save_one_factor_effects(one_factor, figure_dir / 'Fig_one_factor_effects_final')
    blind = None
    if cfg.get('blind_test', {}).get('enabled', False):
        excluded = [X, comparator['X'], study.validation['X']]
        if len(checks):
            excluded.append(space.to_unit(checks[list(space.names)].to_numpy(float)))
        if len(one_factor):
            excluded.append(space.to_unit(one_factor[list(space.names)].drop_duplicates().to_numpy(float)))
        excluded_unit = np.vstack([values for values in excluded if values is not None and len(values)])
        blind_cfg = dict(cfg.get('blind_test', {}))
        blind_cfg['regimes'] = study_cfg.get('regimes', {})
        blind = run_blind_test(model, space, blind_cfg, surrogate, X, y_end, y_min, data_dir / 'blind_test', model_dir, excluded_unit=excluded_unit)
        blind_frame = blind['data']
        blind_metrics = blind['metrics']
        save_blind_validation_panel(blind_frame, blind_metrics, figure_dir / 'Fig_blind_validation_GP')
        save_model_baseline_comparison(blind_metrics, figure_dir / 'Fig_surrogate_baseline_comparison')
    run_manifest = _scan_run_manifest(root)
    representative = _representative_trajectories(root, run_manifest)
    if len(representative):
        save_representative_trajectories(representative, figure_dir / 'Fig_representative_regimes')
    n_adaptive = len(X) - n_initial
    n_comparator = int(study_cfg.get('n_comparator_additional', 72))
    n_validation = len(study.validation['X'])
    n_boundary = len(checks)
    n_one_factor = int(len(one_factor))
    n_blind = int(blind['metrics']['n_blind']) if blind is not None else 0
    total = n_initial + n_adaptive + n_comparator + n_validation + n_boundary + n_one_factor + n_blind
    budget = {'shared_initial': n_initial, 'adaptive_additional': n_adaptive, 'space_filling_additional': n_comparator, 'independent_validation': n_validation, 'regime_boundary_checks': n_boundary, 'one_factor_direct_PDE': n_one_factor, 'untouched_blind_test': n_blind, 'total_unique_PDE_runs': total, 'adaptive_training_size': n_initial + n_adaptive, 'space_filling_training_size': n_initial + n_comparator, 'recommended_adaptive_stop_n_train': adaptive.get('recommended_stop_n_train'), 'direct_sobol_equivalent_evaluations': int(study_cfg.get('sobol_n_base', 16384)) * (space.dim + 2)}
    (report_dir / 'computational_budget.json').write_text(json.dumps(budget, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {'top_mechanisms': sobol_frame.mechanism.head(3).tolist(), 'adaptive_validation': validation_metrics['adaptive'], 'space_filling_validation': validation_metrics['space_filling'], 'blind_test': blind['metrics'] if blind is not None else None, 'budget': budget, 'map_boundary_accuracy': float(checks.correct.mean()) if len(checks) else None, 'n_cached_run_directories': int(len(run_manifest))}
    (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    figures = [('Fig_representative_regimes', 'data/representative_trajectories.csv'), ('Fig_one_factor_effects_final', 'data/one_factor/one_factor_effects.csv'), ('Fig_blind_validation_GP', 'data/blind_test/blind_test_predictions.csv'), ('Fig_surrogate_baseline_comparison', 'data/blind_test/blind_model_comparison.csv'), ('Fig_adaptive_vs_space_filling', 'data/checkpoint_metrics_long.csv'), ('Fig_adaptive_sampling_top_pair', 'data/adaptive/acquisition_history.csv'), ('Fig_Sobol_final', 'data/sobol_indices_final.csv'), ('regime_maps', 'data/regime_maps')]
    pd.DataFrame(figures, columns=['figure_stem', 'source_data']).to_csv(root / 'figure_manifest.csv', index=False)
    _artifact_manifest(root)
    return summary
