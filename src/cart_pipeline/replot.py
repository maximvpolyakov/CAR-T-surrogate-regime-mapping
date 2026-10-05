from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from .config import MechanismSpace, load_yaml
from .plots import save_blind_validation_panel, save_model_baseline_comparison, save_one_factor_effects, save_performance_comparison, save_regime_map, save_representative_trajectories, save_sampling_design, save_sobol_plot

def _load_map(path):
    frame = pd.read_csv(path)
    names = [column for column in frame.columns if column not in {'Y_end_mean', 'Y_min_mean', 'Y_end_std', 'Y_min_std', 'regime'}]
    if len(names) != 2:
        raise ValueError(str(path))
    first, second = names
    x = np.sort(frame[first].unique())
    y = np.sort(frame[second].unique())

    def grid(column):
        return frame.pivot(index=second, columns=first, values=column).reindex(index=y, columns=x).to_numpy()
    space = MechanismSpace()
    return {'name_i': first, 'name_j': second, 'x': x, 'y': y, 'i': space.names.index(first), 'j': space.names.index(second), 'y_end': grid('Y_end_mean'), 'y_min': grid('Y_min_mean'), 'y_end_std': grid('Y_end_std'), 'y_min_std': grid('Y_min_std'), 'regime': grid('regime')}

def rebuild_figures(results_dir, output_dir=None):
    root = Path(results_dir)
    data = root / 'data'
    output = Path(output_dir) if output_dir else root / 'figures_rebuilt'
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(root / 'config_used.yaml')
    space = MechanismSpace(lower=float(cfg['study'].get('lower', 0.6)), upper=float(cfg['study'].get('upper', 1.4)))
    sobol_frame = pd.read_csv(data / 'sobol_indices_final.csv')
    ordered = sobol_frame.set_index('mechanism').reindex(space.names)
    sobol = SimpleNamespace(first=ordered.S_i.to_numpy(), total=ordered.ST_i.to_numpy())
    intervals = {'first_low': ordered.S_i_low.to_numpy(), 'first_high': ordered.S_i_high.to_numpy(), 'total_low': ordered.ST_i_low.to_numpy(), 'total_high': ordered.ST_i_high.to_numpy()}
    save_sobol_plot(space.names, sobol, output / 'Fig_Sobol_final', intervals)
    adaptive_checkpoints = json.loads((data / 'adaptive' / 'adaptive_checkpoints.json').read_text(encoding='utf-8'))
    comparator_checkpoints = json.loads((data / 'comparator' / 'comparator_checkpoints.json').read_text(encoding='utf-8'))
    save_performance_comparison(adaptive_checkpoints, comparator_checkpoints, output / 'Fig_adaptive_vs_space_filling')
    map_paths = sorted((data / 'regime_maps').glob('*.csv'))
    maps = []
    checks_path = data / 'regime_boundary_checks.csv'
    try:
        checks = pd.read_csv(checks_path) if checks_path.exists() and checks_path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        checks = pd.DataFrame()
    map_output = output / 'regime_maps'
    map_output.mkdir(exist_ok=True)
    for path in map_paths:
        regime_map = _load_map(path)
        maps.append(regime_map)
        points = None
        if len(checks):
            selected = checks[checks.map_pair == path.stem]
            if len(selected):
                points = selected[[regime_map['name_i'], regime_map['name_j']]].to_numpy()
        save_regime_map(regime_map, map_output / f'{path.stem}_regimes', points)
    training = pd.read_csv(data / 'adaptive' / 'training_data.csv')
    if maps and 'stage' in training.columns:
        initial = training.loc[training.stage == 'initial', [f'z_{name}' for name in space.names]].to_numpy()
        adaptive = training.loc[training.stage == 'adaptive', [f'z_{name}' for name in space.names]].to_numpy()
        save_sampling_design(space, initial, adaptive, maps[0], output / 'Fig_adaptive_sampling_top_pair')
    trajectories = data / 'representative_trajectories.csv'
    if trajectories.exists():
        save_representative_trajectories(pd.read_csv(trajectories), output / 'Fig_representative_regimes')
    one_factor = data / 'one_factor' / 'one_factor_effects.csv'
    if one_factor.exists():
        save_one_factor_effects(pd.read_csv(one_factor), output / 'Fig_one_factor_effects_final')
    blind_predictions = data / 'blind_test' / 'blind_test_predictions.csv'
    blind_metrics = data / 'blind_test' / 'blind_test_metrics.json'
    if blind_predictions.exists() and blind_metrics.exists():
        frame = pd.read_csv(blind_predictions)
        metrics = json.loads(blind_metrics.read_text(encoding='utf-8'))
        save_blind_validation_panel(frame, metrics, output / 'Fig_blind_validation_GP')
        save_model_baseline_comparison(metrics, output / 'Fig_surrogate_baseline_comparison')

def main():
    parser = ArgumentParser()
    parser.add_argument('--results', default='results')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    rebuild_figures(args.results, args.output)
if __name__ == '__main__':
    main()
