from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .regimes import REGIME_TO_INT

def _save(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=400, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)

def save_sobol_plot(names, sobol, path, intervals=None):
    order = np.argsort(np.asarray(sobol.total))[::-1]
    names = np.asarray(names)[order]
    first = np.asarray(sobol.first)[order]
    total = np.asarray(sobol.total)[order]
    y = np.arange(len(names))
    h = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if intervals is None:
        e1 = e2 = None
    else:
        lo1 = np.asarray(intervals['first_low'])[order]
        hi1 = np.asarray(intervals['first_high'])[order]
        lo2 = np.asarray(intervals['total_low'])[order]
        hi2 = np.asarray(intervals['total_high'])[order]
        e1 = np.vstack([np.maximum(first - lo1, 0), np.maximum(hi1 - first, 0)])
        e2 = np.vstack([np.maximum(total - lo2, 0), np.maximum(hi2 - total, 0)])
    ax.barh(y - h / 2, first, height=h, label='First-order $S_i$', xerr=e1, capsize=2)
    ax.barh(y + h / 2, total, height=h, label='Total-order $S_{T_i}$', xerr=e2, capsize=2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('Sobol index')
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    ax.grid(axis='x', alpha=0.2)
    _save(fig, path)

def save_validation_plot(y_true, y_pred, label, path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=34)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    pad = 0.04 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], '--', linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f'Full PDE: {label}')
    ax.set_ylabel(f'GP prediction: {label}')
    ax.grid(alpha=0.15)
    _save(fig, path)

def save_validation_panel(yend_true, yend_pred, ymin_true, ymin_pred, metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, yt, yp, label, key in [(axes[0], np.asarray(yend_true), np.asarray(yend_pred), '$Y_{end}$', 'Y_end'), (axes[1], np.asarray(ymin_true), np.asarray(ymin_pred), '$Y_{min}$', 'Y_min')]:
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.04 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad
        ax.scatter(yt, yp, s=30)
        ax.plot([lo, hi], [lo, hi], '--', lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.18)
        ax.set_xlabel(f'Full PDE {label}')
        ax.set_ylabel(f'GP prediction {label}')
        m = metrics[key]
        ax.text(0.04, 0.96, f"$R^2={m['r2']:.6f}$" + '\n' + f"NRMSE$={m['nrmse']:.4f}$", transform=ax.transAxes, va='top')
    _save(fig, path)

def save_confusion_matrix(matrix, labels, path):
    cm = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel('GP-predicted regime')
    ax.set_ylabel('Full PDE regime')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{int(cm[i, j])}', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Validation cases')
    _save(fig, path)

def save_regime_map(regime_map, path, checked_points=None):
    regimes = np.asarray(regime_map['regime'])
    unique = np.unique(regimes)
    extent = [regime_map['x'][0], regime_map['x'][-1], regime_map['y'][0], regime_map['y'][-1]]
    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    if len(unique) < 2:
        im = ax.imshow(regime_map['y_end'], origin='lower', extent=extent, aspect='auto', interpolation='bilinear')
        fig.colorbar(im, ax=ax, label='Predicted $Y_{end}$')
        ax.set_title(f'No regime boundary detected ({unique[0]})')
    else:
        code = np.vectorize(REGIME_TO_INT.get)(regimes)
        base = plt.get_cmap('tab10')
        cmap = ListedColormap([base(i) for i in range(4)])
        im = ax.imshow(code, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap=cmap, vmin=-0.5, vmax=3.5)
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cb.ax.set_yticklabels(['DC', 'PR', 'RL', 'Failure'])
    if checked_points is not None and len(checked_points):
        ax.scatter(checked_points[:, 0], checked_points[:, 1], marker='x', s=42, linewidth=1.2, label='Full PDE checks')
        ax.legend(frameon=False, loc='best')
    ax.set_xlabel(regime_map['name_i'])
    ax.set_ylabel(regime_map['name_j'])
    _save(fig, path)

def save_continuous_map(regime_map, field, label, path):
    extent = [regime_map['x'][0], regime_map['x'][-1], regime_map['y'][0], regime_map['y'][-1]]
    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    im = ax.imshow(np.asarray(regime_map[field]), origin='lower', extent=extent, aspect='auto', interpolation='bilinear')
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel(regime_map['name_i'])
    ax.set_ylabel(regime_map['name_j'])
    _save(fig, path)

def save_performance_comparison(adaptive_cp, comparator_cp, path):
    a = pd.DataFrame(adaptive_cp)
    b = pd.DataFrame(comparator_cp)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for df, label in [(a, 'Adaptive'), (b, 'Space-filling')]:
        axes[0].plot(df.n_train, df.nrmse_end, marker='o', label=label)
        axes[1].plot(df.n_train, df.nrmse_min, marker='o', label=label)
        axes[2].plot(df.n_train, df.regime_accuracy, marker='o', label=label)
    axes[0].set_ylabel('NRMSE')
    axes[0].set_title('$Y_{end}$')
    axes[1].set_ylabel('NRMSE')
    axes[1].set_title('$Y_{min}$')
    axes[2].set_ylabel('Regime accuracy')
    axes[2].set_ylim(0, 1.03)
    axes[2].set_title('DC / PR / RL / Failure')
    for ax in axes:
        ax.set_xlabel('Full PDE training runs')
        ax.grid(alpha=0.2)
    axes[2].legend(frameon=False)
    _save(fig, path)

def save_sobol_reference_error(adaptive_cp, comparator_cp, ref_first, ref_total, path):
    rows = []
    for method, cps in [('Adaptive', adaptive_cp), ('Space-filling', comparator_cp)]:
        for r in cps:
            err = float(np.mean(np.r_[np.abs(np.asarray(r['S']) - ref_first), np.abs(np.asarray(r['ST']) - ref_total)]))
            rows.append({'method': method, 'n_train': r['n_train'], 'error': err})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for method, g in df.groupby('method'):
        ax.plot(g.n_train, g.error, marker='o', label=method)
    ax.set_xlabel('Full PDE training runs')
    ax.set_ylabel('Mean absolute Sobol deviation')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, path)
    return df

def save_sobol_convergence(checkpoints, names, path):
    cps = list(checkpoints)
    if_empty = len(cps) == 0
    if if_empty:
        return
    ST = np.asarray([r['ST'] for r in cps])
    n = np.asarray([r['n_train'] for r in cps])
    final = np.abs(ST[-1])
    top = np.argsort(final)[::-1][:4]
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for i in top:
        ax.plot(n, ST[:, i], marker='o', label=names[i])
    ax.set_xlabel('Full PDE training runs')
    ax.set_ylabel('Total-order index $S_{T_i}$')
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    _save(fig, path)

def save_sampling_design(space, initial_X, adaptive_X, map_obj, path):
    i, j = (map_obj['i'], map_obj['j'])
    extent = [map_obj['x'][0], map_obj['x'][-1], map_obj['y'][0], map_obj['y'][-1]]
    fig, ax = plt.subplots(figsize=(6.0, 5.1))
    im = ax.imshow(map_obj['y_end'], origin='lower', extent=extent, aspect='auto', interpolation='bilinear', alpha=0.75)
    fig.colorbar(im, ax=ax, label='GP prediction $Y_{end}$')
    ip = space.from_unit(np.asarray(initial_X))
    ap = space.from_unit(np.asarray(adaptive_X))
    ax.scatter(ip[:, i], ip[:, j], s=20, marker='o', label='Initial LHS')
    if len(ap):
        ax.scatter(ap[:, i], ap[:, j], s=26, marker='x', label='Adaptive additions')
    ax.set_xlabel(map_obj['name_i'])
    ax.set_ylabel(map_obj['name_j'])
    ax.legend(frameon=False)
    _save(fig, path)

def save_representative_trajectories(traj_df, path):
    if traj_df is None or len(traj_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for name, g in traj_df.groupby('scenario', sort=False):
        ax.plot(g.t, g.R, lw=2.1, label=name.replace('_', ' '))
    ax.axhline(1.0, ls=':', lw=1)
    ax.set_xlabel('Time, days')
    ax.set_ylabel('Normalised tumour burden $R(t)$')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, path)

def save_numerical_verification(df, path):
    d = pd.DataFrame(df)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for name, g in d[d.variant.isin(['N40_dt001', 'N50_dt001', 'N60_dt001'])].groupby('scenario'):
        g = g.sort_values('N')
        axes[0].plot(g.N, g.Y_end, marker='o', label=name.replace('_', ' '))
    axes[0].set_xlabel('Grid nodes per dimension')
    axes[0].set_ylabel('$Y_{end}$')
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    dtrows = d[d.variant.isin(['N40_dt001', 'N40_dt0005'])].copy()
    for name, g in dtrows.groupby('scenario'):
        g = g.sort_values('dt')
        axes[1].plot(g.dt, g.Y_end, marker='o', label=name.replace('_', ' '))
    axes[1].set_xlabel('Time step $\\Delta t$')
    axes[1].set_ylabel('$Y_{end}$')
    axes[1].grid(alpha=0.2)
    _save(fig, path)

def save_regime_counts(regimes, path):
    order = ['DC', 'PR', 'RL', 'Failure']
    vals = pd.Series(regimes).value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    ax.bar(vals.index, vals.values)
    ax.set_ylabel('Number of full PDE runs')
    ax.set_xlabel('Computational regime')
    ax.grid(axis='y', alpha=0.2)
    _save(fig, path)

def save_one_factor_effects(df, path):
    d = pd.DataFrame(df).copy()
    ref = d[d['scenario'] == 'reference']
    if len(ref) == 0:
        return
    y0 = float(ref.iloc[0]['Y_end'])
    labels = {'s_cyt': 'Cytotoxicity', 's_esc': 'Antigen escape', 's_mig': 'Chemotaxis', 's_exh': 'Exhaustion', 's_env': 'Immunosuppression', 's_hyp': 'Hypoxia'}
    order = ['s_cyt', 's_esc', 's_mig', 's_exh', 's_env', 's_hyp']
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    y = np.arange(len(order))
    lows = []
    highs = []
    for mech in order:
        q = d[d['mechanism'] == mech]
        lo = q.loc[np.isclose(q['multiplier'], 0.6), 'Y_end']
        hi = q.loc[np.isclose(q['multiplier'], 1.4), 'Y_end']
        lows.append(float(lo.iloc[0]) - y0 if len(lo) else np.nan)
        highs.append(float(hi.iloc[0]) - y0 if len(hi) else np.nan)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    for yi, (lo, hi) in enumerate(zip(lows, highs)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], lw=1, alpha=0.65)
    ax.scatter(lows, y, s=48, marker='o', label='$0.6\\times$ reference')
    ax.scatter(highs, y, s=48, marker='s', label='$1.4\\times$ reference')
    ax.axvline(0, ls='--', lw=1)
    ax.set_yticks(y, [labels[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel('Change in $Y_{end}$ relative to reference')
    ax.legend(frameon=False, loc='best')
    ax.grid(axis='x', alpha=0.18)
    _save(fig, path)

def save_model_baseline_comparison(metrics, path):
    order = ['GP', 'Quadratic', 'Linear']
    keys = {'GP': 'gp', 'Quadratic': 'quadratic', 'Linear': 'linear'}
    x = np.arange(len(order))
    w = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    end = [metrics[keys[k]]['Y_end']['nrmse'] for k in order]
    ymin = [metrics[keys[k]]['Y_min']['nrmse'] for k in order]
    axes[0].bar(x - w / 2, end, width=w, label='$Y_{end}$')
    axes[0].bar(x + w / 2, ymin, width=w, label='$Y_{min}$')
    axes[0].set_xticks(x, order)
    axes[0].set_ylabel('NRMSE')
    axes[0].set_title('Blind-test regression error')
    axes[0].legend(frameon=False)
    axes[0].grid(axis='y', alpha=0.2)
    acc = [metrics[keys[k]]['regime_accuracy'] for k in order]
    axes[1].bar(x, acc, width=0.56)
    axes[1].set_xticks(x, order)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel('Regime accuracy')
    axes[1].set_title('DC / PR / RL / Failure')
    axes[1].grid(axis='y', alpha=0.2)
    _save(fig, path)

def save_blind_validation_panel(df, metrics, path):
    d = pd.DataFrame(df)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, true_col, pred_col, label, key in [(axes[0], 'Y_end_PDE', 'Y_end_GP', '$Y_{end}$', 'Y_end'), (axes[1], 'Y_min_PDE', 'Y_min_GP', '$Y_{min}$', 'Y_min')]:
        yt = d[true_col].to_numpy(float)
        yp = d[pred_col].to_numpy(float)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = 0.04 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad
        ax.scatter(yt, yp, s=30)
        ax.plot([lo, hi], [lo, hi], '--', lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.18)
        ax.set_xlabel(f'Full PDE {label}')
        ax.set_ylabel(f'GP prediction {label}')
        m = metrics['gp'][key]
        cov = metrics['gp'].get(f'{key}_coverage_95', float('nan'))
        ax.text(0.04, 0.96, f"$R^2={m['r2']:.6f}$" + '\n' + f"NRMSE$={m['nrmse']:.4f}$" + '\n' + f'95% coverage$={cov:.3f}$', transform=ax.transAxes, va='top')
    _save(fig, path)
