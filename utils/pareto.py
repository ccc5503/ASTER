import numpy as np
import matplotlib.pyplot as plt

def is_dominated(candidate, others):
    for other in others:
        if np.all(other >= candidate) and np.any(other > candidate):
            return True
    return False

def get_pareto_front(points):
    pareto = []
    for p in points:
        if not is_dominated(p, points):
            pareto.append(p)
    return np.array(pareto)

def plot_pareto_2d(points, pareto, x_idx, y_idx, labels, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, x_idx], points[:, y_idx], alpha=0.3, label='All Points')
    plt.scatter(pareto[:, x_idx], pareto[:, y_idx], color='red', label='Pareto Frontier')
    plt.xlabel(labels[x_idx])
    plt.ylabel(labels[y_idx])
    plt.title(f"Pareto Frontier: {labels[x_idx]} vs {labels[y_idx]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_radar(points, labels, title="Pareto Radar Plot", save_path=None):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for i, p in enumerate(points):
        values = p.tolist() + [p[0]]
        ax.plot(angles, values, label=f"Solution {i+1}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    if save_path:
        plt.savefig(save_path)
    plt.close()