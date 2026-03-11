import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap_pr(df, title, filename=None):
    """Plots a heatmap of best_fitness by parameters p and r.
    
    Args:
        df: DataFrame with columns 'p', 'r', and 'best_fitness'.
        title: Title for the plot.
        filename: Optional filename to save the plot.
    """
    required = {"p", "r", "best_fitness"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    pivot = df.pivot(index="p", columns="r", values="best_fitness").sort_index().sort_index(axis=1)
    values = pivot.to_numpy(dtype=float)
    p_index = pivot.index.to_numpy()
    r_cols = pivot.columns.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("r")
    ax.set_ylabel("p")
    ax.set_xticks(np.arange(len(r_cols)))
    ax.set_yticks(np.arange(len(p_index)))
    ax.set_xticklabels(r_cols)
    ax.set_yticklabels(p_index)

    # Annotations (values in cells)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3g}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, label="best_fitness (final)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_convergence(evals, histories, title, filename=None):
    """Plots the convergence curve (mean ± std) of the best fitness.
    
    Args:
        evals: Array of evaluation counts (x-axis).
        histories: List of arrays containing best fitness history for each run.
        title: Title for the plot.
        filename: Optional filename to save the plot.
    """
    histories = np.asarray(histories)
    mean_best = np.mean(histories, axis=0)
    std_best = np.std(histories, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(evals, mean_best, label='Mean Best Fitness')
    plt.fill_between(evals, mean_best - std_best, mean_best + std_best, alpha=0.2, label='Std Dev')
    plt.title(title)
    plt.xlabel('Evaluations')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_surface_3d(X, Y, Z, title, xlabel='x₁', ylabel='x₂', zlabel='f(x)', filename=None):
    """Plots a 3D surface of a function.
    
    Args:
        X, Y: Meshgrid coordinates.
        Z: Function values.
        title: Plot title.
        xlabel, ylabel, zlabel: Labels for axes.
        filename: Optional filename to save.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_contour(X, Y, Z, title, xlabel='x₁', ylabel='x₂', filename=None, global_min=None):
    """Plots a contour plot of a function.
    
    Args:
        X, Y: Meshgrid coordinates.
        Z: Function values.
        title: Plot title.
        xlabel, ylabel: Labels for axes.
        filename: Optional filename to save.
        global_min: Tuple (x, y) of the global minimum to mark.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if global_min:
        ax.plot(global_min[0], global_min[1], 'r*', markersize=15, label='Global minimum')
        ax.legend()
    fig.colorbar(contour, ax=ax)
    if filename:
        plt.savefig(filename)
    plt.show()
