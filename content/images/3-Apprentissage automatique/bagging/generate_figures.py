"""
Génère 4 figures 3D illustrant l'idée du bagging :
- 3 arbres de décision individuels fittés sur 3 bootstraps différents
- la moyenne des 3 (= ce que fait le bagging)

Usage : python generate_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12,
})

try:
    OUT = Path(__file__).parent
except NameError:
    OUT = Path.cwd()


# ============================================================
# Dataset 2D : deux "cloches" + un creux
# ============================================================
rng = np.random.default_rng(0)
n_train = 80


def true_function(x1, x2):
    """Fonction smooth avec deux pics et un creux."""
    return (
        2.5 * np.exp(-((x1 - 0.3) ** 2 + (x2 - 0.3) ** 2) / 0.08)
        + 1.8 * np.exp(-((x1 - 0.75) ** 2 + (x2 - 0.7) ** 2) / 0.06)
        - 1.2 * np.exp(-((x1 - 0.6) ** 2 + (x2 - 0.2) ** 2) / 0.05)
    )


X1_train = rng.uniform(0, 1, n_train)
X2_train = rng.uniform(0, 1, n_train)
y_train = true_function(X1_train, X2_train) + rng.normal(0, 0.15, n_train)
X_train = np.column_stack([X1_train, X2_train])

# Grille pour le rendu
n_grid = 60
x1g = np.linspace(0, 1, n_grid)
x2g = np.linspace(0, 1, n_grid)
X1G, X2G = np.meshgrid(x1g, x2g)
Xg = np.column_stack([X1G.ravel(), X2G.ravel()])


# ============================================================
# 3 arbres sur 3 bootstraps différents
# ============================================================
M = 3
seeds = [42, 123, 7]
predictions = []

for seed in seeds:
    rng_local = np.random.default_rng(seed)
    idx = rng_local.choice(n_train, size=n_train, replace=True)  # bootstrap
    Xb, yb = X_train[idx], y_train[idx]

    tree = DecisionTreeRegressor(max_depth=3, random_state=seed)
    tree.fit(Xb, yb)
    preds = tree.predict(Xg).reshape(n_grid, n_grid)
    predictions.append(preds)

# Moyenne des 3 arbres = bagging
mean_pred = np.mean(predictions, axis=0)


# ============================================================
# Plot helper — surface 3D + points training en transparence
# ============================================================
all_z = np.concatenate([p.ravel() for p in predictions] + [mean_pred.ravel()])
zmin, zmax = all_z.min(), all_z.max()


def plot_surface(Z, title, filename, color="viridis"):
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X1G, X2G, Z, cmap=color, edgecolor="none",
        alpha=0.9, vmin=zmin, vmax=zmax,
        rstride=1, cstride=1, antialiased=True,
    )

    # Training set en superposition pour le contexte
    ax.scatter(X1_train, X2_train, y_train, c="black", s=8, alpha=0.3,
               depthshade=True)

    ax.set_xlabel("$x_1$", labelpad=6)
    ax.set_ylabel("$x_2$", labelpad=6)
    ax.set_zlabel(r"$\hat f(x)$", labelpad=2)
    ax.set_title(title, pad=10)
    ax.set_zlim(zmin - 0.2, zmax + 0.2)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Génération des 4 figures
# ============================================================
if __name__ == "__main__":
    for i, preds in enumerate(predictions, 1):
        plot_surface(preds, f"Arbre {i} (bootstrap #{i})",
                     f"fig_bag_tree{i}.png", color="viridis")
        print(f"✓ Arbre {i}")

    plot_surface(mean_pred, "Moyenne des 3 arbres (bagging)",
                 "fig_bag_mean.png", color="plasma")
    print("✓ Moyenne")

    print(f"\nFigures sauvegardées dans {OUT}")
