"""
Génère l'image kmeans_iterations.png : K-means en action sur un dataset 2D.

Quatre panneaux : itérations 0, 1, 2, 5. Plus la courbe de distortion J(t).
Les centroïdes (croix) bougent et les points se réorganisent selon leur cluster.

Output : kmeans_iterations.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# ---------------------------------------------------------------------
# Génération du dataset : 3 clusters bien séparés
# ---------------------------------------------------------------------
np.random.seed(42)
N_per_cluster = 80
true_centers = np.array([[-3, -1], [2, 3], [3, -2]])
true_stds = [0.7, 0.9, 0.6]

X = np.vstack([
    np.random.multivariate_normal(c, s**2 * np.eye(2), N_per_cluster)
    for c, s in zip(true_centers, true_stds)
])
N = X.shape[0]

# ---------------------------------------------------------------------
# K-means manuel (pour avoir tous les snapshots intermédiaires)
# ---------------------------------------------------------------------
K = 3

# Init : tirer 3 points du dataset (mauvaise init volontaire, qui force qq itérations)
init_idx = [0, 1, 5]  # tous dans le même cluster en bas à gauche
centroids = X[init_idx].copy()

snapshots = [centroids.copy()]
assignments_history = []
distortions = []

def assign(X, centroids):
    """E-step : assigner chaque point au centroïde le plus proche."""
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)

def update(X, assignments, K):
    """M-step : recalculer les centroïdes."""
    new_centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        mask = (assignments == k)
        if mask.sum() > 0:
            new_centroids[k] = X[mask].mean(axis=0)
    return new_centroids

def distortion(X, assignments, centroids):
    """Calcul de J."""
    total = 0.0
    for k in range(centroids.shape[0]):
        mask = (assignments == k)
        if mask.sum() > 0:
            total += np.sum(np.linalg.norm(X[mask] - centroids[k], axis=1) ** 2)
    return total

# Itération 0 : init seulement, assignments calculés
assignments = assign(X, centroids)
assignments_history.append(assignments.copy())
distortions.append(distortion(X, assignments, centroids))

# Itérations suivantes
for it in range(1, 10):
    centroids = update(X, assignments, K)
    assignments = assign(X, centroids)
    snapshots.append(centroids.copy())
    assignments_history.append(assignments.copy())
    distortions.append(distortion(X, assignments, centroids))

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 5, height_ratios=[3, 2], width_ratios=[1, 1, 1, 1, 1])

# 4 panneaux d'itérations
iters_to_show = [0, 1, 2, 5]
colors = ['#0F6E56', '#185FA5', '#D85A30']

for col, it in enumerate(iters_to_show):
    ax = fig.add_subplot(gs[0, col])
    cents = snapshots[it]
    asgn = assignments_history[it]
    for k in range(K):
        mask = (asgn == k)
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=20, alpha=0.7,
                   edgecolor='white', linewidth=0.3)
    ax.scatter(cents[:, 0], cents[:, 1], marker='X', s=250, c='black',
               edgecolor='white', linewidth=2, zorder=5)
    ax.set_title(f'Itération {it}', fontsize=12)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-4, 5.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# Panneau distortion
ax_d = fig.add_subplot(gs[0, 4])
ax_d.plot(range(len(distortions)), distortions, '-o', color='#A32D2D', linewidth=2,
          markersize=6)
ax_d.set_xlabel('itération', fontsize=11)
ax_d.set_ylabel('distortion $J$', fontsize=11)
ax_d.set_title('Décroissance monotone', fontsize=12)
ax_d.grid(True, alpha=0.3, linestyle=':')
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# Titre global + légende
fig.suptitle('K-means : alternation E-step (assignment) / M-step (update centroïdes)',
             fontsize=13, y=0.98)

fig.text(0.5, 0.32,
         'À chaque itération : E-step (assigner chaque point au centroïde le plus proche), '
         'puis M-step (recalculer chaque centroïde comme la moyenne de ses points).\n'
         'La distortion $J = \\sum_n \\sum_k r_{nk} \\|x_n - \\mu_k\\|^2$ '
         'décroît strictement à chaque étape jusqu\'à convergence.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.18, 1, 0.95])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'kmeans_iterations.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
