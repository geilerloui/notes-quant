"""
Génère l'image gmm_responsibilities.png : visualisation des responsabilités gamma_nk.

Deux panneaux côte à côte :
- À gauche : K-means avec assignments durs (3 couleurs pures).
- À droite : GMM avec responsabilités douces (mélange RGB des 3 couleurs).

Le contraste rend visible la notion d'incertitude qu'EM-GMM gagne sur K-means.

Output : gmm_responsibilities.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

# ---------------------------------------------------------------------
# Génération du dataset : 3 clusters qui se chevauchent un peu
# ---------------------------------------------------------------------
np.random.seed(7)
N_per_cluster = 100
true_centers = np.array([[-2, 0], [1.5, 2], [1.5, -2]])
true_covs = [
    np.array([[0.8, 0.0], [0.0, 0.8]]),
    np.array([[0.7, 0.3], [0.3, 0.7]]),
    np.array([[0.7, -0.2], [-0.2, 0.7]]),
]

X = np.vstack([
    np.random.multivariate_normal(c, cov, N_per_cluster)
    for c, cov in zip(true_centers, true_covs)
])

# ---------------------------------------------------------------------
# Fit K-means manuel (pour assignments durs)
# ---------------------------------------------------------------------
K = 3
np.random.seed(1)
init_idx = np.random.choice(len(X), K, replace=False)
centroids_km = X[init_idx].copy()

for _ in range(20):
    dists = np.linalg.norm(X[:, None, :] - centroids_km[None, :, :], axis=2)
    asgn = np.argmin(dists, axis=1)
    for k in range(K):
        if (asgn == k).sum() > 0:
            centroids_km[k] = X[asgn == k].mean(axis=0)

# ---------------------------------------------------------------------
# Fit GMM manuel (pour responsabilités douces)
# ---------------------------------------------------------------------
# Init avec K-means pour rapide convergence
mus = centroids_km.copy()
Sigmas = [np.eye(2) for _ in range(K)]
pis = np.ones(K) / K

for _ in range(50):
    # E-step : responsabilités
    resp = np.zeros((len(X), K))
    for k in range(K):
        resp[:, k] = pis[k] * multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k])
    resp = resp / resp.sum(axis=1, keepdims=True)

    # M-step
    Nk = resp.sum(axis=0)
    for k in range(K):
        mus[k] = (resp[:, k:k+1] * X).sum(axis=0) / Nk[k]
        diff = X - mus[k]
        Sigmas[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k]
    pis = Nk / len(X)

# Re-calcul des responsabilités finales
resp_final = np.zeros((len(X), K))
for k in range(K):
    resp_final[:, k] = pis[k] * multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k])
resp_final = resp_final / resp_final.sum(axis=1, keepdims=True)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# Couleurs des clusters (RGB-ish pour pouvoir les mélanger)
# Rouge, vert, bleu façon palette adoucie
cluster_colors = np.array([
    [0.85, 0.30, 0.18],  # rouge orangé (cluster 0)
    [0.10, 0.55, 0.30],  # vert (cluster 1)
    [0.15, 0.40, 0.75],  # bleu (cluster 2)
])

# ---- Panneau gauche : K-means (assignments durs) ----
ax = ax1
asgn_km = np.argmin(np.linalg.norm(X[:, None, :] - centroids_km[None, :, :], axis=2), axis=1)
colors_km = cluster_colors[asgn_km]
ax.scatter(X[:, 0], X[:, 1], c=colors_km, s=35, alpha=0.85,
           edgecolor='white', linewidth=0.5)
ax.scatter(centroids_km[:, 0], centroids_km[:, 1], marker='X', s=300, c='black',
           edgecolor='white', linewidth=2, zorder=5)
ax.set_title('K-means : assignment dur\n$r_{nk} \\in \\{0, 1\\}$',
             fontsize=12, pad=10)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---- Panneau droit : GMM (responsabilités douces) ----
ax = ax2
# Chaque point est colorié selon le mélange de ses responsabilités
colors_gmm = resp_final @ cluster_colors
ax.scatter(X[:, 0], X[:, 1], c=colors_gmm, s=35, alpha=0.95,
           edgecolor='white', linewidth=0.5)

# Tracer les ellipses 1-sigma pour visualiser les composantes apprises
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])
for k in range(K):
    L = np.linalg.cholesky(Sigmas[k])
    ellipse = mus[k:k+1].T + L @ circle
    ax.plot(ellipse[0], ellipse[1], color=cluster_colors[k], linewidth=1.5,
            linestyle='--', alpha=0.7)

ax.scatter(mus[:, 0], mus[:, 1], marker='X', s=300, c='black',
           edgecolor='white', linewidth=2, zorder=5)
ax.set_title('GMM : assignment doux\n$\\gamma_{nk} \\in [0, 1]$, $\\sum_k \\gamma_{nk} = 1$',
             fontsize=12, pad=10)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Légende explicative sous le panneau droit
fig.text(0.5, 0.02,
         'À gauche : chaque point a une couleur pure (= un seul cluster, sans nuance).\n'
         'À droite : les points aux frontières des clusters ont des couleurs intermédiaires (= mélange des responsabilités).',
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.07, 1, 1])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gmm_responsibilities.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
