"""
Génère l'image gmm_iterations.png : EM sur GMM en action.

Quatre panneaux : itérations 0 (init), 1, 5, 30 (convergence).
Pour chaque panneau : points colorés par responsabilités, ellipses 1-sigma et 2-sigma,
heatmap de la densité de mélange en arrière-plan.

Output : gmm_iterations.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

# ---------------------------------------------------------------------
# Dataset : 3 clusters ellipsoidaux (formes différentes pour bien voir)
# ---------------------------------------------------------------------
np.random.seed(11)
N_per_cluster = 120
true_centers = np.array([[-2.5, 0.5], [1.5, 2.5], [1.5, -2.0]])
true_covs = [
    np.array([[1.2, -0.3], [-0.3, 0.5]]),  # ellipse oblique
    np.array([[0.5, 0.0], [0.0, 0.5]]),    # cercle
    np.array([[0.8, 0.4], [0.4, 0.6]]),    # ellipse oblique
]

X = np.vstack([
    np.random.multivariate_normal(c, cov, N_per_cluster)
    for c, cov in zip(true_centers, true_covs)
])

# ---------------------------------------------------------------------
# EM manuel avec snapshots
# ---------------------------------------------------------------------
K = 3
N = len(X)

# Init volontairement mauvaise pour voir l'évolution
np.random.seed(3)
mus = X[np.random.choice(N, K, replace=False)].copy()
Sigmas = [np.eye(2) * 2.0 for _ in range(K)]  # init grosse covariance
pis = np.ones(K) / K

snapshots = []  # liste de dicts avec mus, Sigmas, pis, resp

def compute_resp(X, mus, Sigmas, pis):
    """E-step : calcul des responsabilités."""
    K = len(pis)
    resp = np.zeros((len(X), K))
    for k in range(K):
        resp[:, k] = pis[k] * multivariate_normal.pdf(X, mean=mus[k], cov=Sigmas[k])
    return resp / resp.sum(axis=1, keepdims=True)

def m_step(X, resp):
    """M-step : update des paramètres."""
    Nk = resp.sum(axis=0)
    K = resp.shape[1]
    mus = np.zeros((K, X.shape[1]))
    Sigmas = []
    for k in range(K):
        mus[k] = (resp[:, k:k+1] * X).sum(axis=0) / Nk[k]
        diff = X - mus[k]
        Sigmas.append((resp[:, k:k+1] * diff).T @ diff / Nk[k])
    pis = Nk / len(X)
    return mus, Sigmas, pis

# Snapshot initial
resp = compute_resp(X, mus, Sigmas, pis)
snapshots.append({'mus': mus.copy(), 'Sigmas': [s.copy() for s in Sigmas],
                  'pis': pis.copy(), 'resp': resp.copy(), 'it': 0})

# Itérations
for it in range(1, 35):
    mus, Sigmas, pis = m_step(X, resp)
    resp = compute_resp(X, mus, Sigmas, pis)
    snapshots.append({'mus': mus.copy(), 'Sigmas': [s.copy() for s in Sigmas],
                      'pis': pis.copy(), 'resp': resp.copy(), 'it': it})

# ---------------------------------------------------------------------
# Plot : 4 panneaux pour itérations 0, 1, 5, 30
# ---------------------------------------------------------------------
iters_to_show = [0, 1, 5, 30]
fig, axes = plt.subplots(1, 4, figsize=(17, 5))

cluster_colors = np.array([
    [0.85, 0.30, 0.18],
    [0.10, 0.55, 0.30],
    [0.15, 0.40, 0.75],
])

# Grille pour la heatmap de densité
xx, yy = np.meshgrid(np.linspace(-5, 5, 150), np.linspace(-5, 5, 150))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

for ax, it in zip(axes, iters_to_show):
    snap = snapshots[it]
    mus_s = snap['mus']
    Sigmas_s = snap['Sigmas']
    pis_s = snap['pis']
    resp_s = snap['resp']

    # Heatmap de la densité de mélange
    density = np.zeros(len(grid))
    for k in range(K):
        density += pis_s[k] * multivariate_normal.pdf(grid, mean=mus_s[k], cov=Sigmas_s[k])
    density = density.reshape(xx.shape)
    ax.contourf(xx, yy, density, levels=12, cmap='Greys', alpha=0.5)

    # Points colorés par responsabilités
    colors = resp_s @ cluster_colors
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=18, alpha=0.85,
               edgecolor='white', linewidth=0.2)

    # Ellipses 1-sigma et 2-sigma
    for k in range(K):
        try:
            L = np.linalg.cholesky(Sigmas_s[k])
            for scale, lw in [(1.0, 1.8), (2.0, 1.0)]:
                ellipse = mus_s[k:k+1].T + scale * L @ circle
                ax.plot(ellipse[0], ellipse[1], color=cluster_colors[k],
                        linewidth=lw, alpha=0.85)
        except np.linalg.LinAlgError:
            pass

    # Centroïdes
    ax.scatter(mus_s[:, 0], mus_s[:, 1], marker='X', s=180, c='black',
               edgecolor='white', linewidth=1.5, zorder=5)

    ax.set_title(f'Itération {it}' + (' (init)' if it == 0 else
                  ' (convergence)' if it >= 25 else ''),
                 fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('EM-GMM en action : ajustement des 3 gaussiennes du mélange',
             fontsize=13, y=1.02)

fig.text(0.5, -0.03,
         'Couleur des points = mélange RGB des responsabilités. Heatmap grise = densité $p_\\theta(x)$. '
         'Ellipses pleines = contours à 1$\\sigma$, ellipses fines = 2$\\sigma$.\n'
         'À l\'init, les composantes sont mal placées. Après quelques itérations, elles épousent chaque cluster du dataset.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gmm_iterations.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
