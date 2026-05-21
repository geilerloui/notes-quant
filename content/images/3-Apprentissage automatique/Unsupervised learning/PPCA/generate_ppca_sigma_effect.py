"""
Génère ppca_sigma_effect.png : effet de sigma^2 sur la marginale p(x).

3 panneaux côte à côte :
- sigma^2 petit (proche de PCA classique) : ellipse très allongée, presque collée
  sur la droite latente.
- sigma^2 modéré : ellipse oblique, étalée perpendiculairement.
- sigma^2 grand : ellipse quasi-isotropique, le signal latent est noyé.

Sur chaque panneau :
- La droite latente z |-> Wz + mu (verte).
- Les contours de la marginale p(x) = N(mu, WW^T + sigma^2 I) (bleus).
- Un échantillon de points x ~ p(x) (semi-transparents).

Output : ppca_sigma_effect.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import os

# ---------------------------------------------------------------------
# Modèle de base
# ---------------------------------------------------------------------
D = 2
M = 1
mu = np.array([0.0, 0.0])
W = np.array([[1.5], [1.0]])  # D x M
WWT = W @ W.T

# Trois valeurs de sigma^2 à comparer
sigma2_values = [0.05, 0.4, 1.5]
labels = [r'$\sigma^2 = 0.05$  (proche PCA)',
          r'$\sigma^2 = 0.4$  (modéré)',
          r'$\sigma^2 = 1.5$  (signal noyé)']

# ---------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------
color_marginal = '#185FA5'
color_W = '#0F6E56'
color_samples = '#7B3F9E'

# ---------------------------------------------------------------------
# Helper ellipse
# ---------------------------------------------------------------------
def ellipse_from_cov(mean, cov, n_std=1.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return Ellipse(mean, width, height, angle=angle, **kwargs)

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharex=True, sharey=True)

x_lim = (-4.5, 4.5)
y_lim = (-3.5, 3.5)
xx, yy = np.meshgrid(np.linspace(*x_lim, 200), np.linspace(*y_lim, 200))
pos = np.dstack([xx, yy])

# Droite latente (fixe sur les 3 panneaux)
z_range_full = np.linspace(-3, 3, 100)
line_points = np.array([W.flatten() * zr + mu for zr in z_range_full])

for ax, sigma2, label in zip(axes, sigma2_values, labels):
    # Covariance marginale
    C = WWT + sigma2 * np.eye(D)
    rv = multivariate_normal(mu, C)
    zz = rv.pdf(pos)

    # Contours remplis
    ax.contourf(xx, yy, zz, levels=10, cmap='Blues', alpha=0.55)
    ax.contour(xx, yy, zz, levels=6, colors=color_marginal,
               linewidths=1.0, alpha=0.7)

    # Ellipses de niveau
    for n_std, alpha, ls in [(1.0, 0.95, '-'), (2.0, 0.6, '--')]:
        ell = ellipse_from_cov(mu, C, n_std=n_std, fill=False,
                                edgecolor=color_marginal, linewidth=2.0,
                                alpha=alpha, linestyle=ls)
        ax.add_patch(ell)

    # Droite latente
    ax.plot(line_points[:, 0], line_points[:, 1], color=color_W,
            linewidth=2.5, alpha=0.9, zorder=4,
            label=r'hyperplan  $z \mapsto Wz + \mu$' if ax == axes[0] else None)

    # Échantillons
    np.random.seed(42)
    samples = rv.rvs(size=80)
    ax.scatter(samples[:, 0], samples[:, 1], color=color_samples, s=14,
               alpha=0.5, zorder=3, edgecolor='white', linewidth=0.3)

    # Centre
    ax.scatter([mu[0]], [mu[1]], color='black', s=70, zorder=6,
               marker='X', edgecolor='white', linewidth=1.2)

    # Titre du panneau
    ax.set_title(label, fontsize=12, pad=8)
    ax.set_xlabel(r'$x_1$', fontsize=11)
    if ax == axes[0]:
        ax.set_ylabel(r'$x_2$', fontsize=11)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Calculer ratio d'allongement de l'ellipse (eigval ratio) pour annoter
    eigvals = np.linalg.eigvalsh(C)
    ratio = eigvals[-1] / eigvals[0]
    ax.text(0.03, 0.97,
            fr'$\Lambda_{{\max}}/\Lambda_{{\min}} = {ratio:.2f}$',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fafafa',
                      edgecolor='#bbb', linewidth=0.6))

# Légende globale en haut
handles = [
    plt.Line2D([0], [0], color=color_marginal, linewidth=2.0,
               label=r'$p(x) = \mathcal{N}(\mu, WW^\top + \sigma^2 I)$'),
    plt.Line2D([0], [0], color=color_W, linewidth=2.5,
               label=r'hyperplan latent $z \mapsto Wz + \mu$'),
    plt.Line2D([0], [0], marker='o', color=color_samples, linewidth=0, markersize=8,
               markeredgecolor='white', label=r'échantillons $x \sim p(x)$'),
]
fig.legend(handles=handles, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.01), fontsize=11, framealpha=0.95)

# Titre et caption
fig.suptitle(r"Effet de $\sigma^2$ sur la marginale $p(x)$ : du quasi-PCA au signal noyé",
             fontsize=14, y=1.08)

fig.text(0.5, -0.02,
         r"**Gauche** ($\sigma^2$ petit) : l'ellipse est très allongée le long de la droite latente — quasiment un Dirac sur l'hyperplan. C'est la **limite PCA classique** ($\sigma^2 \to 0$ donne PCA). "
         r"**Milieu** : $\sigma^2$ modéré, l'ellipse a une largeur perpendiculaire visible. " + "\n"
         r"**Droite** ($\sigma^2$ grand) : l'ellipse devient quasi-isotropique, le signal latent est **noyé** dans le bruit. "
         r"Le ratio $\Lambda_{\max}/\Lambda_{\min}$ mesure ce niveau d'allongement.",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ppca_sigma_effect.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
