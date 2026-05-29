"""
Génère vi_mean_field_factorization.png : limite de l'approximation mean-field.

Deux panneaux côte à côte :
- Gauche : vraie posterior p(z_1, z_2 | x) gaussienne 2D avec covariance non-diagonale.
  Ellipse de niveau oblique (corrélation forte entre z_1 et z_2).
- Droite : meilleure approximation mean-field q(z_1, z_2) = q(z_1) * q(z_2).
  Ellipse alignée sur les axes — les corrélations sont perdues.

Sur les deux panneaux, on superpose :
- Les contours de la densité 2D.
- Les marginales p(z_1) et p(z_2) sur les bords (haut et droite).

Le mean-field optimal a les MÊMES marginales que la vraie posterior, ce qui montre
visuellement qu'on perd la corrélation mais pas les variances individuelles.

Output : vi_mean_field_factorization.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal, norm
import os

# ---------------------------------------------------------------------
# Vraie posterior : gaussienne 2D corrélée
# ---------------------------------------------------------------------
mu_true = np.array([1.5, 1.0])
sigma_1 = 1.0
sigma_2 = 0.8
rho = 0.75  # corrélation forte

Sigma_true = np.array([
    [sigma_1**2, rho * sigma_1 * sigma_2],
    [rho * sigma_1 * sigma_2, sigma_2**2]
])

# ---------------------------------------------------------------------
# Meilleure approximation mean-field
# Pour KL(q || p) avec p gaussien et q = q_1 * q_2 gaussien :
# la solution optimale a les mêmes moyennes et les mêmes variances diagonales.
# (En réalité c'est plus subtil pour KL(q||p) — on prend les variances marginales
#  exactes, qui sont les éléments diagonaux de Sigma_true.)
# ---------------------------------------------------------------------
mu_mf = mu_true.copy()
Sigma_mf = np.diag([Sigma_true[0, 0], Sigma_true[1, 1]])

# ---------------------------------------------------------------------
# Helpers pour ellipse de niveau
# ---------------------------------------------------------------------
def ellipse_from_cov(mean, cov, n_std=1.0, **kwargs):
    """Retourne un objet Ellipse correspondant à un contour à n_std écart-types."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return Ellipse(mean, width, height, angle=angle, **kwargs)

# ---------------------------------------------------------------------
# Figure : 2 panneaux côte à côte avec marginales sur les bords
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(15, 7))

# GridSpec pour avoir marginales en haut et à droite de chaque panneau
import matplotlib.gridspec as gridspec
outer = gridspec.GridSpec(1, 2, wspace=0.35)

def draw_panel(outer_cell, mu, Sigma, title, fill_color, edge_color,
               show_axes_lines=False):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_cell,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05
    )
    ax_main = plt.subplot(inner[1, 0])
    ax_top = plt.subplot(inner[0, 0], sharex=ax_main)
    ax_right = plt.subplot(inner[1, 1], sharey=ax_main)

    # Grille 2D
    x_lim = (-2.5, 5.5)
    y_lim = (-2.0, 4.5)
    xx, yy = np.meshgrid(np.linspace(*x_lim, 200), np.linspace(*y_lim, 200))
    pos = np.dstack([xx, yy])
    rv = multivariate_normal(mu, Sigma)
    zz = rv.pdf(pos)

    # Contour rempli
    ax_main.contourf(xx, yy, zz, levels=10, cmap='Blues', alpha=0.5)
    ax_main.contour(xx, yy, zz, levels=6, colors=edge_color, linewidths=1.2)

    # Ellipses 1σ et 2σ pour bien marquer la forme
    for n_std, alpha in [(1.0, 0.9), (2.0, 0.6)]:
        ell = ellipse_from_cov(mu, Sigma, n_std=n_std,
                                fill=False, edgecolor=edge_color,
                                linewidth=2.0, alpha=alpha,
                                linestyle='--' if n_std == 2.0 else '-')
        ax_main.add_patch(ell)

    # Centre
    ax_main.scatter([mu[0]], [mu[1]], color=edge_color, s=80, zorder=5,
                    edgecolor='white', linewidth=1.5)

    # Lignes en pointillés sur le centre pour montrer l'alignement (panneau droite)
    if show_axes_lines:
        ax_main.axhline(mu[1], color='gray', linewidth=0.7,
                        linestyle=':', alpha=0.7)
        ax_main.axvline(mu[0], color='gray', linewidth=0.7,
                        linestyle=':', alpha=0.7)

    # Marginale haut : p(z_1)
    z1_range = np.linspace(*x_lim, 300)
    p_z1 = norm.pdf(z1_range, mu[0], np.sqrt(Sigma[0, 0]))
    ax_top.fill_between(z1_range, p_z1, color=fill_color, alpha=0.5)
    ax_top.plot(z1_range, p_z1, color=edge_color, linewidth=2.0)
    ax_top.set_ylim(0, p_z1.max() * 1.2)
    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.set_yticks([])
    ax_top.set_ylabel('$p(z_1)$', fontsize=10, rotation=0, labelpad=20, va='center')

    # Marginale droite : p(z_2)
    z2_range = np.linspace(*y_lim, 300)
    p_z2 = norm.pdf(z2_range, mu[1], np.sqrt(Sigma[1, 1]))
    ax_right.fill_betweenx(z2_range, p_z2, color=fill_color, alpha=0.5)
    ax_right.plot(p_z2, z2_range, color=edge_color, linewidth=2.0)
    ax_right.set_xlim(0, p_z2.max() * 1.2)
    ax_right.tick_params(labelbottom=False, labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.set_xticks([])
    ax_right.set_xlabel('$p(z_2)$', fontsize=10, labelpad=8)

    # Axes principaux
    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.set_xlabel(r'$z_1$', fontsize=12)
    ax_main.set_ylabel(r'$z_2$', fontsize=12)
    ax_main.grid(True, alpha=0.2, linestyle=':')
    ax_main.set_aspect('equal')

    # Titre au-dessus du panneau (au-dessus de la marginale du haut)
    ax_top.set_title(title, fontsize=13, pad=8)

    return ax_main, ax_top, ax_right

# ---------------------------------------------------------------------
# Panneau gauche : vraie posterior corrélée
# ---------------------------------------------------------------------
ax_left, _, _ = draw_panel(
    outer[0, 0],
    mu_true, Sigma_true,
    r"Vraie posterior $p_\theta(z_1, z_2 \mid x)$  — corrélée",
    fill_color='#185FA5', edge_color='#0F3D6E',
    show_axes_lines=False
)
# Annoter la corrélation
ax_left.text(0.98, 0.03,
             fr'$\rho = {rho}$' + '\n(off-diagonal de $\Sigma$)',
             transform=ax_left.transAxes,
             fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor='#0F3D6E', alpha=0.92))

# ---------------------------------------------------------------------
# Panneau droite : approximation mean-field
# ---------------------------------------------------------------------
ax_right_panel, _, _ = draw_panel(
    outer[0, 1],
    mu_mf, Sigma_mf,
    r"Mean-field $q_\phi(z_1, z_2) = q_{\phi_1}(z_1) \cdot q_{\phi_2}(z_2)$  — axis-aligned",
    fill_color='#D85A30', edge_color='#A0421E',
    show_axes_lines=True
)
ax_right_panel.text(0.98, 0.03,
                    r'$\rho = 0$' + '\n(diagonale de $\Sigma$)',
                    transform=ax_right_panel.transAxes,
                    fontsize=10, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor='#A0421E', alpha=0.92))

# ---------------------------------------------------------------------
# Titre global et caption
# ---------------------------------------------------------------------
fig.suptitle(
    "L'approximation mean-field perd les corrélations",
    fontsize=14, y=1.00
)

fig.text(0.5, -0.01,
         r"À gauche : la vraie posterior a une ellipse **oblique** (corrélation $\rho = 0.75$). "
         r"À droite : la meilleure approximation mean-field a la **même moyenne** et les **mêmes marginales** "
         r"(courbes en haut et à droite identiques)," + "\n"
         r"mais son ellipse est **alignée sur les axes** — pas le choix, $q$ se factorise comme produit. "
         r"Le gap résiduel KL$(q \| p)$ mesure exactement cette perte de corrélation.",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'vi_mean_field_factorization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
