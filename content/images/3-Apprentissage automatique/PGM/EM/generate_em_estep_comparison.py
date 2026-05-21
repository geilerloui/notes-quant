"""
Génère em_estep_comparison.png : E-step à l'init vs à convergence.

Deux panneaux 3D côte à côte, même structure que gmm_prior_vs_posterior :
- À gauche : E-step à l'init avec des paramètres MAUVAIS (mu, sigma, pi mal placés).
  Les responsabilités gamma_nk pour x_n = 2.6 sont donc "fausses" par rapport
  à la vraie structure du dataset.
- À droite : E-step à convergence avec des paramètres CORRECTS.
  Les responsabilités sont maintenant cohérentes.

Le message : le E-step n'est qu'une formule mécanique de Bayes appliquée
au theta courant — peu importe sa qualité, on l'applique tel quel.

Output : em_estep_comparison.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Deux jeux de paramètres : mauvais (init) et bons (convergence)
# ---------------------------------------------------------------------
# Init : très mauvais
params_bad = dict(pi1=0.8, pi2=0.2, mu1=-0.5, mu2=3.0, sigma1=1.5, sigma2=0.6)
# Convergence : les vrais paramètres
params_good = dict(pi1=0.4, pi2=0.6, mu1=1.0, mu2=5.0, sigma1=0.5, sigma2=1.2)

x_n = 2.6  # même point pour les deux

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def comp(x, k, p):
    if k == 1:
        return norm.pdf(x, p['mu1'], p['sigma1'])
    return norm.pdf(x, p['mu2'], p['sigma2'])

def mixture(x, p):
    return p['pi1'] * comp(x, 1, p) + p['pi2'] * comp(x, 2, p)

def responsibilities(x, p):
    n1 = p['pi1'] * comp(x, 1, p)
    n2 = p['pi2'] * comp(x, 2, p)
    Z = n1 + n2
    return n1 / Z, n2 / Z

# ---------------------------------------------------------------------
# Fonction qui dessine un panneau 3D avec les paramètres donnés
# ---------------------------------------------------------------------
def draw_panel(ax, p, title):
    color_c1 = '#185FA5'
    color_c2 = '#D85A30'
    color_mix = '#7B3F9E'
    color_post = '#0F6E56'
    color_prior = '#666'

    x_grid = np.linspace(-2, 9, 250)
    N1 = p['pi1'] * norm.pdf(x_grid, p['mu1'], p['sigma1'])
    N2 = p['pi2'] * norm.pdf(x_grid, p['mu2'], p['sigma2'])
    mix = N1 + N2

    x_min, x_max = -2.5, 9.5
    z_min, z_max = 0.5, 3.5
    p_max = 0.55  # un peu plus grand pour absorber les distributions étalées

    y_back = z_max
    x_left = x_min

    g1, g2 = responsibilities(x_n, p)

    # Mise à l'échelle visuelle
    visual_max = 0.35
    scale_prior = visual_max / max(p['pi1'], p['pi2'])
    scale_post = visual_max / max(g1, g2)

    # 1. Jointe : deux rubans
    z1 = np.ones_like(x_grid) * 1.0
    ax.plot(x_grid, z1, N1, color=color_c1, linewidth=2.2, zorder=4,
            label=r'$p(x, z=1)$')

    z2 = np.ones_like(x_grid) * 2.0
    ax.plot(x_grid, z2, N2, color=color_c2, linewidth=2.2, zorder=4,
            label=r'$p(x, z=2)$')

    for xv in np.linspace(x_min + 1, x_max - 0.5, 10):
        n1v = p['pi1'] * norm.pdf(xv, p['mu1'], p['sigma1'])
        n2v = p['pi2'] * norm.pdf(xv, p['mu2'], p['sigma2'])
        if n1v > 0.005:
            ax.plot([xv, xv], [1.0, 1.0], [0, n1v], color=color_c1, linewidth=0.5, alpha=0.3)
        if n2v > 0.005:
            ax.plot([xv, xv], [2.0, 2.0], [0, n2v], color=color_c2, linewidth=0.5, alpha=0.3)

    # 2. Marginale p(x) sur mur fond
    y_back_arr = np.ones_like(x_grid) * y_back
    ax.plot(x_grid, y_back_arr, mix, color=color_mix, linewidth=2.4, zorder=3,
            label=r'$p(x)$ (marginale)')
    for xv in np.linspace(x_min + 0.5, x_max - 0.5, 18):
        mv = mixture(xv, p)
        if mv > 0.005:
            ax.plot([xv, xv], [y_back, y_back], [0, mv],
                    color=color_mix, linewidth=0.4, alpha=0.22)

    # 3. Marginale p(z) sur mur gauche
    for zk, pik, col, k in [(1.0, p['pi1'], color_c1, 1), (2.0, p['pi2'], color_c2, 2)]:
        h = pik * scale_prior
        ax.plot([x_left, x_left], [zk, zk], [0, h],
                color=color_prior, linewidth=4.5, solid_capstyle='butt', alpha=0.85)
        ax.scatter([x_left], [zk], [h], color=col, s=60, edgecolor='black',
                   linewidth=1, zorder=5)
        ax.text(x_left, zk, h + 0.022,
                fr'$\pi_{k}={pik:.2f}$',
                fontsize=9, color='black', ha='center', fontweight='bold')

    # 4. Coupe conditionnelle à x_n
    zz = np.array([z_min, z_max, z_max, z_min])
    pp = np.array([0, 0, p_max, p_max])
    verts = [list(zip([x_n]*4, zz, pp))]
    plane = Poly3DCollection(verts, alpha=0.10, facecolor=color_post, edgecolor='none')
    ax.add_collection3d(plane)

    ax.plot([x_n, x_n], [z_min, z_max], [0, 0], color=color_post, linewidth=1.3,
            linestyle='--', alpha=0.8)
    ax.text(x_n, z_min - 0.35, 0, fr'$x_n={x_n}$', fontsize=10,
            color=color_post, fontweight='bold')

    for zk, gk, col, k in [(1.0, g1, color_c1, 1), (2.0, g2, color_c2, 2)]:
        h = gk * scale_post
        ax.plot([x_n, x_n], [zk, zk], [0, h],
                color=col, linewidth=6.5, solid_capstyle='butt', alpha=0.95)
        ax.scatter([x_n], [zk], [h], color=col, s=100, edgecolor='black',
                   linewidth=1.4, zorder=6)
        ax.text(x_n, zk, h + 0.025,
                fr'$\gamma_{{n,{k}}}={gk:.2f}$',
                fontsize=10, color=col, ha='center', fontweight='bold')

    # Mise en forme
    ax.set_xlabel(r'$x$', fontsize=10, labelpad=8)
    ax.set_ylabel(r'$z$', fontsize=10, labelpad=8)
    ax.set_zlabel(r'densité', fontsize=10, labelpad=8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(0, p_max)
    ax.set_yticks([1, 2])
    ax.set_yticklabels([r'$z=1$', r'$z=2$'])
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=12, pad=12)

    # Légende du panneau (compacte)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.92,
              bbox_to_anchor=(0.0, 0.95))

# ---------------------------------------------------------------------
# Figure avec deux panneaux 3D
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(18, 8))

ax_bad = fig.add_subplot(1, 2, 1, projection='3d')
draw_panel(ax_bad, params_bad,
           r"E-step à l'init : $\theta^{(0)}$ MAUVAIS")

ax_good = fig.add_subplot(1, 2, 2, projection='3d')
draw_panel(ax_good, params_good,
           r"E-step à convergence : $\theta^{(\infty)}$ correct")

# Titre global
fig.suptitle("E-step : la formule de Bayes appliquée au $\\theta$ courant — "
             "peu importe sa qualité",
             fontsize=14, y=0.99)

# Légende globale en bas (sous les panneaux, avant l'encadré de formule)
# Création de proxy artists pour la légende partagée
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], color='#185FA5', linewidth=2.5,
           label=r'$p(x, z=1) = \pi_1\,\mathcal{N}_1(x)$  (jointe, cluster 1)'),
    Line2D([0], [0], color='#D85A30', linewidth=2.5,
           label=r'$p(x, z=2) = \pi_2\,\mathcal{N}_2(x)$  (jointe, cluster 2)'),
    Line2D([0], [0], color='#7B3F9E', linewidth=2.8,
           label=r'$p(x) = \pi_1\mathcal{N}_1(x) + \pi_2\mathcal{N}_2(x)$  (marginale)'),
    Line2D([0], [0], color='#666', linewidth=4,
           label=r'$\pi_k$  (marginale $p(z)$, bâtons à l\'échelle)'),
    Patch(facecolor='#0F6E56', alpha=0.3, label=r'coupe $x = x_n$  (responsabilités $\gamma_{nk}$)'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 0.16), fontsize=9.5, framealpha=0.93,
           frameon=True, columnspacing=1.5)

# Encadré formule + message pédagogique
fig.text(0.5, 0.05,
         r'$\gamma_{nk}^{(t)} = p_{\theta^{(t)}}(z=k \mid x_n) = '
         r'\dfrac{\pi_k^{(t)}\,\mathcal{N}(x_n \mid \mu_k^{(t)}, \sigma_k^{(t)})}'
         r'{\sum_j \pi_j^{(t)}\,\mathcal{N}(x_n \mid \mu_j^{(t)}, \sigma_j^{(t)})}$',
         ha='center', fontsize=13, color='black',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                   edgecolor='#999', linewidth=1.0))

# Comparaison numérique sous l'encadré
g1_bad, g2_bad = responsibilities(x_n, params_bad)
g1_good, g2_good = responsibilities(x_n, params_good)
fig.text(0.5, 0.005,
         f"Pour le même point $x_n = {x_n}$ : à l'init, "
         f"$\\gamma_{{n,1}}={g1_bad:.2f}, \\gamma_{{n,2}}={g2_bad:.2f}$  "
         f"→  à convergence, $\\gamma_{{n,1}}={g1_good:.2f}, \\gamma_{{n,2}}={g2_good:.2f}$.\n"
         "Le E-step est juste la formule de Bayes — il n'a pas d'opinion sur la qualité de $\\theta$.",
         ha='center', fontsize=10, color='#444')

plt.tight_layout(rect=[0, 0.20, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'em_estep_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
