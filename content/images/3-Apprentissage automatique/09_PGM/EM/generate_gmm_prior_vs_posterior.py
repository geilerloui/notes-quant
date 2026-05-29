"""
Génère gmm_prior_vs_posterior.png : vue 3D de la jointe p(x, z) GMM avec
marginales projetées et coupe conditionnelle (responsabilités).

Layout inspiré de la figure classique densité jointe / marginales / conditionnelle.

IMPORTANT — Mise à l'échelle visuelle :
Les densités p(x, z) ont des hauteurs entre 0 et ~0.32, alors que les probabilités
discrètes pi_k et gamma_nk sont entre 0 et 1. Pour que tout tienne dans le même
cadre visuel, on met les bâtons discrets à l'échelle (multipliés par un facteur)
et on annote la VRAIE valeur en étiquette à côté de chaque bâton.

Output : gmm_prior_vs_posterior.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Paramètres du mélange
# ---------------------------------------------------------------------
pi1, pi2 = 0.4, 0.6
mu1, mu2 = 1.0, 5.0
sigma1, sigma2 = 0.5, 1.2

x_n = 2.6  # point de coupe

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def comp(x, k):
    return norm.pdf(x, mu1, sigma1) if k == 1 else norm.pdf(x, mu2, sigma2)

def mixture(x):
    return pi1 * comp(x, 1) + pi2 * comp(x, 2)

def responsibilities(x):
    n1 = pi1 * comp(x, 1)
    n2 = pi2 * comp(x, 2)
    Z = n1 + n2
    return n1 / Z, n2 / Z

g1, g2 = responsibilities(x_n)

# ---------------------------------------------------------------------
# Mise à l'échelle visuelle
# ---------------------------------------------------------------------
# Hauteur cible pour les bâtons (en unité de l'axe "probabilité")
# On veut que les bâtons soient comparables aux pics de densité (~0.32)
visual_max = 0.30
# Facteur d'échelle pour les pi_k (max pi = 0.6 → on veut hauteur visuelle 0.30)
scale_prior = visual_max / max(pi1, pi2)  # = 0.30 / 0.6 = 0.5
# Facteur d'échelle pour les gamma_nk (max gamma = max(g1, g2) → hauteur visuelle 0.30)
scale_post = visual_max / max(g1, g2)

# ---------------------------------------------------------------------
# Figure 3D
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

color_c1 = '#185FA5'
color_c2 = '#D85A30'
color_mix = '#7B3F9E'
color_post = '#0F6E56'
color_prior = '#666'

# Grille x
x_grid = np.linspace(-1.5, 9.5, 250)
N1 = pi1 * norm.pdf(x_grid, mu1, sigma1)
N2 = pi2 * norm.pdf(x_grid, mu2, sigma2)
mix = N1 + N2

# Limites
x_min, x_max = -2.0, 10.0
z_min, z_max = 0.5, 3.5
p_max = 0.40

y_back = z_max
x_left = x_min

# ---------------------------------------------------------------------
# 1. Jointe p(x, z) : deux rubans continus
# ---------------------------------------------------------------------
z1 = np.ones_like(x_grid) * 1.0
ax.plot(x_grid, z1, N1, color=color_c1, linewidth=2.5, zorder=4,
        label=r'$p(x, z=1) = \pi_1\,\mathcal{N}_1(x)$')

z2 = np.ones_like(x_grid) * 2.0
ax.plot(x_grid, z2, N2, color=color_c2, linewidth=2.5, zorder=4,
        label=r'$p(x, z=2) = \pi_2\,\mathcal{N}_2(x)$')

# Lignes verticales pour le volume des rubans
for xv in np.linspace(x_min + 1, x_max - 1, 12):
    n1v = pi1 * norm.pdf(xv, mu1, sigma1)
    n2v = pi2 * norm.pdf(xv, mu2, sigma2)
    if n1v > 0.005:
        ax.plot([xv, xv], [1.0, 1.0], [0, n1v], color=color_c1, linewidth=0.6, alpha=0.35)
    if n2v > 0.005:
        ax.plot([xv, xv], [2.0, 2.0], [0, n2v], color=color_c2, linewidth=0.6, alpha=0.35)

# ---------------------------------------------------------------------
# 2. Marginale p(x) projetée sur le mur du fond
# ---------------------------------------------------------------------
y_back_arr = np.ones_like(x_grid) * y_back
ax.plot(x_grid, y_back_arr, mix, color=color_mix, linewidth=2.8, zorder=3,
        label=r'$p(x) = \pi_1\mathcal{N}_1(x) + \pi_2\mathcal{N}_2(x)$  (marginale)')
for xv in np.linspace(x_min + 0.5, x_max - 0.5, 25):
    mv = mixture(xv)
    if mv > 0.005:
        ax.plot([xv, xv], [y_back, y_back], [0, mv],
                color=color_mix, linewidth=0.5, alpha=0.25)

# ---------------------------------------------------------------------
# 3. Marginale p(z) = pi_k projetée sur le mur de gauche
#    Hauteurs MISES À L'ÉCHELLE pour rester dans le cadre
# ---------------------------------------------------------------------
for zk, pik, col, k in [(1.0, pi1, color_c1, 1), (2.0, pi2, color_c2, 2)]:
    h_visual = pik * scale_prior
    ax.plot([x_left, x_left], [zk, zk], [0, h_visual],
            color=color_prior, linewidth=5.0, solid_capstyle='butt', alpha=0.85)
    ax.scatter([x_left], [zk], [h_visual], color=col, s=70, edgecolor='black',
               linewidth=1, zorder=5)
    # Label avec la VRAIE valeur
    ax.text(x_left, zk, h_visual + 0.018,
            fr'$\pi_{k} = {pik}$',
            fontsize=10, color='black', ha='center', fontweight='bold')

# ---------------------------------------------------------------------
# 4. Coupe conditionnelle à x = x_n
# ---------------------------------------------------------------------
zz = np.array([z_min, z_max, z_max, z_min])
pp = np.array([0, 0, p_max, p_max])
verts = [list(zip([x_n]*4, zz, pp))]
plane = Poly3DCollection(verts, alpha=0.12, facecolor=color_post, edgecolor='none')
ax.add_collection3d(plane)

ax.plot([x_n, x_n], [z_min, z_max], [0, 0], color=color_post, linewidth=1.5,
        linestyle='--', alpha=0.8)
ax.text(x_n, z_min - 0.3, 0, fr'$x_n = {x_n}$', fontsize=11,
        color=color_post, fontweight='bold')

# Bâtons posterior MIS À L'ÉCHELLE
for zk, gk, col, k in [(1.0, g1, color_c1, 1), (2.0, g2, color_c2, 2)]:
    h_visual = gk * scale_post
    ax.plot([x_n, x_n], [zk, zk], [0, h_visual],
            color=col, linewidth=7.0, solid_capstyle='butt', alpha=0.95)
    ax.scatter([x_n], [zk], [h_visual], color=col, s=120, edgecolor='black',
               linewidth=1.5, zorder=6)
    # Label avec la VRAIE valeur
    ax.text(x_n, zk, h_visual + 0.022,
            fr'$\gamma_{{n,{k}}} = {gk:.2f}$',
            fontsize=11, color=col, ha='center', fontweight='bold')

# ---------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------
ax.set_xlabel(r'$x$  (observation)', fontsize=11, labelpad=10)
ax.set_ylabel(r'$z$  (cluster)', fontsize=11, labelpad=10)
ax.set_zlabel(r'densité $p(x, z)$ — bâtons à l\'échelle', fontsize=10, labelpad=10)
ax.set_xlim(x_min, x_max)
ax.set_ylim(z_min, z_max)
ax.set_zlim(0, p_max)

ax.set_yticks([1, 2])
ax.set_yticklabels([r'$z=1$', r'$z=2$'])

ax.view_init(elev=22, azim=-58)
ax.set_title("Jointe $p(x, z)$, marginales $p(x), p(z)$, et posterior $p(z \\mid x_n)$",
             fontsize=13, pad=20)

ax.legend(loc='upper left', fontsize=9.5, framealpha=0.93,
          bbox_to_anchor=(0.0, 0.95))

# ---------------------------------------------------------------------
# Encadré formule + note sur l'échelle
# ---------------------------------------------------------------------
fig.text(0.5, 0.06,
         r'$\gamma_{nk} = p(z=k \mid x_n) = \dfrac{p(x_n, z=k)}{p(x_n)} = '
         r'\dfrac{\pi_k\,\mathcal{N}_k(x_n)}{\sum_j \pi_j\,\mathcal{N}_j(x_n)}$',
         ha='center', fontsize=13, color='black',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                   edgecolor='#999', linewidth=1.0))

fig.text(0.5, 0.005,
         "Les bâtons $\\pi_k$ (gris à gauche) et $\\gamma_{nk}$ (colorés à la coupe verte) "
         "sont des probabilités $\\in [0, 1]$ — leurs hauteurs sont **mises à l\\'échelle** "
         "pour rester comparables aux densités. Les valeurs réelles sont annotées à côté.",
         ha='center', fontsize=9.5, color='#444')

plt.tight_layout(rect=[0, 0.11, 1, 1])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gmm_prior_vs_posterior.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
