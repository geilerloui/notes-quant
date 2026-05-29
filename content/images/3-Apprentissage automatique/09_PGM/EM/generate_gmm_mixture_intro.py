"""
Génère gmm_mixture_intro.png : illustration d'un GMM à 2 composantes en 1D.

Une seule figure montrant :
- Les deux composantes pondérées pi_k * N(x | mu_k, sigma_k) en pointillés colorés.
- Le mélange total p_theta(x) en trait plein violet.
- Quelques points observés sur l'axe avec des flèches verticales qui mesurent
  la densité du mélange en x_i (parallèle direct avec le plot 1 du MLE).
- Encadré listant les paramètres theta = {pi_k, mu_k, sigma_k}.

À placer juste avant §III.B de la note 05_Expectation_Maximization.

Output : gmm_mixture_intro.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Paramètres du mélange à 2 composantes
# ---------------------------------------------------------------------
pi1, pi2 = 0.4, 0.6
mu1, mu2 = 1.0, 5.0
sigma1, sigma2 = 0.5, 1.2

# ---------------------------------------------------------------------
# Points observés : 3 du cluster 1, 3 du cluster 2 (synthétiques, choisis
# pour bien tomber dans chaque mode)
# ---------------------------------------------------------------------
np.random.seed(42)
data = np.concatenate([
    np.random.normal(mu1, sigma1, 3),
    np.random.normal(mu2, sigma2, 3)
])
data = np.sort(data)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

color_c1 = '#185FA5'   # bleu pour composante 1
color_c2 = '#D85A30'   # orange pour composante 2
color_mix = '#7B3F9E'  # violet pour le mélange
color_pts = '#A32D2D'  # rouge pour les points observés
color_arrow = '#666'

x_grid = np.linspace(-1.5, 9, 500)

# Composantes pondérées
comp1 = pi1 * norm.pdf(x_grid, mu1, sigma1)
comp2 = pi2 * norm.pdf(x_grid, mu2, sigma2)
mixture = comp1 + comp2

# Tracer les composantes (pointillés)
ax.plot(x_grid, comp1, color=color_c1, linewidth=1.8, linestyle='--',
        label=fr'$\pi_1 \cdot \mathcal{{N}}(x \mid \mu_1={mu1}, \sigma_1={sigma1})$',
        alpha=0.85)
ax.plot(x_grid, comp2, color=color_c2, linewidth=1.8, linestyle='--',
        label=fr'$\pi_2 \cdot \mathcal{{N}}(x \mid \mu_2={mu2}, \sigma_2={sigma2})$',
        alpha=0.85)

# Tracer le mélange (trait plein, plus épais)
ax.plot(x_grid, mixture, color=color_mix, linewidth=2.8,
        label=r'$p_\theta(x) = \pi_1 \mathcal{N}_1 + \pi_2 \mathcal{N}_2$')
ax.fill_between(x_grid, mixture, alpha=0.08, color=color_mix)

# Points observés + flèches verticales vers le mélange
for xi in data:
    yi = pi1 * norm.pdf(xi, mu1, sigma1) + pi2 * norm.pdf(xi, mu2, sigma2)
    # flèche verticale jusqu'au mélange
    ax.annotate('',
                xy=(xi, yi), xytext=(xi, 0),
                arrowprops=dict(arrowstyle='->', color=color_arrow, lw=1.2,
                                alpha=0.8))
    # point sur l'axe
    ax.scatter([xi], [0], color=color_pts, s=70, zorder=5,
               edgecolor='white', linewidth=1.2)
    # point sur la courbe du mélange
    ax.scatter([xi], [yi], color=color_mix, s=45, zorder=5,
               edgecolor='white', linewidth=1)
    # valeur de p_theta(x_i)
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(xi, yi + 0.012),
                fontsize=8, ha='center', color='#444',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.7))

# Annotation pédagogique : pointer une flèche pour expliquer p_theta(x_i)
xi_demo = data[1]  # un point du cluster 1
yi_demo = pi1 * norm.pdf(xi_demo, mu1, sigma1) + pi2 * norm.pdf(xi_demo, mu2, sigma2)
ax.annotate(r'$p_\theta(x_i) = \pi_1 \mathcal{N}_1(x_i) + \pi_2 \mathcal{N}_2(x_i)$',
            xy=(xi_demo, yi_demo / 2),
            xytext=(xi_demo + 1.5, yi_demo / 2 + 0.05),
            fontsize=11, color=color_arrow, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color_arrow, lw=1.0),
            va='center')

# Label observations
ax.annotate(r'observations $x_1, x_2, \ldots, x_n$',
            xy=(data[len(data)//2], 0),
            xytext=(data[len(data)//2], -0.045),
            fontsize=10, color=color_pts, ha='center', fontweight='bold')

# Mise en forme
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel('densité', fontsize=11)
ax.set_title("GMM à 2 composantes — on évalue $p_\\theta(x_i)$ pour chaque observation",
             fontsize=12, pad=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-0.07, 0.45)
ax.set_xlim(-1.5, 9)

# Encadré paramètres en haut à gauche (3 lignes courtes)
ax.text(0.02, 0.97,
        r'Paramètres $\theta$ du modèle :' + '\n'
        fr'$\pi_1 = {pi1},\ \mu_1 = {mu1},\ \sigma_1 = {sigma1}$' + '\n'
        fr'$\pi_2 = {pi2},\ \mu_2 = {mu2},\ \sigma_2 = {sigma2}$',
        transform=ax.transAxes, fontsize=9, color='#333',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#fafafa',
                  edgecolor='#bbb', linewidth=0.8))

plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gmm_mixture_intro.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
