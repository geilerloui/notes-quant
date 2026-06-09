"""
Génère mle_step1_densities.png : le PREMIER plot du MLE isolé.

Une seule figure montrant :
- une gaussienne fixée (mu et sigma "devinés"),
- les n points du dataset sur l'axe x (cercles rouges),
- les flèches verticales montant de chaque point jusqu'à la courbe,
  illustrant la mesure de la densité p(x_i | mu, sigma).

C'est l'illustration de l'Étape 1 ("fixer les paramètres et mesurer").
À placer entre §II et §III de la note 04_Estimations.

Output : mle_step1_densities.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Dataset synthétique (mêmes paramètres que les autres scripts MLE)
# ---------------------------------------------------------------------
np.random.seed(42)
true_mu, true_sigma = 5.0, 1.5
n = 8
data = np.random.normal(true_mu, true_sigma, size=n)
data = np.sort(data)

mu_show = np.mean(data)
sigma_show = np.std(data)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

color_gauss = '#185FA5'
color_pts = '#A32D2D'
color_arrow = '#666'

x_grid = np.linspace(0, 10, 400)
y_gauss = norm.pdf(x_grid, mu_show, sigma_show)

# Gaussienne
ax.plot(x_grid, y_gauss, color=color_gauss, linewidth=2.5,
        label=fr'$\mathcal{{N}}(x \mid \mu={mu_show:.2f}, \sigma={sigma_show:.2f})$')
ax.fill_between(x_grid, y_gauss, alpha=0.10, color=color_gauss)

# Pour chaque point : flèche verticale + point sur l'axe + point sur la courbe
for xi in data:
    yi = norm.pdf(xi, mu_show, sigma_show)
    ax.annotate('',
                xy=(xi, yi), xytext=(xi, 0),
                arrowprops=dict(arrowstyle='->', color=color_arrow, lw=1.2,
                                alpha=0.8))
    ax.scatter([xi], [0], color=color_pts, s=70, zorder=5,
               edgecolor='white', linewidth=1.2)
    ax.scatter([xi], [yi], color=color_gauss, s=45, zorder=5,
               edgecolor='white', linewidth=1)
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(xi, yi + 0.013),
                fontsize=8, ha='center', color='#444',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.7))

# Annotation pédagogique p(x_i | mu, sigma) à droite d'une flèche centrale
idx_mid = len(data) // 2
xi_mid = data[idx_mid]
yi_mid = norm.pdf(xi_mid, mu_show, sigma_show)
ax.annotate(r'$p(x_i \mid \mu, \sigma)$',
            xy=(xi_mid, yi_mid / 2),
            xytext=(xi_mid + 1.5, yi_mid / 2),
            fontsize=12, color=color_arrow, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color_arrow, lw=1.0),
            va='center')

# Label "observations x_i" sous l'axe
ax.annotate(r'observations $x_1, x_2, \ldots, x_n$',
            xy=(data[len(data)//2], 0),
            xytext=(data[len(data)//2], -0.04),
            fontsize=10, color=color_pts, ha='center', fontweight='bold')

# Mise en forme
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel('densité', fontsize=11)
ax.set_title("Étape 1 — On fixe $(\\mu, \\sigma)$ et on mesure $p(x_i \\mid \\mu, \\sigma)$ "
             "pour chaque observation",
             fontsize=12, pad=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-0.06, 0.40)

# Encadré récapitulatif sur 3 lignes courtes pour rester compact
ax.text(0.02, 0.97,
        'Hauteur de la flèche\n'
        r'= densité $p(x_i \mid \mu, \sigma)$' + '\n'
        r'= "plausibilité" du point $x_i$',
        transform=ax.transAxes, fontsize=9, color='#333',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#fafafa',
                  edgecolor='#bbb', linewidth=0.8))

plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mle_step1_densities.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
