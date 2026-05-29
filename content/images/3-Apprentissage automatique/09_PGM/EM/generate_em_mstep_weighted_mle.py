"""
Génère em_mstep_weighted_mle.png : visualisation du M-step comme MLE pondéré.

Idée : reprendre le plot 1 du MLE classique (gaussienne + points + flèches verticales
mesurant les densités), mais cette fois POUR CHAQUE CLUSTER séparément, avec
les points pondérés par leurs responsabilités gamma_nk.

Deux panneaux côte à côte :
- Gauche : M-step pour le cluster 1 — gaussienne N(mu_1, sigma_1) ajustée,
  tous les points du dataset, mais OPACITÉ = gamma_n1. Les points "appartenant"
  au cluster 1 apparaissent nets, les autres fantomatiques.
- Droite : pareil pour le cluster 2 avec gamma_n2.

C'est exactement ce qui se passe dans le M-step : on fait un MLE classique
pour chaque cluster, mais chaque point a un poids gamma_nk au lieu de poids 1.

Output : em_mstep_weighted_mle.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Paramètres "à convergence" (mêmes que les autres figures GMM)
# ---------------------------------------------------------------------
pi1, pi2 = 0.4, 0.6
mu1, mu2 = 1.0, 5.0
sigma1, sigma2 = 0.5, 1.2

# ---------------------------------------------------------------------
# Dataset synthétique tiré du vrai mélange (n=20 points)
# ---------------------------------------------------------------------
np.random.seed(42)
n_total = 20
# Tirer des points selon les vrais poids
n_from_1 = int(round(n_total * pi1))   # 8
n_from_2 = n_total - n_from_1           # 12
data = np.concatenate([
    np.random.normal(mu1, sigma1, n_from_1),
    np.random.normal(mu2, sigma2, n_from_2)
])
data = np.sort(data)

# ---------------------------------------------------------------------
# Calculer les responsabilités gamma_nk pour chaque point
# ---------------------------------------------------------------------
def responsibilities(x):
    n1 = pi1 * norm.pdf(x, mu1, sigma1)
    n2 = pi2 * norm.pdf(x, mu2, sigma2)
    Z = n1 + n2
    return n1 / Z, n2 / Z

gammas_1 = np.array([responsibilities(xi)[0] for xi in data])
gammas_2 = np.array([responsibilities(xi)[1] for xi in data])

# Vérification : N_k = somme des responsabilités
N_1 = gammas_1.sum()
N_2 = gammas_2.sum()

# Vérification numérique des updates M-step
mu1_new = np.sum(gammas_1 * data) / N_1
mu2_new = np.sum(gammas_2 * data) / N_2
sigma1_new = np.sqrt(np.sum(gammas_1 * (data - mu1_new)**2) / N_1)
sigma2_new = np.sqrt(np.sum(gammas_2 * (data - mu2_new)**2) / N_2)
pi1_new = N_1 / n_total
pi2_new = N_2 / n_total

# ---------------------------------------------------------------------
# Plot — 2 panneaux côte à côte
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

color_c1 = '#185FA5'
color_c2 = '#D85A30'
color_arrow = '#666'

x_grid = np.linspace(-2, 9, 400)

def draw_cluster_panel(ax, mu_k, sigma_k, mu_new, sigma_new, gammas, color, k_idx):
    """Dessine un panneau M-step pour le cluster k."""

    # Gaussienne du cluster (la nouvelle, après l'update du M-step)
    y_gauss = norm.pdf(x_grid, mu_new, sigma_new)
    ax.plot(x_grid, y_gauss, color=color, linewidth=2.5,
            label=fr'$\mathcal{{N}}(x \mid \mu_{k_idx}^{{\mathrm{{new}}}}, \sigma_{k_idx}^{{\mathrm{{new}}}})$')
    ax.fill_between(x_grid, y_gauss, alpha=0.08, color=color)

    # Pour chaque point : flèche verticale + point, opacité = gamma_nk
    for xi, gk in zip(data, gammas):
        yi = norm.pdf(xi, mu_new, sigma_new)

        # Opacité minimale pour qu'on voie au moins la trace des autres points
        # (sinon les points avec gamma=0.01 sont totalement invisibles)
        alpha_arrow = max(gk, 0.06)
        alpha_marker = max(gk, 0.08)

        # Flèche verticale
        ax.annotate('',
                    xy=(xi, yi), xytext=(xi, 0),
                    arrowprops=dict(arrowstyle='->', color=color_arrow,
                                    lw=1.0 + 1.5 * gk,  # épaisseur proportionnelle à gamma
                                    alpha=alpha_arrow))
        # Point sur l'axe (taille proportionnelle à gamma aussi)
        ax.scatter([xi], [0], color=color,
                   s=40 + 80 * gk,
                   alpha=alpha_marker,
                   edgecolor='white', linewidth=1.0, zorder=5)
        # Point sur la courbe (plus discret)
        ax.scatter([xi], [yi], color=color,
                   s=20 + 30 * gk,
                   alpha=alpha_marker,
                   edgecolor='white', linewidth=0.7, zorder=5)

    # Ligne verticale au mu_new
    ax.axvline(mu_new, color=color, linewidth=1.0, linestyle=':', alpha=0.6)
    ax.scatter([mu_new], [norm.pdf(mu_new, mu_new, sigma_new)],
               color='black', s=80, edgecolor='white', linewidth=1.5, zorder=6)
    ax.annotate(fr'$\mu_{k_idx}^{{\mathrm{{new}}}}={mu_new:.2f}$',
                xy=(mu_new, 0.02),
                xytext=(mu_new + 0.5, 0.05),
                fontsize=11, color='black', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # Mise en forme
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel('densité', fontsize=11)
    ax.set_title(fr'M-step pour le cluster {k_idx} — MLE pondéré par $\gamma_{{n,{k_idx}}}$',
                 fontsize=12, pad=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-2, 9)
    ax.set_ylim(-0.04, 0.55)

# Panneau cluster 1
draw_cluster_panel(ax1, mu1, sigma1, mu1_new, sigma1_new, gammas_1, color_c1, 1)

# Encadré formule sous panneau 1
ax1.text(0.02, 0.97,
         r'Update pour le cluster 1 :' + '\n'
         r'$\mu_1^{\mathrm{new}} = \dfrac{\sum_n \gamma_{n,1}\, x_n}{N_1}$' + '\n'
         r'$\sigma_1^{2,\mathrm{new}} = \dfrac{\sum_n \gamma_{n,1}\,(x_n - \mu_1^{\mathrm{new}})^2}{N_1}$' + '\n'
         fr'$N_1 = \sum_n \gamma_{{n,1}} = {N_1:.2f}$',
         transform=ax1.transAxes, fontsize=9, color='#333',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa',
                   edgecolor='#bbb', linewidth=0.8))

# Panneau cluster 2
draw_cluster_panel(ax2, mu2, sigma2, mu2_new, sigma2_new, gammas_2, color_c2, 2)

# Encadré formule sous panneau 2
ax2.text(0.02, 0.97,
         r'Update pour le cluster 2 :' + '\n'
         r'$\mu_2^{\mathrm{new}} = \dfrac{\sum_n \gamma_{n,2}\, x_n}{N_2}$' + '\n'
         r'$\sigma_2^{2,\mathrm{new}} = \dfrac{\sum_n \gamma_{n,2}\,(x_n - \mu_2^{\mathrm{new}})^2}{N_2}$' + '\n'
         fr'$N_2 = \sum_n \gamma_{{n,2}} = {N_2:.2f}$',
         transform=ax2.transAxes, fontsize=9, color='#333',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa',
                   edgecolor='#bbb', linewidth=0.8))

# ---------------------------------------------------------------------
# Titre global + légende sous l'image
# ---------------------------------------------------------------------
fig.suptitle(r"M-step = MLE classique avec poids $\gamma_{nk}$ au lieu de poids 1",
             fontsize=14, y=1.00)

fig.text(0.5, -0.02,
         r'Comparer avec le MLE classique (cf. [[04_Estimations]]) : '
         r'là, $\mu_{\mathrm{MLE}} = \frac{1}{n}\sum_n x_n$ (chaque point compte pour 1).' '\n'
         r'Ici, chaque cluster fait son MLE en pondérant chaque point par sa responsabilité '
         r'$\gamma_{nk}$ — les points "appartenant fortement" au cluster comptent plus.' '\n'
         r'$\pi_k^{\mathrm{new}} = N_k / N$ : la proportion effective du cluster $k$ dans le dataset.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'em_mstep_weighted_mle.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
print(f"Updates calculés : mu1={mu1_new:.3f}, sigma1={sigma1_new:.3f}, "
      f"mu2={mu2_new:.3f}, sigma2={sigma2_new:.3f}")
print(f"pi1_new={pi1_new:.3f}, pi2_new={pi2_new:.3f}")
plt.show()
