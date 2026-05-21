"""
Génère gmm_loglikelihood.png : la vraie log-vraisemblance d'un GMM
en fonction de mu_1 (tous les autres paramètres fixés à leur valeur MLE).

But pédagogique : avant d'introduire la "vue ELBO" qui montre des paraboles
tangentes à log p_theta(x), on veut voir concrètement à quoi ressemble
log p_theta(x) pour un vrai modèle GMM appliqué à un vrai dataset.

Setup :
- Dataset synthétique tiré du vrai mélange (N=200 points).
- On fixe pi_1, pi_2, sigma_1, sigma_2, mu_2 à leurs vraies valeurs.
- On fait varier mu_1 sur l'axe horizontal.
- On calcule l(mu_1) = sum_n log[pi_1 N(x_n | mu_1, sigma_1) + pi_2 N(x_n | mu_2, sigma_2)].

On marque le maximum (mu_1 MLE) et on annote quelques régions :
- Où mu_1 est "trop à gauche" → log-vraisemblance basse.
- Où mu_1 est "à la bonne place" → maximum.
- Où mu_1 est "confondu avec mu_2" → autre maximum local possible (label switching).

Output : gmm_loglikelihood.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Paramètres "vrais" du mélange
# ---------------------------------------------------------------------
pi1, pi2 = 0.4, 0.6
mu1_true, mu2_true = 1.0, 5.0
sigma1, sigma2 = 0.5, 1.2

# ---------------------------------------------------------------------
# Dataset synthétique
# ---------------------------------------------------------------------
np.random.seed(42)
N = 200
n_from_1 = int(round(N * pi1))
n_from_2 = N - n_from_1
data = np.concatenate([
    np.random.normal(mu1_true, sigma1, n_from_1),
    np.random.normal(mu2_true, sigma2, n_from_2)
])

# ---------------------------------------------------------------------
# Calcul de la log-vraisemblance en fonction de mu_1 seul
# ---------------------------------------------------------------------
def log_likelihood(mu1, data):
    """log p_theta(X) avec tous les autres paramètres fixés."""
    comp1 = pi1 * norm.pdf(data, mu1, sigma1)
    comp2 = pi2 * norm.pdf(data, mu2_true, sigma2)
    return np.sum(np.log(comp1 + comp2))

mu1_grid = np.linspace(-3, 8, 500)
ll = np.array([log_likelihood(m, data) for m in mu1_grid])

# Trouver le max
mu1_argmax = mu1_grid[ll.argmax()]
ll_max = ll.max()

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5))

color_main = '#185FA5'
color_max = '#A32D2D'
color_anno = '#666'

# La courbe principale
ax.plot(mu1_grid, ll, color=color_main, linewidth=2.5,
        label=r'$\ell(\mu_1) = \log p_\theta(X)$  avec $\theta = (\pi_k, \sigma_k, \mu_2$ fixés$)$')
ax.fill_between(mu1_grid, ll, ll.min() - 50, alpha=0.05, color=color_main)

# Maximum
ax.scatter([mu1_argmax], [ll_max], color=color_max, s=120, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\mu_1^{{\mathrm{{MLE}}}} \approx {mu1_argmax:.2f}$',
            xy=(mu1_argmax, ll_max),
            xytext=(mu1_argmax + 0.6, ll_max - 30),
            fontsize=12, color=color_max, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color_max, lw=1.2))

# Ligne verticale pointillée à la vraie valeur
ax.axvline(mu1_true, color=color_max, linewidth=0.8, linestyle=':', alpha=0.5)
ax.text(mu1_true, ll.min() - 20,
        fr'$\mu_1^* = {mu1_true}$' + '\n(vraie valeur)',
        fontsize=9, color=color_max, ha='center', va='top', style='italic')

# Annotations sur 3 régions
# Région "trop à gauche"
mu_left = -1.5
ll_left = log_likelihood(mu_left, data)
ax.scatter([mu_left], [ll_left], color=color_anno, s=70, zorder=5,
           edgecolor='white', linewidth=1)
ax.annotate(fr'$\mu_1 = {mu_left}$' + '\n(trop loin)',
            xy=(mu_left, ll_left),
            xytext=(mu_left, ll_left + 80),
            fontsize=9.5, color=color_anno, ha='center',
            arrowprops=dict(arrowstyle='->', color=color_anno, lw=0.8))

# Région "confondu avec mu_2"
mu_right = mu2_true
ll_right = log_likelihood(mu_right, data)
ax.scatter([mu_right], [ll_right], color=color_anno, s=70, zorder=5,
           edgecolor='white', linewidth=1)
ax.annotate(fr'$\mu_1 = {mu_right}$' + '\n(confondu avec $\\mu_2$)',
            xy=(mu_right, ll_right),
            xytext=(mu_right + 0.8, ll_right - 20),
            fontsize=9.5, color=color_anno, ha='left',
            arrowprops=dict(arrowstyle='->', color=color_anno, lw=0.8))

# Mise en forme
ax.set_xlabel(r'$\mu_1$  (moyenne du cluster 1, autres paramètres fixés)', fontsize=12)
ax.set_ylabel(r'$\log p_\theta(X) = \sum_{n=1}^{N} \log p_\theta(x_n)$', fontsize=11)
ax.set_title("La log-vraisemblance GMM en fonction de $\\mu_1$",
             fontsize=13, pad=12)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-3, 8)

# Encadré contextualisant
ax.text(0.02, 0.97,
        f'Dataset : $N = {N}$ points tirés du vrai mélange.' + '\n'
        fr'Fixés : $\pi_1={pi1}, \pi_2={pi2}, \mu_2={mu2_true}, \sigma_1={sigma1}, \sigma_2={sigma2}$.' + '\n'
        r'Seul $\mu_1$ varie sur l\'axe horizontal.',
        transform=ax.transAxes, fontsize=9.5, color='#333',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa',
                  edgecolor='#bbb', linewidth=0.8))

# Légende sous l'image
fig.text(0.5, -0.005,
         r"C'est *cette* fonction $\ell(\mu_1)$ qu'EM cherche à maximiser. "
         r"Elle est typiquement non-convexe (plusieurs maxima locaux possibles selon le dataset)." '\n'
         r"En GMM réel avec tous les paramètres libres, $\ell(\theta)$ vit dans un espace à 6 dimensions \u2014 impossible \u00e0 dessiner directement.",
         ha='center', fontsize=9.5, color='#444')

plt.tight_layout(rect=[0, 0.03, 1, 1])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gmm_loglikelihood.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
print(f"mu_1 MLE (sur la grille) = {mu1_argmax:.3f}")
print(f"log-likelihood max = {ll_max:.2f}")
plt.show()
