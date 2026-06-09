"""
Génère mle_likelihood.png : la vraisemblance L en fonction de mu et de sigma.

Deux plots côte à côte :
- À gauche : L(mu, sigma_fixé) en fonction de mu — sommet à mu_MLE.
- À droite : L(mu_fixé, sigma) en fonction de sigma — sommet à sigma_MLE.

À placer à la fin de §IV (Étape 3 — maximiser) de la note 04_Estimations.

Output : mle_likelihood.png dans le même dossier que ce script.
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

mu_mle = np.mean(data)
sigma_mle = np.std(data)

# ---------------------------------------------------------------------
# Setup figure 1x2
# ---------------------------------------------------------------------
fig, (ax_mu, ax_sigma) = plt.subplots(1, 2, figsize=(13, 5))

color_lik = '#0F6E56'

# ---------------------------------------------------------------------
# Plot gauche : L en fonction de mu
# ---------------------------------------------------------------------
ax = ax_mu
mu_grid = np.linspace(2.5, 7.5, 200)

def likelihood_mu(mu, data, sigma):
    return np.prod(norm.pdf(data, mu, sigma))

L_mu = np.array([likelihood_mu(m, data, sigma_mle) for m in mu_grid])

ax.plot(mu_grid, L_mu, color=color_lik, linewidth=2.4)
ax.fill_between(mu_grid, L_mu, alpha=0.12, color=color_lik)

L_max = L_mu.max()
ax.scatter([mu_mle], [L_max], color='black', s=90, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\mu_{{\mathrm{{MLE}}}} = {mu_mle:.2f}$',
            xy=(mu_mle, L_max),
            xytext=(mu_mle + 0.5, L_max * 0.92),
            fontsize=12, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

ax.axvline(mu_mle, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

ax.set_title(fr'Vraisemblance en fonction de $\mu$  ($\sigma = {sigma_mle:.2f}$ fixé)',
             fontsize=12, pad=10)
ax.set_xlabel(r'$\mu$', fontsize=12)
ax.set_ylabel(r'$L(\mu, \sigma_{\mathrm{fixé}} \mid \mathrm{data})$', fontsize=11)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# Plot droit : L en fonction de sigma
# ---------------------------------------------------------------------
ax = ax_sigma
sigma_grid = np.linspace(0.5, 4.0, 200)

def likelihood_sigma(sigma, data, mu):
    return np.prod(norm.pdf(data, mu, sigma))

L_sigma = np.array([likelihood_sigma(s, data, mu_mle) for s in sigma_grid])

ax.plot(sigma_grid, L_sigma, color=color_lik, linewidth=2.4)
ax.fill_between(sigma_grid, L_sigma, alpha=0.12, color=color_lik)

L_max_s = L_sigma.max()
sigma_argmax = sigma_grid[L_sigma.argmax()]
ax.scatter([sigma_argmax], [L_max_s], color='black', s=90, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\sigma_{{\mathrm{{MLE}}}} = {sigma_argmax:.2f}$',
            xy=(sigma_argmax, L_max_s),
            xytext=(sigma_argmax + 0.5, L_max_s * 0.92),
            fontsize=12, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

ax.axvline(sigma_argmax, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

ax.set_title(fr'Vraisemblance en fonction de $\sigma$  ($\mu = {mu_mle:.2f}$ fixé)',
             fontsize=12, pad=10)
ax.set_xlabel(r'$\sigma$', fontsize=12)
ax.set_ylabel(r'$L(\mu_{\mathrm{fixé}}, \sigma \mid \mathrm{data})$', fontsize=11)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# Titre global
# ---------------------------------------------------------------------
fig.suptitle('Étape 3 — la vraisemblance comme fonction des paramètres',
             fontsize=13, y=1.00)

fig.text(0.5, -0.02,
         r'On fait varier un paramètre, on fixe l\'autre. La courbe a un sommet : '
         r'la valeur du paramètre qui maximise la plausibilité des données observées.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mle_likelihood.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
