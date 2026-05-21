"""
Génère mle_loglikelihood.png : la log-vraisemblance log L en fonction de mu et de sigma.

Deux plots côte à côte :
- À gauche : log L(mu, sigma_fixé) en fonction de mu — sommet à mu_MLE.
- À droite : log L(mu_fixé, sigma) en fonction de sigma — sommet à sigma_MLE.

À placer à la fin de §V (Étape 4 — passer au log) de la note 04_Estimations.

Output : mle_loglikelihood.png dans le même dossier que ce script.
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

color_loglik = '#D85A30'

# ---------------------------------------------------------------------
# Plot gauche : log L en fonction de mu
# ---------------------------------------------------------------------
ax = ax_mu
mu_grid = np.linspace(2.5, 7.5, 200)

def log_likelihood_mu(mu, data, sigma):
    return np.sum(norm.logpdf(data, mu, sigma))

logL_mu = np.array([log_likelihood_mu(m, data, sigma_mle) for m in mu_grid])

ax.plot(mu_grid, logL_mu, color=color_loglik, linewidth=2.4)
ax.fill_between(mu_grid, logL_mu, alpha=0.12, color=color_loglik)

logL_max = logL_mu.max()
ax.scatter([mu_mle], [logL_max], color='black', s=90, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\mu_{{\mathrm{{MLE}}}} = {mu_mle:.2f}$',
            xy=(mu_mle, logL_max),
            xytext=(mu_mle + 0.5, logL_max - 4),
            fontsize=12, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

ax.axvline(mu_mle, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

ax.set_title(fr'Log-vraisemblance en fonction de $\mu$  ($\sigma = {sigma_mle:.2f}$ fixé)',
             fontsize=12, pad=10)
ax.set_xlabel(r'$\mu$', fontsize=12)
ax.set_ylabel(r'$\log L(\mu, \sigma_{\mathrm{fixé}} \mid \mathrm{data})$', fontsize=11)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# Plot droit : log L en fonction de sigma
# ---------------------------------------------------------------------
ax = ax_sigma
sigma_grid = np.linspace(0.5, 4.0, 200)

logL_sigma = np.array([np.sum(norm.logpdf(data, mu_mle, s)) for s in sigma_grid])

ax.plot(sigma_grid, logL_sigma, color=color_loglik, linewidth=2.4)
ax.fill_between(sigma_grid, logL_sigma, alpha=0.12, color=color_loglik)

logL_max_s = logL_sigma.max()
sigma_argmax = sigma_grid[logL_sigma.argmax()]
ax.scatter([sigma_argmax], [logL_max_s], color='black', s=90, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\sigma_{{\mathrm{{MLE}}}} = {sigma_argmax:.2f}$',
            xy=(sigma_argmax, logL_max_s),
            xytext=(sigma_argmax + 0.5, logL_max_s - 4),
            fontsize=12, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

ax.axvline(sigma_argmax, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

ax.set_title(fr'Log-vraisemblance en fonction de $\sigma$  ($\mu = {mu_mle:.2f}$ fixé)',
             fontsize=12, pad=10)
ax.set_xlabel(r'$\sigma$', fontsize=12)
ax.set_ylabel(r'$\log L(\mu_{\mathrm{fixé}}, \sigma \mid \mathrm{data})$', fontsize=11)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# Titre global
# ---------------------------------------------------------------------
fig.suptitle('Étape 4 — la log-vraisemblance : même sommet, calculs plus stables',
             fontsize=13, y=1.00)

fig.text(0.5, -0.02,
         r'Comparer avec la figure précédente (vraisemblance directe) : '
         r'les sommets sont aux mêmes endroits ($\hat\mu_{\mathrm{MLE}}, \hat\sigma_{\mathrm{MLE}}$).' '\n'
         r'La forme de la courbe change (somme au lieu de produit) mais la position du maximum est conservée.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mle_loglikelihood.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
