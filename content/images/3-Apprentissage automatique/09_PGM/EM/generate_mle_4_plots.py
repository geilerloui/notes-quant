"""
Génère l'image mle_4_plots.png : explication visuelle du MLE classique (style StatQuest).

Quatre panneaux :
1. Gaussienne fixée + points du dataset + flèches verticales montrant p(x_i | mu, sigma).
2. Vraisemblance L(mu, sigma_fixé) en fonction de mu — sommet à mu_MLE.
3. Vraisemblance L(mu_fixé, sigma) en fonction de sigma — sommet à sigma_MLE.
4. Log-vraisemblance log L en fonction de mu (et sigma en miniature) — sommet au même endroit.

Output : mle_4_plots.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ---------------------------------------------------------------------
# Dataset synthétique : 8 points tirés d'une gaussienne (peu, pour bien voir)
# ---------------------------------------------------------------------
np.random.seed(42)
true_mu, true_sigma = 5.0, 1.5
n = 8
data = np.random.normal(true_mu, true_sigma, size=n)
# On force des points raisonnablement écartés pour la pédagogie
data = np.sort(data)

# MLE empirique
mu_mle = np.mean(data)
sigma_mle = np.std(data)  # estimateur MLE (divise par n)

# ---------------------------------------------------------------------
# Setup figure 2x2
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# Palette
color_gauss = '#185FA5'
color_pts = '#A32D2D'
color_arrow = '#666'
color_lik = '#0F6E56'
color_loglik = '#D85A30'

# ---------------------------------------------------------------------
# PLOT 1 — Gaussienne fixée + densités p(x_i | mu, sigma) en flèches
# ---------------------------------------------------------------------
ax = ax1
x_grid = np.linspace(0, 10, 400)

# On choisit une gaussienne "devinée" qui passe par les points (= la MLE pour bien visualiser)
mu_show, sigma_show = mu_mle, sigma_mle
y_gauss = norm.pdf(x_grid, mu_show, sigma_show)

ax.plot(x_grid, y_gauss, color=color_gauss, linewidth=2.2,
        label=fr'$\mathcal{{N}}(x \mid \mu={mu_show:.2f}, \sigma={sigma_show:.2f})$')
ax.fill_between(x_grid, y_gauss, alpha=0.10, color=color_gauss)

# Points sur l'axe + flèches verticales jusqu'à la courbe
for xi in data:
    yi = norm.pdf(xi, mu_show, sigma_show)
    ax.vlines(xi, 0, yi, color=color_arrow, linewidth=1.0, linestyle='-', alpha=0.7)
    ax.scatter([xi], [0], color=color_pts, s=50, zorder=5,
               edgecolor='white', linewidth=1)
    ax.scatter([xi], [yi], color=color_gauss, s=35, zorder=5,
               edgecolor='white', linewidth=0.8)
    # petite étiquette au-dessus
    ax.annotate(f'{yi:.2f}', xy=(xi, yi), xytext=(xi, yi + 0.015),
                fontsize=7, ha='center', color='#444')

ax.set_title('① Fixer $(\\mu, \\sigma)$ et mesurer $p(x_i \\mid \\mu, \\sigma)$',
             fontsize=11, pad=8)
ax.set_xlabel('$x$', fontsize=11)
ax.set_ylabel('densité', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.02, 0.95,
        r'$L(\mu, \sigma \mid x_1, \ldots, x_n) = \prod_i p(x_i \mid \mu, \sigma)$',
        transform=ax.transAxes, fontsize=10, color='#333',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa',
                  edgecolor='#bbb', linewidth=0.8))

# ---------------------------------------------------------------------
# PLOT 2 — Vraisemblance L en fonction de mu (sigma fixé)
# ---------------------------------------------------------------------
ax = ax2
mu_grid = np.linspace(2.5, 7.5, 200)

def likelihood_mu(mu, data, sigma):
    """L(mu, sigma fixé | data) = produit des densités."""
    return np.prod(norm.pdf(data, mu, sigma))

L_mu = np.array([likelihood_mu(m, data, sigma_mle) for m in mu_grid])

ax.plot(mu_grid, L_mu, color=color_lik, linewidth=2.2)
ax.fill_between(mu_grid, L_mu, alpha=0.12, color=color_lik)

# Sommet
L_max = L_mu.max()
ax.scatter([mu_mle], [L_max], color='black', s=80, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\mu_{{\mathrm{{MLE}}}} = {mu_mle:.2f}$',
            xy=(mu_mle, L_max),
            xytext=(mu_mle + 0.4, L_max * 0.95),
            fontsize=11, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# Ligne verticale pointillée au max
ax.axvline(mu_mle, color='black', linewidth=0.7, linestyle=':', alpha=0.5)

ax.set_title(fr'② Vraisemblance en fonction de $\mu$ ($\sigma = {sigma_mle:.2f}$ fixé)',
             fontsize=11, pad=8)
ax.set_xlabel(r'$\mu$', fontsize=11)
ax.set_ylabel(r'$L(\mu, \sigma_{\mathrm{fixé}} \mid \mathrm{data})$', fontsize=10)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# PLOT 3 — Vraisemblance L en fonction de sigma (mu fixé)
# ---------------------------------------------------------------------
ax = ax3
sigma_grid = np.linspace(0.5, 4.0, 200)

def likelihood_sigma(sigma, data, mu):
    return np.prod(norm.pdf(data, mu, sigma))

L_sigma = np.array([likelihood_sigma(s, data, mu_mle) for s in sigma_grid])

ax.plot(sigma_grid, L_sigma, color=color_lik, linewidth=2.2)
ax.fill_between(sigma_grid, L_sigma, alpha=0.12, color=color_lik)

L_max_s = L_sigma.max()
sigma_argmax = sigma_grid[L_sigma.argmax()]
ax.scatter([sigma_argmax], [L_max_s], color='black', s=80, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\sigma_{{\mathrm{{MLE}}}} = {sigma_argmax:.2f}$',
            xy=(sigma_argmax, L_max_s),
            xytext=(sigma_argmax + 0.4, L_max_s * 0.95),
            fontsize=11, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.axvline(sigma_argmax, color='black', linewidth=0.7, linestyle=':', alpha=0.5)

ax.set_title(fr'③ Vraisemblance en fonction de $\sigma$ ($\mu = {mu_mle:.2f}$ fixé)',
             fontsize=11, pad=8)
ax.set_xlabel(r'$\sigma$', fontsize=11)
ax.set_ylabel(r'$L(\mu_{\mathrm{fixé}}, \sigma \mid \mathrm{data})$', fontsize=10)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# PLOT 4 — Log-vraisemblance (en mu, avec inset pour sigma)
# ---------------------------------------------------------------------
ax = ax4

# log-likelihood en mu
def log_likelihood_mu(mu, data, sigma):
    return np.sum(norm.logpdf(data, mu, sigma))

logL_mu = np.array([log_likelihood_mu(m, data, sigma_mle) for m in mu_grid])

ax.plot(mu_grid, logL_mu, color=color_loglik, linewidth=2.2, label=r'$\log L(\mu, \sigma_{\mathrm{fixé}})$')
ax.fill_between(mu_grid, logL_mu, alpha=0.12, color=color_loglik)

# Sommet
logL_max = logL_mu.max()
ax.scatter([mu_mle], [logL_max], color='black', s=80, zorder=5,
           edgecolor='white', linewidth=1.5)
ax.annotate(fr'$\hat\mu_{{\mathrm{{MLE}}}} = {mu_mle:.2f}$',
            xy=(mu_mle, logL_max),
            xytext=(mu_mle + 0.4, logL_max - 3),
            fontsize=11, color='black', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax.axvline(mu_mle, color='black', linewidth=0.7, linestyle=':', alpha=0.5)

ax.set_title(r'④ Log-vraisemblance : même sommet, calculs plus stables',
             fontsize=11, pad=8)
ax.set_xlabel(r'$\mu$', fontsize=11)
ax.set_ylabel(r'$\log L(\mu, \sigma_{\mathrm{fixé}})$', fontsize=10)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Inset pour log L en sigma
ax_inset = ax.inset_axes([0.62, 0.10, 0.35, 0.32])
logL_sigma = np.array([np.sum(norm.logpdf(data, mu_mle, s)) for s in sigma_grid])
ax_inset.plot(sigma_grid, logL_sigma, color=color_loglik, linewidth=1.5)
ax_inset.fill_between(sigma_grid, logL_sigma, alpha=0.12, color=color_loglik)
ax_inset.axvline(sigma_argmax, color='black', linewidth=0.6, linestyle=':', alpha=0.5)
ax_inset.scatter([sigma_argmax], [logL_sigma.max()], color='black', s=30, zorder=5,
                 edgecolor='white', linewidth=1)
ax_inset.set_title(r'$\log L$ vs $\sigma$', fontsize=9)
ax_inset.set_xlabel(r'$\sigma$', fontsize=8)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.25, linestyle=':')
ax_inset.spines['top'].set_visible(False)
ax_inset.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# Titre global et légende
# ---------------------------------------------------------------------
fig.suptitle('Maximum Likelihood Estimation (MLE) — cas gaussien classique',
             fontsize=14, y=0.995)

fig.text(0.5, -0.01,
         r'Le MLE cherche $(\hat\mu, \hat\sigma) = \arg\max_{\mu, \sigma} L(\mu, \sigma \mid \mathrm{data})$. '
         r'Géométriquement : la gaussienne qui maximise les densités aux points observés.' '\n'
         r'En pratique on maximise $\log L$ (plot ④) plutôt que $L$ : sommet au même endroit, '
         r'mais somme stable au lieu de produit qui sous-déborde.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mle_4_plots.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
