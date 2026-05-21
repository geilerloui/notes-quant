"""
Génère vi_kl_directions.png : mode-seeking vs mode-covering.

Deux panneaux côte à côte :
- Gauche : posterior bimodale p(z) en bleu + meilleure gaussienne q(z) au sens
  arg min KL(q || p) — la gaussienne se cale sur UN SEUL des deux modes.
- Droite : posterior bimodale identique + meilleure gaussienne au sens
  arg min KL(p || q) — la gaussienne s'étale pour couvrir les deux modes.

L'image montre visuellement la différence de comportement entre les deux
directions de KL : mode-seeking (zero-forcing, exclusive) vs mode-covering
(zero-avoiding, inclusive).

Output : vi_kl_directions.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import os

# ---------------------------------------------------------------------
# La posterior bimodale (modèle de référence)
# ---------------------------------------------------------------------
# p(z) = w * N(z | mu_1, sigma_1) + (1-w) * N(z | mu_2, sigma_2)
w = 0.5
mu_p1, sigma_p1 = -2.0, 0.5
mu_p2, sigma_p2 = 2.0, 0.5

def p(z):
    return w * norm.pdf(z, mu_p1, sigma_p1) + (1 - w) * norm.pdf(z, mu_p2, sigma_p2)

# ---------------------------------------------------------------------
# Grille pour le tracé
# ---------------------------------------------------------------------
z_grid = np.linspace(-5, 5, 1000)
p_vals = p(z_grid)

# ---------------------------------------------------------------------
# Calcul de la meilleure gaussienne au sens KL(q || p)  (mode-seeking)
# Pour KL(q || p), on minimise int q log(q/p) dz par rapport à (mu_q, sigma_q).
# Pas de forme fermée, mais on peut le faire numériquement par quadrature.
# ---------------------------------------------------------------------
def kl_q_p(params, p_func, z_grid):
    mu_q, log_sigma_q = params
    sigma_q = np.exp(log_sigma_q)
    q_vals = norm.pdf(z_grid, mu_q, sigma_q)
    # Éviter log(0)
    eps = 1e-30
    log_ratio = np.log(q_vals + eps) - np.log(p_func(z_grid) + eps)
    # Trapezoidal integration
    integrand = q_vals * log_ratio
    return np.trapz(integrand, z_grid)

# Init proche d'un mode (sinon convergence aléatoire)
init_seek = np.array([-1.5, np.log(0.5)])  # init proche du mode gauche
res_seek = minimize(kl_q_p, init_seek, args=(p, z_grid),
                    method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5})
mu_q_seek, sigma_q_seek = res_seek.x[0], np.exp(res_seek.x[1])

# ---------------------------------------------------------------------
# Calcul de la meilleure gaussienne au sens KL(p || q)  (mode-covering)
# Pour p donné, arg min KL(p || q) sur les gaussiennes a une forme fermée :
# mu_q* = E_p[z] et sigma_q*^2 = Var_p[z]
# (c'est du moment matching)
# ---------------------------------------------------------------------
mu_q_cover = w * mu_p1 + (1 - w) * mu_p2
# Variance : E[z^2] - E[z]^2
ez2 = w * (sigma_p1**2 + mu_p1**2) + (1 - w) * (sigma_p2**2 + mu_p2**2)
var_q_cover = ez2 - mu_q_cover**2
sigma_q_cover = np.sqrt(var_q_cover)

print(f"Mode-seeking : mu = {mu_q_seek:.3f}, sigma = {sigma_q_seek:.3f}")
print(f"Mode-covering : mu = {mu_q_cover:.3f}, sigma = {sigma_q_cover:.3f}")

# ---------------------------------------------------------------------
# Figure : 2 panneaux côte à côte
# ---------------------------------------------------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6.5),
                                          sharex=True, sharey=True)

color_p = '#185FA5'   # bleu pour la posterior
color_seek = '#A32D2D'  # rouge pour mode-seeking
color_cover = '#0F6E56'  # vert pour mode-covering

# ---------------------------------------------------------------------
# Panneau gauche : mode-seeking
# ---------------------------------------------------------------------
ax = ax_left
ax.fill_between(z_grid, p_vals, color=color_p, alpha=0.25, label=None)
ax.plot(z_grid, p_vals, color=color_p, linewidth=2.5,
        label=r'Vraie posterior $p_\theta(z \mid x)$  (bimodale)')

q_seek_vals = norm.pdf(z_grid, mu_q_seek, sigma_q_seek)
ax.plot(z_grid, q_seek_vals, color=color_seek, linewidth=2.5,
        label=fr'$q^*_{{\mathrm{{seek}}}}(z) = \mathcal{{N}}({mu_q_seek:.2f}, {sigma_q_seek:.2f}^2)$')
ax.fill_between(z_grid, q_seek_vals, color=color_seek, alpha=0.18)

# Marquer le mode choisi
ax.scatter([mu_q_seek], [norm.pdf(mu_q_seek, mu_q_seek, sigma_q_seek)],
           color=color_seek, s=100, zorder=5, edgecolor='white', linewidth=1.5)

# Marquer le mode ignoré avec une flèche pédagogique
ignored_x = mu_p2
ignored_y = p(ignored_x)
ax.annotate('mode ignoré',
            xy=(ignored_x, ignored_y),
            xytext=(ignored_x - 0.3, ignored_y + 0.15),
            fontsize=10, color='#666', ha='right',
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.0))

ax.set_xlabel(r'$z$', fontsize=12)
ax.set_ylabel(r'densité', fontsize=11)
ax.set_title(r"$\arg\min_q\; D_{\mathrm{KL}}(q \,\|\, p)$  —  **mode-seeking**",
             fontsize=12, pad=10)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Note sous la légende
ax.text(0.02, 0.55,
        '$q$ se cale sur un mode' + '\net **ignore l\'autre**.' + '\n'
        '"$q$ refuse de mettre' + '\nde la masse là où $p \\approx 0$"',
        transform=ax.transAxes, fontsize=9.5, color='#444',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff5f5',
                  edgecolor=color_seek, alpha=0.9, linewidth=0.8))

# ---------------------------------------------------------------------
# Panneau droite : mode-covering
# ---------------------------------------------------------------------
ax = ax_right
ax.fill_between(z_grid, p_vals, color=color_p, alpha=0.25)
ax.plot(z_grid, p_vals, color=color_p, linewidth=2.5,
        label=r'Vraie posterior $p_\theta(z \mid x)$  (bimodale)')

q_cover_vals = norm.pdf(z_grid, mu_q_cover, sigma_q_cover)
ax.plot(z_grid, q_cover_vals, color=color_cover, linewidth=2.5,
        label=fr'$q^*_{{\mathrm{{cover}}}}(z) = \mathcal{{N}}({mu_q_cover:.2f}, {sigma_q_cover:.2f}^2)$')
ax.fill_between(z_grid, q_cover_vals, color=color_cover, alpha=0.18)

# Marquer le centre de la gaussienne couvrante
ax.scatter([mu_q_cover], [norm.pdf(mu_q_cover, mu_q_cover, sigma_q_cover)],
           color=color_cover, s=100, zorder=5, edgecolor='white', linewidth=1.5)

# Marquer la "vallée" où q met de la masse alors que p ~ 0
valley_x = 0.0
ax.annotate('$q$ met de la masse\ndans la vallée\noù $p \\approx 0$',
            xy=(valley_x, p(valley_x)),
            xytext=(valley_x + 0.5, 0.32),
            fontsize=9.5, color='#444', ha='left',
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.0))

ax.set_xlabel(r'$z$', fontsize=12)
ax.set_title(r"$\arg\min_q\; D_{\mathrm{KL}}(p \,\|\, q)$  —  **mode-covering**",
             fontsize=12, pad=10)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.98, 0.55,
        '$q$ **couvre les deux modes**.' + '\n'
        '"$q$ refuse d\'ignorer' + '\nune région où $p > 0$"',
        transform=ax.transAxes, fontsize=9.5, color='#444',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0faf5',
                  edgecolor=color_cover, alpha=0.9, linewidth=0.8))

# ---------------------------------------------------------------------
# Limites communes
# ---------------------------------------------------------------------
ax_left.set_xlim(-5, 5)
ax_left.set_ylim(0, 0.55)

# ---------------------------------------------------------------------
# Titre global + caption
# ---------------------------------------------------------------------
fig.suptitle("La direction de la KL change tout : mode-seeking vs mode-covering",
             fontsize=14, y=1.00)

fig.text(0.5, -0.01,
         r"VI utilise **$\arg\min_q D_{\mathrm{KL}}(q \| p)$** (gauche) — pour des raisons de tractabilité (§II.D). "
         r"Conséquence : si la vraie posterior est multimodale, VI rate des modes." + "\n"
         r"L'alternative (droite) couvre tout le support mais nécessite de pouvoir échantillonner $p$, "
         r"qu'on ne sait pas faire en VI. **Aucune des deux n'est \"correcte\" — elles encodent des compromis différents.**",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'vi_kl_directions.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
