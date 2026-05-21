"""
Génère ppca_generative_model.png : la figure-phare type Bishop §12.2.

Deux panneaux côte à côte :
- Gauche : l'espace latent 1D. On voit p(z) = N(0, 1), une distribution gaussienne
  univariée standard. Un point z_hat est tiré (matérialisé par un bâton).

- Droite : l'espace observé 2D. On voit :
  1. La droite affine z |-> Wz + mu (l'hyperplan affine de dim M=1).
  2. Le point W*z_hat + mu projeté sur cette droite.
  3. Le nuage gaussien isotropique p(x | z_hat) autour de ce point, de variance sigma^2 I.
  4. Optionnel : contours de la marginale p(x) en arrière-plan (ellipse oblique allongée).

Output : ppca_generative_model.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm, multivariate_normal
import os

# ---------------------------------------------------------------------
# Paramètres du modèle PPCA (illustratifs)
# ---------------------------------------------------------------------
D = 2
M = 1
mu = np.array([2.0, 1.5])
# Direction du sous-espace latent : une droite oblique
W = np.array([[1.5], [1.0]])  # D x M
sigma2 = 0.15

# Covariance marginale C = W W^T + sigma^2 I
C = W @ W.T + sigma2 * np.eye(D)

# Un échantillon spécifique de z (illustratif)
z_hat = 0.7

# ---------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------
color_latent = '#185FA5'   # bleu pour l'espace latent
color_observed = '#7B3F9E' # violet pour la projection sur la droite
color_noise = '#D85A30'    # orange pour le nuage de bruit
color_marginal = '#0F6E56' # vert pour la marginale globale
color_W = '#A32D2D'        # rouge pour la direction W

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(15, 7))

# Layout : panneau gauche étroit (espace latent 1D), panneau droite large (espace obs 2D)
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.25)

# ---------------------------------------------------------------------
# Panneau gauche : espace latent 1D
# ---------------------------------------------------------------------
ax_latent = plt.subplot(gs[0, 0])

z_range = np.linspace(-3, 3, 300)
p_z = norm.pdf(z_range, 0, 1)
ax_latent.fill_betweenx(z_range, 0, p_z, color=color_latent, alpha=0.3)
ax_latent.plot(p_z, z_range, color=color_latent, linewidth=2.5,
               label=r'$p(z) = \mathcal{N}(z \mid 0, 1)$')

# Marquer z_hat
p_z_hat = norm.pdf(z_hat, 0, 1)
ax_latent.scatter([p_z_hat], [z_hat], color=color_latent, s=100, zorder=5,
                  edgecolor='white', linewidth=1.5)
ax_latent.plot([0, p_z_hat], [z_hat, z_hat], color=color_latent,
               linewidth=1.2, linestyle=':', alpha=0.7)
ax_latent.text(p_z_hat + 0.02, z_hat, fr'  $\hat{{z}} = {z_hat}$',
               fontsize=11, color=color_latent, va='center', fontweight='bold')

# Origine
ax_latent.axhline(0, color='gray', linewidth=0.6, linestyle='--', alpha=0.6)
ax_latent.text(0.42, 0.1, '$z = 0$', fontsize=9, color='gray')

ax_latent.set_xlim(0, 0.5)
ax_latent.set_ylim(-3, 3)
ax_latent.set_xlabel(r'$p(z)$', fontsize=11)
ax_latent.set_ylabel(r'$z \in \mathbb{R}^M$  (espace latent, ici $M = 1$)',
                     fontsize=11)
ax_latent.set_title(r"Espace latent  —  $z \sim \mathcal{N}(0, I)$",
                     fontsize=12, pad=10)
ax_latent.spines['top'].set_visible(False)
ax_latent.spines['right'].set_visible(False)
ax_latent.grid(True, alpha=0.25, linestyle=':')

# Flèche conceptuelle : "transformation"
ax_latent.annotate('', xy=(1.05, 0), xytext=(0.6, 0),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='black',
                                    lw=2.5, mutation_scale=22))
ax_latent.text(0.83, 0.05, r'$z \mapsto Wz + \mu + \epsilon$',
               transform=ax_latent.transAxes, fontsize=10,
               ha='center', fontweight='bold')

# ---------------------------------------------------------------------
# Panneau droite : espace observé 2D
# ---------------------------------------------------------------------
ax_obs = plt.subplot(gs[0, 1])

# 1. Marginale p(x) en arrière-plan (contours)
x_lim = (-1.5, 6.5)
y_lim = (-1.5, 5.0)
xx, yy = np.meshgrid(np.linspace(*x_lim, 200), np.linspace(*y_lim, 200))
pos = np.dstack([xx, yy])
rv = multivariate_normal(mu, C)
zz = rv.pdf(pos)

ax_obs.contour(xx, yy, zz, levels=8, colors=color_marginal,
               linewidths=1.0, alpha=0.55)
# Une ellipse de niveau pour bien voir la forme de la marginale
def ellipse_from_cov(mean, cov, n_std=1.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return Ellipse(mean, width, height, angle=angle, **kwargs)

ell_marg = ellipse_from_cov(mu, C, n_std=2.0, fill=False,
                             edgecolor=color_marginal, linewidth=2.0,
                             linestyle='--', alpha=0.8)
ax_obs.add_patch(ell_marg)

# 2. La droite z |-> Wz + mu (hyperplan affine de dim M=1)
z_range_full = np.linspace(-3, 3, 100)
line_points = np.array([W.flatten() * zr + mu for zr in z_range_full])
ax_obs.plot(line_points[:, 0], line_points[:, 1], color=color_W,
            linewidth=2.5, alpha=0.85,
            label=r'hyperplan affine  $z \mapsto Wz + \mu$')

# 3. Le point W*z_hat + mu sur la droite
x_proj = W.flatten() * z_hat + mu
ax_obs.scatter([x_proj[0]], [x_proj[1]], color=color_observed, s=140, zorder=6,
               edgecolor='white', linewidth=1.5,
               label=r'$W\hat{z} + \mu$  (point sur l\'hyperplan)')

# 4. Nuage gaussien autour : contours de p(x | z_hat)
C_cond = sigma2 * np.eye(D)
rv_cond = multivariate_normal(x_proj, C_cond)
zz_cond = rv_cond.pdf(pos)
ax_obs.contourf(xx, yy, zz_cond, levels=8, cmap='Oranges', alpha=0.45)
ax_obs.contour(xx, yy, zz_cond, levels=4, colors=color_noise,
               linewidths=1.2, alpha=0.85)
ell_noise_1 = ellipse_from_cov(x_proj, C_cond, n_std=1.0, fill=False,
                                edgecolor=color_noise, linewidth=2.0,
                                alpha=0.95)
ell_noise_2 = ellipse_from_cov(x_proj, C_cond, n_std=2.0, fill=False,
                                edgecolor=color_noise, linewidth=1.2,
                                alpha=0.6, linestyle=':')
ax_obs.add_patch(ell_noise_1)
ax_obs.add_patch(ell_noise_2)

# 5. Centre mu
ax_obs.scatter([mu[0]], [mu[1]], color='black', s=80, zorder=5,
               marker='X', edgecolor='white', linewidth=1.2)
ax_obs.text(mu[0] + 0.15, mu[1] - 0.35, r'$\mu$', fontsize=12,
            fontweight='bold')

# 6. Tirer quelques échantillons x ~ p(x | z_hat) pour illustration
np.random.seed(42)
samples = rv_cond.rvs(size=15)
ax_obs.scatter(samples[:, 0], samples[:, 1], color=color_noise, s=25,
               alpha=0.5, zorder=4, edgecolor='white', linewidth=0.5)

# Annotations explicatives
ax_obs.annotate(r'$p(x \mid \hat{z}) = \mathcal{N}(W\hat{z} + \mu, \sigma^2 I)$' + '\n(bruit isotropique)',
                xy=(x_proj[0] + 0.4, x_proj[1] + 0.3),
                xytext=(x_proj[0] + 1.4, x_proj[1] + 1.6),
                fontsize=10, color=color_noise, ha='left',
                arrowprops=dict(arrowstyle='->', color=color_noise, lw=1.0),
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff5f0',
                          edgecolor=color_noise, alpha=0.92))

ax_obs.annotate('marginale\n' + r'$p(x) = \mathcal{N}(\mu, WW^\top + \sigma^2 I)$',
                xy=(mu[0] - 1.5, mu[1] - 0.8),
                xytext=(-1.0, -0.5),
                fontsize=10, color=color_marginal, ha='left',
                arrowprops=dict(arrowstyle='->', color=color_marginal, lw=1.0),
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0faf5',
                          edgecolor=color_marginal, alpha=0.92))

ax_obs.set_xlim(x_lim)
ax_obs.set_ylim(y_lim)
ax_obs.set_aspect('equal')
ax_obs.set_xlabel(r'$x_1$', fontsize=11)
ax_obs.set_ylabel(r'$x_2$', fontsize=11)
ax_obs.set_title(r"Espace observé  —  $x = Wz + \mu + \epsilon$  ($D = 2$)",
                  fontsize=12, pad=10)
ax_obs.spines['top'].set_visible(False)
ax_obs.spines['right'].set_visible(False)
ax_obs.grid(True, alpha=0.25, linestyle=':')
ax_obs.legend(loc='upper left', fontsize=9, framealpha=0.95)

# ---------------------------------------------------------------------
# Titre et caption
# ---------------------------------------------------------------------
fig.suptitle("Modèle génératif PPCA : tirer $z$, transformer en $Wz + \\mu$, ajouter du bruit",
             fontsize=14, y=1.00)

fig.text(0.5, -0.01,
         r"À gauche : on tire $\hat{z} \sim \mathcal{N}(0, 1)$ dans l'espace latent 1D. "
         r"À droite : la transformation $z \mapsto Wz + \mu$ trace une **droite** dans $\mathbb{R}^2$ (en rouge). " + "\n"
         r"Pour $\hat{z}$ fixé, on obtient un point sur la droite ($W\hat{z} + \mu$, en violet), "
         r"puis le bruit isotropique $\sigma^2 I$ crée un nuage gaussien autour (orange). "
         r"**Marginalisée sur tous les $z$**, la distribution $p(x)$ devient une gaussienne 2D allongée le long de la droite (vert).",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ppca_generative_model.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
