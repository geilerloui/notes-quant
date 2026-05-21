"""
Génère ppca_posterior_z_given_x.png : la posterior p(z | x) en image.

Deux panneaux :
- Gauche : l'espace observé 2D avec :
  * La droite latente (hyperplan affine).
  * Un point x observé (en dehors de la droite).
  * La projection orthogonale W*E[z|x] + mu sur la droite (centre de la posterior dans l'espace observé).
  * Une ligne en pointillés entre x et la projection.

- Droite : la posterior p(z | x) dans l'espace latent 1D :
  * Une gaussienne centrée sur E[z | x] = M^-1 W^T (x - mu).
  * Variance sigma^2 * M^-1.
  * Optionnel : comparer au prior p(z) = N(0,1) pour montrer l'effet du conditionnement.

Output : ppca_posterior_z_given_x.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm, multivariate_normal
import os

# ---------------------------------------------------------------------
# Paramètres du modèle PPCA (mêmes que ppca_generative_model)
# ---------------------------------------------------------------------
D = 2
M = 1
mu = np.array([2.0, 1.5])
W = np.array([[1.5], [1.0]])  # D x M
sigma2 = 0.3  # un peu plus de bruit ici pour voir l'incertitude

# Posterior gaussienne :
# E[z | x] = M_tilde^-1 W^T (x - mu)
# Cov[z | x] = sigma^2 * M_tilde^-1
# où M_tilde = W^T W + sigma^2 I
M_tilde = W.T @ W + sigma2 * np.eye(M)
M_tilde_inv = np.linalg.inv(M_tilde)

# Un point x observé qui n'est PAS sur la droite (pour voir la projection)
x_obs = np.array([4.8, 2.0])

# Calcul de la posterior
mean_z_given_x = (M_tilde_inv @ W.T @ (x_obs - mu)).flatten()  # scalar (M=1)
var_z_given_x = (sigma2 * M_tilde_inv).flatten()  # scalar

print(f"E[z | x] = {mean_z_given_x[0]:.4f}")
print(f"Var[z | x] = {var_z_given_x[0]:.4f}")
print(f"Std[z | x] = {np.sqrt(var_z_given_x[0]):.4f}")

# Projection de x sur la droite (dans l'espace observé)
x_proj_on_line = (W.flatten() * mean_z_given_x[0] + mu)

# ---------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------
color_prior = '#888888'    # gris pour le prior
color_post = '#185FA5'     # bleu pour la posterior
color_x = '#A32D2D'        # rouge pour x observé
color_proj = '#7B3F9E'     # violet pour la projection
color_W = '#0F6E56'        # vert pour la droite latente

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, (ax_obs, ax_lat) = plt.subplots(1, 2, figsize=(15, 6.5))

# ---------------------------------------------------------------------
# Panneau gauche : espace observé 2D
# ---------------------------------------------------------------------
x_lim = (-1, 7)
y_lim = (-1, 4.5)

# Tracer la droite latente
z_range_full = np.linspace(-3, 3, 100)
line_points = np.array([W.flatten() * zr + mu for zr in z_range_full])
ax_obs.plot(line_points[:, 0], line_points[:, 1], color=color_W,
            linewidth=2.5, alpha=0.85,
            label=r'hyperplan latent  $z \mapsto Wz + \mu$')

# Centre mu
ax_obs.scatter([mu[0]], [mu[1]], color='black', s=80, zorder=5,
               marker='X', edgecolor='white', linewidth=1.2)
ax_obs.text(mu[0] - 0.5, mu[1] - 0.45, r'$\mu$', fontsize=12,
            fontweight='bold')

# Le point x observé
ax_obs.scatter([x_obs[0]], [x_obs[1]], color=color_x, s=180, zorder=6,
               edgecolor='white', linewidth=1.8,
               label=r'$x$ observé', marker='o')
ax_obs.annotate(r'$x$', xy=(x_obs[0], x_obs[1]),
                xytext=(x_obs[0] + 0.3, x_obs[1] - 0.3),
                fontsize=14, color=color_x, fontweight='bold')

# Projection sur la droite
ax_obs.scatter([x_proj_on_line[0]], [x_proj_on_line[1]],
               color=color_proj, s=140, zorder=6,
               edgecolor='white', linewidth=1.5,
               label=r'$W\,\mathbb{E}[z \mid x] + \mu$')
ax_obs.text(x_proj_on_line[0] - 0.7, x_proj_on_line[1] + 0.35,
            r"projection",
            fontsize=10, color=color_proj, ha='left')

# Ligne en pointillés entre x et la projection (la "résidu" orthogonal au sous-espace)
ax_obs.plot([x_obs[0], x_proj_on_line[0]], [x_obs[1], x_proj_on_line[1]],
            color=color_x, linewidth=1.3, linestyle='--', alpha=0.7)
mid = ((x_obs + x_proj_on_line) / 2)
ax_obs.text(mid[0] + 0.2, mid[1] + 0.1,
            'résidu\n(absorbé par $\\epsilon$)',
            fontsize=9, color=color_x, style='italic')

# Annoter E[z | x]
ax_obs.annotate(fr'$\mathbb{{E}}[z \mid x] = {mean_z_given_x[0]:.2f}$',
                xy=(x_proj_on_line[0], x_proj_on_line[1]),
                xytext=(x_proj_on_line[0] + 0.5, x_proj_on_line[1] - 0.8),
                fontsize=10, color=color_proj, ha='left',
                arrowprops=dict(arrowstyle='->', color=color_proj, lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f0fc',
                          edgecolor=color_proj, alpha=0.92))

ax_obs.set_xlim(x_lim)
ax_obs.set_ylim(y_lim)
ax_obs.set_aspect('equal')
ax_obs.set_xlabel(r'$x_1$', fontsize=11)
ax_obs.set_ylabel(r'$x_2$', fontsize=11)
ax_obs.set_title(r"Espace observé : $x$ et sa projection sur l'hyperplan latent",
                  fontsize=12, pad=10)
ax_obs.spines['top'].set_visible(False)
ax_obs.spines['right'].set_visible(False)
ax_obs.grid(True, alpha=0.25, linestyle=':')
ax_obs.legend(loc='lower right', fontsize=9, framealpha=0.95)

# ---------------------------------------------------------------------
# Panneau droite : posterior p(z | x) sur l'espace latent
# ---------------------------------------------------------------------
z_range = np.linspace(-3, 3, 400)

# Prior p(z) = N(0, 1)
p_z_vals = norm.pdf(z_range, 0, 1)
ax_lat.fill_between(z_range, p_z_vals, color=color_prior, alpha=0.25)
ax_lat.plot(z_range, p_z_vals, color=color_prior, linewidth=2.0,
            linestyle='--',
            label=r'prior  $p(z) = \mathcal{N}(0, 1)$')

# Posterior p(z | x)
post_vals = norm.pdf(z_range, mean_z_given_x[0], np.sqrt(var_z_given_x[0]))
ax_lat.fill_between(z_range, post_vals, color=color_post, alpha=0.4)
ax_lat.plot(z_range, post_vals, color=color_post, linewidth=2.5,
            label=fr'posterior  $p(z \mid x) = \mathcal{{N}}({mean_z_given_x[0]:.2f}, {var_z_given_x[0]:.3f})$')

# Marquer la moyenne de la posterior
ax_lat.axvline(mean_z_given_x[0], color=color_post, linewidth=1.5,
               linestyle=':', alpha=0.7)
ax_lat.scatter([mean_z_given_x[0]], [post_vals.max()], color=color_post,
               s=100, zorder=5, edgecolor='white', linewidth=1.5)
ax_lat.text(mean_z_given_x[0] + 0.08, post_vals.max() * 0.95,
            fr'  $\mathbb{{E}}[z \mid x] = {mean_z_given_x[0]:.2f}$',
            fontsize=10, color=color_post, va='top', fontweight='bold')

# Marquer la moyenne du prior (zéro)
ax_lat.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

ax_lat.set_xlim(-3, 3)
ax_lat.set_ylim(0, max(post_vals.max(), p_z_vals.max()) * 1.2)
ax_lat.set_xlabel(r'$z$  (espace latent)', fontsize=11)
ax_lat.set_ylabel(r'densité', fontsize=11)
ax_lat.set_title(r"Espace latent : prior $p(z)$ vs posterior $p(z \mid x)$",
                  fontsize=12, pad=10)
ax_lat.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax_lat.grid(True, alpha=0.25, linestyle=':')
ax_lat.spines['top'].set_visible(False)
ax_lat.spines['right'].set_visible(False)

# Note sur l'incertitude
ax_lat.text(0.97, 0.55,
            r'**Effet du conditionnement :**' + '\n'
            r'  $\bullet$ prior : centré sur 0, variance 1' + '\n'
            r'  $\bullet$ posterior : centrée sur $\mathbb{E}[z \mid x]$,' + '\n'
            fr'    variance $\sigma^2 / (W^\top W + \sigma^2) = {var_z_given_x[0]:.3f}$' + '\n'
            r'  $\bullet$ **bien plus étroite que le prior**' + '\n'
            r'    $\to$ $x$ donne beaucoup d\'info sur $z$',
            transform=ax_lat.transAxes,
            fontsize=9, color='#333', va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fafafa',
                      edgecolor='#bbb', linewidth=0.7))

# ---------------------------------------------------------------------
# Titre et caption
# ---------------------------------------------------------------------
fig.suptitle(r"La posterior $p(z \mid x)$ : projeter $x$ sur le sous-espace latent (+ incertitude)",
             fontsize=14, y=1.00)

fig.text(0.5, -0.02,
         r"À gauche : un point $x$ observé (rouge) est **projeté orthogonalement** sur l'hyperplan latent (vert), donnant le point violet $W\,\mathbb{E}[z \mid x] + \mu$. "
         r"Le résidu (en pointillés) est absorbé par le bruit $\epsilon$." + "\n"
         r"À droite : la posterior $p(z \mid x)$ (bleu) sur l'espace latent. Elle est **gaussienne** de moyenne $\tilde M^{-1} W^\top(x-\mu)$ et de variance $\sigma^2 \tilde M^{-1}$. "
         r"Elle est bien plus concentrée que le prior $\mathcal{N}(0, 1)$ (gris pointillé) — $x$ a apporté beaucoup d'information sur $z$.",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ppca_posterior_z_given_x.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nImage sauvegardée : {output_path}")
plt.show()
