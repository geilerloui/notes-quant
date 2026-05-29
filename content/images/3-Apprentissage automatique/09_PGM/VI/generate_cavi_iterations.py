"""
Génère cavi_iterations.png : trajectoire CAVI sur une posterior gaussienne 2D corrélée.

Setup :
- Vraie posterior p(z_1, z_2) gaussienne 2D corrélée (rho = 0.75) — la même que dans
  vi_mean_field_factorization.png.
- Approximation mean-field q(z_1, z_2) = q_1(z_1) * q_2(z_2), gaussienne par facteur.
- Init mal placée : q_1^(0), q_2^(0) avec des moyennes très loin de la vraie posterior.
- On applique les updates CAVI en forme fermée, alternativement q_1 puis q_2.

Updates CAVI explicites pour notre cas gaussien :
Pour p = N([mu_1, mu_2], [[s11, s12], [s12, s22]]) :
  q_1*(z_1) = N(mu_1 + (s12/s22) * (E_q2[z_2] - mu_2), s11 - s12^2/s22)
  q_2*(z_2) = N(mu_2 + (s12/s11) * (E_q1[z_1] - mu_1), s22 - s12^2/s11)

Les variances sont fixes après le premier update — c'est une propriété du cas gaussien.
Seules les moyennes bougent au fil des itérations.

4 panneaux : init, après 1 update (q_1 seul), après 2 updates (q_1 + q_2), à convergence.

Output : cavi_iterations.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal, norm
import os

# ---------------------------------------------------------------------
# Vraie posterior : gaussienne 2D corrélée (même que vi_mean_field_factorization)
# ---------------------------------------------------------------------
mu_true = np.array([1.5, 1.0])
sigma_1_true = 1.0
sigma_2_true = 0.8
rho_true = 0.75

Sigma_true = np.array([
    [sigma_1_true**2, rho_true * sigma_1_true * sigma_2_true],
    [rho_true * sigma_1_true * sigma_2_true, sigma_2_true**2]
])

s11 = Sigma_true[0, 0]
s22 = Sigma_true[1, 1]
s12 = Sigma_true[0, 1]

# ---------------------------------------------------------------------
# Updates CAVI en forme fermée
# ---------------------------------------------------------------------
# Pour p gaussienne, la posterior conditionnelle z_1 | z_2 = b est gaussienne avec :
#   moyenne = mu_1 + (s12/s22) * (b - mu_2)
#   variance = s11 - s12^2/s22
# L'update CAVI prend la moyenne conditionnelle sous q (on remplace b par E_q[z_2]).

def update_q1(mu_q2):
    """Update CAVI de q_1 sachant q_2 (de moyenne mu_q2)."""
    mu_q1_new = mu_true[0] + (s12 / s22) * (mu_q2 - mu_true[1])
    var_q1_new = s11 - s12**2 / s22
    return mu_q1_new, var_q1_new

def update_q2(mu_q1):
    """Update CAVI de q_2 sachant q_1 (de moyenne mu_q1)."""
    mu_q2_new = mu_true[1] + (s12 / s11) * (mu_q1 - mu_true[0])
    var_q2_new = s22 - s12**2 / s11
    return mu_q2_new, var_q2_new

# ---------------------------------------------------------------------
# Simulation : init mal placée + plusieurs cycles CAVI
# ---------------------------------------------------------------------
# Init avec moyennes très loin et grosses variances pour rendre le mouvement visible
mu_q1, var_q1 = -2.5, 1.5
mu_q2, var_q2 = 3.5, 1.2

states = []
labels = []

# État 0 : init
states.append((mu_q1, var_q1, mu_q2, var_q2))
labels.append(r"$t = 0$  (init)")

# État 1 : après update de q_1
mu_q1, var_q1 = update_q1(mu_q2)
states.append((mu_q1, var_q1, mu_q2, var_q2))
labels.append(r"$t = 1$  (update $q_1$)")

# État 2 : après update de q_2
mu_q2, var_q2 = update_q2(mu_q1)
states.append((mu_q1, var_q1, mu_q2, var_q2))
labels.append(r"$t = 2$  (update $q_2$)")

# Itérer jusqu'à convergence (en réalité, 2 updates suffisent pour ce cas gaussien
# car les variances sont fixes et les moyennes convergent vers le vrai mu)
for _ in range(8):
    mu_q1, var_q1 = update_q1(mu_q2)
    mu_q2, var_q2 = update_q2(mu_q1)

states.append((mu_q1, var_q1, mu_q2, var_q2))
labels.append(r"$t \to \infty$  (convergence)")

print("Trajectoire CAVI :")
for i, (mq1, vq1, mq2, vq2) in enumerate(states):
    print(f"  État {i} : mu_q1={mq1:.3f}, var_q1={vq1:.3f}, "
          f"mu_q2={mq2:.3f}, var_q2={vq2:.3f}")

# ---------------------------------------------------------------------
# Helpers pour ellipses
# ---------------------------------------------------------------------
def ellipse_from_cov(mean, cov, n_std=1.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return Ellipse(mean, width, height, angle=angle, **kwargs)

# ---------------------------------------------------------------------
# Calcul de l'ELBO pour chaque état (pour annotation)
# Pour p gaussienne et q gaussien mean-field, ELBO = log p(x) - KL(q || p).
# Comme p(x) est fixé, on track juste -KL ou la KL directement.
# KL(q || p) entre deux gaussiennes 2D :
#   KL = 0.5 * [tr(Sigma_p^-1 Sigma_q) + (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q) - 2 + log|Sigma_p|/|Sigma_q|]
# ---------------------------------------------------------------------
def kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p):
    d = 2
    Sigma_p_inv = np.linalg.inv(Sigma_p)
    diff = mu_p - mu_q
    tr_term = np.trace(Sigma_p_inv @ Sigma_q)
    quad_term = diff @ Sigma_p_inv @ diff
    logdet_term = np.log(np.linalg.det(Sigma_p)) - np.log(np.linalg.det(Sigma_q))
    return 0.5 * (tr_term + quad_term - d + logdet_term)

kls = []
for (mq1, vq1, mq2, vq2) in states:
    mu_q_vec = np.array([mq1, mq2])
    Sigma_q_mat = np.diag([vq1, vq2])
    kls.append(kl_gaussian(mu_q_vec, Sigma_q_mat, mu_true, Sigma_true))

print("\nKL(q || p) à chaque état :")
for i, kl in enumerate(kls):
    print(f"  État {i} : KL = {kl:.4f}")

# ---------------------------------------------------------------------
# Figure : 4 panneaux côte à côte
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharex=True, sharey=True)

color_p = '#185FA5'      # vraie posterior
color_q = '#D85A30'      # q courant
color_p_edge = '#0F3D6E'
color_q_edge = '#A0421E'

x_lim = (-3.5, 5.5)
y_lim = (-1.5, 4.5)
xx, yy = np.meshgrid(np.linspace(*x_lim, 200), np.linspace(*y_lim, 200))
pos = np.dstack([xx, yy])

# Densité de la vraie posterior (fixe pour tous les panneaux)
rv_p = multivariate_normal(mu_true, Sigma_true)
zz_p = rv_p.pdf(pos)

for idx, (ax, (mq1, vq1, mq2, vq2), label, kl) in enumerate(
        zip(axes, states, labels, kls)):

    # 1. Vraie posterior en arrière-plan (contours + ellipses)
    ax.contour(xx, yy, zz_p, levels=6, colors=color_p, linewidths=1.0, alpha=0.6)
    ell_p_1 = ellipse_from_cov(mu_true, Sigma_true, n_std=1.0,
                                fill=False, edgecolor=color_p_edge,
                                linewidth=2.2, alpha=0.85,
                                label=r'vraie posterior $p$')
    ell_p_2 = ellipse_from_cov(mu_true, Sigma_true, n_std=2.0,
                                fill=False, edgecolor=color_p_edge,
                                linewidth=1.5, alpha=0.5, linestyle='--')
    ax.add_patch(ell_p_1)
    ax.add_patch(ell_p_2)

    # 2. q courant (gaussienne mean-field, axis-aligned)
    mu_q_vec = np.array([mq1, mq2])
    Sigma_q_mat = np.diag([vq1, vq2])

    rv_q = multivariate_normal(mu_q_vec, Sigma_q_mat)
    zz_q = rv_q.pdf(pos)
    ax.contourf(xx, yy, zz_q, levels=8, cmap='Oranges', alpha=0.35)

    ell_q_1 = ellipse_from_cov(mu_q_vec, Sigma_q_mat, n_std=1.0,
                                fill=False, edgecolor=color_q_edge,
                                linewidth=2.5, alpha=0.95,
                                label=r'$q_\phi$ courant (mean-field)')
    ell_q_2 = ellipse_from_cov(mu_q_vec, Sigma_q_mat, n_std=2.0,
                                fill=False, edgecolor=color_q_edge,
                                linewidth=1.5, alpha=0.6, linestyle='--')
    ax.add_patch(ell_q_1)
    ax.add_patch(ell_q_2)

    # Centre de q (point orange)
    ax.scatter([mq1], [mq2], color=color_q, s=80, zorder=5,
               edgecolor='white', linewidth=1.5)
    # Centre de p (point bleu)
    ax.scatter([mu_true[0]], [mu_true[1]], color=color_p, s=80, zorder=5,
               edgecolor='white', linewidth=1.5, marker='X')

    # 3. Mise en forme
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=12, pad=8)
    ax.set_xlabel(r'$z_1$', fontsize=11)
    if idx == 0:
        ax.set_ylabel(r'$z_2$', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. Encadré valeurs courantes
    ax.text(0.02, 0.97,
            fr'$\mu_{{q_1}}={mq1:.2f}, \sigma_{{q_1}}^2={vq1:.2f}$' + '\n'
            fr'$\mu_{{q_2}}={mq2:.2f}, \sigma_{{q_2}}^2={vq2:.2f}$' + '\n'
            fr'$D_{{KL}}(q\|p) = {kl:.3f}$',
            transform=ax.transAxes, fontsize=9, color='#333',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fafafa',
                      edgecolor='#bbb', linewidth=0.7))

    # 5. Sur les panneaux 1 et 2, indiquer ce qu'on vient d'updater
    if idx == 1:
        ax.annotate('$q_1$ vient\nd\'être mis à jour\n(direction $z_1$)',
                    xy=(mq1, mq2), xytext=(mq1 + 0.3, mq2 + 1.3),
                    fontsize=9, color=color_q_edge, ha='left',
                    arrowprops=dict(arrowstyle='->', color=color_q_edge, lw=1.0))
    elif idx == 2:
        ax.annotate('$q_2$ vient\nd\'être mis à jour\n(direction $z_2$)',
                    xy=(mq1, mq2), xytext=(mq1 - 1.5, mq2 - 1.2),
                    fontsize=9, color=color_q_edge, ha='left',
                    arrowprops=dict(arrowstyle='->', color=color_q_edge, lw=1.0))

# Légende globale en haut
handles = [
    plt.Line2D([0], [0], color=color_p_edge, linewidth=2.2,
               label=r'Vraie posterior $p$ (corrélée, ellipse oblique)'),
    plt.Line2D([0], [0], color=color_q_edge, linewidth=2.5,
               label=r'$q_\phi = q_1 \cdot q_2$ mean-field (axis-aligned)'),
    plt.Line2D([0], [0], marker='X', color=color_p, linewidth=0, markersize=10,
               markeredgecolor='white', label=r'centre de $p$'),
    plt.Line2D([0], [0], marker='o', color=color_q, linewidth=0, markersize=10,
               markeredgecolor='white', label=r'centre de $q_\phi$'),
]
fig.legend(handles=handles, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.02), fontsize=10, framealpha=0.95)

# Titre et caption
fig.suptitle("CAVI sur une posterior gaussienne 2D corrélée — mise à jour alternée de $q_1$ et $q_2$",
             fontsize=14, y=1.10)

fig.text(0.5, -0.03,
         r"Init mal placée (mu = (-2.5, 3.5) pour $q$ alors que $p$ centrée en (1.5, 1.0)). "
         r"À chaque itération, on update **un facteur à la fois** ($q_1$ puis $q_2$)." + "\n"
         r"Le centre de $q$ (point orange) se rapproche du centre de $p$ (croix bleue), et la KL décroît monotonement. "
         r"**Convergence en quelques itérations**, mais $q$ reste axis-aligned — la corrélation de $p$ est perdue (gap KL résiduel).",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'cavi_iterations.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nImage sauvegardée : {output_path}")
plt.show()
