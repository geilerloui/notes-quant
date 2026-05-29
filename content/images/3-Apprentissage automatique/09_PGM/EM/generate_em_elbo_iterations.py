"""
Génère em_elbo_iterations.png : la vue ELBO d'Andrew Ng appliquée à un vrai GMM.

Un seul panneau : la vraie log-vraisemblance ℓ(μ_1) (courbe bleue) avec 3 paraboles
ELBO successives (vert foncé → vert clair), chacune tangente à ℓ en μ_1^(t).

Setup pédagogique : 3 itérations EM sur μ_1 seul (autres paramètres fixés à MLE).
Init très mauvaise (μ_1^(0) = -1) pour que la progression soit visible.

Output : em_elbo_iterations.png
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
# Dataset
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
# Fonctions
# ---------------------------------------------------------------------
def log_likelihood(mu1, data):
    """ℓ(μ_1) = somme des log p_theta(x_n)."""
    c1 = pi1 * norm.pdf(data, mu1, sigma1)
    c2 = pi2 * norm.pdf(data, mu2_true, sigma2)
    return np.sum(np.log(c1 + c2))

def responsibilities_at(mu1, data):
    """γ_nk pour chaque point, sachant μ_1 courant."""
    c1 = pi1 * norm.pdf(data, mu1, sigma1)
    c2 = pi2 * norm.pdf(data, mu2_true, sigma2)
    Z = c1 + c2
    return c1 / Z, c2 / Z

def elbo(mu1_query, gammas_fixed, data):
    """
    ELBO(μ_1 | q) = E_q[log p_theta(x, z)] - E_q[log q(z)]
                  = Σ_n Σ_k γ_nk log[π_k N_k(x_n; μ_k)] - Σ_n Σ_k γ_nk log γ_nk
    avec γ fixé (= responsabilités calculées au point μ_1^(t)).
    μ_1_query varie pour tracer la parabole.
    """
    g1, g2 = gammas_fixed
    log_pi1_N1 = np.log(pi1) + norm.logpdf(data, mu1_query, sigma1)
    log_pi2_N2 = np.log(pi2) + norm.logpdf(data, mu2_true, sigma2)
    eps = 1e-300
    H_q = -np.sum(g1 * np.log(g1 + eps) + g2 * np.log(g2 + eps))
    return np.sum(g1 * log_pi1_N1 + g2 * log_pi2_N2) + H_q

# ---------------------------------------------------------------------
# Simulation de 3 itérations EM (sur μ_1 seul)
# ---------------------------------------------------------------------
mu1_init = -1.0
mu1_iters = [mu1_init]
gammas_iters = []

n_iters = 3
for t in range(n_iters):
    mu1_t = mu1_iters[-1]
    g1_t, g2_t = responsibilities_at(mu1_t, data)
    gammas_iters.append((g1_t, g2_t))
    mu1_new = np.sum(g1_t * data) / np.sum(g1_t)
    mu1_iters.append(mu1_new)

# ---------------------------------------------------------------------
# Calcul de la courbe ℓ et des 3 paraboles ELBO
# ---------------------------------------------------------------------
mu1_grid = np.linspace(-3, 6, 400)
ll_curve = np.array([log_likelihood(m, data) for m in mu1_grid])

elbo_curves = []
for t in range(n_iters):
    gammas_fixed = gammas_iters[t]
    e = np.array([elbo(m, gammas_fixed, data) for m in mu1_grid])
    elbo_curves.append(e)

# ---------------------------------------------------------------------
# Plot — un seul panneau
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

# Courbe ℓ
ax.plot(mu1_grid, ll_curve, color='#185FA5', linewidth=2.8, zorder=4,
        label=r'$\ell(\mu_1) = \log p_\theta(X)$  (vraie log-vraisemblance)')

# Couleurs pour les 3 ELBO (vert foncé → vert clair)
elbo_colors = ['#0F6E56', '#5BA85B', '#A5D6A7']

# Tracer les 3 paraboles ELBO
for t, (e_curve, col) in enumerate(zip(elbo_curves, elbo_colors)):
    mask = e_curve > ll_curve.min() - 50
    ax.plot(mu1_grid[mask], e_curve[mask], color=col, linewidth=2.2,
            linestyle='--', alpha=0.92, zorder=3,
            label=fr'ELBO $g_{t}(\mu_1) = \mathcal{{L}}(\mu_1 \mid q^{{({t})}})$')

# Marquer les points μ_1^(t) sur la courbe ℓ
point_colors = ['#A32D2D', '#D85A30', '#C77A30', '#1B5E20']
for t in range(len(mu1_iters)):
    mu_t = mu1_iters[t]
    ll_t = log_likelihood(mu_t, data)
    col = point_colors[t]
    ax.scatter([mu_t], [ll_t], color=col, s=130, zorder=6,
               edgecolor='white', linewidth=2)
    offset_y = 20 if t % 2 == 0 else -45
    ax.annotate(fr'$\mu_1^{{({t})}}={mu_t:.2f}$',
                xy=(mu_t, ll_t),
                xytext=(mu_t + 0.15, ll_t + offset_y),
                fontsize=11, color=col, fontweight='bold')
    ax.axvline(mu_t, color=col, linewidth=0.7, linestyle=':', alpha=0.4)

# Mise en forme
ax.set_xlabel(r'$\mu_1$  (paramètre à optimiser, autres fixés)', fontsize=12)
ax.set_ylabel(r'log-vraisemblance  /  ELBO', fontsize=11)
ax.set_title("EM = grimper sur $\\ell(\\mu_1)$ via une suite d'ELBO tangentes",
             fontsize=13, pad=12)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-3, 6)
ax.set_ylim(ll_curve.min() - 30, ll_curve.max() + 70)

# Caption sous l'image
fig.text(0.5, -0.01,
         r"À chaque itération $t$ : le E-step construit la parabole $g_t(\mu_1)$ tangente à $\ell$ en $\mu_1^{(t)}$. "
         r"Le M-step monte au sommet de $g_t$ → on obtient $\mu_1^{(t+1)}$ sur la courbe bleue." '\n'
         r"On répète : nouvelle ELBO tangente à $\mu_1^{(t+1)}$, nouveau sommet, etc. "
         r"La log-vraisemblance croît à chaque itération.",
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.03, 1, 1])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'em_elbo_iterations.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
print(f"μ_1 itérations : {[round(m, 3) for m in mu1_iters]}")
plt.show()
