"""
Figure D pour la note 03 : test Tracy-Widom applique a la detection de signal.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_bbp_tw_test.py

Le calcul prend ~1 minute (4000 GOE).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

OUTPUT_FILE = "fig_bbp_tw_test.png"
SEED = 99

rng = np.random.default_rng(seed=SEED)

N = 200
T = 400
q = N / T
lp = (1 + np.sqrt(q))**2

# Distribution sous H0
n_h0 = 4000
print("Simulations sous H0 (bruit pur)...")
lam_max_h0 = np.empty(n_h0)
for k in range(n_h0):
    Z = rng.standard_normal(size=(N, T))
    W = (Z @ Z.T) / T
    lam_max_h0[k] = np.linalg.eigvalsh(W)[-1]

# Deux scenarios
theta_test_1 = 0.4  # sous-critique
u = rng.standard_normal(N); u /= np.linalg.norm(u)
Sigma1 = np.eye(N) + theta_test_1 * np.outer(u, u)
Z = rng.standard_normal(size=(N, T))
X = np.linalg.cholesky(Sigma1) @ Z
lam_obs_1 = np.linalg.eigvalsh((X @ X.T) / T)[-1]

theta_test_2 = 1.5  # sur-critique
u = rng.standard_normal(N); u /= np.linalg.norm(u)
Sigma2 = np.eye(N) + theta_test_2 * np.outer(u, u)
Z = rng.standard_normal(size=(N, T))
X = np.linalg.cholesky(Sigma2) @ Z
lam_obs_2 = np.linalg.eigvalsh((X @ X.T) / T)[-1]

p_value_1 = np.mean(lam_max_h0 >= lam_obs_1)
p_value_2 = np.mean(lam_max_h0 >= lam_obs_2)

# Figure
fig, ax = plt.subplots(figsize=(11, 5.5))

ax.hist(lam_max_h0, bins=60, density=True, alpha=0.4, color='steelblue',
        edgecolor='white', linewidth=0.5,
        label=f'distribution de $\\lambda_{{\\max}}$ sous H$_0$\n(bruit pur, {n_h0} simulations)')

kde = gaussian_kde(lam_max_h0, bw_method=0.3)
x_grid = np.linspace(lam_max_h0.min() - 0.05, max(lam_max_h0.max(), lam_obs_2) + 0.05, 500)
ax.plot(x_grid, kde(x_grid), color='steelblue', linewidth=2, alpha=0.7)

q_95 = np.quantile(lam_max_h0, 0.95)
q_99 = np.quantile(lam_max_h0, 0.99)
ax.axvline(q_95, color='orange', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'seuil 5% : $\\lambda = {q_95:.3f}$')
ax.axvline(q_99, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'seuil 1% : $\\lambda = {q_99:.3f}$')

y_obs = 5.0
ax.plot([lam_obs_1, lam_obs_1], [0, y_obs], color='darkgray', linewidth=2.5)
ax.scatter([lam_obs_1], [y_obs], color='darkgray', s=100, zorder=5, marker='v')
ax.text(lam_obs_1, y_obs + 0.5,
        f'$\\theta = {theta_test_1}$\n(sous-critique)\n$\\lambda_{{\\rm obs}} = {lam_obs_1:.2f}$\np = {p_value_1:.3f}\n=> non significatif',
        ha='center', fontsize=9, color='darkgray', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke',
                  edgecolor='darkgray', alpha=0.9))

ax.plot([lam_obs_2, lam_obs_2], [0, y_obs], color='darkgreen', linewidth=2.5)
ax.scatter([lam_obs_2], [y_obs], color='darkgreen', s=100, zorder=5, marker='v')
ax.text(lam_obs_2, y_obs + 0.5,
        f'$\\theta = {theta_test_2}$\n(sur-critique)\n$\\lambda_{{\\rm obs}} = {lam_obs_2:.2f}$\np = {p_value_2:.4f}\n=> SIGNAL detecte',
        ha='center', fontsize=9, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                  edgecolor='darkgreen', alpha=0.9))

ax.axvspan(lam_max_h0.min() - 0.05, q_99, alpha=0.06, color='steelblue')

ax.set_xlabel(r'$\lambda_{\max}$ (plus grande valeur propre empirique)', fontsize=11)
ax.set_ylabel('densite', fontsize=11)
ax.set_title(
    f'Test de detection : la valeur propre observee est-elle compatible avec du bruit ?\n'
    f'($N={N}$, $T={T}$, $q={q}$, $\\theta_c = \\sqrt{{q}} = {np.sqrt(q):.3f}$)',
    fontsize=11
)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(lam_max_h0.min() - 0.05, max(lam_max_h0.max(), lam_obs_2) + 0.3)
ax.set_ylim(0, 9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
