"""
Figure C pour la note 03 : diagramme de bifurcation BBP.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_bbp_bifurcation.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_bbp_bifurcation.png"
SEED = 123

rng = np.random.default_rng(seed=SEED)

N = 200
T = 400
q = N / T
theta_c = np.sqrt(q)
lp = (1 + np.sqrt(q))**2

theta_grid = np.concatenate([
    np.linspace(0.01, theta_c - 0.1, 12),
    np.linspace(theta_c - 0.1, theta_c + 0.1, 8),
    np.linspace(theta_c + 0.1, 4.0, 18)
])

n_trials = 40
results = []

for theta in theta_grid:
    for _ in range(n_trials):
        u = rng.standard_normal(N)
        u = u / np.linalg.norm(u)
        Sigma_true = np.eye(N) + theta * np.outer(u, u)
        Sigma_sqrt = np.linalg.cholesky(Sigma_true)
        Z = rng.standard_normal(size=(N, T))
        X = Sigma_sqrt @ Z
        W = (X @ X.T) / T
        results.append((theta, np.linalg.eigvalsh(W)[-1]))

results = np.array(results)

theta_smooth = np.linspace(0.01, 4.0, 500)
lam_theory = np.where(
    theta_smooth < theta_c,
    lp,
    (1 + theta_smooth) * (1 + q / theta_smooth)
)
vrai_lambda = 1 + theta_smooth

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.scatter(results[:, 0], results[:, 1], s=10, alpha=0.25, color='steelblue',
           label=f'simulations empiriques\n({n_trials} tirages par valeur de $\\theta$)')
ax.plot(theta_smooth, lam_theory, color='crimson', linewidth=2.5,
        label=r'prediction BBP : $\lambda_1^{\rm emp}$')
ax.plot(theta_smooth, vrai_lambda, color='darkgreen', linewidth=2, linestyle='--',
        label=r'spectre vrai : $1 + \theta$')
ax.axvline(theta_c, color='black', linestyle=':', linewidth=1.5, alpha=0.8,
           label=fr'seuil BBP : $\theta_c = \sqrt{{q}} = {theta_c:.3f}$')
ax.axhline(lp, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax.text(3.85, lp + 0.07, fr'$\lambda_+ = {lp:.2f}$', fontsize=9, color='gray', ha='right')

ax.axvspan(0, theta_c, alpha=0.07, color='red')
ax.axvspan(theta_c, 4, alpha=0.07, color='green')

ax.text(theta_c / 2, 4.5, 'INVISIBLE\n($\\theta < \\theta_c$)\nspike noye\ndans le bulk',
        ha='center', va='center', fontsize=10, color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose',
                  edgecolor='darkred', alpha=0.85))

ax.text(2.5, 2.6, 'DETECTABLE\n($\\theta > \\theta_c$)\nspike sort du bulk',
        ha='center', va='center', fontsize=10, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew',
                  edgecolor='darkgreen', alpha=0.85))

ax.set_xlabel(r'$\theta$ (force du vrai spike)', fontsize=11)
ax.set_ylabel(r'$\lambda_1^{\rm emp}$ (plus grande valeur propre empirique)', fontsize=11)
ax.set_title(
    f'Transition BBP : bifurcation de $\\lambda_1^{{\\rm emp}}$ en fonction de la force $\\theta$\n'
    f'($N={N}$, $T={T}$, $q={q}$)',
    fontsize=11
)
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0, 4)
ax.set_ylim(2, 6.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
