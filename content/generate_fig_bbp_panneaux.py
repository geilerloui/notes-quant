"""
Figure B (piece maitresse) pour la note 03 : 4 panneaux montrant le spectre
empirique pour theta croissant, a q = 0.5 fixe.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_bbp_panneaux.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_bbp_panneaux.png"
SEED = 42

rng = np.random.default_rng(seed=SEED)

N = 300
T = 600
q = N / T
theta_c = np.sqrt(q)
lp = (1 + np.sqrt(q))**2
lm = (1 - np.sqrt(q))**2

theta_values = [0.3, 0.7, 1.5, 3.0]
labels = ['sous-critique', 'pres du seuil', 'sur-critique', 'sur-critique fort']
colors = ['#888888', '#d4a017', '#2ca02c', '#1f77b4']

def mp_density(lam, q):
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2
    out = np.zeros_like(lam)
    mask = (lam > lm) & (lam < lp)
    out[mask] = np.sqrt((lp - lam[mask]) * (lam[mask] - lm)) / (2 * np.pi * q * lam[mask])
    return out

lam_plot = np.linspace(0.001, max(theta_values) + 4, 1500)
rho_mp = mp_density(lam_plot, q)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharey=True)

for idx, (theta, ax, color, lab) in enumerate(zip(theta_values, axes, colors, labels)):
    u = rng.standard_normal(N)
    u = u / np.linalg.norm(u)
    Sigma_true = np.eye(N) + theta * np.outer(u, u)
    Sigma_sqrt = np.linalg.cholesky(Sigma_true)
    Z = rng.standard_normal(size=(N, T))
    X = Sigma_sqrt @ Z
    W = (X @ X.T) / T
    spectrum = np.linalg.eigvalsh(W)
    lambda_max = spectrum[-1]
    
    xmax_plot = max(lp + 0.5, lambda_max + 0.5)
    
    ax.hist(spectrum[spectrum < lp + 0.3], bins=35, density=True, alpha=0.5,
            color='steelblue', edgecolor='white', linewidth=0.5,
            label='spectre empirique')
    ax.plot(lam_plot, rho_mp, color='crimson', linewidth=2.0,
            label=r'$\rho_{\rm MP}$')
    
    is_out = lambda_max > lp + 0.05
    marker_color = 'darkgreen' if is_out else 'darkred'
    marker_height = 1.3
    ax.plot([lambda_max, lambda_max], [0, marker_height], color=marker_color, linewidth=2.2)
    ax.scatter([lambda_max], [marker_height], color=marker_color, s=70, zorder=5)
    ax.text(lambda_max, marker_height + 0.08,
            fr'$\lambda_1 = {lambda_max:.2f}$',
            ha='center', fontsize=9, color=marker_color, fontweight='bold')
    
    ax.axvline(lp, color='black', linestyle='--', linewidth=1, alpha=0.6)
    
    if theta > theta_c:
        lam_bbp = (1 + theta) * (1 + q / theta)
        ax.axvline(lam_bbp, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=fr'BBP: $(1+\theta)(1+q/\theta) = {lam_bbp:.2f}$')
    
    relation = '<' if theta < theta_c else '\\approx' if abs(theta - theta_c) < 0.1 else '>'
    ax.set_title(
        fr'$\theta = {theta}$  ({lab})' + '\n'
        + fr'$\theta {relation} \theta_c = \sqrt{{q}} = {theta_c:.2f}$',
        fontsize=10
    )
    ax.set_xlabel(r'$\lambda$')
    if idx == 0:
        ax.set_ylabel('densite')
    ax.set_xlim(0, xmax_plot)
    ax.set_ylim(0, 1.6)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Spectre empirique pour des spikes de force $\\theta$ croissante  (N={N}, T={T}, q={q})\n'
    + 'Pour $\\theta$ petit : le spike est invisible (noye dans le bulk MP). Pour $\\theta$ grand : il sort.',
    fontsize=11, y=1.02
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
