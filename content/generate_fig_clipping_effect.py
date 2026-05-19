"""
Figure B pour la note 04 : effet visuel du clipping.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_clipping_effect.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_clipping_effect.png"
SEED = 42

rng = np.random.default_rng(seed=SEED)

N = 200
T = 500
q = N / T
lp = (1 + np.sqrt(q))**2
lm = (1 - np.sqrt(q))**2

strengths = [6.0, 3.0, 1.5]
factors = []
for _ in range(3):
    u = rng.standard_normal(N); u /= np.linalg.norm(u)
    factors.append(u)

Sigma_true = np.eye(N)
for theta, u in zip(strengths, factors):
    Sigma_true += theta * np.outer(u, u)

Z = rng.standard_normal(size=(N, T))
X = np.linalg.cholesky(Sigma_true) @ Z
W = (X @ X.T) / T
emp_spectrum = np.sort(np.linalg.eigvalsh(W))[::-1]

clipped_spectrum = emp_spectrum.copy()
bulk_mask = clipped_spectrum <= lp
bulk_mean = clipped_spectrum[bulk_mask].mean()
clipped_spectrum[bulk_mask] = bulk_mean

def mp_density(lam, q):
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2
    out = np.zeros_like(lam)
    mask = (lam > lm) & (lam < lp)
    out[mask] = np.sqrt((lp - lam[mask]) * (lam[mask] - lm)) / (2 * np.pi * q * lam[mask])
    return out

lam_grid = np.linspace(0.001, 10, 500)
rho_mp = mp_density(lam_grid, q)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
bulk = emp_spectrum[emp_spectrum < lp + 0.3]
ax.hist(bulk, bins=30, density=True, alpha=0.5, color='steelblue',
        edgecolor='white', linewidth=0.5)
ax.plot(lam_grid, rho_mp, color='crimson', linewidth=1.8, label=r'$\rho_{\rm MP}$')
y_top = 1.4
outliers = emp_spectrum[emp_spectrum > lp + 0.1]
for lam_o in outliers:
    ax.plot([lam_o, lam_o], [0, y_top], color='darkgreen', linewidth=2.0)
    ax.scatter([lam_o], [y_top], color='darkgreen', s=50, zorder=5)
ax.axvline(lp, color='black', linestyle='--', linewidth=1, alpha=0.7,
           label=fr'$\lambda_+={lp:.2f}$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('densite')
ax.set_title(r'$(a)$ Spectre empirique BRUT' + '\n' +
             'bulk etale + outliers biaises',
             fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.7)
ax.grid(True, alpha=0.3)

ax = axes[1]
n_bulk = bulk_mask.sum()
ax.axvline(bulk_mean, color='steelblue', linewidth=8, alpha=0.7,
           ymin=0, ymax=0.85, label=f'bulk -> {bulk_mean:.2f}\n({n_bulk} val. propres)')
for lam_o in outliers:
    ax.plot([lam_o, lam_o], [0, y_top], color='darkgreen', linewidth=2.0)
    ax.scatter([lam_o], [y_top], color='darkgreen', s=50, zorder=5)
ax.axvline(lp, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel(r'$\lambda$')
ax.set_title(r'$(b)$ Apres CLIPPING' + '\n' +
             'bulk remplace par moyenne, outliers conserves',
             fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.7)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.axvline(1.0, color='steelblue', linewidth=8, alpha=0.7,
           ymin=0, ymax=0.85, label=f'baseline : $\\lambda=1$\n({N-3} val. propres)')
for theta in sorted(strengths, reverse=True):
    lam_vrai = 1 + theta
    ax.plot([lam_vrai, lam_vrai], [0, y_top], color='darkgreen', linewidth=2.0, linestyle='--')
    ax.scatter([lam_vrai], [y_top], color='darkgreen', s=50, zorder=5, marker='s')
    ax.text(lam_vrai, y_top + 0.05, f'{lam_vrai:.1f}',
            ha='center', fontsize=9, color='darkgreen')
ax.set_xlabel(r'$\lambda$')
ax.set_title(r'$(c)$ Spectre VRAI (reference)' + '\n' +
             f'{N-3} a 1 + {len(strengths)} spikes a $1+\\theta_k$',
             fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.7)
ax.grid(True, alpha=0.3)

fig.suptitle(f'Effet du clipping sur le spectre (N={N}, T={T}, q={q:.2f})',
             fontsize=12, y=1.02)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
