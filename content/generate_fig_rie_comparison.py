"""
Figure C pour la note 04 : comparaison des 4 estimateurs de spectre.
BRUT, CLIPPING, RIE, VRAI sur la meme simulation.

Implementation RIE pragmatique :
- Outliers : inversion BBP pour debiaser
- Bulk : shrinkage doux vers la moyenne

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_rie_comparison.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_rie_comparison.png"
SEED = 42

rng = np.random.default_rng(seed=SEED)

N = 200
T = 500
q = N / T
lp = (1 + np.sqrt(q))**2

strengths = [6.0, 3.0, 1.5]
factors = []
for _ in range(3):
    u = rng.standard_normal(N); u /= np.linalg.norm(u)
    factors.append(u)
Sigma_true = np.eye(N)
for theta, u in zip(strengths, factors):
    Sigma_true += theta * np.outer(u, u)
true_spectrum_sorted = np.sort(np.linalg.eigvalsh(Sigma_true))[::-1]

Z = rng.standard_normal(size=(N, T))
X = np.linalg.cholesky(Sigma_true) @ Z
W = (X @ X.T) / T
emp_spectrum_sorted = np.sort(np.linalg.eigvalsh(W))[::-1]

clipped_spectrum = emp_spectrum_sorted.copy()
bulk_mask = clipped_spectrum <= lp
bulk_mean = clipped_spectrum[bulk_mask].mean()
clipped_spectrum[bulk_mask] = bulk_mean

def invert_bbp(lam_emp, q):
    """Inversion BBP : retourne 1+theta a partir de la valeur empirique du spike."""
    b = 1 + q - lam_emp
    disc = b**2 - 4*q
    if disc < 0: return 1.0
    return 1 + (-b + np.sqrt(disc)) / 2

rie_spectrum = emp_spectrum_sorted.copy()
for i, lam in enumerate(emp_spectrum_sorted):
    if lam > lp:
        rie_spectrum[i] = invert_bbp(lam, q)
    else:
        rie_spectrum[i] = bulk_mean + 0.3 * (lam - bulk_mean)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
positions = np.arange(N) + 1
ax.semilogy(positions, emp_spectrum_sorted, 'o', color='crimson', markersize=3.5,
            label='BRUT $\\hat\\Sigma$', alpha=0.7)
ax.semilogy(positions, clipped_spectrum, 's', color='darkorange', markersize=3.5,
            label='CLIPPING', alpha=0.7)
ax.semilogy(positions, rie_spectrum, '^', color='purple', markersize=3.5,
            label='RIE', alpha=0.7)
ax.semilogy(positions, true_spectrum_sorted, '-', color='steelblue', linewidth=2,
            label='VRAI $\\Sigma_{\\rm vrai}$')
ax.axhline(lp, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax.text(N - 30, lp * 1.1, fr'$\lambda_+={lp:.2f}$', fontsize=9, color='gray')
ax.axhline(1.0, color='lightblue', linestyle=':', linewidth=1, alpha=0.6)
ax.set_xlabel('rang de la valeur propre', fontsize=11)
ax.set_ylabel(r'$\lambda$ (log)', fontsize=11)
ax.set_title(r'$(a)$ Toutes les valeurs propres', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
n_zoom = 10
positions_zoom = np.arange(n_zoom) + 1
ax.plot(positions_zoom, emp_spectrum_sorted[:n_zoom], 'o-', color='crimson', markersize=10,
        label='BRUT', linewidth=2, alpha=0.7)
ax.plot(positions_zoom, clipped_spectrum[:n_zoom], 's-', color='darkorange', markersize=10,
        label='CLIPPING', linewidth=2, alpha=0.7)
ax.plot(positions_zoom, rie_spectrum[:n_zoom], '^-', color='purple', markersize=10,
        label='RIE', linewidth=2, alpha=0.7)
ax.plot(positions_zoom, true_spectrum_sorted[:n_zoom], 'D-', color='steelblue', markersize=10,
        label='VRAI', linewidth=2.5)

for k, val in enumerate(true_spectrum_sorted[:3]):
    ax.annotate(f'vrai = {val:.2f}',
                xy=(k+1, val), xytext=(k+1.3, val + 0.7),
                fontsize=9, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.8))

ax.axhline(lp, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax.set_xlabel('rang (1 = plus grande)', fontsize=11)
ax.set_ylabel(r'$\lambda$', fontsize=11)
ax.set_title(r'$(b)$ Zoom top-10' + '\n' +
             'RIE rapproche les outliers du VRAI', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(positions_zoom)

fig.suptitle(f'Comparaison des estimateurs de spectre (N={N}, T={T}, q={q:.2f})',
             fontsize=12, y=1.02)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
