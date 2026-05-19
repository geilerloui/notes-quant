"""
Figure 3 pour la note 05 : produit de matrices et dynamical isotropy.

Pour differentes profondeurs L, on calcule le spectre singulier de J = W_L * ... * W_1
avec deux initialisations : Xavier (gaussien) vs Orthogonale (Haar).

Resultat spectaculaire : a L=50, Xavier a un conditionnement de ~10^18 (reseau
strictement non entrainable), alors qu'orthogonale garde s_i = 1 exactement.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_deep_init.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_deep_init.png"
SEED = 2024

rng = np.random.default_rng(seed=SEED)
N = 200
depths = [1, 5, 20, 50]

def xavier_layer(N, rng):
    return rng.standard_normal((N, N)) / np.sqrt(N)

def orthogonal_layer(N, rng):
    M = rng.standard_normal((N, N))
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))  # correction pour vraie Haar
    return Q

results_xavier = {}
results_ortho = {}

for L in depths:
    J_xavier = np.eye(N)
    for _ in range(L):
        J_xavier = xavier_layer(N, rng) @ J_xavier
    results_xavier[L] = np.linalg.svd(J_xavier, compute_uv=False)
    
    J_ortho = np.eye(N)
    for _ in range(L):
        J_ortho = orthogonal_layer(N, rng) @ J_ortho
    results_ortho[L] = np.linalg.svd(J_ortho, compute_uv=False)
    
    print(f"L={L:3d} | Xavier: ratio={results_xavier[L].max()/results_xavier[L].min():.2e} | Ortho: ratio={results_ortho[L].max()/results_ortho[L].min():.2e}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

ax = axes[0]
for L, c in zip(depths, colors):
    s = results_xavier[L]
    ax.semilogy(np.arange(len(s)) / len(s), np.sort(s)[::-1],
                color=c, linewidth=2, label=f'$L = {L}$')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6,
           label=r'$\sigma = 1$ (ideal)')
ax.set_xlabel('rang relatif (0 = plus grande val. singuliere)', fontsize=11)
ax.set_ylabel(r'$\sigma_i$ (echelle log)', fontsize=11)
ax.set_title(
    r'$(a)$ XAVIER : $W_\ell \sim \mathcal{N}(0, 1/N)$' + '\n'
    + 'dispersion qui explose avec la profondeur',
    fontsize=11
)
ax.legend(loc='center right', fontsize=10)
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
for L, c in zip(depths, colors):
    s = results_ortho[L]
    ax.semilogy(np.arange(len(s)) / len(s), np.sort(s)[::-1],
                color=c, linewidth=2, label=f'$L = {L}$')
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6,
           label=r'$\sigma = 1$ (ideal)')
ax.set_xlabel('rang relatif', fontsize=11)
ax.set_ylabel(r'$\sigma_i$ (echelle log)', fontsize=11)
ax.set_title(
    r'$(b)$ ORTHOGONALE : $W_\ell \sim$ Haar$(O(N))$' + '\n'
    + r'$\sigma_i = 1$ a toute profondeur — dynamical isotropy parfaite',
    fontsize=11
)
ax.legend(loc='center right', fontsize=10)
ax.grid(True, alpha=0.3, which='both')

y_min = min(min(results_xavier[depths[-1]].min(), results_ortho[depths[-1]].min()), 1e-3)
y_max = max(results_xavier[depths[-1]].max(), results_ortho[depths[-1]].max()) * 2
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

fig.suptitle(
    f'Spectre singulier du produit $J = W_L \\cdots W_1$ pour differentes profondeurs $L$  ($N={N}$)\n'
    + 'Xavier disperse exponentiellement avec $L$. Orthogonale preserve l\'isotropie a toute profondeur.',
    fontsize=11, y=1.04
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
