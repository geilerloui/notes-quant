"""
Figure 1 pour la note 05 : independance classique vs freeness.

On illustre empiriquement la difference entre :
- "non libres" : A et B partagent la meme base propre
- "libres" : B = U^T D U avec U uniforme sur le groupe orthogonal

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_freeness.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

OUTPUT_FILE = "fig_freeness.png"
SEED = 42

rng = np.random.default_rng(seed=SEED)
N = 500

eig_A = rng.uniform(0, 2, N)
eig_B = rng.uniform(0, 2, N)

# Cas non libres : A et B diagonales dans meme base => spectre de A+B = (a_i + b_i)
eig_sum_aligned = eig_A + eig_B

# Cas libres : B = U^T diag(eig_B) U avec U orthogonale uniforme
A_libre = np.diag(eig_A)
M = rng.standard_normal((N, N))
U, _ = np.linalg.qr(M)
B_libre = U.T @ np.diag(eig_B) @ U
eig_sum_libre = np.linalg.eigvalsh(A_libre + B_libre)

# Densite triangulaire (convolution classique de deux U[0,2])
def triangular_density(x, a, b):
    out = np.zeros_like(x)
    c = a + b
    h = b - a
    mask = (x >= 2*a) & (x <= 2*b)
    out[mask] = (1.0 / h**2) * np.where(x[mask] <= c, x[mask] - 2*a, 2*b - x[mask])
    return out

x_grid = np.linspace(0, 4, 500)
triangular = triangular_density(x_grid, 0, 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(eig_sum_aligned, bins=40, density=True, alpha=0.6, color='steelblue',
        edgecolor='white', linewidth=0.5,
        label='spectre empirique de $A+B$')
ax.plot(x_grid, triangular, color='crimson', linewidth=2.2,
        label='convolution CLASSIQUE\n(triangulaire)')
ax.set_xlabel(r'$\lambda$', fontsize=11)
ax.set_ylabel('densite', fontsize=11)
ax.set_title(
    r'$(a)$ NON LIBRES : $A$ et $B$ diagonales dans la meme base' + '\n'
    + 'spectre de $A+B$ = CONVOLUTION CLASSIQUE des spectres',
    fontsize=10
)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 4)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(eig_sum_libre, bins=40, density=True, alpha=0.6, color='purple',
        edgecolor='white', linewidth=0.5,
        label='spectre empirique de $A+B$')
ax.plot(x_grid, triangular, color='crimson', linewidth=2.2, linestyle='--', alpha=0.6,
        label='convolution CLASSIQUE\n(ce qu\'on aurait sans freeness)')
kde_libre = gaussian_kde(eig_sum_libre, bw_method=0.15)
ax.plot(x_grid, kde_libre(x_grid), color='darkblue', linewidth=2.2,
        label='convolution LIBRE\n(loi limite)')
ax.set_xlabel(r'$\lambda$', fontsize=11)
ax.set_ylabel('densite', fontsize=11)
ax.set_title(
    r'$(b)$ LIBRES : $B$ tournee aleatoirement par rapport a $A$' + '\n'
    + 'spectre suit la CONVOLUTION LIBRE — distribution differente',
    fontsize=10
)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, 4)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Independance classique vs freeness  (N={N})\n'
    + 'Memes spectres marginaux de $A$ et $B$ -> spectres de $A+B$ DIFFERENTS selon la relation entre bases propres',
    fontsize=11, y=1.04
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
