"""
Figure 2 pour la note 05 : convolution libre additive sur GOE.

Demi-cercle + demi-cercle = demi-cercle PLUS LARGE de rayon sqrt(2) fois plus grand.
C'est l'analogue libre de "somme de deux gaussiennes = gaussienne plus large".

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_free_addition.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

OUTPUT_FILE = "fig_free_addition.png"
SEED = 7

rng = np.random.default_rng(seed=SEED)
N = 1000

def goe(N, rng):
    M = rng.standard_normal((N, N))
    M = (M + M.T) / np.sqrt(2)
    return M / np.sqrt(N)

A = goe(N, rng)
B = goe(N, rng)
eig_A = np.linalg.eigvalsh(A)
eig_B = np.linalg.eigvalsh(B)
eig_sum = np.linalg.eigvalsh(A + B)

def semicircle(x, R=2):
    out = np.zeros_like(x)
    mask = np.abs(x) < R
    out[mask] = (2.0 / (np.pi * R**2)) * np.sqrt(R**2 - x[mask]**2)
    return out

x_grid = np.linspace(-4, 4, 500)
rho_single = semicircle(x_grid, R=2)
rho_free_sum = semicircle(x_grid, R=2*np.sqrt(2))
dx = x_grid[1] - x_grid[0]
rho_classical_conv = fftconvolve(rho_single, rho_single, mode='same') * dx

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(eig_A, bins=40, density=True, alpha=0.5, color='steelblue',
        edgecolor='white', linewidth=0.5,
        label=r'spectre empirique de $A$')
ax.hist(eig_B, bins=40, density=True, alpha=0.3, color='darkorange',
        edgecolor='white', linewidth=0.5,
        label=r'spectre empirique de $B$')
ax.plot(x_grid, rho_single, color='crimson', linewidth=2.2,
        label=r'demi-cercle $\rho_{\rm sc}$ sur $[-2, 2]$')
ax.set_xlabel(r'$\lambda$', fontsize=11)
ax.set_ylabel('densite', fontsize=11)
ax.set_title(
    r'$(a)$ Deux GOE independantes $A$ et $B$ (de meme loi)' + '\n'
    + r'spectre = demi-cercle sur $[-2, 2]$',
    fontsize=11
)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(eig_sum, bins=50, density=True, alpha=0.5, color='purple',
        edgecolor='white', linewidth=0.5,
        label=r'spectre empirique de $A+B$')
ax.plot(x_grid, rho_free_sum, color='darkblue', linewidth=2.5,
        label=r'convolution LIBRE : demi-cercle sur $[-2\sqrt{2}, 2\sqrt{2}]$')
ax.plot(x_grid, rho_classical_conv, color='crimson', linewidth=2, linestyle='--', alpha=0.6,
        label=r'convolution CLASSIQUE (incorrecte ici)')
ax.axvline(2*np.sqrt(2), color='gray', linestyle=':', alpha=0.6)
ax.axvline(-2*np.sqrt(2), color='gray', linestyle=':', alpha=0.6)
ax.text(2*np.sqrt(2), 0.33, fr'  $2\sqrt{{2}} \approx {2*np.sqrt(2):.2f}$',
        fontsize=9, color='gray')
ax.set_xlabel(r'$\lambda$', fontsize=11)
ax.set_ylabel('densite', fontsize=11)
ax.set_title(
    r'$(b)$ Spectre de $A+B$ = demi-cercle PLUS LARGE' + '\n'
    + r'"somme de deux demi-cercles = demi-cercle $\sqrt{2}$ fois plus grand"',
    fontsize=11
)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Convolution libre additive : $\\mu_A \\boxplus \\mu_B$ pour deux GOE  (N={N})\n'
    + r'Analogue libre de "somme de gaussiennes = gaussienne". Ici : demi-cercle + demi-cercle = demi-cercle.',
    fontsize=11, y=1.04
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
