"""
Génère la figure pédagogique "Dirac vs histogramme" pour la note RMT 01_Fondamentaux.

Deux panneaux côte à côte pour une GOE normalisée :
- Gauche  : N pics verticaux de hauteur 1/N à chaque valeur propre (mesure mu_N brute)
- Droite  : histogramme par bins + courbe limite du demi-cercle

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_dirac_vs_histogramme.py

Sortie : fig_dirac_vs_histogramme.png dans le repertoire courant.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --- Parametres ---------------------------------------------------------------
N = 200
SEED = 42
N_BINS = 30
OUTPUT_FILE = "fig_dirac_vs_histogramme.png"

rng = np.random.default_rng(seed=SEED)

# --- Tirage d'une GOE normalisee ----------------------------------------------
# Convention : entrees hors-diag N(0,1), diag N(0,2), puis on divise par sqrt(N)
A = rng.standard_normal(size=(N, N))
M = (A + A.T) / np.sqrt(2)  # variance hors-diag = 1, diag = 2
W = M / np.sqrt(N)

eigvals = np.linalg.eigvalsh(W)

# --- Densite theorique : demi-cercle ------------------------------------------
lam = np.linspace(-2.0, 2.0, 500)
rho_sc = (1.0 / (2.0 * np.pi)) * np.sqrt(np.maximum(4.0 - lam**2, 0.0))

# --- Figure -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)

# PANNEAU GAUCHE : la mesure mu_N comme somme de Dirac
ax = axes[0]
height = 1.0 / N
ax.vlines(eigvals, ymin=0, ymax=height, colors='steelblue', linewidth=0.8, alpha=0.85)
ax.scatter(eigvals, np.full_like(eigvals, height), s=8, color='steelblue', zorder=3)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.0005, height * 2.5)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'masse')
ax.set_title(
    r'$(a)$ Mesure spectrale empirique $\mu_N = \frac{1}{N}\sum_{i=1}^{N}\delta_{\lambda_i}$' + '\n'
    + f'$N={N}$ pics, chacun de hauteur $1/N = {height:.4f}$',
    fontsize=10
)
ax.text(0, height * 2.0,
        'tiges denses pres de 0,\nclairsemees pres de $\\pm 2$',
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='goldenrod', alpha=0.8))

# PANNEAU DROIT : on agrege les Dirac dans des bins -> histogramme
ax = axes[1]
ax.hist(
    eigvals, bins=N_BINS, density=True,
    color='steelblue', alpha=0.55, edgecolor='white', linewidth=0.8,
    label='histogramme (Dirac agreges par intervalle)'
)
ax.plot(lam, rho_sc, color='crimson', linewidth=2.2, label=r'densite limite $\rho_{\rm sc}(\lambda)$')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'densite')
ax.set_title(
    r'$(b)$ On regroupe les Dirac dans des intervalles' + '\n'
    + r'$\Rightarrow$ apparait la densite $\rho_{\rm sc}(\lambda)=\frac{1}{2\pi}\sqrt{4-\lambda^2}$',
    fontsize=10
)
ax.legend(loc='upper right', fontsize=9, frameon=True)

# Fleche entre les deux panneaux
fig.text(0.495, 0.5, r'$\longrightarrow$', ha='center', va='center',
         fontsize=20, color='gray')

plt.tight_layout(rect=[0, 0, 1, 1])

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
