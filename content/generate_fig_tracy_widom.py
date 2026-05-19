"""
Genere la figure pedagogique "Tracy-Widom" pour la note RMT 01_Fondamentaux.

Deux panneaux cote a cote :
- Gauche : distribution brute de lambda_max pour differents N
           (montre pourquoi on doit rescaler : tout s'ecrase sur 2)
- Droite : apres rescaling xi = N^{2/3}*(lambda_max - 2), les histogrammes
           se superposent a la densite limite Tracy-Widom

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_tracy_widom.py

Sortie : fig_tracy_widom.png dans le repertoire courant.
Le calcul prend ~1-2 minutes (tirage de plusieurs milliers de GOE).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --- Parametres -------------------------------------------------------------
SEED = 2024
OUTPUT_FILE = "fig_tracy_widom.png"

N_ref = 500  ; n_ref = 4000   # reference "TW limite" a grand N
N_med = 100  ; n_med = 4000
N_sml = 20   ; n_sml = 4000

rng = np.random.default_rng(seed=SEED)

def sample_lambda_max_goe(N, n_samples, rng):
    """Tire n_samples valeurs de lambda_max pour une GOE normalisee NxN."""
    lambdas = np.empty(n_samples)
    for k in range(n_samples):
        A = rng.standard_normal(size=(N, N))
        M = (A + A.T) / np.sqrt(2)
        W = M / np.sqrt(N)
        lambdas[k] = np.linalg.eigvalsh(W)[-1]
    return lambdas

print(f"Echantillonnage N_sml = {N_sml} ...")
lam_sml = sample_lambda_max_goe(N_sml, n_sml, rng)
print(f"Echantillonnage N_med = {N_med} ...")
lam_med = sample_lambda_max_goe(N_med, n_med, rng)
print(f"Echantillonnage N_ref = {N_ref} ...")
lam_ref = sample_lambda_max_goe(N_ref, n_ref, rng)

# Rescaling : xi = N^{2/3} * (lambda_max - 2)
xi_sml = N_sml**(2/3) * (lam_sml - 2)
xi_med = N_med**(2/3) * (lam_med - 2)
xi_ref = N_ref**(2/3) * (lam_ref - 2)

# Densite limite par KDE sur l'echantillon de reference
kde = gaussian_kde(xi_ref, bw_method=0.25)
x_curve = np.linspace(-6, 4, 400)
density_tw = kde(x_curve)

# --- Figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# PANNEAU GAUCHE : sans rescaling
ax = axes[0]
ax.hist(lam_sml, bins=40, density=True, alpha=0.5, color='steelblue',
        label=f'$N={N_sml}$', edgecolor='white')
ax.hist(lam_med, bins=40, density=True, alpha=0.5, color='darkorange',
        label=f'$N={N_med}$', edgecolor='white')
ax.hist(lam_ref, bins=40, density=True, alpha=0.5, color='crimson',
        label=f'$N={N_ref}$', edgecolor='white')
ax.axvline(2.0, color='black', linestyle='--', linewidth=1, label=r'$\lambda_{\max} \to 2$')
ax.set_xlabel(r'$\lambda_{\max}$ (valeur brute, sans rescaling)')
ax.set_ylabel('densite')
ax.set_title(
    r'$(a)$ Sans rescaling : la distribution se contracte sur $\lambda=2$' + '\n'
    + r'plus $N$ grandit, plus les fluctuations sont etroites',
    fontsize=10
)
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(1.4, 2.8)

# PANNEAU DROIT : avec rescaling
ax = axes[1]
ax.hist(xi_sml, bins=40, density=True, alpha=0.5, color='steelblue',
        label=f'$N={N_sml}$', edgecolor='white')
ax.hist(xi_med, bins=40, density=True, alpha=0.5, color='darkorange',
        label=f'$N={N_med}$', edgecolor='white')
ax.plot(x_curve, density_tw, color='black', linewidth=2.2,
        label=r'densite limite Tracy-Widom')
ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.set_xlabel(r'$\xi = N^{2/3}(\lambda_{\max} - 2)$')
ax.set_ylabel('densite')
ax.set_title(
    r'$(b)$ Apres rescaling : $\xi = N^{2/3}(\lambda_{\max} - 2)$' + '\n'
    + r'les histogrammes pour differents $N$ se superposent a $\rho_{\rm TW}$',
    fontsize=10
)
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-6, 4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
