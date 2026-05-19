"""
Genere la figure pedagogique "sine kernel" pour la note RMT 01_Fondamentaux.

On tire des matrices GOE, on collecte les espacements entre valeurs propres
consecutives au centre du spectre, on les normalise, et on compare leur
distribution a deux courbes theoriques :
  - le sine kernel (loi RMT, avec repulsion : p(0) = 0)
  - la densite exponentielle / Poisson (points independants : p(0) = 1)

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_sine_kernel.py

Sortie : fig_sine_kernel.png dans le repertoire courant.
Le calcul prend ~30 secondes (200 tirages de GOE 500x500).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Parametres -------------------------------------------------------------
N = 500
N_MATRICES = 200
CENTER_WINDOW = 0.4
SEED = 7
OUTPUT_FILE = "fig_sine_kernel.png"

rng = np.random.default_rng(seed=SEED)

# --- Tirage et collecte des espacements -------------------------------------
all_spacings = []
for k in range(N_MATRICES):
    A = rng.standard_normal(size=(N, N))
    M = (A + A.T) / np.sqrt(2)
    W = M / np.sqrt(N)
    eigvals = np.linalg.eigvalsh(W)
    
    # On garde les valeurs propres au centre du spectre
    mask = np.abs(eigvals) < CENTER_WINDOW
    central = eigvals[mask]
    if len(central) < 2:
        continue
    
    spacings = np.diff(central)
    # Normalisation : on divise par l'espacement moyen local
    # (au centre du demi-cercle, densite rho(0) = 1/pi, donc espacement moyen = pi/N)
    mean_spacing = np.pi / N
    s = spacings / mean_spacing
    all_spacings.extend(s)

all_spacings = np.array(all_spacings)
print(f"Nombre d'espacements collectes : {len(all_spacings)}")
print(f"Moyenne des espacements normalises (devrait etre ~1) : {all_spacings.mean():.3f}")

# --- Densites theoriques ---------------------------------------------------
s_grid = np.linspace(0, 4, 500)

# Conjecture de Wigner pour beta=1 (GOE) : approximation tres precise du
# vrai sine kernel pour beta=1
wigner_surmise = (np.pi / 2) * s_grid * np.exp(-np.pi * s_grid**2 / 4)

# Poisson (= points independants, pas de repulsion)
poisson = np.exp(-s_grid)

# --- Figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(all_spacings, bins=50, density=True, alpha=0.55, color='steelblue',
        edgecolor='white', linewidth=0.6,
        label=f'espacements empiriques\n({N_MATRICES} GOE de taille $N={N}$)')

ax.plot(s_grid, wigner_surmise, color='crimson', linewidth=2.4,
        label=r'sine kernel (repulsion RMT)' + '\n' + r'$p(s) \approx \frac{\pi}{2} s\, e^{-\pi s^2/4}$')

ax.plot(s_grid, poisson, color='darkgreen', linewidth=2.4, linestyle='--',
        label=r'Poisson (points independants)' + '\n' + r'$p(s) = e^{-s}$ - pas de repulsion')

# Annotation : la repulsion au voisinage de 0
ax.annotate('',
            xy=(0.05, 0.1), xytext=(0.05, 0.95),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax.text(0.15, 0.55,
        'repulsion :\n$p(0) = 0$\npour RMT',
        fontsize=10, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='goldenrod', alpha=0.85))

ax.set_xlabel(r'espacement normalise $s = (\lambda_{i+1} - \lambda_i) \,/\, \bar{s}$')
ax.set_ylabel(r'densite $p(s)$')
ax.set_title(
    "Espacements entre valeurs propres consecutives - au centre du spectre\n"
    "Les valeurs propres se repoussent (pas comme des points independants)",
    fontsize=11
)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 1.15)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
