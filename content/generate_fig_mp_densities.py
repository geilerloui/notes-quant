"""
Genere la figure 1 pour la note 02_Marchenko_Pastur :
densites Marchenko-Pastur pour differentes valeurs de q.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_mp_densities.py

Sortie : fig_mp_densities.png dans le repertoire courant.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_mp_densities.png"

def mp_density(lam, q):
    """Densite de Marchenko-Pastur (cas q <= 1, sans masse de Dirac)."""
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2
    out = np.zeros_like(lam)
    mask = (lam > lm) & (lam < lp)
    out[mask] = np.sqrt((lp - lam[mask]) * (lam[mask] - lm)) / (2 * np.pi * q * lam[mask])
    return out

lam = np.linspace(0.001, 6.5, 2000)
q_values = [0.1, 0.3, 0.5, 0.8]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(10, 5.5))

for q, c in zip(q_values, colors):
    rho = mp_density(lam, q)
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2
    ax.plot(lam, rho, color=c, linewidth=2.2,
            label=fr'$q={q}$ : support $[{lm:.2f},\,{lp:.2f}]$')

ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
           label=r'spectre vrai : $\{1, 1, \dots\}$')

ax.set_xlabel(r'$\lambda$ (valeur propre)', fontsize=11)
ax.set_ylabel(r'densite $\rho_{\rm MP}(\lambda;q)$', fontsize=11)
ax.set_title(
    "Loi de Marchenko-Pastur pour differentes valeurs de $q = N/T$\n"
    r"Vraie covariance = $\mathbb{I}_N$ (spectre concentre sur $\lambda=1$)",
    fontsize=11
)
ax.legend(loc='upper right', fontsize=10, frameon=True)
ax.set_xlim(0, 5)
ax.set_ylim(0, 2.5)
ax.grid(True, alpha=0.3)

ax.annotate(
    'plus $q$ augmente,\nplus le spectre s\'etale\nautour de la vraie valeur 1',
    xy=(1.0, 0.05), xytext=(2.8, 1.5),
    fontsize=10, style='italic',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='goldenrod', alpha=0.85),
    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2,
                    connectionstyle='arc3,rad=0.2')
)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
