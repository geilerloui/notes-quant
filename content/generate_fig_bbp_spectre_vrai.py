"""
Figure A pour la note 03 : spectre vrai d'une covariance avec un spike rang-1.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_bbp_spectre_vrai.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_bbp_spectre_vrai.png"
N = 20
theta = 3.0

fig, ax = plt.subplots(figsize=(10, 4.2))

ax.plot([1.0, 1.0], [0, N-1], color='steelblue', linewidth=4, alpha=0.8)
ax.scatter([1.0], [N-1], color='steelblue', s=80, zorder=3)
ax.text(1.0, N - 1 + 1.0, f'multiplicite $N-1 = {N-1}$',
        ha='center', fontsize=10, color='steelblue', fontweight='bold')

ax.plot([1 + theta, 1 + theta], [0, 1], color='crimson', linewidth=4)
ax.scatter([1 + theta], [1], color='crimson', s=80, zorder=3)
ax.text(1 + theta, 1 + 1.0, f'multiplicite 1\n(le spike)',
        ha='center', fontsize=10, color='crimson', fontweight='bold')

ax.annotate(
    'directions baseline\n(toutes equivalentes)',
    xy=(1.0, (N-1) / 2), xytext=(0.3, 12),
    fontsize=10, color='steelblue', style='italic',
    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1, alpha=0.6)
)

ax.annotate(
    "direction $u$ perturbee\n(le 'vrai facteur')",
    xy=(1 + theta, 0.5), xytext=(2.3, 7),
    fontsize=10, color='crimson', style='italic',
    arrowprops=dict(arrowstyle='->', color='crimson', lw=1, alpha=0.6)
)

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlim(0.2, 1 + theta + 0.8)
ax.set_ylim(-0.5, N + 1.5)
ax.set_xlabel(r'$\lambda$ (valeur propre)', fontsize=11)
ax.set_ylabel('multiplicite', fontsize=11)
ax.set_title(
    r'Spectre VRAI de $\Sigma_{\rm vrai} = \mathbb{I}_N + \theta\, u u^\top$' +
    f'   (ici $N={N}, \\theta={theta}$)\n' +
    r'$N-1$ valeurs propres a $\lambda=1$  (degenerescence)  $+$  une valeur propre a $\lambda = 1+\theta$',
    fontsize=11
)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
