"""
Génère vi_elbo_decomposition.png : la décomposition log p(x) = ELBO + KL
visualisée comme un "thermomètre".

Deux barres verticales côte à côte :
- Gauche : à l'init, ELBO petit + KL grand = log p(x).
- Droite : après optimisation, ELBO grand + KL petit = log p(x).

La hauteur totale est FIXE (= log p(x), ne dépend pas de phi).
Maximiser l'ELBO = descendre la KL d'autant.

Output : vi_elbo_decomposition.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ---------------------------------------------------------------------
# Valeurs (illustratives)
# ---------------------------------------------------------------------
log_px = 10.0  # log p(x), constante

# Avant optimisation : ELBO bas, KL grand
elbo_before = 3.0
kl_before = log_px - elbo_before

# Après optimisation : ELBO haut, KL petit
elbo_after = 8.5
kl_after = log_px - elbo_after

# ---------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------
color_elbo = '#185FA5'   # bleu pour ELBO
color_kl = '#D85A30'     # orange pour KL
color_total = '#7B3F9E'  # violet pour log p(x)

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.6
positions = [1.0, 3.0]
labels = [r"Avant optimisation : $\phi^{(0)}$",
          r"Après optimisation : $\phi^{*}$"]
elbo_values = [elbo_before, elbo_after]
kl_values = [kl_before, kl_after]

# Tracer les deux thermomètres
for i, (pos, lab, elbo_val, kl_val) in enumerate(
        zip(positions, labels, elbo_values, kl_values)):
    # ELBO (en bas)
    bar_elbo = ax.bar(pos, elbo_val, bar_width,
                      color=color_elbo, edgecolor='black', linewidth=1.2,
                      alpha=0.85, label=r'ELBO $\mathcal{L}(\phi, \theta)$' if i == 0 else None)
    # KL (au-dessus de l'ELBO)
    bar_kl = ax.bar(pos, kl_val, bar_width, bottom=elbo_val,
                    color=color_kl, edgecolor='black', linewidth=1.2,
                    alpha=0.85, label=r'KL$(q_\phi \| p_\theta(z|x))$' if i == 0 else None,
                    hatch='//')

    # Annotations valeurs ELBO et KL
    ax.text(pos, elbo_val / 2,
            fr'$\mathcal{{L}} = {elbo_val:.1f}$',
            ha='center', va='center', fontsize=12, color='white',
            fontweight='bold')
    ax.text(pos, elbo_val + kl_val / 2,
            fr'KL = {kl_val:.1f}',
            ha='center', va='center', fontsize=12, color='white',
            fontweight='bold')

    # Label sous la barre
    ax.text(pos, -0.5, lab, ha='center', va='top',
            fontsize=11, fontweight='bold')

# ---------------------------------------------------------------------
# Ligne horizontale en pointillé pour la hauteur totale log p(x)
# ---------------------------------------------------------------------
ax.axhline(y=log_px, color=color_total, linewidth=2.0,
           linestyle='--', alpha=0.7, zorder=1)
ax.text(4.1, log_px,
        fr'$\log p_\theta(x) = {log_px}$' + '\n(constante en $\\phi$)',
        ha='left', va='center', fontsize=12, color=color_total,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=color_total, linewidth=1.2))

# ---------------------------------------------------------------------
# Flèche entre les deux barres pour signifier "optimisation"
# ---------------------------------------------------------------------
ax.annotate('', xy=(positions[1] - bar_width/2 - 0.05, log_px / 2),
            xytext=(positions[0] + bar_width/2 + 0.05, log_px / 2),
            arrowprops=dict(arrowstyle='->', color='black',
                            lw=2.5, mutation_scale=25))
ax.text((positions[0] + positions[1]) / 2, log_px / 2 + 0.3,
        r'maximiser $\mathcal{L}$ sur $\phi$',
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        style='italic')
ax.text((positions[0] + positions[1]) / 2, log_px / 2 - 0.3,
        r'$\Rightarrow$ KL descend d\'autant',
        ha='center', va='top', fontsize=10, style='italic', color='#555')

# ---------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------
ax.set_ylabel(r'décomposition de $\log p_\theta(x)$', fontsize=12)
ax.set_xlim(0, 5.5)
ax.set_ylim(-1.5, log_px + 1.8)
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.grid(True, alpha=0.2, linestyle=':', axis='y')

ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.set_title(r"$\log p_\theta(x) = \mathcal{L}(\phi, \theta) + \mathrm{KL}(q_\phi \| p_\theta(z|x))$ — la somme est fixe",
             fontsize=13, pad=15)

# Caption sous l'image
fig.text(0.5, 0.01,
         r"La hauteur totale (violet pointillé) est $\log p_\theta(x)$, **fixe** pour $\theta, x$ donnés. "
         r"À gauche : avant optimisation, l'ELBO (bleu) est petit, la KL (orange) prend la majeure partie." '\n'
         r"À droite : après maximisation de l'ELBO sur $\phi$, l'ELBO a monté, la KL a fondu d'exactement la même quantité. "
         r"**On ne calcule jamais la KL directement** — on optimise l'ELBO et la KL descend mécaniquement.",
         ha='center', fontsize=9.5, color='#333')

plt.tight_layout(rect=[0, 0.05, 1, 1])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'vi_elbo_decomposition.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
