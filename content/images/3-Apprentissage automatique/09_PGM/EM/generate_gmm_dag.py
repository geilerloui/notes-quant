"""
Génère l'image gmm_dag.png : le DAG du modèle GMM.

Deux panneaux :
- À gauche : version dépliée avec 2 observations explicites (pas 3, pour rester lisible).
  Paramètres positionnés au-dessus de leurs cibles : pi au-dessus des z, mu/Sigma au-dessus
  des x. Les flèches partent en oblique sans se croiser.
- À droite : plate notation compacte avec mu/Sigma décalés à droite pour que leurs flèches
  vers x_n ne traversent pas z_n.

Conventions : cercles blancs = latent, cercles grisés = observé,
carrés = paramètres (constantes du point de vue du graphe).

Output : gmm_dag.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import os

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def add_node(ax, x, y, label, observed=False, param=False, radius=0.32, fontsize=12):
    """Ajoute un nœud (cercle pour variable, carré pour paramètre)."""
    if param:
        rect = Rectangle((x - radius, y - radius), 2 * radius, 2 * radius,
                         facecolor='white', edgecolor='black', linewidth=1.5,
                         zorder=4)
        ax.add_patch(rect)
    else:
        face = '#cccccc' if observed else 'white'
        circ = Circle((x, y), radius, facecolor=face, edgecolor='black',
                      linewidth=1.5, zorder=4)
        ax.add_patch(circ)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            style='italic', zorder=5)

def add_arrow(ax, x1, y1, x2, y2, r_start=0.32, r_end=0.32):
    """Flèche entre deux nœuds, en raccourcissant aux bords (rayons éventuellement différents)."""
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L == 0:
        return
    ux, uy = dx / L, dy / L
    arrow = FancyArrowPatch(
        (x1 + ux * r_start, y1 + uy * r_start),
        (x2 - ux * r_end, y2 - uy * r_end),
        arrowstyle='-|>', mutation_scale=14, linewidth=1.2, color='black',
        zorder=2
    )
    ax.add_patch(arrow)

# ---------------------------------------------------------------------
# Panneau gauche : version dépliée avec 2 observations seulement (plus lisible)
# ---------------------------------------------------------------------
ax = ax1
ax.set_xlim(-0.5, 6)
ax.set_ylim(-0.5, 5.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Version dépliée\n(2 observations explicites)', fontsize=12, pad=10)

# Positions des observations
x1_pos, x2_pos = 1.5, 4.0

# Paramètres en haut, positionnés près de leurs cibles :
#   pi -> z_n (à gauche-centre)
#   mu, Sigma -> x_n (à droite)
pi_x, pi_y = 0.5, 4.5
mu_x, mu_y = 2.75, 4.5
Sigma_x, Sigma_y = 5.0, 4.5

add_node(ax, pi_x, pi_y, r'$\pi$', param=True, radius=0.28, fontsize=11)
add_node(ax, mu_x, mu_y, r'$\mu$', param=True, radius=0.28, fontsize=11)
add_node(ax, Sigma_x, Sigma_y, r'$\Sigma$', param=True, radius=0.28, fontsize=11)

# z_1, z_2 en milieu
z1_y = 2.6
add_node(ax, x1_pos, z1_y, '$z_1$', observed=False, radius=0.32, fontsize=12)
add_node(ax, x2_pos, z1_y, '$z_2$', observed=False, radius=0.32, fontsize=12)

# x_1, x_2 en bas
x_y = 0.6
add_node(ax, x1_pos, x_y, '$x_1$', observed=True, radius=0.32, fontsize=12)
add_node(ax, x2_pos, x_y, '$x_2$', observed=True, radius=0.32, fontsize=12)

# Flèches pi -> z_1, pi -> z_2
add_arrow(ax, pi_x, pi_y, x1_pos, z1_y, r_start=0.28, r_end=0.32)
add_arrow(ax, pi_x, pi_y, x2_pos, z1_y, r_start=0.28, r_end=0.32)

# Flèches z_1 -> x_1, z_2 -> x_2
add_arrow(ax, x1_pos, z1_y, x1_pos, x_y, r_start=0.32, r_end=0.32)
add_arrow(ax, x2_pos, z1_y, x2_pos, x_y, r_start=0.32, r_end=0.32)

# Flèches mu -> x_1, mu -> x_2
add_arrow(ax, mu_x, mu_y, x1_pos, x_y, r_start=0.28, r_end=0.32)
add_arrow(ax, mu_x, mu_y, x2_pos, x_y, r_start=0.28, r_end=0.32)

# Flèches Sigma -> x_1, Sigma -> x_2
add_arrow(ax, Sigma_x, Sigma_y, x1_pos, x_y, r_start=0.28, r_end=0.32)
add_arrow(ax, Sigma_x, Sigma_y, x2_pos, x_y, r_start=0.28, r_end=0.32)

# ---------------------------------------------------------------------
# Panneau droit : plate notation, mu/Sigma décalés à droite
# ---------------------------------------------------------------------
ax = ax2
ax.set_xlim(-0.5, 6)
ax.set_ylim(-0.5, 5.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Plate notation\n(compacte, $n = 1, \\ldots, N$)', fontsize=12, pad=10)

# Positions des nœuds dans la plate
zn_x, zn_y = 2.0, 2.8
xn_x, xn_y = 2.0, 0.8

# Paramètres
#   pi : au-dessus de z_n (axe vertical)
#   mu : à droite, au-dessus de x_n
#   Sigma : encore plus à droite
pi_x, pi_y = 2.0, 4.6
mu_x, mu_y = 3.8, 4.6
Sigma_x, Sigma_y = 5.0, 4.6

add_node(ax, pi_x, pi_y, r'$\pi$', param=True, radius=0.28, fontsize=11)
add_node(ax, mu_x, mu_y, r'$\mu$', param=True, radius=0.28, fontsize=11)
add_node(ax, Sigma_x, Sigma_y, r'$\Sigma$', param=True, radius=0.28, fontsize=11)

# Plate (rectangle pointillé autour de z_n et x_n)
plate = Rectangle((1.0, 0.0), 2.0, 3.6, fill=False, edgecolor='black',
                  linewidth=1.0, linestyle='--', zorder=1)
ax.add_patch(plate)
ax.text(2.9, 0.1, r'$n = 1, \ldots, N$', ha='right', va='bottom', fontsize=10)

# Nœuds dans la plate
add_node(ax, zn_x, zn_y, '$z_n$', observed=False, radius=0.32, fontsize=12)
add_node(ax, xn_x, xn_y, '$x_n$', observed=True, radius=0.32, fontsize=12)

# Flèches :
# pi -> z_n (vertical, ne traverse rien)
add_arrow(ax, pi_x, pi_y, zn_x, zn_y, r_start=0.28, r_end=0.32)
# z_n -> x_n (vertical dans la plate)
add_arrow(ax, zn_x, zn_y, xn_x, xn_y, r_start=0.32, r_end=0.32)
# mu -> x_n (oblique, contourne z_n par la droite)
add_arrow(ax, mu_x, mu_y, xn_x, xn_y, r_start=0.28, r_end=0.32)
# Sigma -> x_n (oblique, contourne z_n par la droite, encore plus à droite)
add_arrow(ax, Sigma_x, Sigma_y, xn_x, xn_y, r_start=0.28, r_end=0.32)

# ---------------------------------------------------------------------
# Légende globale
# ---------------------------------------------------------------------
fig.text(0.5, 0.02,
         'Cercle blanc : variable latente. Cercle grisé : variable observée. Carré : paramètre.\n'
         '$z_n$ encode le cluster d\'origine de $x_n$ (1-of-K). '
         'Les paramètres $\\pi, \\mu, \\Sigma$ gouvernent toutes les observations.',
         ha='center', fontsize=10, color='#333')

plt.tight_layout(rect=[0, 0.06, 1, 1])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gmm_dag.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
