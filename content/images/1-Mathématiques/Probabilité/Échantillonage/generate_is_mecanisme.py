"""
Génère une image qui illustre le mécanisme d'Importance Sampling.

Idée : on a une cible p et une proposale q. On tire des échantillons x_i ~ q,
et chacun reçoit un poids w(x_i) = p(x_i)/q(x_i). La taille des points (et
la hauteur des barres) encode visuellement le poids — pas d'accept/reject,
juste de la pondération.

Output : is_mecanisme.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------
# Setup : cible p et proposale q
# ---------------------------------------------------------------------
# Cible p : mélange de deux gaussiennes (bimodale) — typique
#   d'un cas où on ne sait pas inverser la CDF.
# Proposale q : une gaussienne large qui couvre le support de p,
#   mais pas alignée — donc des poids variables, justement ce qu'on
#   veut illustrer.

def p_target(x):
    """Cible : mélange de deux gaussiennes."""
    return 0.4 * stats.norm.pdf(x, loc=-2, scale=1.0) + \
           0.6 * stats.norm.pdf(x, loc=3, scale=1.2)

def q_proposal(x):
    """Proposale : une gaussienne large centrée entre les deux modes."""
    return stats.norm.pdf(x, loc=0.5, scale=2.5)

# ---------------------------------------------------------------------
# Tirages depuis q + calcul des poids
# ---------------------------------------------------------------------
np.random.seed(42)
n_samples = 30  # peu de samples pour que ça reste lisible
samples = np.random.normal(loc=0.5, scale=2.5, size=n_samples)

weights = p_target(samples) / q_proposal(samples)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

# Axe x : grille fine pour les courbes
x_grid = np.linspace(-7, 8, 500)

# Tracer p et q
ax.plot(x_grid, p_target(x_grid), color='#0F6E56', linewidth=2.0,
        label=r'cible  $p(x)$', zorder=3)
ax.fill_between(x_grid, p_target(x_grid), alpha=0.15, color='#0F6E56', zorder=2)

ax.plot(x_grid, q_proposal(x_grid), color='#185FA5', linewidth=2.0,
        linestyle='--', label=r'proposale  $q(x)$', zorder=3)
ax.fill_between(x_grid, q_proposal(x_grid), alpha=0.10, color='#185FA5', zorder=1)

# Tracer les samples avec une barre dont la hauteur encode le poids w(x_i).
# On normalise les poids pour que la barre la plus haute aille jusqu'à un
# niveau visuellement comparable aux courbes (purement esthétique, pour
# que l'œil voie la variation relative).
max_weight = weights.max()
bar_scale = 0.20 / max_weight  # 0.20 = hauteur max ~ courbe q

# Ramp de couleur : on utilise plasma inversé pour avoir
#   - petit poids = bleu/violet foncé (lisible)
#   - gros poids = orange/jaune (lisible)
# Tout reste contrasté sur fond blanc, plus de gris pâle illisible.
cmap = plt.cm.plasma

for x_i, w_i in zip(samples, weights):
    height = w_i * bar_scale
    intensity = w_i / max_weight  # entre 0 et 1
    # On reste dans la portion lisible du cmap : 0.15 -> 0.85
    color = cmap(0.15 + 0.70 * intensity)
    ax.vlines(x_i, 0, height, color=color, linewidth=2.5, zorder=4,
              alpha=0.9)
    # petit point au sommet pour mettre en valeur
    ax.scatter([x_i], [height], s=40 + 200 * intensity, color=color,
               edgecolor='black', linewidth=0.5, zorder=5)

# Marqueurs des x_i sur l'axe (en bas)
ax.scatter(samples, np.zeros_like(samples) - 0.005, marker='|',
           color='#333', s=80, zorder=6, alpha=0.7)

# ---------------------------------------------------------------------
# Annotations pédagogiques
# ---------------------------------------------------------------------
# Trouver un sample avec gros poids (proche d'un mode de p, loin du centre de q)
# et un sample avec petit poids (où q est grand mais p faible).
i_big = np.argmax(weights)
i_small_pool = np.where(np.abs(samples) < 4)[0]
i_small_real = i_small_pool[np.argmin(weights[i_small_pool])]

# Couleur du gros poids = couleur "haute" du cmap
big_color = cmap(0.85)
# Couleur du petit poids = couleur "basse" du cmap (violet foncé, lisible)
small_color = cmap(0.15)

# Annotation gros poids
ax.annotate(
    r'gros poids $w(x_i) = p/q$ élevé' '\n' r'$p$ grand ici, $q$ faible',
    xy=(samples[i_big], weights[i_big] * bar_scale),
    xytext=(samples[i_big] + 1.5, weights[i_big] * bar_scale + 0.05),
    fontsize=10, color=big_color, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=big_color, lw=1.2),
    ha='left'
)

# Annotation petit poids — couleur foncée bien lisible
ax.annotate(
    r'petit poids $w(x_i)$ faible' '\n' r'$q$ grand ici mais $p$ faible',
    xy=(samples[i_small_real], weights[i_small_real] * bar_scale),
    xytext=(samples[i_small_real] - 3.0, 0.15),
    fontsize=10, color=small_color, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=small_color, lw=1.2),
    ha='left'
)

# ---------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------
ax.set_xlim(-7, 8)
ax.set_ylim(-0.02, 0.32)
ax.set_xlabel(r'$x$', fontsize=12)
ax.set_ylabel('densité', fontsize=12)
ax.set_title(
    r'Importance sampling : tirages $x_i \sim q$, pondérés par '
    r'$w(x_i) = p(x_i) / q(x_i)$',
    fontsize=12, pad=15
)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Légende manuelle pour expliquer la hauteur des barres
ax.text(
    0.98, 0.97,
    'Hauteur de barre = poids $w(x_i)$\n'
    'Pas de rejet : tous les tirages sont gardés,\nmais ils comptent différemment',
    transform=ax.transAxes, ha='right', va='top',
    fontsize=10, color='#222',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa',
              edgecolor='#bbb', linewidth=0.8)
)

# Grille et style général
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Sauvegarde
import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'is_mecanisme.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")

plt.show()
