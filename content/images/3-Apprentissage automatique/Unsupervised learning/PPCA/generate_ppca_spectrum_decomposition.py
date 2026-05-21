"""
Génère ppca_spectrum_decomposition.png : décomposition spectrale signal/bruit
de la solution MLE de PPCA.

L'idée :
- On a une matrice de covariance empirique S = U Lambda U^T avec valeurs propres lambda_1 >= ... >= lambda_D.
- On choisit M = nombre de composantes à garder.
- En VERT : les M premières valeurs propres -> captures par W W^T (signal).
- En ORANGE : les D - M dernières -> moyennées en sigma^2 (bruit).
- On trace une ligne horizontale à sigma^2 = moyenne(lambda_{M+1}, ..., lambda_D).

Output : ppca_spectrum_decomposition.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------
# Spectre illustratif
# ---------------------------------------------------------------------
np.random.seed(42)
D = 12
M = 4

# Spectre réaliste : quelques grandes valeurs propres + queue
true_eigs = np.array([8.5, 6.2, 4.8, 3.5,
                       0.9, 0.75, 0.65, 0.58, 0.52, 0.48, 0.42, 0.38])

assert len(true_eigs) == D
lambdas = true_eigs

# sigma^2 ML = moyenne des D-M plus petites
sigma2_ml = np.mean(lambdas[M:])

print(f"Valeurs propres signal (top {M}) : {lambdas[:M]}")
print(f"Valeurs propres bruit (bottom {D-M}) : {lambdas[M:]}")
print(f"sigma^2 ML = moyenne des bruit = {sigma2_ml:.4f}")

# ---------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------
color_signal = '#2E7D5F'
color_noise = '#D85A30'
color_sigma_line = '#A32D2D'

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.5))

x_positions = np.arange(1, D + 1)
bar_width = 0.7

# Bars
for i, val in enumerate(lambdas):
    if i < M:
        color = color_signal
        label = r'signal (top $M = %d$)' % M if i == 0 else None
    else:
        color = color_noise
        label = r'bruit (bottom $D - M = %d$)' % (D - M) if i == M else None

    ax.bar(x_positions[i], val, bar_width, color=color, alpha=0.85,
           edgecolor='black', linewidth=0.8, label=label, zorder=3)

# Valeurs annotées au-dessus des barres
for i, val in enumerate(lambdas):
    ax.text(x_positions[i], val + 0.15, f'{val:.2f}',
            ha='center', va='bottom', fontsize=8.5, color='#444')

# Ligne sigma^2 pointillée
ax.axhline(sigma2_ml, color=color_sigma_line, linewidth=2.0,
           linestyle='--', alpha=0.8, zorder=4,
           label=r'$\sigma^2_{\mathrm{ML}} = \frac{1}{D-M} \sum_{j > M} \lambda_j = %.3f$' % sigma2_ml)

# Zone sigma^2 : remplir une bande horizontale pour visualiser l'isotropie
ax.axhspan(0, sigma2_ml, alpha=0.08, color=color_sigma_line, zorder=1)

# Séparation visuelle entre signal et bruit
ax.axvline(M + 0.5, color='gray', linewidth=1.2, linestyle=':', alpha=0.7, zorder=2)

# Annotations textuelles : "capture par W W^T" et "modelise comme bruit isotropique"
ax.annotate('', xy=(M + 0.4, lambdas[0] * 0.9), xytext=(1 - 0.4, lambdas[0] * 0.9),
            arrowprops=dict(arrowstyle='<->', color=color_signal, lw=1.5))
ax.text((M + 1) / 2, lambdas[0] * 0.95,
        r'$W W^\top$ capture ces directions',
        ha='center', fontsize=11, color=color_signal, fontweight='bold')

ax.annotate('', xy=(D + 0.4, sigma2_ml * 2.2), xytext=(M + 1 - 0.4, sigma2_ml * 2.2),
            arrowprops=dict(arrowstyle='<->', color=color_noise, lw=1.5))
ax.text((M + 1 + D) / 2, sigma2_ml * 2.5,
        r'modelise comme bruit $\sigma^2 I$',
        ha='center', fontsize=11, color=color_noise, fontweight='bold')

# Encadre avec la formule de WW^T (PAS de \boxed - matplotlib mathtext ne le connait pas)
formula_text = (
    r'$W W^\top = U_M (\Lambda_M - \sigma^2 I) U_M^\top$' + '\n\n'
    + r"En d'autres termes :" + '\n'
    + r'  $\bullet$ on prend les $M$ premieres directions propres' + '\n'
    + r'  $\bullet$ on les pondere par $(\lambda_j - \sigma^2)^{1/2}$' + '\n'
    + r'    (en enlevant ce qui est attribue au bruit)' + '\n'
    + r'  $\bullet$ le reste tombe dans $\sigma^2 I$'
)
ax.text(0.97, 0.97, formula_text,
        transform=ax.transAxes, fontsize=10, color='#333',
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef4fa',
                  edgecolor='#2E7D5F', linewidth=1.5))

# Mise en forme
ax.set_xlabel(r'index $j$  (valeurs propres de $S$ triees par ordre decroissant)',
              fontsize=11)
ax.set_ylabel(r'$\lambda_j$  (valeurs propres)', fontsize=11)
ax.set_title(r"Decomposition spectrale : la solution MLE de PPCA separe signal et bruit",
             fontsize=13, pad=12)
ax.set_xticks(x_positions)
ax.set_xticklabels([r'$\lambda_{%d}$' % (i+1) for i in range(D)], fontsize=10)
ax.set_ylim(0, max(lambdas) * 1.15)
ax.set_xlim(0.3, D + 0.7)
ax.grid(True, alpha=0.25, linestyle=':', axis='y', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper center', fontsize=10, framealpha=0.95, ncol=3,
          bbox_to_anchor=(0.5, -0.13))

# Caption
fig.text(0.5, -0.10,
         r"Le spectre de $S$ (covariance empirique) est trie en ordre decroissant. La solution MLE de PPCA dit : "
         + r"les $M$ plus grandes valeurs propres (vert) sont capturees par $W W^\top$, " + "\n"
         + r"et les $D - M$ restantes (orange) sont modelisees par $\sigma^2 I$, avec $\sigma^2$ = moyenne de ces queues. "
         + r"La separation a $M$ est un choix de l'utilisateur - plus $M$ est petit, plus la barre verte retrecit et plus $\sigma^2$ grossit.",
         ha='center', fontsize=9.5, color='#333')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ppca_spectrum_decomposition.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nImage sauvegardee : {output_path}")
plt.show()
