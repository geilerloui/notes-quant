"""
Génère l'image em_elbo_view.png : la vue géométrique d'EM comme coordinate ascent.

Une seule figure avec :
- La vraie log-vraisemblance log p_theta(x) (courbe bleue continue) en fonction de theta.
- L'ELBO L(q^old, theta) (courbe verte) tangente à log p en theta_old (après E-step).
- L'ELBO L(q^new, theta) (courbe orange) tangente à log p en theta_new (après nouveau E-step).
- Les points et flèches qui matérialisent les itérations.

Output : em_elbo_view.png dans le même dossier que ce script.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------
def log_p(theta):
    """Vraie log-vraisemblance (fonction objectif concave, max en theta=4)."""
    return -0.05 * (theta - 4) ** 4 - 0.3 * (theta - 4) ** 2 + 3

def log_p_prime(theta):
    return -0.2 * (theta - 4) ** 3 - 0.6 * (theta - 4)

def elbo(theta, theta0, curvature=0.3):
    """
    ELBO tangente à log_p en theta0 (parabole de courbure douce).
    Au point theta0 : ELBO(theta0) = log_p(theta0), ELBO'(theta0) = log_p'(theta0).
    Partout ailleurs : ELBO <= log_p (concavité de la borne).
    """
    a = log_p(theta0)
    b = log_p_prime(theta0)
    return a + b * (theta - theta0) - 0.5 * curvature * (theta - theta0) ** 2

# ---------------------------------------------------------------------
# Points d'EM
# ---------------------------------------------------------------------
theta_old = 1.8

# theta_new = argmax_theta elbo(theta, theta_old)
# Sommet de parabole : theta_new = theta_old + b/c
curvature = 0.3
theta_new = theta_old + log_p_prime(theta_old) / curvature

# Vérification : on veut que theta_new soit avant le sommet (4) pour que l'image reste lisible
# Si log_p_prime(1.8) > 0 (on est à gauche du sommet), alors theta_new > theta_old.
# Avec curvature=0.3 et theta_old=1.8 : log_p_prime(1.8) = -0.2*(-2.2)^3 - 0.6*(-2.2) = 2.13+1.32 = 3.45
# Donc theta_new = 1.8 + 3.45/0.3 = 13.3 ... bien trop loin !
#
# Il faut une courbure plus forte pour que theta_new reste proche. Réajustons.
#
# On veut theta_new ≈ 3.5 (juste avant le sommet à 4), donc on veut b/c ≈ 1.7.
# Avec b = 3.45 (en theta_old=1.8), il faut c ≈ 2.0. Mais c=2 redonne une parabole trop pentue.
#
# Solution propre : prendre un theta_old plus proche du sommet, par exemple 2.8.
# log_p_prime(2.8) = -0.2*(-1.2)^3 - 0.6*(-1.2) = 0.346 + 0.72 = 1.066
# Avec c = 1.0, theta_new = 2.8 + 1.066/1.0 = 3.87 (proche du sommet à 4) -> parfait.

theta_old = 2.6
curvature = 1.0
theta_new = theta_old + log_p_prime(theta_old) / curvature
# log_p_prime(2.6) = -0.2*(-1.4)^3 - 0.6*(-1.4) = 0.549 + 0.84 = 1.389
# theta_new = 2.6 + 1.389 = 3.99 (presque le max, ok)

# Pour l'ELBO suivante (orange), centrée en theta_new : sa pente sera plus faible
# (proche du sommet), donc elle ne montera pas trop. Parfait.

theta_grid = np.linspace(0.5, 6.5, 400)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))

# Vraie log-vraisemblance
ax.plot(theta_grid, log_p(theta_grid), color='#185FA5', linewidth=2.5,
        label=r'$\log p_\theta(x)$ — vraie log-vraisemblance', zorder=3)

# ELBO 1 (après E-step à theta_old) — verte tangente en theta_old
elbo1 = elbo(theta_grid, theta_old, curvature=curvature)
ax.plot(theta_grid, elbo1, color='#0F6E56', linewidth=2,
        linestyle='--',
        label=r"$\mathcal{L}(q^{\mathrm{old}}, \theta)$ — ELBO après E-step à $\theta^{\mathrm{old}}$",
        zorder=2)

# ELBO 2 (après E-step à theta_new) — orange tangente en theta_new
elbo2 = elbo(theta_grid, theta_new, curvature=curvature)
ax.plot(theta_grid, elbo2, color='#D85A30', linewidth=2,
        linestyle='--',
        label=r"$\mathcal{L}(q^{\mathrm{new}}, \theta)$ — ELBO après E-step à $\theta^{\mathrm{new}}$",
        zorder=2)

# --- Points clés ---

# theta_old : log p et ELBO 1 se touchent (gap = 0)
y_old = log_p(theta_old)
ax.scatter([theta_old], [y_old], s=100, color='#185FA5',
           edgecolor='black', linewidth=1.5, zorder=5)
ax.annotate(r'$\theta^{\mathrm{old}}$', xy=(theta_old, y_old),
            xytext=(theta_old - 0.05, y_old - 0.5),
            fontsize=13, ha='right', color='#185FA5', fontweight='bold')

# theta_new : argmax de ELBO 1 (M-step)
y_new_on_elbo = elbo(theta_new, theta_old, curvature=curvature)
ax.scatter([theta_new], [y_new_on_elbo], s=100, color='#0F6E56',
           edgecolor='black', linewidth=1.5, zorder=5)
ax.annotate('argmax ELBO\n(M-step)',
            xy=(theta_new, y_new_on_elbo),
            xytext=(theta_new + 0.6, y_new_on_elbo - 0.3),
            fontsize=10, ha='left', color='#0F6E56',
            arrowprops=dict(arrowstyle='->', color='#0F6E56', lw=1))

# theta_new sur log p (après nouveau E-step, gap = 0 à nouveau)
y_new = log_p(theta_new)
ax.scatter([theta_new], [y_new], s=100, color='#185FA5',
           edgecolor='black', linewidth=1.5, zorder=5)
ax.annotate(r'$\theta^{\mathrm{new}}$', xy=(theta_new, y_new),
            xytext=(theta_new + 0.05, y_new + 0.4),
            fontsize=13, ha='left', color='#185FA5', fontweight='bold')

# --- Flèche "log p augmente" entre theta_old et theta_new sur la courbe bleue ---
ax.annotate('',
            xy=(theta_new - 0.05, y_new),
            xytext=(theta_old + 0.05, y_old),
            arrowprops=dict(arrowstyle='-|>', color='#A32D2D',
                            lw=2.0, connectionstyle='arc3,rad=-0.3'))
# Texte au-dessus de l'arc
mid_x = (theta_old + theta_new) / 2
mid_y = max(y_old, y_new) + 0.7
ax.text(mid_x, mid_y, r"$\log p$ augmente", fontsize=11, color='#A32D2D',
        ha='center', fontweight='bold')

# --- Indicateurs de gap ---

# Gap = 0 en theta_old (E-step exact rend la borne serrée)
ax.annotate("", xy=(theta_old - 0.15, y_old),
            xytext=(theta_old - 0.15, elbo(theta_old, theta_old, curvature=curvature)),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax.text(theta_old - 0.25, y_old, 'gap = 0\n(E-step exact)',
        fontsize=9, ha='right', va='center', color='#666')

# Gap > 0 en theta_new pour ELBO 1 (avant nouveau E-step)
gap_top = y_new
gap_bot = elbo(theta_new, theta_old, curvature=curvature)
ax.annotate("", xy=(theta_new + 0.15, gap_top),
            xytext=(theta_new + 0.15, gap_bot),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1))
ax.text(theta_new + 0.25, (gap_top + gap_bot) / 2,
        'gap > 0\n(KL > 0)', fontsize=9, ha='left', va='center', color='#666')

# --- Mise en forme ---
ax.set_xlabel(r'$\theta$', fontsize=13)
ax.set_ylabel('valeur', fontsize=11)
ax.set_title("EM = coordinate ascent sur l'ELBO",
             fontsize=13, pad=15)
ax.legend(loc='lower center', fontsize=10, framealpha=0.95,
          bbox_to_anchor=(0.5, -0.03))
ax.grid(True, alpha=0.25, linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(-2.5, 4.5)

# Légende du déroulé (sous l'image, ne chevauche plus l'axe)
fig.text(0.5, -0.04,
         "À $\\theta^{\\mathrm{old}}$ : E-step rend l'ELBO tangente à $\\log p$ (KL = 0). "
         "M-step : on monte sur l'ELBO jusqu'à $\\theta^{\\mathrm{new}}$.\n"
         "$\\log p$ a augmenté. À $\\theta^{\\mathrm{new}}$ : nouveau E-step → nouvelle ELBO "
         "tangente (orange), et ainsi de suite jusqu'à convergence.",
         ha='center', fontsize=10, color='#333')

plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'em_elbo_view.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Image sauvegardée : {output_path}")
plt.show()
