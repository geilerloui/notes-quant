"""Test pour les figures Lévy area avant de les ajouter au script principal."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = Path(__file__).parent


def fbm_cholesky(H, n, T=1.0, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T, n + 1)
    ts = t[1:]
    s_grid, t_grid = np.meshgrid(ts, ts)
    K = 0.5 * (s_grid**(2*H) + t_grid**(2*H) - np.abs(s_grid - t_grid)**(2*H))
    L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
    z = rng.standard_normal(n)
    B_inner = L @ z
    B = np.concatenate([[0], B_inner])
    return t, B


# =========================================================================
# Figure : Formule de l'aire de Green (cas classique régulier)
# =========================================================================
def fig_area_classical():
    """Pédagogie : montrer la formule A = (1/2) ∮ (x dy - y dx) sur une ellipse."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Ellipse paramétrée : x = a cos(t), y = b sin(t), t ∈ [0, 2π]
    a, b = 2.0, 1.2
    t = np.linspace(0, 2*np.pi, 500)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # === Panneau gauche : la boucle avec son aire colorée ===
    ax = axes[0]
    ax.fill(x, y, color="#3498db", alpha=0.25, label=fr"Aire $= \pi a b = {np.pi*a*b:.2f}$")
    ax.plot(x, y, color="#2c3e50", lw=2, label="Chemin fermé (ellipse)")

    # Vecteurs tangents en quelques points pour montrer dx, dy
    for t_arrow in np.linspace(0, 2*np.pi, 12, endpoint=False):
        xa = a * np.cos(t_arrow)
        ya = b * np.sin(t_arrow)
        dxa = -a * np.sin(t_arrow) * 0.15
        dya = b * np.cos(t_arrow) * 0.15
        ax.annotate("", xy=(xa+dxa, ya+dya), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))

    ax.scatter([0], [0], color="black", s=40, zorder=5, label="Origine")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_title(r"Aire d'une boucle : $A = \frac{1}{2}\oint(x\,dy - y\,dx)$",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # === Panneau droit : décomposition de la formule ===
    ax = axes[1]
    # On discrétise et on calcule l'aire empiriquement
    n_disc = 200
    t_d = np.linspace(0, 2*np.pi, n_disc + 1)
    x_d = a * np.cos(t_d)
    y_d = b * np.sin(t_d)
    dx_d = np.diff(x_d)
    dy_d = np.diff(y_d)

    # Contributions x_i * dy_i et y_i * dx_i
    x_mid = 0.5 * (x_d[:-1] + x_d[1:])
    y_mid = 0.5 * (y_d[:-1] + y_d[1:])

    contrib_xdy = x_mid * dy_d
    contrib_ydx = y_mid * dx_d
    half_area = 0.5 * np.sum(contrib_xdy - contrib_ydx)

    # Plot des contributions cumulées
    cum_xdy = np.concatenate([[0], np.cumsum(contrib_xdy)])
    cum_ydx = np.concatenate([[0], np.cumsum(contrib_ydx)])
    cum_diff = 0.5 * (cum_xdy - cum_ydx)

    ax.plot(t_d, cum_xdy, color="#2980b9", lw=1.8, label=r"$\int_0^t x\,dy$")
    ax.plot(t_d, cum_ydx, color="#c0392b", lw=1.8, label=r"$\int_0^t y\,dx$")
    ax.plot(t_d, cum_diff, color="#27ae60", lw=2.5,
            label=fr"$\frac{{1}}{{2}}(\int x\,dy - \int y\,dx) \to {half_area:.2f}$")
    ax.axhline(np.pi*a*b, color="black", linestyle=":", lw=1.5, alpha=0.6,
               label=fr"Aire vraie $\pi ab = {np.pi*a*b:.2f}$")
    ax.set_xlabel("$t$ (paramètre)")
    ax.set_ylabel("intégrale partielle")
    ax.set_title("Construction de l'aire par intégration le long du chemin",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        r"Avant de définir l'aire de Lévy : rappel de la formule de Green pour un chemin lisse",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig5_area_classical.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 5 : aire classique")


# =========================================================================
# Figure : Aire de Lévy - 2 chemins 2D avec trajectoires visuellement similaires
# mais aires de Lévy très différentes
# =========================================================================
def fig_levy_area():
    """L'image-clef : 2 chemins 2D qui ont trajectoire similaire mais
    aires de Lévy très différentes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Construction : 2 chemins qui partent de (0,0) et finissent à environ (2, 1)
    # Chemin A : trajectoire "directe" qui va doucement de (0,0) vers (2,1)
    # Chemin B : même trajectoire enveloppe MAIS oscille dans un sens créant
    # une grande aire balayée

    n = 1000
    T = 1.0
    t = np.linspace(0, T, n)

    # Chemin A : ligne presque droite avec petites fluctuations symétriques
    rng = np.random.default_rng(42)
    noise_A_x = 0.15 * np.sin(8 * np.pi * t) + 0.05 * rng.standard_normal(n)
    noise_A_y = -0.15 * np.sin(8 * np.pi * t) + 0.05 * rng.standard_normal(n)

    X_A = 2 * t + noise_A_x
    Y_A = 1 * t + noise_A_y

    # Chemin B : avance pareil mais avec rotation (forme une spirale autour
    # de la diagonale)
    radius = 0.25
    theta_B = 6 * np.pi * t  # 3 tours
    X_B = 2 * t + radius * np.cos(theta_B)
    Y_B = 1 * t + radius * np.sin(theta_B)

    # Calcul de l'aire de Lévy : A = (1/2) ∫ (X dY - Y dX)
    # On centre sur le point de départ pour éviter le décalage
    def levy_area(X, Y):
        X_c = X - X[0]
        Y_c = Y - Y[0]
        dX = np.diff(X_c)
        dY = np.diff(Y_c)
        X_mid = 0.5 * (X_c[:-1] + X_c[1:])
        Y_mid = 0.5 * (Y_c[:-1] + Y_c[1:])
        return 0.5 * np.sum(X_mid * dY - Y_mid * dX)

    area_A = levy_area(X_A, Y_A)
    area_B = levy_area(X_B, Y_B)

    # === Panneau gauche : Chemin A ===
    ax = axes[0]
    ax.plot(X_A, Y_A, color="#2c3e50", lw=1.2, alpha=0.7)
    ax.scatter([X_A[0]], [Y_A[0]], color="green", s=80, zorder=5, label="Départ")
    ax.scatter([X_A[-1]], [Y_A[-1]], color="red", s=80, zorder=5, label="Arrivée")
    # Ligne droite "trajectoire moyenne" pour comparaison
    ax.plot([X_A[0], X_A[-1]], [Y_A[0], Y_A[-1]], color="gray",
            linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("$X^1$")
    ax.set_ylabel("$X^2$")
    ax.set_title(fr"Chemin A — aire de Lévy $\approx {area_A:+.3f}$ (faible)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # === Panneau droit : Chemin B ===
    ax = axes[1]
    ax.plot(X_B, Y_B, color="#c0392b", lw=1.2, alpha=0.7)
    ax.scatter([X_B[0]], [Y_B[0]], color="green", s=80, zorder=5, label="Départ")
    ax.scatter([X_B[-1]], [Y_B[-1]], color="red", s=80, zorder=5, label="Arrivée")
    ax.plot([X_B[0], X_B[-1]], [Y_B[0], Y_B[-1]], color="gray",
            linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("$X^1$")
    ax.set_ylabel("$X^2$")
    ax.set_title(fr"Chemin B — aire de Lévy $\approx {area_B:+.3f}$ (élevée, spirale CCW)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Deux chemins 2D : trajectoire visuellement similaire (départ et arrivée alignés), aires de Lévy très différentes",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig6_levy_area.png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Fig 6 : Lévy area (A: {area_A:.3f}, B: {area_B:.3f})")


if __name__ == "__main__":
    fig_area_classical()
    fig_levy_area()
