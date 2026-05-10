"""
Génère des figures pour la note Rough Paths :
- comparaison brownien fractionnaire avec H = 0.2, 0.5, 0.8
- effet de H sur la rugosité, la persistance, la régularité

Usage : python generate_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

try:
    OUT = Path(__file__).parent
except NameError:
    OUT = Path.cwd()


def fbm_cholesky(H, n, T=1.0, seed=None):
    """
    Génère une trajectoire de mouvement brownien fractionnaire B^H sur [0, T]
    via la méthode de Cholesky : B^H ~ N(0, K) où K_{ij} = (s^{2H} + t^{2H} - |s-t|^{2H}) / 2.
    
    Renvoie (t, B) avec t = grille temporelle, B = trajectoire (B[0] = 0).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T, n + 1)
    # Construit la matrice de covariance pour t_1, ..., t_n (on saute t_0 = 0)
    ts = t[1:]  # exclut le 0
    s_grid, t_grid = np.meshgrid(ts, ts)
    K = 0.5 * (s_grid**(2*H) + t_grid**(2*H) - np.abs(s_grid - t_grid)**(2*H))
    # Cholesky
    L = np.linalg.cholesky(K + 1e-10 * np.eye(n))  # petit jitter pour stabilité
    z = rng.standard_normal(n)
    B_inner = L @ z
    B = np.concatenate([[0], B_inner])
    return t, B


# =========================================================================
# Figure 1 : 3 trajectoires côte à côte avec H = 0.2, 0.5, 0.8
# =========================================================================
def fig_compare_H():
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    
    n = 1000
    Hs = [0.2, 0.5, 0.8]
    titles = [
        r"$H = 0.2$ — anti-persistant, **plus rugueux** que le brownien",
        r"$H = 0.5$ — brownien standard ($W_t$)",
        r"$H = 0.8$ — persistant, **plus lisse** que le brownien",
    ]
    colors = ["#c0392b", "#2c3e50", "#27ae60"]
    
    # Même seed pour comparaison
    base_seed = 42
    
    for ax, H, title, color in zip(axes, Hs, titles, colors):
        # 3 trajectoires par H pour montrer la variabilité
        for k in range(3):
            t, B = fbm_cholesky(H, n, seed=base_seed + 100*k)
            ax.plot(t, B, color=color, lw=1.0, alpha=0.7 - k*0.15)
        ax.axhline(0, color="black", lw=0.4, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(r"$B^H_t$")
    
    axes[-1].set_xlabel("$t$")
    fig.suptitle(
        r"Mouvement brownien fractionnaire $B^H_t$ pour 3 valeurs de l'exposant de Hurst",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_compare_H.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 1 : comparaison H")


# =========================================================================
# Figure 2 : Auto-similarité / scaling
# =========================================================================
def fig_self_similarity():
    """Montrer que B^H_{ct} = c^H * B^H_t (même loi)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    
    # Générer une trajectoire fine pour H = 0.2 et H = 0.8
    for ax, H, color in zip(axes, [0.2, 0.8], ["#c0392b", "#27ae60"]):
        # Trajectoire complète sur [0, 1]
        t, B = fbm_cholesky(H, 2000, T=1.0, seed=42)
        ax.plot(t, B, color=color, lw=1.2, alpha=0.6, label=r"$B^H_t$ sur $[0, 1]$")
        
        # Zoom sur [0, 0.1] : extrait + rescaling par 10^H
        idx_zoom = t <= 0.1
        t_zoom = t[idx_zoom]
        B_zoom = B[idx_zoom]
        
        # Rescaling : on plot 10*t_zoom (échelle x) et B_zoom / 10^H (échelle y)
        ax.plot(10 * t_zoom, B_zoom / (10**H), color="black", lw=1.2, alpha=0.9,
                linestyle="--",
                label=fr"Zoom $[0, 0.1]$ rescalé : $t \to 10t$, $B \to B/10^{{{H}}}$")
        
        ax.axhline(0, color="black", lw=0.4, alpha=0.5)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$B^H_t$")
        ax.set_title(fr"$H = {H}$ : auto-similarité $B^H_{{ct}} \sim c^H B^H_t$ (même loi)",
                     fontsize=11)
        ax.legend(fontsize=9, loc="best")
    
    fig.suptitle(
        r"Propriété d'auto-similarité du fBm : un zoom rescalé ressemble à l'original",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_self_similarity.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 2 : auto-similarité")


# =========================================================================
# Figure 3 : Variation quadratique — pourquoi H < 1/2 casse Itô
# =========================================================================
def fig_quadratic_variation():
    """
    Pour H = 1/2, sum (Delta B)^2 -> t (variation quadratique finie).
    Pour H < 1/2, sum (Delta B)^2 -> +infini (ne converge pas).
    Pour H > 1/2, sum (Delta B)^2 -> 0.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    Hs = [0.2, 0.5, 0.8]
    colors = ["#c0392b", "#2c3e50", "#27ae60"]
    labels = [
        r"$H = 0.2$ : $\sum (\Delta B^H)^2 \to +\infty$ (variation quadratique infinie)",
        r"$H = 0.5$ : $\sum (\Delta B^H)^2 \to t$ (variation quadratique finie ✓)",
        r"$H = 0.8$ : $\sum (\Delta B^H)^2 \to 0$ (trop lisse)",
    ]
    
    n_grid = [50, 100, 200, 500, 1000, 2000, 5000]
    T = 1.0
    
    # Pour chaque H, calculer sum (Delta B)^2 sur la trajectoire entière à différentes résolutions
    for H, color, label in zip(Hs, colors, labels):
        # On utilise UNE trajectoire fine, on subdivise plus ou moins
        n_max = 5000
        t_fine, B_fine = fbm_cholesky(H, n_max, T=T, seed=7)
        
        QV_values = []
        for n in n_grid:
            idx = np.linspace(0, n_max, n + 1, dtype=int)
            B_sub = B_fine[idx]
            QV = np.sum(np.diff(B_sub)**2)
            QV_values.append(QV)
        
        ax.plot(n_grid, QV_values, "o-", color=color, lw=1.8, markersize=8, label=label)
    
    ax.axhline(T, color="black", linestyle=":", lw=1.5, alpha=0.6,
               label=fr"$T = {T}$ (variation quadratique du brownien)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$ (subdivisions de $[0, T]$, échelle log)")
    ax.set_ylabel(r"$\sum_{i=0}^{n-1} (B^H_{t_{i+1}} - B^H_{t_i})^2$")
    ax.set_title(
        r"Variation quadratique du fBm : pourquoi le calcul d'Itô casse pour $H \neq 1/2$",
        fontsize=11
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    
    fig.tight_layout()
    fig.savefig(OUT / "fig3_quadratic_variation.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 3 : variation quadratique")


# =========================================================================
# Figure 4 : Rough volatility — modèle Bayer-Friz-Gatheral 2016
# =========================================================================
def fig_rough_vol():
    """
    Comparaison volatilité Heston classique (H=0.5 implicite via OU) vs rough vol (H ≈ 0.1).
    Pédagogique : on plot deux trajectoires de log-vol pour montrer la différence.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    n = 2000
    T = 1.0
    sigma0 = 0.2
    eta = 1.5
    
    # Trajectoire vol "classique" (brownien standard, H = 0.5)
    t, B_classic = fbm_cholesky(0.5, n, T=T, seed=33)
    sigma_classic = sigma0 * np.exp(eta * B_classic)
    
    # Trajectoire vol "rough" (H = 0.1)
    t, B_rough = fbm_cholesky(0.1, n, T=T, seed=33)
    sigma_rough = sigma0 * np.exp(eta * B_rough)
    
    ax = axes[0]
    ax.plot(t, sigma_classic, color="#2c3e50", lw=1.2,
            label=r"Vol. classique ($H = 0.5$, type Heston)")
    ax.set_ylabel(r"$\sigma_t$")
    ax.set_title(r"Volatilité \"classique\" : $\sigma_t = \sigma_0 \exp(\eta B^{0.5}_t)$",
                 fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.set_yscale("log")
    
    ax = axes[1]
    ax.plot(t, sigma_rough, color="#c0392b", lw=1.2,
            label=r"Rough vol. ($H = 0.1$, Bayer-Friz-Gatheral 2016)")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\sigma_t$")
    ax.set_title(r"Volatilité \"rough\" : $\sigma_t = \sigma_0 \exp(\eta B^{0.1}_t)$ — beaucoup plus rugueuse !",
                 fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.set_yscale("log")
    
    fig.suptitle(
        r"Application en finance : la volatilité empirique a un Hurst $H \approx 0.1$",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_rough_vol.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 4 : rough volatility")


# =========================================================================
# Figure 5 : Formule de l'aire de Green (cas classique régulier)
# =========================================================================
def fig_area_classical():
    """Pédagogie : montrer la formule A = (1/2) ∮ (x dy - y dx) sur une ellipse."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    a, b = 2.0, 1.2
    t = np.linspace(0, 2*np.pi, 500)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # === Panneau gauche : la boucle avec son aire colorée ===
    ax = axes[0]
    ax.fill(x, y, color="#3498db", alpha=0.25, label=fr"Aire $= \pi a b = {np.pi*a*b:.2f}$")
    ax.plot(x, y, color="#2c3e50", lw=2, label="Chemin fermé (ellipse)")

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
    n_disc = 200
    t_d = np.linspace(0, 2*np.pi, n_disc + 1)
    x_d = a * np.cos(t_d)
    y_d = b * np.sin(t_d)
    dx_d = np.diff(x_d)
    dy_d = np.diff(y_d)

    x_mid = 0.5 * (x_d[:-1] + x_d[1:])
    y_mid = 0.5 * (y_d[:-1] + y_d[1:])

    contrib_xdy = x_mid * dy_d
    contrib_ydx = y_mid * dx_d
    half_area = 0.5 * np.sum(contrib_xdy - contrib_ydx)

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
# Figure 6 : Aire de Lévy - 2 chemins 2D
# =========================================================================
def fig_levy_area():
    """L'image-clef : 2 chemins 2D qui ont trajectoire similaire mais
    aires de Lévy très différentes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    n = 1000
    T = 1.0
    t = np.linspace(0, T, n)

    rng = np.random.default_rng(42)
    noise_A_x = 0.15 * np.sin(8 * np.pi * t) + 0.05 * rng.standard_normal(n)
    noise_A_y = -0.15 * np.sin(8 * np.pi * t) + 0.05 * rng.standard_normal(n)
    X_A = 2 * t + noise_A_x
    Y_A = 1 * t + noise_A_y

    radius = 0.25
    theta_B = 6 * np.pi * t
    X_B = 2 * t + radius * np.cos(theta_B)
    Y_B = 1 * t + radius * np.sin(theta_B)

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

    ax = axes[0]
    ax.plot(X_A, Y_A, color="#2c3e50", lw=1.2, alpha=0.7)
    ax.scatter([X_A[0]], [Y_A[0]], color="green", s=80, zorder=5, label="Départ")
    ax.scatter([X_A[-1]], [Y_A[-1]], color="red", s=80, zorder=5, label="Arrivée")
    ax.plot([X_A[0], X_A[-1]], [Y_A[0], Y_A[-1]], color="gray",
            linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("$X^1$")
    ax.set_ylabel("$X^2$")
    ax.set_title(fr"Chemin A — aire de Lévy $\approx {area_A:+.3f}$ (faible)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

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
        r"Deux chemins 2D : trajectoire similaire (même départ et arrivée), aires de Lévy très différentes",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig6_levy_area.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 6 : Lévy area")


# =========================================================================
# Figure 7 : Décomposition en triangles, deux chemins comparés
# =========================================================================
def fig_triangles():
    """Visualisation explicite des petits triangles balayés."""
    from matplotlib.patches import Polygon

    def signed_triangle_area(p_start, p_a, p_b):
        v1 = p_a - p_start
        v2 = p_b - p_start
        return 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))

    n = 8
    t = np.linspace(0, 1, n)

    rng = np.random.default_rng(42)
    X_A = 2 * t + 0.15 * np.sin(5 * np.pi * t)
    Y_A = 1 * t - 0.15 * np.sin(5 * np.pi * t)

    radius = 0.3
    theta_B = 3 * np.pi * t
    X_B = 2 * t + radius * np.cos(theta_B) - radius
    Y_B = 1 * t + radius * np.sin(theta_B)

    for idx, (X, Y, title_chemin, color_main) in enumerate([
        (X_A, Y_A, "Chemin A — quasi droit", "#2c3e50"),
        (X_B, Y_B, "Chemin B — spirale CCW", "#c0392b"),
    ]):
        ax = axes[idx]

        X_s = np.array([X[0], Y[0]])
        triangles = []
        signed_areas = []
        for k in range(n - 1):
            p_a = np.array([X[k], Y[k]])
            p_b = np.array([X[k + 1], Y[k + 1]])
            triangles.append((X_s, p_a, p_b))
            signed_areas.append(signed_triangle_area(X_s, p_a, p_b))

        total_area = sum(signed_areas)

        for tri, area in zip(triangles, signed_areas):
            poly = Polygon(np.array(tri), closed=True,
                           facecolor=("#27ae60" if area > 0 else "#e67e22"),
                           alpha=0.35, edgecolor="black", linewidth=0.5)
            ax.add_patch(poly)

        ax.plot(X, Y, color=color_main, lw=1.8, alpha=0.9, zorder=3)

        for k in range(n):
            ax.plot([X[0], X[k]], [Y[0], Y[k]],
                    color="gray", lw=0.7, alpha=0.5, linestyle="--", zorder=2)

        ax.scatter(X, Y, color=color_main, s=50, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.scatter([X[0]], [Y[0]], color="green", s=180, zorder=6,
                   edgecolor="black", linewidth=1.5, marker="*",
                   label=r"$X_s$ (départ)")
        ax.scatter([X[-1]], [Y[-1]], color="red", s=120, zorder=6,
                   edgecolor="black", linewidth=1.5, marker="s",
                   label=r"$X_t$ (arrivée)")

        ax.set_xlabel(r"$X^1$")
        ax.set_ylabel(r"$X^2$")
        ax.set_title(
            fr"{title_chemin} — somme des aires triangles $\approx {total_area:+.3f}$",
            fontsize=11
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.margins(0.15)

    fig.suptitle(
        r"Aire de Lévy = somme des aires signées des petits triangles ($X_s$, $X_r$, $X_{r+1}$)",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig7_triangles.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 7 : décomposition en triangles")


# =========================================================================
# Figure 8 : Cumul de l'aire de Lévy au fil du temps
# =========================================================================
def fig_levy_cumulative():
    """Aire de Lévy cumulée A(s,r) pour r dans [s, t]."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    n = 1000
    t = np.linspace(0, 1, n)

    rng = np.random.default_rng(42)
    noise_A_x = 0.15 * np.sin(8 * np.pi * t) + 0.05 * rng.standard_normal(n)
    noise_A_y = -0.15 * np.sin(8 * np.pi * t) + 0.05 * rng.standard_normal(n)
    X_A = 2 * t + noise_A_x
    Y_A = 1 * t + noise_A_y

    radius = 0.25
    theta_B = 6 * np.pi * t
    X_B = 2 * t + radius * np.cos(theta_B)
    Y_B = 1 * t + radius * np.sin(theta_B)

    def cumulative_levy_area(X, Y):
        X_c = X - X[0]
        Y_c = Y - Y[0]
        dX = np.diff(X_c)
        dY = np.diff(Y_c)
        X_mid = 0.5 * (X_c[:-1] + X_c[1:])
        Y_mid = 0.5 * (Y_c[:-1] + Y_c[1:])
        contribs = 0.5 * (X_mid * dY - Y_mid * dX)
        return np.concatenate([[0], np.cumsum(contribs)])

    cum_A = cumulative_levy_area(X_A, Y_A)
    cum_B = cumulative_levy_area(X_B, Y_B)

    ax = axes[0]
    ax.plot(t, cum_A, color="#2c3e50", lw=1.5, label="Chemin A (presque droit)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("$r$ (variable d'intégration)")
    ax.set_ylabel(r"$A_{s,r}$ (aire de Lévy cumulée)")
    ax.set_title(r"Chemin A : oscille autour de 0 — final $\approx 0$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, cum_B, color="#c0392b", lw=1.5, label="Chemin B (spirale CCW)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("$r$ (variable d'intégration)")
    ax.set_ylabel(r"$A_{s,r}$ (aire de Lévy cumulée)")
    ax.set_title(fr"Chemin B : croît monotone — final $\approx {cum_B[-1]:+.3f}$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"L'aire de Lévy se construit progressivement le long du chemin",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig8_levy_cumulative.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 8 : aire cumulative")


if __name__ == "__main__":
    fig_compare_H()
    fig_self_similarity()
    fig_quadratic_variation()
    fig_rough_vol()
    fig_area_classical()
    fig_levy_area()
    fig_triangles()
    fig_levy_cumulative()
    print(f"\nFigures sauvegardées dans {OUT}")
