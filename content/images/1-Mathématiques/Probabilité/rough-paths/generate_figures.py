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


if __name__ == "__main__":
    fig_compare_H()
    fig_self_similarity()
    fig_quadratic_variation()
    fig_rough_vol()
    print(f"\nFigures sauvegardées dans {OUT}")
