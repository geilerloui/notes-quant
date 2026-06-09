"""
Génère les figures pour la note 'Mouvement Brownien'.
Exécuter depuis ce dossier : python generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- style cohérent avec le reste du vault ----
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.2,
})

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

# Palette sobre
C_BLUE   = "#2E5C8A"
C_RED    = "#B23A48"
C_GREEN  = "#4A7C59"
C_GREY   = "#8a8784"
C_ORANGE = "#D08C3F"

# ============================================================
# Fig 1 — Marche aléatoire simple : 3 trajectoires
# ============================================================
def fig_marche_simple():
    n = 200
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for color in [C_BLUE, C_RED, C_GREEN]:
        eps = rng.choice([-1, 1], size=n)
        X = np.concatenate([[0], np.cumsum(eps)])
        ax.step(np.arange(n + 1), X, where="post", color=color, alpha=0.85)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("nombre de pas $n$")
    ax.set_ylabel("$X_n$")
    ax.set_title("Marche aléatoire simple : 3 trajectoires de longueur 200")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_marche_simple.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 2 — Le problème : variance qui explose
# ============================================================
def fig_variance_explose():
    """Si on raffine sans scaler : amplitude explose."""
    T = 1.0
    ns = [10, 100, 1000, 10000]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=False)
    for ax, n in zip(axes, ns):
        eps = rng.choice([-1, 1], size=n)
        X = np.concatenate([[0], np.cumsum(eps)])
        t = np.linspace(0, T, n + 1)
        ax.plot(t, X, color=C_BLUE, lw=0.9)
        ax.set_title(f"$n={n}$  •  $\\sigma \\approx {np.sqrt(n):.0f}$")
        ax.set_xlabel("$t$")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("$X$")
    fig.suptitle("Sans rescaling : $\\mathrm{Var}(X_n) = n \\to \\infty$ quand on raffine",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_variance_explose.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 3 — Marche scaled : convergence vers le brownien
# ============================================================
def fig_scaled_convergence():
    """Même seed → même 'forme', mais on raffine la grille."""
    T = 1.0
    ns = [10, 100, 1000, 10000]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    # On utilise le MÊME bruit gaussien sous-jacent pour montrer la convergence
    base_n = max(ns)
    base_eps = rng.choice([-1, 1], size=base_n)
    for ax, n in zip(axes, ns):
        # sous-échantillonnage : on prend un eps tous les base_n//n
        step = base_n // n
        eps = base_eps[:n * step].reshape(n, step).sum(axis=1) / np.sqrt(step)
        # on a maintenant n incréments d'écart-type 1
        dt = T / n
        W = np.concatenate([[0], np.cumsum(eps * np.sqrt(dt))])
        t = np.linspace(0, T, n + 1)
        ax.plot(t, W, color=C_BLUE, lw=0.9)
        ax.set_title(f"$n={n}$,  $\\Delta t = {dt:g}$")
        ax.set_xlabel("$t$")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("$W^{(n)}(t)$")
    fig.suptitle("Marche aléatoire rescalée : convergence vers le mouvement brownien",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_scaled_convergence.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 4 — 50 trajectoires browniennes + enveloppe ±√t, ±2√t
# ============================================================
def fig_enveloppe():
    T = 1.0
    n = 1000
    M = 50
    dt = T / n
    t = np.linspace(0, T, n + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for _ in range(M):
        increments = rng.normal(0, np.sqrt(dt), size=n)
        W = np.concatenate([[0], np.cumsum(increments)])
        ax.plot(t, W, color=C_BLUE, alpha=0.25, lw=0.7)
    ax.plot(t,  np.sqrt(t), color=C_RED, lw=1.5, label=r"$\pm\sqrt{t}$")
    ax.plot(t, -np.sqrt(t), color=C_RED, lw=1.5)
    ax.plot(t,  2*np.sqrt(t), color=C_RED, lw=1.5, ls="--", label=r"$\pm 2\sqrt{t}$")
    ax.plot(t, -2*np.sqrt(t), color=C_RED, lw=1.5, ls="--")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$W_t$")
    ax.set_title(f"{M} trajectoires browniennes sur $[0, {T:g}]$ avec enveloppe gaussienne")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_enveloppe.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 5 — Non-différentiabilité : zoom successif reste rugueux
# ============================================================
def fig_zoom():
    T = 1.0
    n = 100_000
    dt = T / n
    increments = rng.normal(0, np.sqrt(dt), size=n)
    W = np.concatenate([[0], np.cumsum(increments)])
    t = np.linspace(0, T, n + 1)

    # 4 niveaux de zoom autour de t = 0.5
    centre = 0.5
    largeurs = [1.0, 0.1, 0.01, 0.001]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, L in zip(axes, largeurs):
        mask = (t >= centre - L/2) & (t <= centre + L/2)
        ax.plot(t[mask], W[mask], color=C_BLUE, lw=0.9)
        ax.set_title(f"largeur = {L:g}")
        ax.set_xlabel("$t$")
    axes[0].set_ylabel("$W_t$")
    fig.suptitle("Zoom successif sur une trajectoire brownienne : la rugosité ne disparaît jamais",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_zoom_non_differentiable.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 6 — Variation quadratique converge vers t
# ============================================================
def fig_variation_quadratique():
    T = 1.0
    ns = [10, 100, 1000, 10000]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    # Une trajectoire fine de référence
    n_max = max(ns)
    dt_max = T / n_max
    inc = rng.normal(0, np.sqrt(dt_max), size=n_max)
    t_max = np.linspace(0, T, n_max + 1)
    for n, color in zip(ns, [C_GREY, C_GREEN, C_ORANGE, C_BLUE]):
        step = n_max // n
        # Sous-échantillonner la même trajectoire
        idx = np.arange(0, n_max + 1, step)
        W_sub = np.concatenate([[0], np.cumsum(inc.reshape(n, step).sum(axis=1))])
        t_sub = t_max[idx]
        Q = np.concatenate([[0], np.cumsum(np.diff(W_sub) ** 2)])
        ax.plot(t_sub, Q, color=color, lw=1.3, label=f"$n = {n}$")
    ax.plot([0, T], [0, T], color="black", lw=1.2, ls="--", label="$y = t$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\sum_{t_i \le t}(W_{t_{i+1}} - W_{t_i})^2$")
    ax.set_title("Variation quadratique : converge vers $t$ quand $n \\to \\infty$")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_variation_quadratique.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_marche_simple()
    fig_variance_explose()
    fig_scaled_convergence()
    fig_enveloppe()
    fig_zoom()
    fig_variation_quadratique()
    print("OK — 6 figures générées dans", OUT)
