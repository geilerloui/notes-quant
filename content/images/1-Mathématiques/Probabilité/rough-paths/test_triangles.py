"""Test : visualiser les petits triangles balayés."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = Path(__file__).parent


def signed_triangle_area(p_start, p_a, p_b):
    """Aire signée d'un triangle de sommets p_start, p_a, p_b.
    Positive si parcouru en sens anti-horaire."""
    v1 = p_a - p_start
    v2 = p_b - p_start
    return 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])


# =========================================================================
# Figure 7 : Décomposition en triangles, deux chemins comparés
# =========================================================================
def fig_triangles():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))

    # === On prend peu de points pour voir clairement chaque triangle ===
    n = 8
    t = np.linspace(0, 1, n)

    # Chemin A : presque droit avec petites oscillations symétriques
    rng = np.random.default_rng(42)
    X_A = 2 * t + 0.15 * np.sin(5 * np.pi * t)
    Y_A = 1 * t - 0.15 * np.sin(5 * np.pi * t)

    # Chemin B : spirale anti-horaire autour de la diagonale
    radius = 0.3
    theta_B = 3 * np.pi * t
    X_B = 2 * t + radius * np.cos(theta_B) - radius
    Y_B = 1 * t + radius * np.sin(theta_B)

    for ax, X, Y, title_chemin, color_main in zip(
        axes,
        [(X_A, Y_A), (X_B, Y_B)],
        [(X_A, Y_A), (X_B, Y_B)],  # placeholder non utilisé
        ["Chemin A — quasi droit", "Chemin B — spirale CCW"],
        ["#2c3e50", "#c0392b"]
    ):
        Xc, Yc = X[0], X[1]
        # placeholders ignorés, on va refaire propre

    # OK je refais proprement
    for idx, (X, Y, title_chemin, color_main) in enumerate([
        (X_A, Y_A, "Chemin A — quasi droit", "#2c3e50"),
        (X_B, Y_B, "Chemin B — spirale CCW", "#c0392b"),
    ]):
        ax = axes[idx]

        # Calcule les aires de chaque triangle (X_s, X_r, X_{r+1})
        X_s = np.array([X[0], Y[0]])
        triangles = []
        signed_areas = []
        for k in range(n - 1):
            p_a = np.array([X[k], Y[k]])
            p_b = np.array([X[k + 1], Y[k + 1]])
            triangles.append((X_s, p_a, p_b))
            signed_areas.append(signed_triangle_area(X_s, p_a, p_b))

        total_area = sum(signed_areas)

        # Dessine les triangles avec couleur selon signe
        for k, (tri, area) in enumerate(zip(triangles, signed_areas)):
            poly = Polygon(np.array(tri), closed=True,
                           facecolor=("#27ae60" if area > 0 else "#e67e22"),
                           alpha=0.35, edgecolor="black", linewidth=0.5)
            ax.add_patch(poly)

        # Dessine le chemin
        ax.plot(X, Y, color=color_main, lw=1.8, alpha=0.9, zorder=3)

        # Dessine les rayons depuis X_s vers chaque X_r
        for k in range(n):
            ax.plot([X[0], X[k]], [Y[0], Y[k]],
                    color="gray", lw=0.7, alpha=0.5, linestyle="--", zorder=2)

        # Marque les points
        ax.scatter(X, Y, color=color_main, s=50, zorder=5,
                   edgecolor="white", linewidth=1.2)

        # Marque le point de départ X_s en gros
        ax.scatter([X[0]], [Y[0]], color="green", s=180, zorder=6,
                   edgecolor="black", linewidth=1.5, marker="*",
                   label=r"$X_s$ (départ)")

        # Marque l'arrivée
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
        # marge
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
    """Pour chaque chemin, tracer A(s,r) = aire de Lévy entre s et r,
    en fonction de r. Pour chemin A on doit voir un truc qui oscille autour de 0,
    pour chemin B un truc qui croît monotone."""
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
        """Retourne A(s, r) cumulée pour r ∈ [s, t]."""
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
    ax.set_title(r"Chemin A : oscille autour de 0 — final $\approx 0$",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, cum_B, color="#c0392b", lw=1.5, label="Chemin B (spirale CCW)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("$r$ (variable d'intégration)")
    ax.set_ylabel(r"$A_{s,r}$ (aire de Lévy cumulée)")
    ax.set_title(fr"Chemin B : croît monotone — final $\approx {cum_B[-1]:+.3f}$",
                 fontsize=11)
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
    fig_triangles()
    fig_levy_cumulative()
