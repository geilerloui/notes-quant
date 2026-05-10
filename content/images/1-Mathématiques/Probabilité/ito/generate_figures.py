"""
Génère 8 figures pour la note Calcul d'Itô.

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

rng = np.random.default_rng(42)


def brownian(T, n, seed=None):
    """Génère une trajectoire brownienne sur [0, T] à n+1 points."""
    r = np.random.default_rng(seed) if seed is not None else rng
    dt = T / n
    dW = r.normal(0, np.sqrt(dt), n)
    W = np.concatenate([[0], np.cumsum(dW)])
    t = np.linspace(0, T, n + 1)
    return t, W


# =========================================================================
# Figure 1 : Filtration — le cône d'information
# =========================================================================
def fig_filtration():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t, W = brownian(1.0, 500, seed=10)

    t_obs = 0.45
    idx_obs = int(t_obs * 500)

    ax.plot(t[:idx_obs+1], W[:idx_obs+1], color="#2c3e50", lw=1.8,
            label=r"$W_s$ pour $s \leq t$ — connu")
    ax.plot(t[idx_obs:], W[idx_obs:], color="gray", lw=1, alpha=0.3,
            label=r"$W_s$ pour $s > t$ — inconnu")

    for i in range(15):
        _, W_alt = brownian(1.0 - t_obs, 500 - idx_obs, seed=100 + i)
        W_alt = W_alt + W[idx_obs]
        t_alt = t[idx_obs:]
        ax.plot(t_alt, W_alt, color="#e74c3c", lw=0.7, alpha=0.25)

    ax.axvline(t_obs, color="black", linestyle="--", alpha=0.6)
    ax.text(t_obs + 0.01, ax.get_ylim()[1] * 0.95, r"$t$ (maintenant)", fontsize=10, va="top")

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    y_box = ymin + yrange * 0.95
    ax.text(0.15, y_box, r"$\mathcal{F}_t$ = info disponible", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="gray"),
            va="top")
    ax.text(0.65, y_box, "Futurs possibles", fontsize=11, color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fadbd8", edgecolor="#c0392b"),
            va="top")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$W_t$")
    ax.set_title(r"Filtration $\mathcal{F}_t$ — ce que je sais à l'instant $t$, et tous les futurs encore possibles")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_filtration.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 1 : filtration")


# =========================================================================
# Figure 2 : Adapté vs anticipé
# =========================================================================
def fig_adapted_vs_anticipating():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t, W = brownian(1.0, 500, seed=42)
    t_show = 0.4
    idx = int(t_show * 500)
    T = 1.0

    # Haut : adapté
    ax = axes[0]
    ax.plot(t, W, color="#2c3e50", lw=1.4, label=r"$W_s$")
    ax.plot(t, W, color="#27ae60", lw=2.5, alpha=0.5, label=r"$H_s = W_s$ (adapté)")
    ax.scatter([t_show], [W[idx]], color="#27ae60", s=120, zorder=5, edgecolor="black")
    ax.annotate(r"$H_{0.4} = W_{0.4}$ — info passée uniquement ✓",
                xy=(t_show, W[idx]), xytext=(t_show + 0.05, W[idx] + 0.4),
                fontsize=10, color="#27ae60",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", edgecolor="#27ae60"),
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.axvline(t_show, color="gray", linestyle=":", alpha=0.6)
    ax.set_title(r"Adapté ✓ — à $s=0.4$, $H_s = W_s$ ne dépend que de $W_u$ pour $u \leq s$",
                 color="#27ae60", fontsize=11)
    ax.set_ylabel("valeur")
    ax.legend(loc="lower left")

    # Bas : anticipé
    ax = axes[1]
    ax.plot(t, W, color="#2c3e50", lw=1.4, label=r"$W_s$")
    ax.axhline(W[-1], color="#e74c3c", lw=2.5, alpha=0.6, label=r"$H_s = W_T$ (anticipé)")
    ax.scatter([t_show], [W[-1]], color="#e74c3c", s=120, zorder=5, edgecolor="black")
    ax.scatter([T], [W[-1]], color="black", s=80, zorder=5, marker="*", label=r"$W_T$ — futur !")
    ax.annotate(r"$H_{0.4} = W_T$ — utilise le futur ✗",
                xy=(t_show, W[-1]), xytext=(t_show + 0.05, W[-1] - 0.3),
                fontsize=10, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#c0392b"),
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.axvline(t_show, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(T, color="black", linestyle="--", alpha=0.4)
    ax.set_title(r"Anticipé ✗ — à $s=0.4$, $H_s = W_T$ regarde déjà la fin",
                 color="#c0392b", fontsize=11)
    ax.set_xlabel("$s$")
    ax.set_ylabel("valeur")
    ax.legend(loc="lower left")

    fig.suptitle(r"Processus adapté à $\mathcal{F}_s$ : à chaque $s$, $H_s$ ne doit dépendre que du passé",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_adapted.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 2 : adapté vs anticipé")


# =========================================================================
# Figure 3 : Exemple manuel n=4 + convergence S_n
# =========================================================================
def fig_riemann_sum():
    """2 panneaux côte à côte :
       - gauche : l'exemple manuel à n=4 avec les valeurs exactes du tableau
       - droite : convergence de S_n sur une vraie trajectoire brownienne fine
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ============= PANNEAU GAUCHE : exemple manuel n=4 =============
    ax = axes[0]

    # Les 5 valeurs du tableau pédagogique (mêmes que dans la note)
    s_pts = np.array([0, 0.25, 0.5, 0.75, 1.0])
    W_pts = np.array([0.0, -0.3, 0.4, 0.1, 0.6])

    # Trajectoire = interpolation linéaire pour visualiser
    ax.plot(s_pts, W_pts, color="lightgray", lw=1.5, alpha=0.7, zorder=1,
            label=r"trajectoire $W_s(\omega)$ (interp. lin.)")

    # Points avec annotations
    ax.scatter(s_pts, W_pts, color="#2c3e50", s=90, zorder=5,
               edgecolor="white", linewidth=1.5)
    for i, (s, W) in enumerate(zip(s_pts, W_pts)):
        ax.annotate(f"$W_{{s_{i}}}={W}$",
                    xy=(s, W), xytext=(8, 12), textcoords="offset points",
                    fontsize=9.5, color="#2c3e50")

    # Flèches verticales pour les incréments dW_i, colorées selon contrib
    contribs = []
    for i in range(4):
        dW = W_pts[i+1] - W_pts[i]
        contrib = W_pts[i] * dW
        contribs.append(contrib)
        color = "#27ae60" if contrib > 0 else ("#c0392b" if contrib < 0 else "gray")
        # flèche de (s_i, W_i) à (s_{i+1}, W_{i+1})
        ax.annotate("",
                    xy=(s_pts[i+1], W_pts[i+1]),
                    xytext=(s_pts[i+1], W_pts[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0, alpha=0.85))
        # Label de la contribution au milieu de la flèche verticale
        x_mid = s_pts[i+1] + 0.015
        y_mid = (W_pts[i] + W_pts[i+1]) / 2
        ax.text(x_mid, y_mid, f"$W_{{s_{i}}}\\cdot\\Delta W_{i}={contrib:+.2f}$",
                fontsize=9, color=color, va="center")

    # Ligne horizontale de palier H_si = W_si
    for i in range(4):
        ax.plot([s_pts[i], s_pts[i+1]], [W_pts[i], W_pts[i]],
                color="#2980b9", lw=2.5, alpha=0.6, zorder=3)

    ax.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$W_s(\omega)$")

    S4 = sum(contribs)
    ax.set_title(
        rf"Exemple manuel à $n = 4$ pas : $S_4 = {S4:+.2f}$",
        fontsize=11
    )
    ax.set_xlim(-0.05, 1.25)
    ax.set_ylim(-0.65, 0.95)
    ax.legend(loc="upper left", fontsize=9)

    # ============= PANNEAU DROIT : convergence sur vraie trajectoire =============
    ax = axes[1]

    T = 1.0
    ns = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    n_ultra = 100000
    _, W_ultra = brownian(T, n_ultra, seed=7)

    S_values = []
    for n in ns:
        idx2 = np.linspace(0, n_ultra, n + 1, dtype=int)
        W_sub2 = W_ultra[idx2]
        dWs2 = np.diff(W_sub2)
        H_left2 = W_sub2[:-1]
        S_values.append(np.sum(H_left2 * dWs2))

    I_exact = (W_ultra[-1]**2 - T) / 2

    ax.plot(ns, S_values, "o-", color="#2980b9", lw=1.8, markersize=10,
            label=r"$S_n$ (somme partielle)")
    ax.axhline(I_exact, color="#c0392b", linestyle="--", lw=2,
               label=fr"Valeur exacte (formule d'Itô) : $(W_T^2-T)/2 = {I_exact:.3f}$")
    ax.set_xscale("log")
    ax.set_xlabel("$n$ (nombre de subdivisions, échelle log)")
    ax.set_ylabel(r"$S_n$")
    ax.set_title(
        r"Convergence de $S_n$ vers l'intégrale d'Itô (autre trajectoire fine)",
        fontsize=11
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        r"Construction de l'intégrale d'Itô : exemple manuel + convergence empirique",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_riemann.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 3 : exemple n=4 + convergence")


# =========================================================================
# Figure 4 : Itô vs Stratonovich
# =========================================================================
def fig_ito_vs_strato():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_traj = 5000
    n_steps = 1000
    T = 1.0

    I_ito = np.zeros(n_traj)
    I_strato = np.zeros(n_traj)

    for k in range(n_traj):
        _, W = brownian(T, n_steps, seed=1000 + k)
        dW = np.diff(W)
        H_left = W[:-1]
        I_ito[k] = np.sum(H_left * dW)
        H_mid = 0.5 * (W[:-1] + W[1:])
        I_strato[k] = np.sum(H_mid * dW)

    ax = axes[0]
    bins = np.linspace(-1.5, 4, 60)
    ax.hist(I_ito, bins=bins, alpha=0.6, color="#2980b9",
            label=f"Itô (gauche) : moy = {I_ito.mean():.3f}", density=True)
    ax.hist(I_strato, bins=bins, alpha=0.6, color="#c0392b",
            label=f"Stratonovich (milieu) : moy = {I_strato.mean():.3f}", density=True)
    ax.axvline(I_ito.mean(), color="#2980b9", linestyle="--", lw=2)
    ax.axvline(I_strato.mean(), color="#c0392b", linestyle="--", lw=2)
    ax.axvline(0, color="black", linestyle=":", alpha=0.5, label=r"$0$")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label=r"$T/2 = 0.5$")
    ax.set_xlabel(r"Valeur de $\int_0^T W_s\,dW_s$ sur 5000 trajectoires")
    ax.set_ylabel("densité")
    ax.set_title("Distribution de l'intégrale selon le choix du point d'évaluation")
    ax.legend(fontsize=9)

    ax = axes[1]
    diff = I_strato - I_ito
    ax.hist(diff, bins=40, color="#27ae60", alpha=0.7, edgecolor="black")
    ax.axvline(diff.mean(), color="#27ae60", linestyle="--", lw=2,
               label=f"moy empirique = {diff.mean():.3f}")
    ax.axvline(0.5, color="black", linestyle=":", lw=2, label=r"$T/2 = 0.5$ (théorie)")
    ax.set_xlabel("Stratonovich − Itô")
    ax.set_ylabel("comptage")
    ax.set_title(r"La différence vaut exactement $T/2$ — pas de hasard, c'est systématique")
    ax.legend()

    fig.suptitle(r"Itô vs Stratonovich sur $\int_0^1 W_s\,dW_s$ — le choix du point change la valeur de $T/2$ !",
                 fontsize=13, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ito_strato.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 4 : Itô vs Stratonovich")


# =========================================================================
# Figure 5 : Espérance nulle
# =========================================================================
def fig_expectation_zero():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    n_traj = 1000
    n_steps = 500
    T = 1.0

    I_traj = np.zeros((n_traj, n_steps + 1))
    for k in range(n_traj):
        _, W = brownian(T, n_steps, seed=2000 + k)
        dW = np.diff(W)
        H_left = W[:-1]
        I_traj[k, 1:] = np.cumsum(H_left * dW)

    t = np.linspace(0, T, n_steps + 1)

    ax = axes[0]
    for k in range(50):
        ax.plot(t, I_traj[k], color="lightgray", lw=0.5, alpha=0.6)
    ax.plot(t, I_traj.mean(axis=0), color="#c0392b", lw=2.5,
            label=r"moy. empirique sur 1000 traj.")
    ax.axhline(0, color="black", linestyle="--", lw=1.5,
               label=r"$\mathbb{E}[I_t] = 0$ (théorie)")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$I_t = \int_0^t W_s\,dW_s$")
    ax.set_title(r"L'intégrale d'Itô a espérance nulle à chaque $t$")
    ax.legend()

    ax = axes[1]
    ax.hist(I_traj[:, -1], bins=40, color="#3498db", alpha=0.7, edgecolor="black", density=True)
    ax.axvline(I_traj[:, -1].mean(), color="#c0392b", linestyle="--", lw=2,
               label=f"moy. empirique = {I_traj[:, -1].mean():.3f}")
    ax.axvline(0, color="black", linestyle=":", lw=2, label=r"théorie : $0$")
    ax.set_xlabel(r"$I_T = \int_0^T W_s\,dW_s$")
    ax.set_ylabel("densité")
    ax.set_title(r"Distribution de $I_T$ : centrée en $0$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "fig5_expectation.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 5 : espérance nulle")


# =========================================================================
# Figure 6 : Isométrie d'Itô
# =========================================================================
def fig_ito_isometry():
    fig, ax = plt.subplots(figsize=(9, 5))

    Ts = np.linspace(0.2, 2.0, 8)
    n_traj = 3000
    n_steps = 500

    emp_var_I = []
    emp_E_int_H2 = []

    for T in Ts:
        I_T = np.zeros(n_traj)
        int_H2 = np.zeros(n_traj)
        for k in range(n_traj):
            _, W = brownian(T, n_steps, seed=3000 + k)
            dW = np.diff(W)
            H = W[:-1]
            I_T[k] = np.sum(H * dW)
            int_H2[k] = np.sum(H**2 * (T / n_steps))
        emp_var_I.append(I_T.var())
        emp_E_int_H2.append(int_H2.mean())

    emp_var_I = np.array(emp_var_I)
    emp_E_int_H2 = np.array(emp_E_int_H2)

    ax.scatter(emp_E_int_H2, emp_var_I, s=80, color="#9b59b6", zorder=5, edgecolor="black",
               label="Mesures empiriques (Monte Carlo)")
    ax.plot([0, max(emp_var_I.max(), emp_E_int_H2.max()) * 1.1],
            [0, max(emp_var_I.max(), emp_E_int_H2.max()) * 1.1],
            color="black", linestyle="--", label="$y = x$ (isométrie parfaite)")

    for T, x, y in zip(Ts, emp_E_int_H2, emp_var_I):
        ax.annotate(f"$T={T:.1f}$", xy=(x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color="#7d3c98")

    ax.set_xlabel(r"$\mathbb{E}\left[\int_0^T H_s^2\,ds\right]$ (empirique)")
    ax.set_ylabel(r"$\mathrm{Var}\left(\int_0^T H_s\,dW_s\right)$ (empirique)")
    ax.set_title(r"Isométrie d'Itô : $\mathbb{E}\left[\left(\int H\,dW\right)^2\right] = \mathbb{E}\left[\int H^2\,ds\right]$" +
                 "\n(ici $H_s = W_s$, pour différents $T$)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_isometry.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 6 : isométrie")


# =========================================================================
# Figure 7 : Martingalité
# =========================================================================
def fig_martingale():
    fig, ax = plt.subplots(figsize=(10, 5))

    T = 1.0
    n_steps = 500
    n_branches = 100

    t_obs = 0.4
    idx_obs = int(t_obs * n_steps)

    _, W_past = brownian(T, n_steps, seed=11)
    dW_past = np.diff(W_past[:idx_obs+1])
    I_past = np.concatenate([[0], np.cumsum(W_past[:idx_obs] * dW_past)])
    t_past = np.linspace(0, t_obs, idx_obs + 1)

    ax.plot(t_past, I_past, color="black", lw=2.5, label=r"$I_s$ pour $s \leq t$ (connu)")
    ax.scatter([t_obs], [I_past[-1]], color="black", s=100, zorder=10)

    I_T_futures = []
    t_future = np.linspace(t_obs, T, n_steps - idx_obs + 1)

    for b in range(n_branches):
        _, W_branch = brownian(T - t_obs, n_steps - idx_obs, seed=5000 + b)
        W_full_branch = W_past[idx_obs] + W_branch
        dW_branch = np.diff(W_full_branch)
        I_branch_inc = np.cumsum(W_full_branch[:-1] * dW_branch)
        I_branch = np.concatenate([[I_past[-1]], I_past[-1] + I_branch_inc])
        ax.plot(t_future, I_branch, color="#e74c3c", lw=0.5, alpha=0.3)
        I_T_futures.append(I_branch[-1])

    I_T_futures = np.array(I_T_futures)

    ax.axhline(I_past[-1], color="#27ae60", lw=2.5, linestyle="--",
               label=fr"$I_t = {I_past[-1]:.3f}$ (valeur actuelle)")
    ax.scatter([T], [I_T_futures.mean()], color="#27ae60", s=200, zorder=10, marker="*",
               label=fr"$\mathbb{{E}}[I_T \mid \mathcal{{F}}_t] \approx {I_T_futures.mean():.3f}$ (moy. de 100 futurs)")

    ax.axvline(t_obs, color="gray", linestyle=":", alpha=0.6)
    ax.text(t_obs + 0.005, ax.get_ylim()[0] * 0.95, "$t$", fontsize=11)
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$I_s = \int_0^s W_u\,dW_u$")
    ax.set_title(r"Martingalité : $\mathbb{E}[I_T \mid \mathcal{F}_t] = I_t$" +
                 "\n(la meilleure prédiction du futur de l'intégrale, c'est sa valeur actuelle)",
                 fontsize=11)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "fig7_martingale.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 7 : martingale")


# =========================================================================
# Figure 8 : Variation quadratique
# =========================================================================
def fig_quadratic_variation():
    fig, ax = plt.subplots(figsize=(10, 5))

    T = 1.0
    n_steps = 5000
    _, W = brownian(T, n_steps, seed=99)
    dW = np.diff(W)
    H = W[:-1]
    dt = T / n_steps

    I_t = np.concatenate([[0], np.cumsum(H * dW)])
    dI = np.diff(I_t)
    quad_var_I = np.concatenate([[0], np.cumsum(dI**2)])
    int_H2 = np.concatenate([[0], np.cumsum(H**2 * dt)])

    t = np.linspace(0, T, n_steps + 1)

    ax.plot(t, quad_var_I, color="#9b59b6", lw=2,
            label=r"$\sum_i (\Delta I_i)^2$ — variation quadratique empirique")
    ax.plot(t, int_H2, color="black", lw=2, linestyle="--",
            label=r"$\int_0^t H_s^2\,ds = \int_0^t W_s^2\,ds$ — théorie")

    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$[I]_t$")
    ax.set_title(r"Variation quadratique de $I_t = \int_0^t W_s\,dW_s$ : empirique vs théorique",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig8_quad_var.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 8 : variation quadratique")


# =========================================================================
# Figure 9 : Exemple résolu — \int_0^t s dW_s
# =========================================================================
def fig_example_t_dW():
    """4 panneaux pour visualiser \int_0^t s\,dW_s :
       (a) la trajectoire brownienne W_s
       (b) le poids h(s) = s
       (c) les contributions s_i * dW_i (barres colorées)
       (d) le processus intégrale I_t accumulé + vérification par formule d'Itô
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    T = 1.0
    n_steps = 1000
    t, W = brownian(T, n_steps, seed=33)
    s = t  # alias pour l'integrand

    # --- Sous-grille pour visualiser les contributions ---
    n_sub = 25
    idx = np.linspace(0, n_steps, n_sub + 1, dtype=int)
    s_sub = t[idx]
    W_sub = W[idx]
    H_sub = s_sub  # ici H_si = s_i
    dW_sub = np.diff(W_sub)
    contributions = H_sub[:-1] * dW_sub

    # === (a) Trajectoire W_s ===
    ax = axes[0, 0]
    ax.plot(t, W, color="#2c3e50", lw=1.5)
    ax.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$W_s(\omega)$")
    ax.set_title(r"(a) Une trajectoire brownienne $W_s$", fontsize=11)

    # === (b) Poids h(s) = s ===
    ax = axes[0, 1]
    ax.plot(t, t, color="#9b59b6", lw=2)
    ax.fill_between(t, 0, t, color="#9b59b6", alpha=0.15)
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$H_s = s$")
    ax.set_title(r"(b) Le poids : $H_s = s$ — croît linéairement", fontsize=11)
    ax.text(0.5, 0.15, "Plus $s$ est grand,\nplus le saut $\\Delta W$\ncomptera lourd",
            ha="center", fontsize=10, color="#7d3c98",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4ecf7", edgecolor="#9b59b6"))

    # === (c) Contributions s_i * dW_i ===
    ax = axes[1, 0]
    bar_width = T / n_sub * 0.85
    colors = ["#27ae60" if c > 0 else "#c0392b" for c in contributions]
    ax.bar(s_sub[:-1] + bar_width / 2, contributions, width=bar_width,
           color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("$s_i$")
    ax.set_ylabel(r"$s_i \cdot \Delta W_i$")
    ax.set_title(r"(c) Contributions $s_i \cdot \Delta W_i$ — plus larges à droite (poids élevé)",
                 fontsize=11)

    # === (d) Intégrale accumulée + vérif Ito ===
    ax = axes[1, 1]
    # Calcul à résolution fine
    dW_fine = np.diff(W)
    H_fine = t[:-1]
    I_traj = np.concatenate([[0], np.cumsum(H_fine * dW_fine)])

    # Vérification : t W_t - \int_0^t W_s ds
    integral_W = np.concatenate([[0], np.cumsum(W[:-1] * (T / n_steps))])
    I_verify = t * W - integral_W

    ax.plot(t, I_traj, color="#2980b9", lw=2,
            label=r"$I_t = \int_0^t s\,dW_s$ (somme directe)")
    ax.plot(t, I_verify, color="#e67e22", lw=2, linestyle="--",
            label=r"$tW_t - \int_0^t W_s\,ds$ (formule d'Itô)")
    ax.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$I_t$")
    ax.set_title(r"(d) L'intégrale $I_t$ accumulée — vérifiée par la formule d'Itô",
                 fontsize=11)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        r"Exemple résolu : $\int_0^t s\,dW_s$ sur une trajectoire fixée $\omega$",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig9_example_t_dW.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 9 : exemple \\int s dW")


if __name__ == "__main__":
    fig_filtration()
    fig_adapted_vs_anticipating()
    fig_riemann_sum()
    fig_ito_vs_strato()
    fig_expectation_zero()
    fig_ito_isometry()
    fig_martingale()
    fig_quadratic_variation()
    fig_example_t_dW()
    print(f"\nFigures sauvegardées dans {OUT}")
