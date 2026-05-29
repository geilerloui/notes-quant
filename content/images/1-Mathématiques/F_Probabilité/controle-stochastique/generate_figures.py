"""
Génère 4 figures pour la note Contrôle Stochastique.

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


# =========================================================================
# Figure 1 : Setup générique - contrôler = modifier l'EDS
# =========================================================================
def fig_setup_generic():
    """Système 1D avec contrôle. On compare :
    - Pas de contrôle (alpha = 0) : système libre
    - Contrôle constant alpha = -0.5 : tire vers le bas
    - Contrôle feedback alpha(x) = -x : ramène vers 0
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    T = 2.0
    n_steps = 1000
    dt = T / n_steps
    n_traj = 30
    sigma = 0.5

    rng = np.random.default_rng(42)
    # Même bruit pour toutes les politiques pour comparaison équitable
    noise = rng.standard_normal((n_traj, n_steps))
    t = np.linspace(0, T, n_steps + 1)

    def simulate(control_fn):
        X = np.zeros((n_traj, n_steps + 1))
        X[:, 0] = 1.0  # condition initiale
        for k in range(n_steps):
            x_curr = X[:, k]
            alpha = control_fn(x_curr)
            X[:, k+1] = x_curr + alpha * dt + sigma * np.sqrt(dt) * noise[:, k]
        return X

    # Politique 1 : pas de contrôle
    X_no = simulate(lambda x: np.zeros_like(x))
    # Politique 2 : contrôle constant
    X_const = simulate(lambda x: -0.5 * np.ones_like(x))
    # Politique 3 : contrôle feedback
    X_feedback = simulate(lambda x: -2.0 * x)

    titles = [
        r"$\alpha = 0$ (pas de contrôle)",
        r"$\alpha = -0.5$ (contrôle constant)",
        r"$\alpha(x) = -2x$ (contrôle feedback)"
    ]
    Xs = [X_no, X_const, X_feedback]
    colors = ["#7f8c8d", "#e67e22", "#27ae60"]

    for ax, X, title, color in zip(axes, Xs, titles, colors):
        for i in range(n_traj):
            ax.plot(t, X[i], color=color, lw=0.5, alpha=0.4)
        # Moyenne
        ax.plot(t, X.mean(axis=0), color=color, lw=2.5, label="moyenne")
        ax.axhline(0, color="black", lw=0.4, alpha=0.5, linestyle="--")
        ax.set_xlabel("$t$")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, loc="upper right")

    axes[0].set_ylabel(r"$X_t$")

    fig.suptitle(
        r"Contrôler une EDS = choisir $\alpha_t$ qui modifie le drift : $dX_t = \alpha_t\,dt + \sigma\,dW_t$",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_setup.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 1 : setup générique")


# =========================================================================
# Figure 2 : Merton — allocation de portefeuille
# =========================================================================
def fig_merton():
    """3 allocations : sous-investi, optimal, sur-investi.
    Montre que l'optimal n'est pas trivial."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Paramètres
    T = 5.0
    n_steps = 500
    dt = T / n_steps
    n_traj = 200
    mu = 0.10  # rendement actif risqué
    r = 0.02   # taux sans risque
    sigma_S = 0.20  # volatilité actif
    gamma = 3.0  # aversion au risque (CRRA)

    # Ratio de Merton optimal
    alpha_optimal = (mu - r) / (sigma_S**2 * gamma)
    alphas = [0.2 * alpha_optimal, alpha_optimal, 2.5 * alpha_optimal]
    labels = [
        f"Sous-investi : $\\alpha = {alphas[0]:.2f}$",
        f"Optimal Merton : $\\alpha^* = {alphas[1]:.2f}$",
        f"Sur-investi : $\\alpha = {alphas[2]:.2f}$"
    ]
    colors = ["#3498db", "#27ae60", "#c0392b"]

    rng = np.random.default_rng(2025)
    noise = rng.standard_normal((n_traj, n_steps))
    t = np.linspace(0, T, n_steps + 1)

    def simulate_wealth(alpha):
        W = np.zeros((n_traj, n_steps + 1))
        W[:, 0] = 1.0
        for k in range(n_steps):
            # dW/W = (r + alpha(mu-r)) dt + alpha sigma dW
            drift = (r + alpha * (mu - r)) * dt
            diffusion = alpha * sigma_S * np.sqrt(dt) * noise[:, k]
            W[:, k+1] = W[:, k] * (1 + drift + diffusion)
        return W

    # === Gauche : trajectoires ===
    ax = axes[0]
    for alpha, label, color in zip(alphas, labels, colors):
        W = simulate_wealth(alpha)
        # Quelques trajectoires
        for i in range(15):
            ax.plot(t, W[i], color=color, lw=0.5, alpha=0.25)
        # Médiane
        ax.plot(t, np.median(W, axis=0), color=color, lw=2.5, label=label)

    ax.set_xlabel("$t$ (années)")
    ax.set_ylabel(r"Richesse $W_t$")
    ax.set_title("Trajectoires de richesse pour 3 allocations", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_yscale("log")

    # === Droite : utilité espérée en fonction de alpha ===
    ax = axes[1]
    alpha_grid = np.linspace(0, 3 * alpha_optimal, 30)
    expected_utility = []
    for alpha in alpha_grid:
        W = simulate_wealth(alpha)
        # Utilité CRRA : U(W) = W^(1-gamma) / (1-gamma)
        U = W[:, -1]**(1 - gamma) / (1 - gamma)
        expected_utility.append(U.mean())

    ax.plot(alpha_grid, expected_utility, color="#2c3e50", lw=2)
    ax.axvline(alpha_optimal, color="#27ae60", lw=2, linestyle="--",
               label=fr"$\alpha^* = (\mu - r)/(\sigma^2\gamma) = {alpha_optimal:.2f}$")
    ax.set_xlabel(r"Allocation $\alpha$ (fraction en actif risqué)")
    ax.set_ylabel(r"$\mathbb{E}[U(W_T)]$ (utilité espérée)")
    ax.set_title("Utilité espérée — maximum atteint en $\\alpha^*$ Merton",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        r"Problème de Merton : maximiser $\mathbb{E}[U(W_T)]$ avec $U(W) = W^{1-\gamma}/(1-\gamma)$, $\gamma = 3$",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_merton.png", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Fig 2 : Merton (alpha* = {alpha_optimal:.3f})")


# =========================================================================
# Figure 3 : Optimal execution — Almgren-Chriss
# =========================================================================
def fig_optimal_execution():
    """3 schedules : naïf (tout au début), uniforme, optimal."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Paramètres
    T = 1.0  # horizon (1 jour de trading)
    X0 = 100  # quantité à liquider
    n_steps = 100
    t = np.linspace(0, T, n_steps + 1)

    # Coût d'impact temporaire : eta (per share squared)
    # Coût de risque (variance du prix) : lambda * sigma^2
    eta = 0.5
    sigma = 0.3
    lam = 5.0

    # Schedule optimal Almgren-Chriss : décroissance exponentielle
    # x(t) = X0 * sinh(kappa(T-t)) / sinh(kappa T)
    # avec kappa = sqrt(lambda*sigma^2 / eta)
    kappa = np.sqrt(lam * sigma**2 / eta)
    x_optimal = X0 * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    x_uniform = X0 * (1 - t / T)
    x_naive = np.where(t < 0.1, X0 * (1 - t / 0.1), 0)

    # === Gauche : schedules ===
    ax = axes[0]
    ax.plot(t, x_naive, color="#c0392b", lw=2.5,
            label="Naïf : tout vendu en 10% du temps")
    ax.plot(t, x_uniform, color="#3498db", lw=2.5, label="Uniforme : $X_0(1 - t/T)$")
    ax.plot(t, x_optimal, color="#27ae60", lw=2.5,
            label=fr"Optimal Almgren-Chriss ($\kappa = {kappa:.2f}$)")
    ax.set_xlabel("$t$ (jour)")
    ax.set_ylabel(r"$X_t$ (shares restantes à liquider)")
    ax.set_title("Schedules de liquidation : combien de shares restent à vendre",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")

    # === Droite : coûts totaux ===
    ax = axes[1]
    # On calcule pour différentes valeurs de lambda (aversion au risque)
    lams_grid = np.linspace(0.1, 50, 50)
    costs_optimal = []
    costs_uniform = []
    costs_naive = []

    for lam_val in lams_grid:
        kappa_val = np.sqrt(lam_val * sigma**2 / eta)
        # Schedules
        x_opt = X0 * np.sinh(kappa_val * (T - t)) / np.sinh(kappa_val * T)
        x_uni = X0 * (1 - t / T)
        x_nai = np.where(t < 0.1, X0 * (1 - t / 0.1), 0)
        # Vitesses de vente (taux de liquidation)
        v_opt = -np.gradient(x_opt, t)
        v_uni = -np.gradient(x_uni, t)
        v_nai = -np.gradient(x_nai, t)
        # Coût total = coût impact + coût risque
        # Impact : eta * integral(v^2 dt)
        # Risque : lambda * sigma^2 * integral(x^2 dt)
        impact_opt = eta * np.trapz(v_opt**2, t)
        risk_opt = lam_val * sigma**2 * np.trapz(x_opt**2, t)
        costs_optimal.append(impact_opt + risk_opt)

        impact_uni = eta * np.trapz(v_uni**2, t)
        risk_uni = lam_val * sigma**2 * np.trapz(x_uni**2, t)
        costs_uniform.append(impact_uni + risk_uni)

        impact_nai = eta * np.trapz(v_nai**2, t)
        risk_nai = lam_val * sigma**2 * np.trapz(x_nai**2, t)
        costs_naive.append(impact_nai + risk_nai)

    ax.plot(lams_grid, costs_naive, color="#c0392b", lw=2, label="Naïf")
    ax.plot(lams_grid, costs_uniform, color="#3498db", lw=2, label="Uniforme")
    ax.plot(lams_grid, costs_optimal, color="#27ae60", lw=2, label="Optimal HJB")
    ax.set_xlabel(r"$\lambda$ (aversion au risque de prix)")
    ax.set_ylabel(r"Coût total : impact + risque")
    ax.set_title(r"Coût total selon l'aversion $\lambda$ — l'optimal HJB domine",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_yscale("log")

    fig.suptitle(
        r"Optimal execution (Almgren-Chriss) : minimiser impact d'exécution + risque de prix",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_execution.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 3 : optimal execution")


# =========================================================================
# Figure 4 : Dualité HJB / Fokker-Planck — schéma conceptuel
# =========================================================================
def fig_duality():
    """Schéma : HJB et Fokker-Planck partagent le même générateur,
    mais résolvent des problèmes différents."""
    fig, ax = plt.subplots(figsize=(13, 7))

    # Désactiver les axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # === Centre : générateur infinitésimal ===
    ax.add_patch(plt.Rectangle((3.5, 3.5), 3, 1, facecolor="#f4ecf7",
                                edgecolor="#8e44ad", linewidth=2))
    ax.text(5, 4, r"Générateur $\mathcal{L} = \mu \partial_x + \frac{1}{2}\sigma^2 \partial_{xx}$",
            ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(5, 3.2, "(la même opération appliquée à des objets différents)",
            ha="center", va="center", fontsize=10, style="italic", color="#7d3c98")

    # === Haut gauche : Fokker-Planck (densité) ===
    ax.add_patch(plt.Rectangle((0.5, 5.5), 4, 2, facecolor="#d5f5e3",
                                edgecolor="#27ae60", linewidth=2))
    ax.text(2.5, 7.2, "Fokker-Planck (forward)",
            ha="center", fontsize=12, fontweight="bold", color="#196f3d")
    ax.text(2.5, 6.5, r"$\partial_t p_t = \mathcal{L}^* p_t$",
            ha="center", fontsize=12)
    ax.text(2.5, 6, "Comment évolue la densité $p_t(x)$ ?",
            ha="center", fontsize=10, color="#196f3d")
    ax.text(2.5, 5.7, "(point de vue distribution)",
            ha="center", fontsize=9, style="italic", color="#196f3d")

    # === Haut droite : HJB (valeur) ===
    ax.add_patch(plt.Rectangle((5.5, 5.5), 4, 2, facecolor="#fadbd8",
                                edgecolor="#c0392b", linewidth=2))
    ax.text(7.5, 7.2, "Hamilton-Jacobi-Bellman (backward)",
            ha="center", fontsize=12, fontweight="bold", color="#922b21")
    ax.text(7.5, 6.5, r"$\partial_t V + \sup_\alpha \{ \mathcal{L}^\alpha V + f \} = 0$",
            ha="center", fontsize=12)
    ax.text(7.5, 6, "Comment évolue la fonction valeur $V(t,x)$ ?",
            ha="center", fontsize=10, color="#922b21")
    ax.text(7.5, 5.7, "(point de vue optimisation)",
            ha="center", fontsize=9, style="italic", color="#922b21")

    # === Bas : applications ===
    ax.add_patch(plt.Rectangle((0.5, 0.5), 4, 2.3, facecolor="#ebf5fb",
                                edgecolor="#2980b9", linewidth=1.5))
    ax.text(2.5, 2.5, "Applications Fokker-Planck", ha="center", fontsize=11,
            fontweight="bold", color="#21618c")
    ax.text(2.5, 1.9, r"• Sampling (Langevin, [[04_Dynamique de Langevin]])", ha="center", fontsize=9)
    ax.text(2.5, 1.5, "• Reverse-time SDE (diffusion models)", ha="center", fontsize=9)
    ax.text(2.5, 1.1, "• Distribution stationnaire", ha="center", fontsize=9)
    ax.text(2.5, 0.7, r"• Évolution de probabilités $p_t$", ha="center", fontsize=9)

    ax.add_patch(plt.Rectangle((5.5, 0.5), 4, 2.3, facecolor="#fef5e7",
                                edgecolor="#d68910", linewidth=1.5))
    ax.text(7.5, 2.5, "Applications HJB", ha="center", fontsize=11,
            fontweight="bold", color="#9c640c")
    ax.text(7.5, 1.9, "• Allocation Merton (finance)", ha="center", fontsize=9)
    ax.text(7.5, 1.5, "• Optimal execution (trading)", ha="center", fontsize=9)
    ax.text(7.5, 1.1, "• Smart grid, demand response (énergie)", ha="center", fontsize=9)
    ax.text(7.5, 0.7, "• Reinforcement Learning (RL continu)", ha="center", fontsize=9)

    # Flèches du générateur vers FP et HJB
    ax.annotate("", xy=(2.5, 5.5), xytext=(4, 4.3),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
    ax.annotate("", xy=(7.5, 5.5), xytext=(6, 4.3),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
    # Flèches FP -> appli, HJB -> appli
    ax.annotate("", xy=(2.5, 2.8), xytext=(2.5, 5.3),
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5))
    ax.annotate("", xy=(7.5, 2.8), xytext=(7.5, 5.3),
                arrowprops=dict(arrowstyle="->", color="#d68910", lw=1.5))

    fig.suptitle(
        r"Dualité Fokker-Planck $\leftrightarrow$ HJB : le même générateur, deux questions différentes",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_duality.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 4 : dualité HJB / Fokker-Planck")


if __name__ == "__main__":
    fig_setup_generic()
    fig_merton()
    fig_optimal_execution()
    fig_duality()
    print(f"\nFigures sauvegardées dans {OUT}")
