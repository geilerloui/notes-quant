"""
Génère les 6 figures pour la note Équations Différentielles Stochastiques.
À exécuter depuis ce dossier — les PNG seront créés dedans.

Usage : python generate_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "grid.linestyle": "-", "grid.linewidth": 0.6, "lines.linewidth": 1.2,
})

try:
    OUT = Path(__file__).parent
except NameError:
    OUT = Path.cwd()

C_BLUE   = "#2E5C8A"
C_RED    = "#B23A48"
C_GREEN  = "#4A7C59"
C_GREY   = "#8a8784"
C_ORANGE = "#D08C3F"


# ============================================================
# Fig 1 : ABM — trajectoires + gaussienne tournée
# ============================================================
def fig_abm_trajectoires():
    X0 = 7318.5
    mu_a = 581
    sigma_a = 860
    T = 50/252
    n_steps = 50
    M = 80
    dt = T / n_steps

    rng = np.random.default_rng(42)
    t = np.linspace(0, T, n_steps + 1)
    trajectories = np.zeros((M, n_steps + 1))
    trajectories[:, 0] = X0
    for i in range(n_steps):
        Z = rng.normal(0, 1, size=M)
        trajectories[:, i + 1] = trajectories[:, i] + mu_a * dt + sigma_a * np.sqrt(dt) * Z

    mean_T = X0 + mu_a * T
    std_T = sigma_a * np.sqrt(T)

    fig, (ax_traj, ax_dist) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        sharey=True
    )
    for i in range(M):
        ax_traj.plot(t, trajectories[i], color=C_BLUE, alpha=0.25, lw=0.7)
    ax_traj.plot(t, X0 + mu_a * t, color=C_RED, lw=1.5, label=r"moyenne $X_0 + \mu t$")
    ax_traj.axhline(X0, color="black", lw=0.5, alpha=0.4, ls="--")
    ax_traj.set_xlabel("$t$ (années)")
    ax_traj.set_ylabel("$X_t$")
    ax_traj.set_title(f"{M} trajectoires d'ABM sur 50 jours (FTSE 100, $X_0 = {X0}$)")
    ax_traj.legend(loc="upper left", fontsize=10, framealpha=0.9)

    y_range = np.linspace(mean_T - 4 * std_T, mean_T + 4 * std_T, 300)
    pdf = norm.pdf(y_range, mean_T, std_T)
    ax_dist.fill_betweenx(y_range, 0, pdf, color=C_RED, alpha=0.25)
    ax_dist.plot(pdf, y_range, color=C_RED, lw=1.5)
    ax_dist.axhline(mean_T, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_dist.set_xlabel("densité")
    ax_dist.set_xticks([])
    ax_dist.set_title(f"$X_T \\sim \\mathcal{{N}}({mean_T:.0f}, {std_T:.0f}^2)$", fontsize=10)
    ax_dist.annotate(
        f"$\\mu_T = {mean_T:.0f}$\n$\\sigma_T = {std_T:.0f}$",
        xy=(0.5, 0.05), xycoords="axes fraction",
        fontsize=10, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.9)
    )
    fig.suptitle(
        r"ABM : $dX_t = \mu\,dt + \sigma\,dW_t$ — trajectoires et distribution finale",
        y=1.02, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_eds_abm_trajectoires.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 2 : ABM — évolution de la distribution dans le temps
# ============================================================
def fig_abm_evolution():
    X0 = 7318.5
    mu_a = 581
    sigma_a = 860
    horizons = [0.25, 0.5, 0.75, 1.0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]
    labels_horizon = ["3 mois", "6 mois", "9 mois", "1 an"]
    x_range = np.linspace(X0 - 3000, X0 + 5000, 500)

    for T, color, lbl in zip(horizons, colors, labels_horizon):
        mean_T = X0 + mu_a * T
        std_T = sigma_a * np.sqrt(T)
        pdf = norm.pdf(x_range, mean_T, std_T)
        ax.plot(x_range, pdf, color=color, lw=1.5, label=f"{lbl} ($T = {T}$)")
        ax.fill_between(x_range, 0, pdf, color=color, alpha=0.10)
        ax.axvline(mean_T, color=color, lw=0.6, alpha=0.5, ls=":")

    ax.axvline(X0, color="black", lw=0.5, alpha=0.5, ls="--", label=f"$X_0 = {X0}$")
    ax.set_xlabel("$X_T$")
    ax.set_ylabel("densité")
    ax.set_title(r"ABM : évolution de la distribution $\mathcal{N}(X_0 + \mu T, \sigma^2 T)$ avec $T$")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eds_abm_evolution.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 3 : MBG — trajectoires + log-normale tournée
# ============================================================
def fig_mbg_trajectoires():
    X0 = 7318.5
    mu = 0.0865
    sigma2 = 0.013
    sigma = np.sqrt(sigma2)
    T = 50/252
    n_steps = 50
    M = 80
    dt = T / n_steps

    rng = np.random.default_rng(42)
    t = np.linspace(0, T, n_steps + 1)
    trajectories = np.zeros((M, n_steps + 1))
    trajectories[:, 0] = X0
    for i in range(n_steps):
        Z = rng.normal(0, 1, size=M)
        trajectories[:, i + 1] = trajectories[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    log_mean = np.log(X0) + (mu - 0.5 * sigma**2) * T
    log_std = sigma * np.sqrt(T)
    mean_T = X0 * np.exp(mu * T)

    fig, (ax_traj, ax_dist) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        sharey=True
    )
    for i in range(M):
        ax_traj.plot(t, trajectories[i], color=C_BLUE, alpha=0.25, lw=0.7)
    ax_traj.plot(t, X0 * np.exp(mu * t), color=C_RED, lw=1.5,
                 label=r"moyenne $X_0 e^{\mu t}$")
    ax_traj.axhline(X0, color="black", lw=0.5, alpha=0.4, ls="--")
    ax_traj.set_xlabel("$t$ (années)")
    ax_traj.set_ylabel("$X_t$")
    ax_traj.set_title(f"{M} trajectoires de MBG sur 50 jours (FTSE 100, $X_0 = {X0}$)")
    ax_traj.legend(loc="upper left", fontsize=10, framealpha=0.9)

    y_range = np.linspace(trajectories.min() * 0.98, trajectories.max() * 1.02, 400)
    pdf = lognorm.pdf(y_range, s=log_std, scale=np.exp(log_mean))
    ax_dist.fill_betweenx(y_range, 0, pdf, color=C_RED, alpha=0.25)
    ax_dist.plot(pdf, y_range, color=C_RED, lw=1.5)
    ax_dist.axhline(mean_T, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_dist.set_xlabel("densité")
    ax_dist.set_xticks([])
    ax_dist.set_title(r"$X_T \sim \mathcal{LN}$", fontsize=10)
    ax_dist.annotate(
        f"$\\mathbb{{E}}[X_T] = {mean_T:.0f}$",
        xy=(0.5, 0.05), xycoords="axes fraction",
        fontsize=10, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.9)
    )
    fig.suptitle(
        r"MBG : $dX_t = \mu X_t\,dt + \sigma X_t\,dW_t$ — trajectoires log-normales",
        y=1.02, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_eds_mbg_trajectoires.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 4 : MBG vs ABM côte à côte (positivité)
# ============================================================
def fig_mbg_vs_abm():
    X0 = 100.0
    T = 5.0
    n_steps = 1000
    M = 30
    dt = T / n_steps

    rng = np.random.default_rng(7)
    t = np.linspace(0, T, n_steps + 1)
    increments = rng.normal(0, np.sqrt(dt), size=(M, n_steps))

    mu_abm = 5
    sigma_abm = 30
    abm = np.zeros((M, n_steps + 1))
    abm[:, 0] = X0
    for i in range(n_steps):
        abm[:, i + 1] = abm[:, i] + mu_abm * dt + sigma_abm * increments[:, i]

    mu_mbg = 0.05
    sigma_mbg = 0.30
    mbg = np.zeros((M, n_steps + 1))
    mbg[:, 0] = X0
    for i in range(n_steps):
        mbg[:, i + 1] = mbg[:, i] * np.exp(
            (mu_mbg - 0.5 * sigma_mbg**2) * dt + sigma_mbg * increments[:, i]
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i in range(M):
        axes[0].plot(t, abm[i], color=C_BLUE, alpha=0.4, lw=0.8)
    axes[0].axhline(0, color=C_RED, lw=1.0, ls="--", label="zéro")
    axes[0].axhline(X0, color="black", lw=0.5, alpha=0.4, ls="--")
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$X_t$")
    axes[0].set_title(r"ABM : $X_t$ peut devenir négatif")
    axes[0].legend(loc="upper left", fontsize=9)

    for i in range(M):
        axes[1].plot(t, mbg[i], color=C_BLUE, alpha=0.4, lw=0.8)
    axes[1].axhline(0, color=C_RED, lw=1.0, ls="--", label="zéro")
    axes[1].axhline(X0, color="black", lw=0.5, alpha=0.4, ls="--")
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("$X_t$")
    axes[1].set_title(r"MBG : $X_t > 0$ toujours")
    axes[1].legend(loc="upper left", fontsize=9)

    fig.suptitle("Comparaison ABM vs MBG : positivité du processus",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eds_mbg_vs_abm.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 5 : OU — mean-reversion (effet de kappa)
# ============================================================
def fig_ou_kappa():
    X0 = 5.0
    theta = 0.0
    sigma = 1.0
    T = 5.0
    n_steps = 2000
    M = 5
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    kappas = [0.5, 2.0, 10.0]
    rng = np.random.default_rng(2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, kappa in zip(axes, kappas):
        for m in range(M):
            X = np.zeros(n_steps + 1)
            X[0] = X0
            increments = rng.normal(0, np.sqrt(dt), size=n_steps)
            for i in range(n_steps):
                X[i + 1] = X[i] + kappa * (theta - X[i]) * dt + sigma * increments[i]
            ax.plot(t, X, color=C_BLUE, alpha=0.6, lw=0.9)
        ax.axhline(theta, color=C_RED, lw=1.5, ls="--", label=fr"$\theta = {theta}$")
        ax.axhline(X0, color="black", lw=0.5, alpha=0.4, ls=":")
        ax.set_xlabel("$t$")
        H = np.log(2) / kappa
        ax.set_title(fr"$\kappa = {kappa}$  •  half-life = ${H:.2f}$")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    axes[0].set_ylabel("$X_t$")

    fig.suptitle(r"OU : effet de $\kappa$ sur la vitesse de mean-reversion vers $\theta$",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eds_ou_kappa.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 6 : OU — distribution stationnaire
# ============================================================
def fig_ou_stationnaire():
    X0 = 3.0
    theta = 0.0
    kappa = 1.0
    sigma = 1.0
    T = 10.0
    n_steps = 5000
    M = 100
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    rng = np.random.default_rng(3)
    trajectories = np.zeros((M, n_steps + 1))
    trajectories[:, 0] = X0
    for i in range(n_steps):
        Z = rng.normal(0, 1, size=M)
        trajectories[:, i + 1] = (
            trajectories[:, i] + kappa * (theta - trajectories[:, i]) * dt
            + sigma * np.sqrt(dt) * Z
        )

    var_stat = sigma**2 / (2 * kappa)
    std_stat = np.sqrt(var_stat)

    fig, (ax_traj, ax_dist) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05},
        sharey=True
    )
    for i in range(M):
        ax_traj.plot(t, trajectories[i], color=C_BLUE, alpha=0.15, lw=0.6)
    ax_traj.axhline(theta, color=C_RED, lw=1.5, ls="--", label=fr"$\theta = {theta}$")
    ax_traj.axhline(theta + std_stat, color=C_RED, lw=0.8, ls=":", alpha=0.7,
                    label=fr"$\theta \pm \sigma_\infty$")
    ax_traj.axhline(theta - std_stat, color=C_RED, lw=0.8, ls=":", alpha=0.7)
    ax_traj.axhline(theta + 2*std_stat, color=C_RED, lw=0.6, ls=":", alpha=0.5)
    ax_traj.axhline(theta - 2*std_stat, color=C_RED, lw=0.6, ls=":", alpha=0.5)
    ax_traj.set_xlabel("$t$")
    ax_traj.set_ylabel("$X_t$")
    ax_traj.set_title(fr"{M} trajectoires d'OU ($\kappa={kappa}$, $\theta={theta}$, $\sigma={sigma}$)")
    ax_traj.legend(loc="upper right", fontsize=9, framealpha=0.9)

    y_range = np.linspace(-4, 4, 300)
    pdf = norm.pdf(y_range, theta, std_stat)
    ax_dist.fill_betweenx(y_range, 0, pdf, color=C_RED, alpha=0.25)
    ax_dist.plot(pdf, y_range, color=C_RED, lw=1.5)
    ax_dist.axhline(theta, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax_dist.set_xlabel("densité")
    ax_dist.set_xticks([])
    ax_dist.set_title(r"$X_\infty \sim \mathcal{N}\left(\theta, \frac{\sigma^2}{2\kappa}\right)$",
                      fontsize=10)
    ax_dist.annotate(
        fr"$\sigma_\infty = {std_stat:.2f}$",
        xy=(0.5, 0.05), xycoords="axes fraction",
        fontsize=10, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.9)
    )

    fig.suptitle(r"OU : distribution stationnaire $\mathcal{N}(\theta, \sigma^2/(2\kappa))$",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_eds_ou_stationnaire.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_abm_trajectoires()
    print("✓ ABM trajectoires")
    fig_abm_evolution()
    print("✓ ABM évolution")
    fig_mbg_trajectoires()
    print("✓ MBG trajectoires")
    fig_mbg_vs_abm()
    print("✓ MBG vs ABM")
    fig_ou_kappa()
    print("✓ OU kappa")
    fig_ou_stationnaire()
    print("✓ OU stationnaire")
    print(f"\nFigures sauvegardées dans {OUT}")
