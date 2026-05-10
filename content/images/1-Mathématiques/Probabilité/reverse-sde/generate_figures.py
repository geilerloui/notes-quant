"""
Génère les figures pour la note Reverse-time SDE et Fokker-Planck.

Usage : python generate_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
# Fig 1 — Fokker-Planck : densité d'un OU vers gaussienne stationnaire
# ============================================================
def fig_fokker_planck_ou():
    theta = 0.0
    kappa = 1.0
    sigma = 1.0
    X0 = 3.0

    times = [0.05, 0.3, 1.0, 3.0]
    var_stat = sigma**2 / (2 * kappa)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_range = np.linspace(-3, 5, 500)
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]

    for t, col in zip(times, colors):
        m_t = X0 * np.exp(-kappa * t) + theta * (1 - np.exp(-kappa * t))
        v_t = var_stat * (1 - np.exp(-2 * kappa * t))
        s_t = np.sqrt(v_t)
        pdf = norm.pdf(x_range, m_t, s_t)
        ax.plot(x_range, pdf, color=col, lw=1.8, label=f"$t = {t}$")
        ax.fill_between(x_range, 0, pdf, color=col, alpha=0.12)

    pdf_stat = norm.pdf(x_range, theta, np.sqrt(var_stat))
    ax.plot(x_range, pdf_stat, color="black", lw=1.5, ls="--",
            label=r"stationnaire $\mathcal{N}(\theta, \sigma^2/(2\kappa))$")

    ax.axvline(X0, color=C_GREY, lw=0.6, alpha=0.5, ls=":", label=f"$X_0 = {X0}$")
    ax.axvline(theta, color="black", lw=0.6, alpha=0.4, ls=":")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$p_t(x)$")
    ax.set_title(r"Fokker-Planck pour OU : la densité $p_t(x)$ converge vers la gaussienne stationnaire")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rev_fokker_planck_ou.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 2 — Score function
# ============================================================
def fig_score_function():
    x_range = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    mu, s = 0.0, 1.0
    pdf = norm.pdf(x_range, mu, s)
    score = -(x_range - mu) / s**2

    axes[0, 0].plot(x_range, pdf, color=C_BLUE, lw=1.8)
    axes[0, 0].fill_between(x_range, 0, pdf, color=C_BLUE, alpha=0.15)
    axes[0, 0].axvline(mu, color=C_RED, lw=1.0, ls="--", alpha=0.6, label=f"mode $\\mu = {mu}$")
    axes[0, 0].set_title(r"(a) Densité $p(x) = \mathcal{N}(0, 1)$")
    axes[0, 0].set_ylabel(r"$p(x)$")
    axes[0, 0].legend(fontsize=9, loc="upper right")

    axes[0, 1].plot(x_range, score, color=C_RED, lw=1.8)
    axes[0, 1].axhline(0, color="black", lw=0.5, alpha=0.4)
    axes[0, 1].axvline(mu, color=C_RED, lw=1.0, ls="--", alpha=0.6)
    axes[0, 1].set_title(r"(b) Score $\nabla \log p(x) = -(x - \mu)/\sigma^2$")
    axes[0, 1].set_ylabel(r"$\nabla \log p(x)$")

    mus = [-2.0, 2.0]
    ss = [0.7, 0.7]
    weights = [0.5, 0.5]
    pdf_mix = sum(w * norm.pdf(x_range, m, s) for w, m, s in zip(weights, mus, ss))
    score_mix = np.zeros_like(x_range)
    for w, m, s in zip(weights, mus, ss):
        score_mix += w * norm.pdf(x_range, m, s) * (-(x_range - m) / s**2)
    score_mix /= pdf_mix

    axes[1, 0].plot(x_range, pdf_mix, color=C_BLUE, lw=1.8)
    axes[1, 0].fill_between(x_range, 0, pdf_mix, color=C_BLUE, alpha=0.15)
    for m in mus:
        axes[1, 0].axvline(m, color=C_RED, lw=1.0, ls="--", alpha=0.6)
    axes[1, 0].set_title(r"(c) Densité bimodale (mélange de 2 gaussiennes)")
    axes[1, 0].set_xlabel("$x$")
    axes[1, 0].set_ylabel(r"$p(x)$")

    axes[1, 1].plot(x_range, score_mix, color=C_RED, lw=1.8)
    axes[1, 1].axhline(0, color="black", lw=0.5, alpha=0.4)
    for m in mus:
        axes[1, 1].axvline(m, color=C_RED, lw=1.0, ls="--", alpha=0.6)
    axes[1, 1].set_title(r"(d) Score : positif vers la gauche, négatif vers la droite")
    axes[1, 1].set_xlabel("$x$")
    axes[1, 1].set_ylabel(r"$\nabla \log p(x)$")

    fig.suptitle(r"Le score $\nabla \log p(x)$ : une 'force' qui pointe vers les modes de la densité",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rev_score.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 3 — Forward + Reverse
# ============================================================
def fig_forward_reverse():
    rng = np.random.default_rng(42)
    n_samples = 2000
    samples_a = rng.normal(-2, 0.5, size=n_samples // 2)
    samples_b = rng.normal(2, 0.5, size=n_samples // 2)
    X0 = np.concatenate([samples_a, samples_b])

    T = 4.0
    n_steps = 400
    dt = T / n_steps

    X_forward = np.zeros((n_samples, n_steps + 1))
    X_forward[:, 0] = X0
    for i in range(n_steps):
        Z = rng.normal(0, 1, size=n_samples)
        X_forward[:, i + 1] = X_forward[:, i] - 0.5 * X_forward[:, i] * dt + np.sqrt(dt) * Z

    times_to_show = [0, 0.5, 1.5, 4.0]
    idx_to_show = [int(t / dt) for t in times_to_show]

    fig, axes = plt.subplots(2, len(times_to_show), figsize=(13, 5.5), sharex=True)

    for j, (t, idx) in enumerate(zip(times_to_show, idx_to_show)):
        ax = axes[0, j]
        ax.hist(X_forward[:, idx], bins=50, density=True, color=C_BLUE, alpha=0.5,
                edgecolor="none")
        ax.set_title(f"$t = {t:.1f}$")
        ax.set_xlim(-5, 5)
        if j == 0:
            ax.set_ylabel("Forward\n$p_t(x)$", fontsize=11)

    for j, (t, idx) in enumerate(zip(times_to_show, idx_to_show)):
        ax = axes[1, j]
        idx_rev = n_steps - idx
        ax.hist(X_forward[:, idx_rev], bins=50, density=True, color=C_RED, alpha=0.5,
                edgecolor="none")
        ax.set_title(f"$\\tau = {t:.1f}$ (= $T - t$)")
        ax.set_xlim(-5, 5)
        ax.set_xlabel("$x$")
        if j == 0:
            ax.set_ylabel("Reverse\n$p_{T-\\tau}(x)$", fontsize=11)

    fig.text(0.5, 0.97, "Forward SDE : on bruite (bimodale → gaussienne)",
             ha="center", fontsize=12, color=C_BLUE, fontweight="bold")
    fig.text(0.5, 0.49, "Reverse-time SDE : on dé-bruite (gaussienne → bimodale)",
             ha="center", fontsize=12, color=C_RED, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_rev_forward_reverse.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 4 — Trajectoires forward et reverse
# ============================================================
def fig_trajectoires_reverse():
    rng = np.random.default_rng(7)
    n_samples = 8
    init = np.concatenate([rng.normal(-2, 0.4, n_samples // 2),
                           rng.normal(2, 0.4, n_samples // 2)])

    T = 5.0
    n_steps = 500
    dt = T / n_steps
    t_arr = np.linspace(0, T, n_steps + 1)

    X_fwd = np.zeros((n_samples, n_steps + 1))
    X_fwd[:, 0] = init
    for i in range(n_steps):
        Z = rng.normal(0, 1, size=n_samples)
        X_fwd[:, i + 1] = X_fwd[:, i] - 0.5 * X_fwd[:, i] * dt + np.sqrt(dt) * Z

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for i in range(n_samples):
        axes[0].plot(t_arr, X_fwd[i], color=C_BLUE, alpha=0.6, lw=0.9)
    axes[0].axhline(0, color="black", lw=0.4, alpha=0.4, ls="--")
    axes[0].set_xlabel("$t$ (forward)")
    axes[0].set_ylabel("$X_t$")
    axes[0].set_title("Forward SDE : trajectoires partent\nde la bimodale et finissent gaussiennes")

    for i in range(n_samples):
        axes[1].plot(T - t_arr, X_fwd[i], color=C_RED, alpha=0.6, lw=0.9)
    axes[1].axhline(0, color="black", lw=0.4, alpha=0.4, ls="--")
    axes[1].set_xlabel(r"$\tau = T - t$ (reverse)")
    axes[1].set_title("Reverse-time SDE : on parcourt\nles mêmes trajectoires en sens inverse")
    axes[1].invert_xaxis()

    fig.suptitle("Forward et reverse : mêmes trajectoires, lues dans des sens opposés",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rev_trajectoires.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Fig 5 — Score guides sampling (Langevin)
# ============================================================
def fig_score_guides_sampling():
    rng = np.random.default_rng(3)
    mus = [-2.0, 2.0]
    ss = [0.5, 0.5]

    def target_pdf(x):
        return 0.5 * norm.pdf(x, mus[0], ss[0]) + 0.5 * norm.pdf(x, mus[1], ss[1])

    def target_score(x):
        p = target_pdf(x)
        s = 0.5 * norm.pdf(x, mus[0], ss[0]) * (-(x - mus[0]) / ss[0]**2)
        s += 0.5 * norm.pdf(x, mus[1], ss[1]) * (-(x - mus[1]) / ss[1]**2)
        return s / p

    n_samples = 800
    n_steps = 300
    eps = 0.02

    X = rng.normal(0, 3, size=n_samples)
    history = [X.copy()]
    snap_steps = [0, 30, 100, 300]

    for k in range(n_steps):
        score = target_score(X)
        Z = rng.normal(0, 1, size=n_samples)
        X = X + eps * score + np.sqrt(2 * eps) * Z
        if (k + 1) in snap_steps:
            history.append(X.copy())

    fig, axes = plt.subplots(1, len(snap_steps), figsize=(13, 3.5), sharex=True)
    x_range = np.linspace(-5, 5, 400)
    pdf_target = target_pdf(x_range)

    labels = ["initial (bruit)", f"étape {snap_steps[1]}",
              f"étape {snap_steps[2]}", f"étape {snap_steps[3]}"]
    colors_h = [C_GREY, C_ORANGE, C_GREEN, C_RED]

    for j, (h, lbl, col) in enumerate(zip(history, labels, colors_h)):
        ax = axes[j]
        ax.hist(h, bins=50, density=True, color=col, alpha=0.5, edgecolor="none")
        ax.plot(x_range, pdf_target, color="black", lw=1.2, ls="--", alpha=0.7,
                label="cible")
        ax.set_title(lbl)
        ax.set_xlabel("$x$")
        ax.set_xlim(-5, 5)
        if j == 0:
            ax.set_ylabel("densité")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(r"Sampling guidé par le score : on part du bruit, le score tire vers les modes de la cible",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rev_score_sampling.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_fokker_planck_ou()
    print("✓ Fokker-Planck OU")
    fig_score_function()
    print("✓ Score function")
    fig_forward_reverse()
    print("✓ Forward + Reverse")
    fig_trajectoires_reverse()
    print("✓ Trajectoires")
    fig_score_guides_sampling()
    print("✓ Score sampling")
    print(f"\nFigures sauvegardées dans {OUT}")
