"""
Génère 6 figures pour la note Dynamique de Langevin.

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
# Helpers
# =========================================================================
def double_well_U(x):
    """Potentiel double-puits : U(x) = (x^2 - 1)^2."""
    return (x**2 - 1)**2


def double_well_grad(x):
    """Gradient : ∇U(x) = 4x(x^2 - 1)."""
    return 4 * x * (x**2 - 1)


def quadratic_U(x):
    """Potentiel quadratique : U(x) = x^2 / 2."""
    return x**2 / 2


def quadratic_grad(x):
    """Gradient : ∇U(x) = x."""
    return x


def langevin_step(x, grad_U, h, rng):
    """Un pas d'Euler-Maruyama pour Langevin.
    
    dX_t = -∇U(X_t) dt + sqrt(2) dW_t
    
    Discrétisé : X_{k+1} = X_k - h ∇U(X_k) + sqrt(2h) ε_k
    """
    return x - h * grad_U(x) + np.sqrt(2 * h) * rng.standard_normal(size=np.shape(x))


def langevin_trajectory(x0, grad_U, h, n_steps, seed=None):
    """Génère une trajectoire de Langevin de longueur n_steps."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps + 1)
    x[0] = x0
    for k in range(n_steps):
        x[k+1] = langevin_step(x[k], grad_U, h, rng)
    return x


# =========================================================================
# Figure 1 : Potentiel double-puits + trajectoire Langevin
# =========================================================================
def fig_double_well():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1, 1.3])

    # === Haut : le potentiel U(x) ===
    ax = axes[0]
    x_grid = np.linspace(-2, 2, 500)
    U_grid = double_well_U(x_grid)
    ax.plot(x_grid, U_grid, color="#2c3e50", lw=2)
    ax.fill_between(x_grid, 0, U_grid, color="#3498db", alpha=0.15)
    
    ax.scatter([-1, 1], [0, 0], color="#27ae60", s=100, zorder=5, edgecolor="black",
               label=r"Minima : $x = \pm 1$")
    ax.scatter([0], [1], color="#c0392b", s=100, zorder=5, edgecolor="black",
               label=r"Col (barrière) : $x = 0$, $U(0) = 1$")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$U(x) = (x^2 - 1)^2$")
    ax.set_title("Le potentiel double-puits — la particule veut tomber dans un des deux minima")
    ax.legend(loc="upper center")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.3, 4.5)

    # === Bas : trajectoire Langevin x(t) ===
    ax = axes[1]
    x_traj = langevin_trajectory(x0=-1.0, grad_U=double_well_grad,
                                  h=0.001, n_steps=20000, seed=42)
    t = np.linspace(0, 20, len(x_traj))
    
    ax.plot(t, x_traj, color="#2c3e50", lw=0.8, alpha=0.85)
    ax.axhline(-1, color="#27ae60", linestyle="--", lw=1, alpha=0.6)
    ax.axhline(1, color="#27ae60", linestyle="--", lw=1, alpha=0.6)
    ax.axhline(0, color="#c0392b", linestyle=":", lw=1, alpha=0.5)
    ax.text(20.3, -1, "$x = -1$", color="#27ae60", va="center", fontsize=10)
    ax.text(20.3, 1, "$x = +1$", color="#27ae60", va="center", fontsize=10)
    ax.text(20.3, 0, "barrière", color="#c0392b", va="center", fontsize=10)
    
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$X_t$")
    ax.set_title(r"Trajectoire Langevin partant de $X_0 = -1$ — la particule oscille puis saute par-dessus la barrière")
    ax.set_xlim(0, 21.5)

    fig.suptitle(
        r"Dynamique de Langevin sur un potentiel double-puits : $dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t$",
        fontsize=12, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_double_well.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 1 : double well")


# =========================================================================
# Figure 2 : Convergence vers la distribution stationnaire
# =========================================================================
def fig_stationary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    n_traj = 2000
    h = 0.01
    rng = np.random.default_rng(123)
    
    times_obs = [10, 100, 2000]
    titles = [r"$t \approx 0.1$ (loin de stationnaire)",
              r"$t \approx 1$ (en chemin)",
              r"$t \approx 20$ (proche stationnaire)"]
    
    x_grid = np.linspace(-2.5, 2.5, 300)
    pi_unnorm = np.exp(-double_well_U(x_grid))
    Z = np.trapz(pi_unnorm, x_grid)
    pi_norm = pi_unnorm / Z
    
    X = np.full(n_traj, -1.5)
    
    step_count = 0
    for ax, t_obs, title in zip(axes, times_obs, titles):
        while step_count < t_obs:
            X = langevin_step(X, double_well_grad, h, rng)
            step_count += 1
        
        ax.hist(X, bins=50, density=True, color="#3498db", alpha=0.6,
                edgecolor="black", linewidth=0.4, label="Empirique (2000 traj.)")
        ax.plot(x_grid, pi_norm, color="#c0392b", lw=2.5,
                label=r"$\pi(x) \propto e^{-U(x)}$")
        ax.set_xlabel("$x$")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-2.5, 2.5)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("densité")
    fig.suptitle(
        r"Convergence vers la distribution stationnaire $\pi(x) \propto e^{-U(x)}$",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_stationary.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 2 : convergence vers stationnaire")


# =========================================================================
# Figure 3 : Cas quadratique = OU
# =========================================================================
def fig_quadratic_OU():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x_grid = np.linspace(-3, 3, 200)
    U_grid = quadratic_U(x_grid)
    ax.plot(x_grid, U_grid, color="#2c3e50", lw=2,
            label=r"$U(x) = x^2/2$")
    ax.fill_between(x_grid, 0, U_grid, color="#9b59b6", alpha=0.15)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$U(x)$")
    ax.set_title(r"Potentiel quadratique : $U(x) = x^2/2$ ⟹ Langevin = OU",
                 fontsize=11)
    ax.legend()
    ax.set_xlim(-3, 3)

    ax = axes[1]
    n_steps = 2000
    h = 0.005
    t = np.linspace(0, n_steps * h, n_steps + 1)
    
    for k, x0 in enumerate([-2.5, 2.0, 1.5, -1.0, 0.5]):
        x_traj = langevin_trajectory(x0=x0, grad_U=quadratic_grad,
                                      h=h, n_steps=n_steps, seed=10+k)
        ax.plot(t, x_traj, lw=0.8, alpha=0.7,
                label=fr"$X_0 = {x0}$" if k < 3 else None)
    
    ax.axhline(0, color="black", linestyle="--", lw=1.2, alpha=0.5,
               label=r"$\mathbb{E}[X_t] \to 0$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$X_t$")
    ax.set_title(r"Trajectoires : retour à la moyenne",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, t[-1])

    fig.suptitle(
        r"Cas particulier : $U(x) = x^2/2$ ⟹ $dX_t = -X_t\,dt + \sqrt{2}\,dW_t$ (OU avec $\theta=1$, $\sigma=\sqrt{2}$)",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_quadratic_OU.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 3 : quadratique = OU")


# =========================================================================
# Figure 4 : Échantillonnage 2D mixture gaussienne
# =========================================================================
def fig_2d_sampling():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    centers = np.array([[-1.5, -1.0], [1.5, -1.0], [0.0, 1.5]])
    weights = np.array([0.4, 0.4, 0.2])
    var = 0.3
    
    def grad_U(x):
        x = np.atleast_2d(x)
        log_terms = np.array([
            np.log(w) - 0.5 * np.sum((x - c)**2, axis=-1) / var
            for c, w in zip(centers, weights)
        ])
        log_terms_norm = log_terms - np.max(log_terms, axis=0)
        weights_soft = np.exp(log_terms_norm)
        weights_soft /= weights_soft.sum(axis=0)
        
        grad_log = np.zeros_like(x)
        for k, c in enumerate(centers):
            grad_log += weights_soft[k][:, None] * (-(x - c) / var)
        
        return -grad_log

    def pi_density(x):
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        for c, w in zip(centers, weights):
            d2 = np.sum((x - c)**2, axis=-1)
            result += w * np.exp(-0.5 * d2 / var) / (2 * np.pi * var)
        return result

    grid = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(grid, grid)
    points = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    pi_grid = pi_density(points).reshape(X1.shape)
    
    ax = axes[0]
    cs = ax.contourf(X1, X2, pi_grid, levels=20, cmap="Blues")
    ax.scatter(centers[:, 0], centers[:, 1], color="red", s=80, marker="x",
               linewidth=2.5, label="Modes")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(r"Densité cible $\pi(x) \propto e^{-U(x)}$ — mixture 3 modes")
    ax.legend()
    ax.set_aspect("equal")

    ax = axes[1]
    n_traj = 1000
    n_steps = 2000
    h = 0.05
    rng = np.random.default_rng(0)
    
    X = rng.normal(0, 0.5, size=(n_traj, 2))
    
    for _ in range(n_steps):
        X = X - h * grad_U(X) + np.sqrt(2 * h) * rng.standard_normal(X.shape)
    
    ax.contour(X1, X2, pi_grid, levels=10, colors="red", linewidths=0.8, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.4, color="#2980b9")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"1000 samples Langevin après {n_steps} pas — couvre les 3 modes")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    fig.suptitle(
        r"Échantillonnage par Langevin : on sample $\pi$ en suivant son score $\nabla \log \pi = -\nabla U$",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_2d_sampling.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 4 : sampling 2D")


# =========================================================================
# Figure 5 : Discrétisation — effet du pas h
# =========================================================================
def fig_discretization():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_traj = 5000
    hs = [0.01, 0.15]
    titles = [
        r"$h = 0.01$ (pas fin) — distribution proche de $\pi$",
        r"$h = 0.15$ (pas grossier) — biais visible"
    ]
    
    x_grid = np.linspace(-2.5, 2.5, 300)
    pi_unnorm = np.exp(-double_well_U(x_grid))
    Z = np.trapz(pi_unnorm, x_grid)
    pi_norm = pi_unnorm / Z
    
    for ax, h, title in zip(axes, hs, titles):
        rng = np.random.default_rng(456)
        X = rng.normal(0, 0.3, n_traj)
        
        n_steps = max(int(20 / h), 100)
        for _ in range(n_steps):
            X = langevin_step(X, double_well_grad, h, rng)
        
        ax.hist(X, bins=50, density=True, color="#3498db", alpha=0.6,
                edgecolor="black", linewidth=0.4, label="Empirique")
        ax.plot(x_grid, pi_norm, color="#c0392b", lw=2.5, label=r"$\pi(x)$ vraie")
        ax.set_xlabel("$x$")
        ax.set_ylabel("densité")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right")
        ax.set_xlim(-2.5, 2.5)

    fig.suptitle(
        r"Discrétisation par Euler-Maruyama : $X_{k+1} = X_k - h\nabla U(X_k) + \sqrt{2h}\,\varepsilon_k$",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig5_discretization.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 5 : discrétisation")


# =========================================================================
# Figure 6 : Lien avec score-based diffusion models
# =========================================================================
def fig_score_link():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x_grid = np.linspace(-2.5, 2.5, 300)
    U = double_well_U(x_grid)
    grad = double_well_grad(x_grid)
    
    ax.plot(x_grid, U, color="#2c3e50", lw=2, label=r"$U(x) = -\log \pi(x)$ (à constante près)")
    ax.plot(x_grid, -grad, color="#27ae60", lw=2, linestyle="--",
            label=r"score $\nabla \log \pi(x) = -\nabla U(x)$")
    ax.axhline(0, color="black", lw=0.4, alpha=0.5)
    ax.set_xlabel("$x$")
    ax.set_title("Le score pointe vers les modes — il guide la trajectoire Langevin")
    ax.legend(fontsize=10, loc="upper center")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-9, 5)
    
    for x_ann in [-1.7, 1.7]:
        score_val = -double_well_grad(x_ann)
        ax.annotate("", xy=(x_ann + 0.3 * np.sign(score_val), -3),
                    xytext=(x_ann, -3),
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
        ax.text(x_ann, -3.7, f"score = {score_val:.1f}", color="#27ae60",
                ha="center", fontsize=9)

    ax = axes[1]
    pi_grid = np.exp(-double_well_U(x_grid))
    pi_grid /= np.trapz(pi_grid, x_grid)
    ax.fill_between(x_grid, 0, pi_grid, color="#3498db", alpha=0.15)
    ax.plot(x_grid, pi_grid, color="#3498db", lw=2,
            label=r"$\pi(x)$ (densité cible)")
    
    rng = np.random.default_rng(2025)
    n_traj_show = 4
    for k, x0 in enumerate([-2.0, -0.3, 0.5, 2.2]):
        x_traj = langevin_trajectory(x0=x0, grad_U=double_well_grad,
                                      h=0.005, n_steps=4000, seed=20+k)
        # Histogramme des positions visitées par cette trajectoire après burn-in
        burn_in = 1000
        ax.scatter(np.full(20, x0), [0.05] * 20, s=80, marker="o",
                   color=f"C{k}", edgecolor="black", zorder=5,
                   label=fr"$X_0 = {x0}$" if k < 2 else None)
        # Trajectoire visualisée par densité de présence (KDE simple)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x_traj[burn_in:])
        ax.plot(x_grid, kde(x_grid), lw=0.8, alpha=0.5, color=f"C{k}")

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"densité")
    ax.set_title("Plusieurs trajectoires Langevin convergent toutes vers $\pi$ (KDE des trajs.)")
    ax.legend(fontsize=8, loc="upper center")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 0.8)

    fig.suptitle(
        r"Pont avec les diffusion models : $\pi$ inconnue ⟹ on apprend le score $\nabla \log \pi$ ⟹ on sample par Langevin",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig6_score_link.png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Fig 6 : score-based link")


if __name__ == "__main__":
    fig_double_well()
    fig_stationary()
    fig_quadratic_OU()
    fig_2d_sampling()
    fig_discretization()
    fig_score_link()
    print(f"\nFigures sauvegardées dans {OUT}")
