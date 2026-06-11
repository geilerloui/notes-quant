"""
Figures pour la note 01 : Analyse des performances d'un portefeuille.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_analyse_performances.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

OUTPUT_DIR = os.path.join("images", "5-Finance", "0_analyse_performances")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ============================================================
# DATA SIMULATION
# ============================================================
T = 2500  # ~10 ans de jours ouvrés

# --- Prix : random walk avec drift + GARCH(1,1)-like vol clustering ---
mu = 0.0004
omega, alpha, beta = 0.000015, 0.18, 0.80
sigma2 = np.full(T, omega / (1 - alpha - beta))
eps = rng.standard_normal(T)
returns = np.zeros(T)
for t in range(1, T):
    sigma2[t] = omega + alpha * (returns[t-1] - mu)**2 + beta * sigma2[t-1]
    returns[t] = mu + np.sqrt(sigma2[t]) * eps[t]

price = 100 * np.cumprod(1 + returns)

# --- Multifactor data pour les sections régression ---
# 5 features autocorrélées avec deux clusters de colinéarité (factor zoo)
n_assets = 5

# Deux facteurs latents AR(1) indépendants
phi_lat = 0.92
lat1 = np.zeros(T); lat2 = np.zeros(T)
for t in range(1, T):
    lat1[t] = phi_lat * lat1[t-1] + rng.standard_normal()
    lat2[t] = phi_lat * lat2[t-1] + rng.standard_normal()

# Bruit idiosyncratique AR(1)
phi_idio = 0.6
idio = np.zeros((T, n_assets))
for j in range(n_assets):
    for t in range(1, T):
        idio[t, j] = phi_idio * idio[t-1, j] + rng.standard_normal()

# X1, X2 chargent lat1 ; X3, X4 chargent lat2 ; X5 = bruit pur
features = np.column_stack([
    0.90 * lat1 + 0.35 * idio[:, 0],
    0.85 * lat1 + 0.35 * idio[:, 1],
    0.92 * lat2 + 0.35 * idio[:, 2],
    0.80 * lat2 + 0.35 * idio[:, 3],
    1.0 * idio[:, 4],
]) * 0.012

# Target return : combinaison + résidu AR(1) + hétéroscédastique
true_betas = np.array([0.20, 0.15, -0.10, -0.05, 0.0])
phi_res = 0.40
eps = np.zeros(T)
v = np.sqrt(sigma2) * rng.standard_normal(T)
for t in range(1, T):
    eps[t] = phi_res * eps[t-1] + v[t]
y = features @ true_betas + eps


def save(name):
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, name)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# ============================================================
# FIG 1 — Prix vs Returns côte à côte
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(price, color="tab:blue", lw=0.9)
axes[0].set_title("Série des prix $P_t$")
axes[0].set_xlabel("jour"); axes[0].set_ylabel("prix")
axes[1].plot(returns * 100, color="tab:orange", lw=0.5)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("Série des rendements $r_t$ (%)")
axes[1].set_xlabel("jour"); axes[1].set_ylabel("rendement (%)")
save("im1.png")


# ============================================================
# FIG 2 — Stationnarité : rolling mean & std prix vs returns
# ============================================================
W = 250  # 1 an
p_ser = pd.Series(price)
r_ser = pd.Series(returns)

fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
axes[0, 0].plot(p_ser.rolling(W).mean(), color="tab:blue")
axes[0, 0].set_title("Moyenne glissante (1 an) — prix")
axes[0, 0].set_ylabel("moyenne")
axes[0, 1].plot(p_ser.rolling(W).std(), color="tab:blue")
axes[0, 1].set_title("Écart-type glissant (1 an) — prix")
axes[0, 1].set_ylabel("std")
axes[1, 0].plot(r_ser.rolling(W).mean() * 100, color="tab:orange")
axes[1, 0].axhline(0, color="black", lw=0.5)
axes[1, 0].set_title("Moyenne glissante (1 an) — rendements (%)")
axes[1, 0].set_xlabel("jour"); axes[1, 0].set_ylabel("moyenne (%)")
axes[1, 1].plot(r_ser.rolling(W).std() * 100, color="tab:orange")
axes[1, 1].set_title("Écart-type glissant (1 an) — rendements (%)")
axes[1, 1].set_xlabel("jour"); axes[1, 1].set_ylabel("std (%)")
save("im2.png")


# ============================================================
# FIG 3 — ACF et PACF des rendements
# ============================================================
max_lag = 30
acf_r = acf(returns, nlags=max_lag, fft=True)
pacf_r = pacf(returns, nlags=max_lag)
conf = 1.96 / np.sqrt(T)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(range(max_lag + 1), acf_r, color="tab:orange", alpha=0.8)
axes[0].axhspan(-conf, conf, color="gray", alpha=0.2, label="IC 95%")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_title("ACF des rendements $r_t$")
axes[0].set_xlabel("lag"); axes[0].set_ylabel("autocorrélation")
axes[0].legend()
axes[1].bar(range(max_lag + 1), pacf_r, color="tab:orange", alpha=0.8)
axes[1].axhspan(-conf, conf, color="gray", alpha=0.2, label="IC 95%")
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("PACF des rendements $r_t$")
axes[1].set_xlabel("lag"); axes[1].set_ylabel("autocorrélation partielle")
axes[1].legend()
save("im3.png")


# ============================================================
# FIG 4 — Volatility clustering : |r|, r², et ACF des r²
# ============================================================
acf_r2 = acf(returns**2, nlags=max_lag, fft=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(np.abs(returns) * 100, color="tab:red", lw=0.4)
axes[0].set_title("|$r_t$| (%) — clusters de volatilité visibles")
axes[0].set_xlabel("jour"); axes[0].set_ylabel("|$r_t$| (%)")
axes[1].bar(range(max_lag + 1), acf_r2, color="tab:red", alpha=0.8)
axes[1].axhspan(-conf, conf, color="gray", alpha=0.2, label="IC 95%")
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("ACF de $r_t^2$ — très significative")
axes[1].set_xlabel("lag"); axes[1].set_ylabel("autocorrélation")
axes[1].legend()
save("im4.png")


# ============================================================
# FIG 5 — Matrice de corrélation des features (multicolinéarité)
# ============================================================
feat_names = [f"$X_{{{i+1}}}$" for i in range(n_assets)]
df_feat = pd.DataFrame(features, columns=feat_names)
corr = df_feat.corr()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_assets)); ax.set_xticklabels(feat_names)
ax.set_yticks(range(n_assets)); ax.set_yticklabels(feat_names)
for i in range(n_assets):
    for j in range(n_assets):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(corr.values[i, j]) > 0.5 else "black",
                fontsize=10)
ax.set_title("Corrélation entre régresseurs — multicolinéarité")
plt.colorbar(im, ax=ax, shrink=0.7)
ax.grid(False)
save("im5.png")


# ============================================================
# FIG 6 — ACF des résidus d'une régression naïve
# ============================================================
# OLS standard
X = features
beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
residuals = y - X @ beta_hat

acf_res = acf(residuals, nlags=max_lag, fft=True)

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(max_lag + 1), acf_res, color="tab:purple", alpha=0.8)
ax.axhspan(-conf, conf, color="gray", alpha=0.2, label="IC 95% sous iid")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("ACF des résidus OLS — violation de l'iid de Gauss-Markov")
ax.set_xlabel("lag"); ax.set_ylabel("autocorrélation")
ax.legend()
save("im6.png")


# ============================================================
# FIG 7 — Hétéroscédasticité : résidus² dans le temps
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(residuals, color="tab:purple", lw=0.4)
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_title("Résidus $\\hat{\\varepsilon}_t$")
axes[0].set_xlabel("jour"); axes[0].set_ylabel("résidu")
axes[1].plot(residuals**2, color="tab:purple", lw=0.4)
axes[1].set_title("Résidus au carré $\\hat{\\varepsilon}_t^2$ — variance qui bouge")
axes[1].set_xlabel("jour"); axes[1].set_ylabel("$\\hat{\\varepsilon}_t^2$")
save("im7.png")


# ============================================================
# FIG 8 — QQ-plot des résidus vs Normale (fat tails)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 6))
res_std = (residuals - residuals.mean()) / residuals.std()
osm, osr = stats.probplot(res_std, dist="norm", fit=False)
ax.scatter(osm, osr, s=8, color="tab:purple", alpha=0.6)
lim = max(abs(osm).max(), abs(osr).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, ls="--", label="droite gaussienne")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("quantiles théoriques (Normale)")
ax.set_ylabel("quantiles empiriques (résidus)")
ax.set_title("QQ-plot — fat tails : queues plus épaisses que la Normale")
ax.legend()
ax.set_aspect("equal")
save("im8.png")

print("\nDone — toutes les figures générées dans :", OUTPUT_DIR)
