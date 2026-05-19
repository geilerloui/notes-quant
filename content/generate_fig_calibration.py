"""
Figure E pour la note 04 : test de calibration (vol predite vs realisee).

Backtest sur donnees simulees avec rebalancement mensuel.
4 strategies : Brut, Clipping, Ledoit-Wolf, RIE.

Le critere "vol predite = vol realisee" est LE test d'un risk manager :
la sous-estimation systematique (ratio >> 1) est ce qui condamne l'estimation brute.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_calibration.py

Le calcul prend ~1 minute (~130 rebalancements de matrice 200x200).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_calibration.png"
SEED = 2024

rng = np.random.default_rng(seed=SEED)

N = 200
T_estim = 250
T_total = 3000
rebal_freq = 21
q = N / T_estim
lp = (1 + np.sqrt(q))**2

print(f"N = {N}, T = {T_estim}, q = {q:.2f}, lambda+ = {lp:.2f}")

strengths = [6.0, 2.0, 1.5, 1.2, 0.8]
factors = []
u_market = np.abs(rng.standard_normal(N)) + 0.5
u_market = u_market / np.linalg.norm(u_market)
factors.append(u_market)
for _ in range(len(strengths) - 1):
    u = rng.standard_normal(N); u /= np.linalg.norm(u)
    factors.append(u)
Sigma_true = np.eye(N)
for theta, u in zip(strengths, factors):
    Sigma_true += theta * np.outer(u, u)
Sigma_sqrt = np.linalg.cholesky(Sigma_true)
returns = (Sigma_sqrt @ rng.standard_normal(size=(N, T_total))).T * 0.01

def compute_weights(Sigma):
    Sigma_reg = Sigma + 1e-6 * np.trace(Sigma) / N * np.eye(N)
    inv = np.linalg.solve(Sigma_reg, np.ones(N))
    return inv / inv.sum()

def invert_bbp(lam, q):
    b = 1 + q - lam
    disc = b**2 - 4*q
    if disc < 0: return 1.0
    return 1 + (-b + np.sqrt(disc)) / 2

def clean_clipping(Sigma_emp):
    eigvals, eigvecs = np.linalg.eigh(Sigma_emp)
    mean_eigval = np.mean(eigvals)
    bulk = eigvals <= lp * mean_eigval
    eigvals_clean = eigvals.copy()
    if bulk.sum() > 0:
        eigvals_clean[bulk] = eigvals[bulk].mean()
    return eigvecs @ np.diag(eigvals_clean) @ eigvecs.T

def clean_ledoit_wolf(Sigma_emp, alpha=0.5):
    target = np.trace(Sigma_emp) / N * np.eye(N)
    return (1 - alpha) * Sigma_emp + alpha * target

def clean_rie(Sigma_emp):
    eigvals, eigvecs = np.linalg.eigh(Sigma_emp)
    mean_eigval = np.mean(eigvals)
    eigvals_norm = eigvals / mean_eigval
    eigvals_clean_norm = eigvals_norm.copy()
    bulk_mask = eigvals_norm <= lp
    bulk_mean = eigvals_norm[bulk_mask].mean() if bulk_mask.sum() > 0 else 1.0
    for i, lam in enumerate(eigvals_norm):
        if lam > lp:
            eigvals_clean_norm[i] = invert_bbp(lam, q)
        else:
            eigvals_clean_norm[i] = bulk_mean + 0.3 * (lam - bulk_mean)
    return eigvecs @ np.diag(eigvals_clean_norm * mean_eigval) @ eigvecs.T

methods = ['Brut', 'Clipping', 'Ledoit-Wolf', 'RIE']
colors_m = ['crimson', 'darkorange', 'green', 'purple']
vol_predicted = {m: [] for m in methods}
vol_realized = {m: [] for m in methods}

t = T_estim
n_rebal = 0
while t + rebal_freq < T_total:
    window = returns[t - T_estim:t]
    Sigma_emp = np.cov(window.T)
    estimators = {
        'Brut': Sigma_emp,
        'Clipping': clean_clipping(Sigma_emp),
        'Ledoit-Wolf': clean_ledoit_wolf(Sigma_emp),
        'RIE': clean_rie(Sigma_emp)
    }
    for m in methods:
        Sigma_clean = estimators[m]
        w = compute_weights(Sigma_clean)
        var_pred = max(w @ Sigma_clean @ w, 1e-12)
        vol_predicted[m].append(np.sqrt(var_pred) * np.sqrt(rebal_freq))
        future = returns[t:t+rebal_freq]
        vol_realized[m].append(np.std(future @ w) * np.sqrt(rebal_freq))
    t += rebal_freq
    n_rebal += 1

print(f"Rebalancements : {n_rebal}")
for m in methods:
    ratio_mean = (np.array(vol_realized[m]) / np.array(vol_predicted[m])).mean()
    print(f"  {m:12s} : ratio realise / predit = {ratio_mean:.2f}x")

# --- Figure ----
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
all_vals = []
for m, c in zip(methods, colors_m):
    pred = np.array(vol_predicted[m])
    real = np.array(vol_realized[m])
    ax.scatter(pred, real, color=c, alpha=0.5, s=20, label=m)
    all_vals.extend(list(pred) + list(real))
v_min, v_max = min(all_vals) * 0.9, max(all_vals) * 1.1
ax.plot([v_min, v_max], [v_min, v_max], 'k--', linewidth=1.5, alpha=0.7,
        label='calibration parfaite')
ax.set_xlabel('vol predite (in-sample)', fontsize=11)
ax.set_ylabel('vol realisee (out-of-sample)', fontsize=11)
ax.set_title(r'$(a)$ Calibration : vol predite vs realisee' + '\n' +
             'Un bon estimateur => points proches de la diagonale', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(v_min, v_max); ax.set_ylim(v_min, v_max)
ax.set_aspect('equal')

ax = axes[1]
data_box = [np.array(vol_realized[m]) / np.array(vol_predicted[m]) for m in methods]
bp = ax.boxplot(data_box, tick_labels=methods, patch_artist=True)
for patch, c in zip(bp['boxes'], colors_m):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)
ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='calibration parfaite')

for i, m in enumerate(methods):
    ratio_mean = (np.array(vol_realized[m]) / np.array(vol_predicted[m])).mean()
    color_text = colors_m[i]
    ax.text(i+1, ax.get_ylim()[1] * 0.92,
            f'moyenne\n{ratio_mean:.2f}x',
            ha='center', fontsize=10, fontweight='bold', color=color_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color_text, alpha=0.9))

ax.set_ylabel('ratio  vol realisee / vol predite', fontsize=11)
ax.set_title(r'$(b)$ Distribution du ratio' + '\n' +
             'Brut sous-estime massivement le risque', fontsize=11)
ax.legend(loc='center right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"\nSaved: {out_path}")
