"""
Figure A pour la note 04 : desastre de Markowitz avec covariance brute.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_markowitz_disaster.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_markowitz_disaster.png"
SEED = 42

rng = np.random.default_rng(seed=SEED)

N = 200
T_train = 250
q = N / T_train

# Structure factorielle
strengths = [4.0, 1.5, 1.0, 0.8]
factors = []
u_market = np.abs(rng.standard_normal(N)) + 0.5
u_market = u_market / np.linalg.norm(u_market)
factors.append(u_market)
for _ in range(3):
    u = rng.standard_normal(N); u /= np.linalg.norm(u)
    factors.append(u)

Sigma_true = np.eye(N)
for theta, u in zip(strengths, factors):
    Sigma_true += theta * np.outer(u, u)

Sigma_sqrt = np.linalg.cholesky(Sigma_true)
X_train = Sigma_sqrt @ rng.standard_normal(size=(N, T_train))
Sigma_emp = (X_train @ X_train.T) / T_train

ones = np.ones(N)
w_true = np.linalg.solve(Sigma_true, ones)
w_true /= w_true.sum()
w_emp = np.linalg.solve(Sigma_emp, ones)
w_emp /= w_emp.sum()

vol_true_out = np.sqrt(w_true @ Sigma_true @ w_true) * np.sqrt(250)
vol_emp_in = np.sqrt(w_emp @ Sigma_emp @ w_emp) * np.sqrt(250)
vol_emp_out = np.sqrt(w_emp @ Sigma_true @ w_emp) * np.sqrt(250)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
sorted_idx = np.argsort(np.abs(w_emp))[::-1]
positions = np.arange(N)
ax.bar(positions, w_emp[sorted_idx], color='crimson', alpha=0.7,
       label='Markowitz BRUT $\\hat\\Sigma$', width=0.7)
ax.bar(positions, w_true[sorted_idx], color='steelblue', alpha=0.7,
       label='Markowitz VRAI $\\Sigma_{\\rm vrai}$', width=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('actif (trie par |poids empirique| decroissant)', fontsize=11)
ax.set_ylabel('poids dans le portefeuille', fontsize=11)
ax.set_title(f'$(a)$ Distribution des poids\n$N={N}$, $T={T_train}$ -> $q={q:.2f}$', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

max_w_emp = np.abs(w_emp).max()
max_w_true = np.abs(w_true).max()
ax.text(0.05, 0.55,
        f'max |poids brut| = {max_w_emp:.3f}\n'
        f'max |poids vrai| = {max_w_true:.3f}\n'
        f'ratio = {max_w_emp/max_w_true:.1f}x',
        transform=ax.transAxes, fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='darkorange', alpha=0.9))

ax = axes[1]
categories = ['Markowitz VRAI\n(reference)',
              'Markowitz BRUT\nin-sample\n(ce qu\'il "promet")',
              'Markowitz BRUT\nout-of-sample\n(ce qu\'on subit)']
values = [vol_true_out, vol_emp_in, vol_emp_out]
colors_bar = ['steelblue', '#ffaa44', 'crimson']
bars = ax.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('volatilite annualisee', fontsize=11)
ax.set_title(f'$(b)$ Volatilite predite vs realisee\n'
             f'Sous-estimation : {(1 - vol_emp_in / vol_emp_out) * 100:.0f}%',
             fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

ax.annotate('', xy=(2, vol_emp_out * 0.95), xytext=(1, vol_emp_in * 1.05),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(1.5, (vol_emp_in + vol_emp_out) / 2 + 0.05,
        f'+{(vol_emp_out/vol_emp_in - 1)*100:.0f}%',
        ha='center', fontsize=11, fontweight='bold', color='darkred')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"Saved: {out_path}")
