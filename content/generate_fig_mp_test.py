"""
Genere la figure 2 pour la note 02_Marchenko_Pastur :
test de bruit pur avec 3 facteurs caches.

On simule une covariance "vraie" = I + somme de spikes (facteurs caches)
de forces differentes, on tire des donnees, on calcule la covariance
empirique, et on superpose son spectre a la densite MP theorique.

Utilisation :
    cd C:\\Users\\geile\\mon-site\\content
    python generate_fig_mp_test.py

Sortie : fig_mp_test.png dans le repertoire courant.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = "fig_mp_test.png"
SEED = 2024

rng = np.random.default_rng(seed=SEED)

# --- Setup ----------------------------------------------------------------
N = 200
T = 600
q = N / T

spike_strengths = [8.0, 3.5, 1.2]

print(f"N = {N}, T = {T}, q = {q:.3f}")
print(f"Seuil BBP theta_c = sqrt(q) = {np.sqrt(q):.3f}")
print(f"Forces des spikes : {spike_strengths}")

# Vecteurs propres orthonormes pour les spikes
K = len(spike_strengths)
A = rng.standard_normal(size=(N, K))
Q_basis, _ = np.linalg.qr(A)

# Construction de la vraie covariance : I + sum theta_k u_k u_k^T
Sigma_true = np.eye(N)
for k, theta in enumerate(spike_strengths):
    u = Q_basis[:, k:k+1]
    Sigma_true += theta * (u @ u.T)

# Racine carree pour generer les donnees
Sigma_sqrt = np.linalg.cholesky(Sigma_true)

# Tirage et covariance empirique
Z = rng.standard_normal(size=(N, T))
X = Sigma_sqrt @ Z
W = (X @ X.T) / T

empirical_spectrum = np.linalg.eigvalsh(W)
print(f"\nTop 5 valeurs propres empiriques : {empirical_spectrum[-5:][::-1]}")

# --- Densite MP theorique --------------------------------------------------
def mp_density(lam, q):
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2
    out = np.zeros_like(lam)
    mask = (lam > lm) & (lam < lp)
    out[mask] = np.sqrt((lp - lam[mask]) * (lam[mask] - lm)) / (2 * np.pi * q * lam[mask])
    return out

lam_plot = np.linspace(0.001, 3.0, 1000)
rho_mp = mp_density(lam_plot, q)
lp = (1 + np.sqrt(q))**2
lm = (1 - np.sqrt(q))**2

# --- Figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))

# Histogramme du bulk
bulk_mask = empirical_spectrum < 3.0
ax.hist(empirical_spectrum[bulk_mask], bins=40, density=True, alpha=0.5,
        color='steelblue', edgecolor='white', linewidth=0.6,
        label=f'spectre empirique (bulk)\n(N={N}, T={T}, q={q:.2f})')

# Courbe MP
ax.plot(lam_plot, rho_mp, color='crimson', linewidth=2.4,
        label=fr'densite MP $\rho_{{\rm MP}}(\lambda; q={q:.2f})$' + '\n'
              + fr'support $[{lm:.2f},\,{lp:.2f}]$')

# Edge MP
ax.axvline(lp, color='black', linestyle='--', linewidth=1.2, alpha=0.8,
           label=fr'$\lambda_+ = {lp:.2f}$ (edge MP)')

# Outliers (tiges)
outliers = empirical_spectrum[empirical_spectrum > lp + 0.1]
y_outlier = 1.4
for lam_o in sorted(outliers, reverse=True):
    ax.plot([lam_o, lam_o], [0, y_outlier], color='darkgreen', linewidth=2.5, alpha=0.85)
    ax.scatter([lam_o], [y_outlier], color='darkgreen', s=60, zorder=5)
    ax.annotate(
        f'$\\lambda \\approx {lam_o:.2f}$',
        xy=(lam_o, y_outlier), xytext=(lam_o, y_outlier + 0.08),
        ha='center', fontsize=9, color='darkgreen', fontweight='bold'
    )

# Zones et annotations
ax.axvspan(lm, lp, alpha=0.08, color='red', label='_nolegend_')
ax.text(
    (lm + lp) / 2, 1.2,
    'BULK\n(bruit)',
    ha='center', va='center', fontsize=12, fontweight='bold',
    color='crimson', alpha=0.6
)
ax.text(
    empirical_spectrum.max() * 0.85, 0.6,
    'SIGNAL\n(facteurs caches qui\nsortent du bulk)',
    ha='center', va='center', fontsize=10, fontweight='bold',
    color='darkgreen',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew',
              edgecolor='darkgreen', alpha=0.85)
)

ax.set_xlabel(r'$\lambda$ (valeur propre)', fontsize=11)
ax.set_ylabel('densite', fontsize=11)
ax.set_title(
    "Test de bruit pur : 3 facteurs caches dans une covariance simulee\n"
    fr"vraie covariance = $\mathbb{{I}} + \sum_k \theta_k u_k u_k^\top$ avec $\theta = {spike_strengths}$",
    fontsize=11
)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(0, empirical_spectrum.max() * 1.1)
ax.set_ylim(0, 1.7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
plt.savefig(out_path, dpi=160, bbox_inches='tight')
print(f"\nSaved: {out_path}")
