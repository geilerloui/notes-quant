"""
Génère les figures pour 3-Apprentissage-automatique/Unsupervised Learning/Réduction de Dimension/PCA.md

Figures produites :
  fig01_nuage_donnees.png       — nuage 2D centré avec axes propres
  fig02_rotation.png            — avant/après changement de base Y = XV
  fig03_cercle_correlations.png — cercle des corrélations dataset météo
  fig04_scree_plot.png          — scree plot avec coude
  fig05_pca_vs_nonlineaire.png  — swiss roll : PCA vs Isomap
  fig06_whitening.png           — raw -> PCA -> whitening

Lancer depuis content/ : python generate_fig_pca.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

OUT_DIR = os.path.join(
    "images",
    "3-Apprentissage-automatique",
    "Unsupervised Learning",
    "Réduction de Dimension",
)
os.makedirs(OUT_DIR, exist_ok=True)

# Style commun
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

C_DATA   = "#3a7ca5"  # bleu : données
C_PC1    = "#d62828"  # rouge : PC1
C_PC2    = "#2a9d8f"  # vert : PC2
C_AXIS   = "#222222"  # noir : axes coordonnées


def save(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path)
    print(f"  -> {path}")
    plt.close()


# --------------------------------------------------------------------------
# fig01 — Nuage 2D centré avec axes propres
# --------------------------------------------------------------------------
def fig01():
    rng = np.random.default_rng(42)
    n = 300
    # Nuage allongé : variance 4 selon une direction, 0.5 selon l'orthogonale
    cov = np.array([[3.0, 1.8], [1.8, 1.5]])
    X = rng.multivariate_normal([0, 0], cov, size=n)
    X = X - X.mean(axis=0)

    pca = PCA(n_components=2).fit(X)
    v1, v2 = pca.components_
    s1, s2 = np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.5, color=C_DATA, label="données")

    # Axes propres (longueur = écart-type)
    ax.annotate("", xy=2*s1*v1, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PC1, lw=2.5))
    ax.annotate("", xy=2*s2*v2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PC2, lw=2.5))
    ax.text(*(2.3*s1*v1), r"$v_1$", color=C_PC1, fontsize=14, ha="center")
    ax.text(*(2.5*s2*v2), r"$v_2$", color=C_PC2, fontsize=14, ha="center")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Nuage centré et ses axes propres")
    save("fig01_nuage_donnees.png")


# --------------------------------------------------------------------------
# fig02 — Rotation Y = XV
# --------------------------------------------------------------------------
def fig02():
    rng = np.random.default_rng(42)
    n = 300
    cov = np.array([[3.0, 1.8], [1.8, 1.5]])
    X = rng.multivariate_normal([0, 0], cov, size=n)
    X = X - X.mean(axis=0)

    pca = PCA(n_components=2).fit(X)
    Y = pca.transform(X)
    v1, v2 = pca.components_
    s1, s2 = np.sqrt(pca.explained_variance_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Gauche : repère original
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.5, color=C_DATA)
    ax.annotate("", xy=2*s1*v1, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PC1, lw=2.5))
    ax.annotate("", xy=2*s2*v2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PC2, lw=2.5))
    ax.text(*(2.3*s1*v1), r"$v_1$", color=C_PC1, fontsize=14)
    ax.text(*(2.5*s2*v2), r"$v_2$", color=C_PC2, fontsize=14)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Repère original $(e_1, e_2)$")

    # Droite : repère propre, Y = XV
    ax = axes[1]
    ax.scatter(Y[:, 0], Y[:, 1], s=14, alpha=0.5, color=C_DATA)
    # Axes du nouveau repère (alignés)
    ax.annotate("", xy=(2*s1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PC1, lw=2.5))
    ax.annotate("", xy=(0, 2*s2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_PC2, lw=2.5))
    ax.text(2.3*s1, 0.1, r"PC$_1$", color=C_PC1, fontsize=14)
    ax.text(0.1, 2.3*s2, r"PC$_2$", color=C_PC2, fontsize=14)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    # Mêmes limites que la fig de gauche pour la comparaison
    lim = max(abs(X).max(), abs(Y).max()) * 1.1
    axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
    axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    ax.set_xlabel(r"PC$_1$")
    ax.set_ylabel(r"PC$_2$")
    ax.set_title(r"Repère propre — après $Y = XV$")

    plt.suptitle("Le même nuage vu sous deux angles", fontsize=13, y=1.02)
    save("fig02_rotation.png")


# --------------------------------------------------------------------------
# fig03 — Cercle des corrélations (dataset météo)
# --------------------------------------------------------------------------
def fig03():
    # Dataset météo 6 villes (valeurs approximatives mais réalistes)
    data = np.array([
        # pluie, tmax, tmin
        [ 750, 14.5,  6.5],   # Lille
        [ 640, 15.5,  7.0],   # Paris
        [ 670, 14.8,  5.5],   # Strasbourg
        [1200, 14.8,  7.8],   # Brest
        [ 660, 18.0,  8.0],   # Toulouse
        [ 770, 19.5, 11.5],   # Nice
    ])
    villes = ["Lille", "Paris", "Strasbourg", "Brest", "Toulouse", "Nice"]
    variables = ["pluie", r"$t_{\max}$", r"$t_{\min}$"]

    # Centrer-réduire
    Xc = (data - data.mean(axis=0)) / data.std(axis=0, ddof=0)
    pca = PCA(n_components=2).fit(Xc)

    # Corrélations variable / composante = loading * sqrt(lambda)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    # Ici Xc est standardisé donc loadings = corrélations directement

    fig, ax = plt.subplots(figsize=(7, 7))
    circle = Circle((0, 0), 1, fill=False, color="black", lw=1)
    ax.add_patch(circle)
    # Cercle intermédiaire
    ax.add_patch(Circle((0, 0), 0.5, fill=False, color="gray", lw=0.5, ls=":"))

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    colors = ["#1f77b4", "#d62728", "#ff7f0e"]
    for i, (label, color) in enumerate(zip(variables, colors)):
        x, y = loadings[i]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        # Décalage du label
        offset = 0.1
        ax.text(x + offset*np.sign(x) if abs(x) > 0.1 else x,
                y + offset*np.sign(y) if abs(y) > 0.1 else y + offset,
                label, color=color, fontsize=13, ha="center")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    var_exp = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC$_1$ ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC$_2$ ({var_exp[1]:.1f}%)")
    ax.set_title("Cercle des corrélations — dataset météo")
    save("fig03_cercle_correlations.png")


# --------------------------------------------------------------------------
# fig04 — Scree plot avec coude
# --------------------------------------------------------------------------
def fig04():
    # Spectre simulé : 3 valeurs propres dominantes puis plateau bruité
    np.random.seed(0)
    lambdas = np.array([5.2, 3.8, 2.1, 0.45, 0.40, 0.35, 0.32, 0.28, 0.26, 0.22])
    k = np.arange(1, len(lambdas) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Gauche : scree plot
    ax = axes[0]
    ax.plot(k, lambdas, "o-", color=C_PC1, lw=2, markersize=8)
    ax.axvline(3.5, color="gray", ls="--", alpha=0.6)
    ax.annotate("coude", xy=(3.5, 0.5), xytext=(4.5, 1.5),
                fontsize=12, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel(r"composante $k$")
    ax.set_ylabel(r"valeur propre $\lambda_k$")
    ax.set_title("Scree plot")
    ax.set_xticks(k)

    # Droite : variance expliquée cumulée
    ax = axes[1]
    cumvar = np.cumsum(lambdas) / lambdas.sum() * 100
    ax.plot(k, cumvar, "o-", color=C_PC2, lw=2, markersize=8)
    ax.axhline(80, color="gray", ls="--", alpha=0.6)
    ax.axhline(95, color="gray", ls=":", alpha=0.4)
    ax.text(8.5, 81, "80%", color="gray", fontsize=10)
    ax.text(8.5, 96, "95%", color="gray", fontsize=10)
    ax.set_xlabel(r"composantes gardées $k$")
    ax.set_ylabel("variance expliquée cumulée (%)")
    ax.set_title("Variance expliquée cumulée")
    ax.set_xticks(k)
    ax.set_ylim(0, 105)

    plt.suptitle("Combien de composantes garder ?", fontsize=13, y=1.02)
    save("fig04_scree_plot.png")


# --------------------------------------------------------------------------
# fig05 — PCA vs non-linéaire sur swiss roll
# --------------------------------------------------------------------------
def fig05():
    X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)

    # PCA 2D
    X_pca = PCA(n_components=2).fit_transform(X)
    # Isomap 2D (alternative à UMAP sans dépendance extra)
    X_iso = Isomap(n_components=2, n_neighbors=10).fit_transform(X)

    fig = plt.figure(figsize=(15, 5))

    # 3D : swiss roll original
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="Spectral", s=8)
    ax.set_title("Swiss roll (manifold 2D dans $\\mathbb{R}^3$)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.view_init(elev=10, azim=-70)

    # PCA
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap="Spectral", s=8)
    ax.set_title("PCA — écrase la spirale")
    ax.set_xlabel("PC$_1$"); ax.set_ylabel("PC$_2$")
    ax.set_aspect("equal")

    # Isomap
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap="Spectral", s=8)
    ax.set_title("Isomap — déroule la spirale")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_aspect("equal")

    plt.suptitle("PCA vs méthode non-linéaire sur un manifold courbé", fontsize=13, y=1.02)
    save("fig05_pca_vs_nonlineaire.png")


# --------------------------------------------------------------------------
# fig06 — Whitening (3 étapes)
# --------------------------------------------------------------------------
def fig06():
    rng = np.random.default_rng(7)
    n = 500
    cov = np.array([[4.0, 2.5], [2.5, 2.0]])
    X = rng.multivariate_normal([0, 0], cov, size=n)
    X = X - X.mean(axis=0)

    pca = PCA(n_components=2, whiten=False).fit(X)
    Y = pca.transform(X)
    # Whitening manuel
    Z = Y / np.sqrt(pca.explained_variance_)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Données brutes (centrées)",
              r"Après PCA — $Y = XV$",
              r"Après whitening — $Z = \Lambda^{-1/2} V^\top X$"]
    nuages = [X, Y, Z]
    lims = [np.max(np.abs(arr)) * 1.1 for arr in nuages]

    for ax, data, title, lim in zip(axes, nuages, titles, lims):
        ax.scatter(data[:, 0], data[:, 1], s=12, alpha=0.5, color=C_DATA)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title)

    plt.suptitle("Whitening — décorrélation puis normalisation des variances",
                 fontsize=13, y=1.02)
    save("fig06_whitening.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Génération des figures dans : {OUT_DIR}")
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    print("Terminé.")
