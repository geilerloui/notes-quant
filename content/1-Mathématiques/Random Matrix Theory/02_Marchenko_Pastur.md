---
title: Loi de Marchenko-Pastur
date: 2026-05-12
tags: [random-matrix-theory, marchenko-pastur, covariance, wishart]
---

> [!warning] Prérequis
> Cette note suppose lue [[01_Fondamentaux]]. On y reprend la matrice de Wishart $W = \frac{1}{T} X X^\top$, le ratio $q = N/T$, et l'exemple "covariance empirique qui ment" sur $N=3, T=5$.

## I. Du constat empirique à la loi limite

En section V de la note précédente, on a vu que la covariance empirique étale les valeurs propres autour de la vraie covariance, et que cet étalement dépend du ratio $q = N/T$. La question naturelle : **peut-on décrire précisément cet étalement ?** La réponse — oui, et de façon déterministe — est la loi de Marchenko-Pastur.

> [!warning] Théorème (Marchenko-Pastur, 1967)
> Soit $W = \frac{1}{T} X X^\top$ avec $X \in \mathbb{R}^{N \times T}$ à colonnes i.i.d. $\mathcal{N}(0, \mathbb{I}_N)$. Quand $N, T \to \infty$ à ratio $q = N/T$ fixé, la mesure spectrale empirique de $W$ converge faiblement vers la **loi de Marchenko-Pastur** de densité
> $$\rho_{\rm MP}(\lambda; q) = \frac{1}{2\pi q \lambda} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)} \,\mathbb{1}_{[\lambda_-, \lambda_+]}(\lambda)$$
> avec **bornes du support** :
> $$\lambda_\pm = (1 \pm \sqrt{q})^2$$
>
> Si $q > 1$, il y a en plus une masse de Dirac en $0$ de poids $1 - 1/q$ (voir section IV).

**À retenir d'emblée.** Sous l'hypothèse "vraie covariance = identité" (donc spectre vrai = $\{1, 1, \dots, 1\}$), le spectre empirique n'est *pas* concentré sur 1 — il s'étale sur tout l'intervalle $[\lambda_-, \lambda_+]$, et la largeur de cet intervalle est entièrement contrôlée par $q$. C'est *le* résultat à intérioriser.

## II. Lire la formule

La densité ressemble à celle du demi-cercle mais asymétrique. Décortiquons les pièces.

### Le support $[\lambda_-, \lambda_+]$

$$\lambda_- = (1 - \sqrt{q})^2, \qquad \lambda_+ = (1 + \sqrt{q})^2$$

Quelques valeurs numériques pour fixer les idées :

| $q = N/T$ | $\lambda_-$ | $\lambda_+$ | Largeur $\lambda_+ - \lambda_-$ |
|---|---|---|---|
| $0$ (limite : $T \gg N$) | $1$ | $1$ | $0$ |
| $0.1$ | $0.47$ | $1.73$ | $1.26$ |
| $0.5$ | $0.09$ | $2.91$ | $2.83$ |
| $1$ | $0$ | $4$ | $4$ |
| $2$ | $0.17$ | $5.83$ | (cas $q>1$, voir IV) |

**Lecture.** Quand $q \to 0$ (beaucoup de données par rapport à la dimension), le support se concentre sur le point $\{1\}$ — on retrouve la vraie covariance. Quand $q$ augmente, le support s'étale de plus en plus, et la covariance empirique devient de plus en plus dispersée.

### La densité elle-même

$$\rho_{\rm MP}(\lambda) = \frac{1}{2\pi q \lambda} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}$$

Forme caractéristique : la racine carrée $\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}$ s'annule aux bornes (comme pour le demi-cercle), et le facteur $1/\lambda$ casse la symétrie — la densité est plus haute près de $\lambda_-$ que de $\lambda_+$.

![[fig_mp_densities.png]]
*Figure 1. Densité de Marchenko-Pastur pour différentes valeurs de $q$. Plus $q$ augmente, plus le spectre s'étale autour de la vraie valeur $1$. La courbe verte ($q=1$) touche $0$ — la matrice empirique commence à être singulière.*

## III. Comparaison avec le demi-cercle

| | Demi-cercle (Wigner) | Marchenko-Pastur |
|---|---|---|
| Matrice | GOE normalisée $M/\sqrt{N}$ | Wishart $\frac{1}{T} X X^\top$ |
| Symétrie | $M = M^\top$ | $W = W^\top$ |
| Valeurs propres | Réelles, signe quelconque | Réelles, $\geq 0$ |
| Paramètre limite | $N \to \infty$ | $N, T \to \infty$, ratio $q$ fixé |
| Support | $[-2, 2]$ | $[\lambda_-, \lambda_+]$ |
| Symétrie de la densité | Oui (autour de 0) | Non |
| Cas dégénéré | Aucun | Masse de Dirac en 0 si $q > 1$ |

**Le lien profond.** MP et demi-cercle sont des **cousins**. Le demi-cercle apparaît quand on prend $W = \frac{X + X^\top}{\sqrt{2N}}$ avec $X$ rectangulaire et on fait $N \to \infty$. MP apparaît quand on prend $W = \frac{1}{T} X X^\top$ et on fait $N, T \to \infty$ à ratio fixé. Les deux résultats se démontrent par la même technique (équation auto-cohérente sur la résolvante).

## IV. Cas $q > 1$ : matrice singulière

Si $q > 1$, autrement dit $N > T$ — plus d'actifs que de jours d'observation — alors la matrice empirique $W$ est **singulière** (déterminant nul) parce que son rang est au plus $T < N$. Concrètement :

- **$T$ valeurs propres** suivent la loi MP sur $[\lambda_-, \lambda_+]$
- **$N - T$ valeurs propres** valent exactement $0$

La loi limite incorpore donc une **masse de Dirac en $0$** de poids $1 - 1/q$ (la proportion de zéros), et la densité MP usuelle pour la fraction $1/q$ restante.

**Exemple S&P 500.** $N = 500$ actifs, $T = 250$ jours (une année) : $q = 2$. La matrice empirique a au moins $N - T = 250$ valeurs propres nulles, et les 250 autres sont étalées sur $[\lambda_-, \lambda_+] = [0.17, 5.83]$. **La moitié de ta covariance empirique est mathématiquement du néant.**

> [!warning] Conséquence opérationnelle
> Tu **ne peux pas inverser** une covariance empirique quand $N > T$. Pour Markowitz qui demande $\Sigma^{-1}$, c'est rédhibitoire. Il faut :
> - réduire $N$ (sélection d'actifs), ou
> - augmenter $T$ (historique plus long, mais on perd la pertinence temporelle), ou
> - **régulariser** (shrinkage, RIE — voir note 04).

## V. Application : test de bruit pur

Voici la première utilisation concrète de MP. Tu disposes d'une matrice de covariance empirique sur $N$ actifs et $T$ jours. Tu veux savoir : **mes valeurs propres sont-elles compatibles avec du bruit pur, ou y a-t-il du signal ?**

**Procédure :**

1. Calculer $q = N/T$ et les bornes $\lambda_\pm = (1 \pm \sqrt{q})^2$ (en supposant que la vraie covariance est l'identité — hypothèse nulle "bruit pur").
2. Calculer le spectre empirique $\{\lambda_1, \dots, \lambda_N\}$.
3. Tracer l'histogramme empirique et superposer la densité MP.
4. **Diagnostic** : les valeurs propres dans $[\lambda_-, \lambda_+]$ sont compatibles avec du bruit. Celles qui en sortent (vers la droite) sont des candidates pour du signal réel.

> [!example] Cas concret S&P 500
> *À compléter avec un exemple numérique réel quand on fera la figure correspondante.*
>
> Idée typique : sur des rendements journaliers du S&P 500, on observe souvent **1 grosse valeur propre** très en dehors du bulk MP (le facteur "marché global"), puis quelques valeurs propres modérément à l'extérieur (facteurs sectoriels), puis le reste qui s'aligne avec MP (le bruit).

![[fig_mp_test.png]]
*Figure 2. (À générer.) Spectre empirique d'une covariance simulée avec 3 vrais facteurs cachés, superposé à la densité MP théorique pour les mêmes $(N, T)$. Les trois valeurs propres sortantes correspondent aux signaux ; le bulk colle parfaitement à MP.*

## VI. Ce que MP ne dit pas encore

MP te donne le **support** du bruit. Mais deux questions restent ouvertes :

1. **Quel seuil exact pour la détection ?** Combien au-dessus de $\lambda_+$ une valeur propre doit-elle être pour être *significativement* du signal ? C'est l'objet de la **transition BBP** (note 03) et de Tracy-Widom.
2. **Comment nettoyer la covariance ?** Une fois qu'on sait que le bulk est du bruit, comment l'éliminer pour obtenir une estimation utilisable de la vraie covariance ? C'est l'objet du **clipping et RIE** (note 04).

## Ce qui suit

[[03_Transition_BBP|Note 03 — Transition BBP et détection de facteurs]] : à partir de quel seuil un facteur sort-il du bulk ? Réponse précise via Tracy-Widom.
