---
title: Nettoyage de covariance — Clipping, RIE et Ledoit-Wolf
date: 2026-05-12
tags: [random-matrix-theory, covariance-cleaning, rie, markowitz, portfolio]
---

> [!warning] Prérequis
> [[01_Fondamentaux]], [[02_Marchenko_Pastur]] et idéalement [[03_Transition_BBP]]. On utilise tous les résultats : bornes MP $[\lambda_-, \lambda_+]$, biais positif des outliers (BBP), et le test de détection.

## I. Le problème : Markowitz et la covariance pourrie

### Rappel : le portefeuille de Markowitz

L'aboutissement classique de la théorie du portefeuille est la solution de **minimum-variance** : étant donné une covariance $\Sigma$ et une contrainte sur la somme des poids (par exemple $\sum_i w_i = 1$), le portefeuille de variance minimale est
$$w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}$$
où $\mathbf{1} = (1, 1, \dots, 1)^\top$. **Tout passe par l'inverse de $\Sigma$.**

> [!example] Pourquoi $\Sigma^{-1}$ est central
> Le portefeuille min-variance cherche à investir le plus dans les directions de **plus faible** variance vraie. Or les "directions" sont les vecteurs propres de $\Sigma$, et les variances dans ces directions sont les valeurs propres. L'inverse $\Sigma^{-1}$ a les mêmes vecteurs propres, mais avec valeurs propres $1/\lambda_i$ — donc il *amplifie* les petites $\lambda_i$. C'est mathématiquement ce qui dit "il faut investir en priorité où la variance est la plus petite".

### Le drame quand on remplace $\Sigma$ par $\hat\Sigma$

Que se passe-t-il si on remplace $\Sigma_{\rm vrai}$ par la covariance empirique $\hat\Sigma = \frac{1}{T} X X^\top$ ? On a vu (notes 02 et 03) que :

1. Le spectre empirique s'étale sur $[\lambda_-, \lambda_+]$ — il y a beaucoup de **petites valeurs propres parasites** dans le bulk MP.
2. Pour $q$ grand, $\lambda_-$ peut être très petit (par exemple, à $q=0.8$, $\lambda_- \approx 0.01$).
3. Quand on inverse, ces petites valeurs propres deviennent **très grandes** : $1/\lambda_- \to \infty$.

Conséquence : $\hat\Sigma^{-1}$ amplifie démesurément les directions de bruit, et le portefeuille calculé sur cette base prend des paris énormes sur des directions purement aléatoires.

### Démonstration du désastre

Pour rendre ça visceral, faisons l'expérience. On simule un univers de $N=200$ actifs avec une structure réaliste (1 facteur marché + 3 facteurs sectoriels), $T=250$ jours d'observation (donc $q = 0.8$). On calcule deux portefeuilles min-variance :
- Markowitz **VRAI** : utilise $\Sigma_{\rm vrai}$ (référence inaccessible en pratique)
- Markowitz **BRUT** : utilise $\hat\Sigma$ empirique

![[fig_markowitz_disaster.png]]
*Figure 1. **(a)** Distribution des poids pour les deux portefeuilles, triés par valeur absolue décroissante. Le portefeuille brut prend des positions 3.2x plus grandes que le vrai — il sur-investit massivement dans les directions de bruit. **(b)** Volatilité prédite vs réalisée. Markowitz brut "promet" une vol de 0.76 sur la base de $\hat\Sigma$, mais en réalité la vol subie est de 3.86 — soit 80% de sous-estimation.*

> [!warning] Le diagnostic
> Markowitz brut produit systématiquement :
> - **Risque massivement sous-estimé in-sample** (la vol prédite est ridiculement basse).
> - **Risque qui explose out-of-sample** (la vol réelle est plusieurs fois la promise).
> - **Concentration sur quelques actifs** aux comportements aléatoires.
> - **Turnover absurde** : à chaque rebalancement, les positions changent radicalement parce que le bruit d'estimation change.
>
> Ce n'est pas un problème d'algorithme — Markowitz résout exactement le bon problème *si on lui donne la bonne $\Sigma$*. C'est un problème d'**estimation** : on lui donne du bruit en pensant lui donner du signal.

Référence empirique : Laloux–Cizeau–Bouchaud–Potters (1999) *"Noise dressing of financial correlation matrices"* — le papier fondateur qui a documenté ce désastre sur données réelles du S&P.

## II. Intuition guidante : RMT comme "Fourier pour la covariance"

Avant de plonger dans les solutions, une analogie qui éclaire tout le pipeline qu'on va dérouler.

> [!example] Le pattern commun avec le débruitage par Fourier
> Le pipeline qu'on va appliquer à la covariance est **exactement** celui du débruitage par transformée de Fourier en traitement du signal. Le pattern est universel :
>
> 1. **Signal observé** = vrai signal + bruit, mélangés inextricablement dans la représentation naturelle.
> 2. **Transformation** vers une base où la séparation signal/bruit devient lisible.
> 3. **Filtrage** dans la nouvelle base (mise à zéro ou atténuation des composantes "bruit").
> 4. **Inversion** vers la représentation originale, nettoyée.
>
> | | Fourier (signal temporel) | RMT (covariance) |
> |---|---|---|
> | Observation | $x(t)$ dans le temps | $\hat\Sigma$ dans la base canonique |
> | Transformation | TF : passage en fréquences | Diagonalisation $\hat\Sigma = U \Lambda U^\top$ |
> | Critère bruit | seuil fréquentiel (passe-bas, etc.) | seuil spectral ($\lambda \in [\lambda_-, \lambda_+]$ MP) |
> | Filtrage | atténuation des fréquences | nettoyage des valeurs propres |
> | Reconstruction | TF inverse | $\hat\Sigma_{\rm clean} = U \Lambda_{\rm clean} U^\top$ |
>
> **Le pont mathématique n'est pas une analogie superficielle.** Pour un signal stationnaire, la matrice de covariance $C_{ij} = c(i-j)$ est de type Toeplitz, et le théorème de Szegő affirme que les matrices de Toeplitz sont **asymptotiquement diagonalisées par la transformée de Fourier**. Donc dans le cas stationnaire, diagonaliser la covariance = faire une TF.
>
> **Où RMT généralise Fourier.** Fourier nettoie bien quand la base "naturelle du bruit" est connue a priori (les fréquences). RMT s'applique quand cette base est **inconnue et doit être estimée depuis les données**. Pour une covariance financière, les vrais facteurs n'ont aucune raison de s'aligner sur des fréquences temporelles — ils définissent leur propre base, qu'on découvre via les vecteurs propres empiriques de $\hat\Sigma$. RMT te dit en plus : *dans cette base, le bruit a la signature spectrale universelle de MP*.

Garder cette analogie en tête : **clipping = passe-haut spectral grossier**, **RIE = filtre de Wiener optimal** sur le spectre de la covariance.

## III. Solution 1 — Eigenvalue clipping

### Procédure pas à pas

L'idée la plus simple, introduite par Bouchaud-Potters dans les années 2000 :

> [!warning] Procédure de clipping
> 1. **Diagonaliser** la covariance empirique : $\hat\Sigma = U \Lambda U^\top$ avec $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_N)$.
> 2. **Identifier le bulk MP** : toutes les valeurs propres dans $[\lambda_-, \lambda_+]$ sont considérées comme du bruit.
> 3. **Remplacer le bulk** par sa moyenne $\bar\lambda_{\rm bulk} = \frac{1}{|\rm bulk|} \sum_{\lambda_i \in \rm bulk} \lambda_i$.
> 4. **Garder les outliers** ($\lambda_i > \lambda_+$) tels quels — ce sont les facteurs détectés.
> 5. **Reconstruire** : $\hat\Sigma_{\rm clipped} = U \Lambda_{\rm clipped} U^\top$ avec $\Lambda_{\rm clipped}$ contenant la nouvelle diagonale.

### Effet visuel sur le spectre

![[fig_clipping_effect.png]]
*Figure 2. Effet visuel du clipping sur le spectre. **(a)** Brut : bulk étalé MP + outliers biaisés vers le haut. **(b)** Après clipping : bulk écrasé à une valeur constante $\bar\lambda_{\rm bulk} \approx 1$, outliers préservés. **(c)** Vrai (référence) : on voit que les outliers du clipping (positions 3.45, 4.49, 7.63) ne correspondent **pas** aux vraies valeurs propres (2.5, 4.0, 7.0).*

### Variantes

| Variante | Principe |
|---|---|
| **Hard clipping** (Bouchaud-Potters) | bulk remplacé par sa moyenne constante |
| **Shrinkage de Ledoit-Wolf** (2003) | $\hat\Sigma_{\rm LW} = (1-\alpha) \hat\Sigma + \alpha \cdot \bar\lambda \mathbb{I}$ — interpolation continue entre l'identité et l'empirique |
| **Soft clipping** | toutes les val. propres < $\lambda_+$ tirées vers la moyenne avec un paramètre de shrinkage |

Ledoit-Wolf est en fait un **clipping uniforme** : il tire toutes les valeurs propres vers une cible commune sans distinguer signal et bruit. C'est plus simple à mettre en œuvre (un seul paramètre $\alpha$) mais moins précis.

### Limites du clipping

Le clipping a un défaut conceptuel : il traite **toutes les valeurs propres du bulk identiquement**, alors qu'elles ont des positions différentes dans le spectre MP. Une valeur propre près de $\lambda_+$ et une valeur propre près de $\lambda_-$ contiennent des informations différentes sur la structure de la covariance, mais le clipping les écrase toutes à la même moyenne.

**Analogie Fourier** : c'est comme un filtre "tout ou rien" (passe-haut binaire) là où on devrait appliquer un **filtre de Wiener** qui atténue chaque fréquence proportionnellement à son rapport signal/bruit local.

De plus, le clipping **ne corrige pas le biais positif des outliers**. Comme on l'a vu en figure 3 de la note 03, les vraies valeurs propres des spikes sont $(1 + \theta_k)$ mais on observe $(1+\theta_k)(1 + q/\theta_k) > 1 + \theta_k$. Le clipping garde la valeur observée biaisée.

C'est ces deux limites que le RIE résout.

## IV. Solution 2 — RIE (Rotationally Invariant Estimator)

### Le principe

> [!warning] Le RIE en une phrase
> On garde les **mêmes vecteurs propres** que $\hat\Sigma$ (invariance par rotation — on ne touche pas aux directions identifiées par l'analyse), mais on remplace **chaque** valeur propre $\hat\lambda_i$ par sa valeur "nettoyée" $\xi_i$ selon une formule qui dépend de sa position dans le spectre.

La formule oracle (Ledoit-Péché 2011, généralisée par Bun-Bouchaud-Potters 2017) est :
$$\xi_i = \frac{\hat\lambda_i}{\big|1 - q + q \, \hat\lambda_i \, g(\hat\lambda_i)\big|^2}$$
où $g(z)$ est la transformée de Stieltjes du spectre empirique, évaluée avec une petite régularisation imaginaire.

### Lecture intuitive façon Wiener

Cette formule peut paraître mystérieuse mais elle est **exactement un filtre de Wiener appliqué dans la base propre**. En traitement du signal, le Wiener fréquentiel atténue chaque fréquence $f$ par
$$H(f) = \frac{S_{\rm signal}(f)}{S_{\rm signal}(f) + S_{\rm bruit}(f)}$$
Ici, le rôle des spectres est joué par la position de $\hat\lambda_i$ par rapport à la densité MP : si $\hat\lambda_i$ est en plein cœur du bulk MP, c'est essentiellement du bruit, donc $\xi_i$ est très atténué (proche de la moyenne). Si $\hat\lambda_i$ est un outlier, c'est du signal, donc $\xi_i$ reste proche de $\hat\lambda_i$ (mais corrigé du biais BBP).

### Implémentation pragmatique

En pratique, on peut implémenter le RIE en deux régimes selon que la valeur propre est dans le bulk ou un outlier :

> [!example] RIE en deux régimes
> **Pour les outliers** ($\hat\lambda_i > \lambda_+$) : on **inverse la formule BBP** pour retrouver la vraie valeur. Rappel : si $\theta$ est la force du vrai spike, on observe $\hat\lambda = (1+\theta)(1+q/\theta)$. En résolvant cette équation du second degré pour $\theta$ :
> $$\theta = \frac{(\hat\lambda - 1 - q) + \sqrt{(\hat\lambda - 1 - q)^2 - 4q}}{2}$$
> et la valeur propre nettoyée est $\xi_i = 1 + \theta$.
>
> **Pour le bulk** ($\hat\lambda_i \leq \lambda_+$) : on applique un shrinkage doux vers la moyenne du bulk :
> $$\xi_i = \bar\lambda_{\rm bulk} + \alpha \cdot (\hat\lambda_i - \bar\lambda_{\rm bulk})$$
> avec $\alpha$ petit (typiquement 0.2-0.4) pour atténuer fortement les déviations.
>
> Cette implémentation simplifiée capture l'essentiel : **débiaisage des outliers + lissage du bulk**.

### Comparaison visuelle des trois estimateurs

![[fig_rie_comparison.png]]
*Figure 3. Comparaison des estimateurs de spectre. **(a)** Vue d'ensemble en échelle log : BRUT s'étale sur 3 décades, CLIPPING écrase tout le bulk à $\bar\lambda \approx 1$, RIE et VRAI sont quasi-superposés. **(b)** Zoom sur les top-10 : RIE colle parfaitement aux vraies valeurs propres (7.03, 3.97, 2.49), alors que CLIPPING garde les valeurs brutes biaisées (7.63, 4.49, 3.45) — le biais positif visible dans la figure 3 de la note 03 est corrigé par RIE et pas par CLIPPING.*

C'est précisément la valeur ajoutée du RIE par rapport au clipping : **corriger le biais des outliers**, pas seulement écraser le bulk.

### Optimalité

> [!note]- Pourquoi le RIE est optimal
> Sous l'hypothèse RMT, le RIE oracle minimise l'erreur quadratique
> $$\|\hat\Sigma_{\rm clean} - \Sigma_{\rm vrai}\|_F^2$$
> parmi tous les estimateurs invariants par rotation (c'est-à-dire ceux qui gardent les vecteurs propres empiriques). C'est l'analogue du théorème d'optimalité du filtre de Wiener en traitement du signal.
>
> Le terme "oracle" indique qu'on suppose connaître la transformée de Stieltjes vraie — en pratique on l'estime depuis les données, ce qui introduit une petite erreur résiduelle. Mais les écarts à l'oracle sont d'ordre $1/\sqrt{N}$, négligeables pour les applications.

## V. Benchmark portefeuille — la preuve par les résultats

Voici le test critique : ces différents nettoyages améliorent-ils réellement le portefeuille en pratique ? On fait un backtest complet :

**Setup** : 
- $N=200$ actifs, $T=250$ jours d'estimation (donc $q=0.8$, régime difficile)
- 5 facteurs cachés dans la vraie covariance (1 marché + 4 sectoriels)
- 3000 jours de données simulées
- Rebalancement mensuel (21 jours), 130 rebalancements
- 4 méthodes : Brut, Clipping, Ledoit-Wolf, RIE

**La métrique clé** : pour un risk manager, la question n'est pas *"quel P&L cumulé ?"* (trop bruité, dépend du hasard) mais *"ma volatilité prédite est-elle proche de la volatilité réalisée ?"*. C'est le test de **calibration** — un estimateur qui sous-estime systématiquement le risque est dangereux indépendamment du P&L.

![[fig_calibration.png]]
*Figure 4. **(a)** Diagramme calibration : vol prédite vs vol réalisée sur chaque rebalancement. La diagonale pointillée représente la calibration parfaite. Brut (rouge) est dramatiquement hors-diagonale — il sous-estime systématiquement le risque réel d'un facteur 5. Les 3 méthodes nettoyées (orange/vert/violet) sont quasi-sur la diagonale. **(b)** Distribution du ratio réalisé / prédit : Brut a un ratio moyen de **4.96x**, alors que les méthodes nettoyées tournent autour de **1.2–1.4x** — quasi-calibrées.*

> [!warning] Résultat opérationnel
> | Méthode | Ratio réalisé/prédit | Diagnostic |
> |---|---|---|
> | **Brut** | **4.96x** | Désastreux : la vol réelle est 5x la promise |
> | Clipping | 1.24x | Bien calibré |
> | Ledoit-Wolf | 1.42x | Acceptable |
> | **RIE** | **1.29x** | Bien calibré et débiaisé sur les outliers |
>
> Sur ce setup ($q=0.8$), les trois méthodes nettoyées donnent des résultats similaires en termes de calibration. La différence entre clipping et RIE devient plus visible dans deux situations :
> 1. Quand on veut **estimer la vraie covariance** (pas juste son inverse pour Markowitz) — RIE est meilleur car débiaisé.
> 2. Quand on a beaucoup d'outliers (univers avec beaucoup de facteurs réels), où le clipping perd de l'information sur la hiérarchie des facteurs.

### Autres métriques typiques

Les benchmarks complets de la littérature (Bun-Bouchaud-Potters 2017, Bartz-Hatz 2021) comparent en plus :

- **Sharpe ratio out-of-sample** : ordre typique RIE > LW > Clipping > Brut, avec écarts notables sur des univers larges.
- **Maximum drawdown** : Brut produit des drawdowns 2-3x plus importants que les nettoyés.
- **Turnover annualisé** : Brut a un turnover ~3x supérieur, car les positions de bruit changent à chaque rebalancement.
- **Concentration (HHI sur les poids)** : Brut concentre 50% du portefeuille sur 10% des actifs, RIE diversifie naturellement.

## VI. Extensions et limites

### Non-stationnarité

Les rendements financiers ne sont pas stationnaires : les corrélations changent avec les régimes (calme vs crise, taux montants vs descendants). Deux extensions standard :

- **EWMA + RMT** : pondération exponentielle des observations passées avec $\lambda$ de l'ordre de 0.94-0.97. Le ratio effectif $q_{\rm eff} = q \cdot \frac{1+\lambda}{1-\lambda}$ doit être recalibré.
- **Régimes** : détecter des changements de régime via les outliers du spectre et adapter le nettoyage en conséquence.

### Modèles à facteurs explicites + RIE sur résidus

Au lieu de chercher les facteurs dans les données (sujet à BBP — limite des facteurs invisibles), on peut **les imposer** : Barra (facteurs style, industrie), Fama-French (taille, valeur, momentum), modèles factoriels macro. On retire la part expliquée par ces facteurs, puis on applique RIE sur la **covariance des résidus idiosyncratiques** pour nettoyer les corrélations résiduelles entre actifs.

C'est l'approche standard chez les gros gérants quantitatifs — et c'est exactement le pipeline qui te concerne (voir section VII).

### Cross-validation pour le seuil $\lambda_+$

En pratique, $\lambda_+ = (1 + \sqrt{q})^2$ est théorique : il suppose vraie covariance = identité, ce qui n'est jamais exactement le cas. On peut estimer un seuil empirique par cross-validation : tester plusieurs seuils, mesurer la calibration out-of-sample sur une fenêtre de validation, retenir le seuil optimal.

### Cas $q > 1$

Quand $N > T$ (plus d'actifs que d'observations), $\hat\Sigma$ est singulière et son inverse n'existe pas. Solutions : pseudoinverse régularisé de Moore-Penrose, ou réduire $N$ par sélection d'actifs préalable.

## VII. Lien avec ton expérience MS — RIE comme couche additionnelle

> [!example] Application potentielle : RIE sur les résidus Barra
> Le Barra factor model que tu connais bien décompose la covariance comme
> $$\Sigma = B \Omega B^\top + D$$
> où $B$ est la matrice des expositions aux facteurs, $\Omega$ la covariance des rendements de facteurs, et $D$ la covariance des **résidus idiosyncratiques** — typiquement supposée diagonale (variance spécifique par actif, pas de corrélation résiduelle).
>
> **Le problème** : en pratique, $D$ n'est pas parfaitement diagonale. Il y a des corrélations résiduelles entre actifs après extraction des facteurs Barra, notamment via :
> - des micro-structures (actifs du même groupe d'arbitrage)
> - des effets liquidité partagés
> - des facteurs cachés non capturés par Barra
>
> **L'idée** : appliquer **RIE sur la matrice de covariance des résidus** au lieu de l'imposer diagonale. On combine les forces :
> - Barra capture la structure macroscopique (facteurs explicites, économiquement interprétables, robustes).
> - RIE capture les corrélations résiduelles propres aux données, sans surfit.
>
> C'est exactement la "structure additionnelle a priori + nettoyage RMT" qui contourne la limite BBP des facteurs invisibles.

### Pourquoi c'est intéressant comme portfolio piece

Ce sujet hybride RMT/Barra est *exactement* ce qui te démarque sur le marché quant parisien :

1. **Tu as l'expérience Morgan Stanley** sur Barra (factor models, P&L decomposition, Greek P&L). C'est crédible quand tu en parles.
2. **Tu maîtrises RMT** (cette série de notes en témoigne). C'est rare combiné au point 1.
3. **L'intersection des deux** est un sujet de recherche actif et pas encore standardisé en industrie. C'est le genre de projet qui montre que tu sais penser au-delà des manuels.

**Comment le présenter** :
- Un **notebook complet** sur ton GitHub : extraction Barra-like de facteurs explicites sur S&P 500, calcul de la covariance résiduelle, comparaison "résiduelle diagonale" vs "résiduelle + RIE", backtest min-variance avec les deux.
- Un **post de blog** sur ton site Quartz, ciblant les recruteurs QRT/CFM/Squarepoint qui vérifient les profils techniques.
- Mention en **cover letter** comme exemple concret d'application de tes compétences théoriques.

Le côté hybride est ton angle : la plupart des candidats sont soit "Barra-only" (vétérans actions) soit "RMT-only" (jeunes diplômés théoriciens). Toi tu fais le pont, et c'est ce que les fonds cherchent pour l'évolution de leurs modèles internes.

## Ce qui suit

Les notes 01-04 couvrent l'intégralité du cœur opérationnel RMT en finance : théorie (01), résultat clé MP (02), détection (03), nettoyage (04). Tu peux t'arrêter ici et avoir un pipeline complet.

[[05_Free_Probability|Note 05 — Free probability et applications avancées]] : pour aller plus loin théoriquement (covariances non-identité, deep learning, graph signal processing).
