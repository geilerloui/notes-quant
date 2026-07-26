---
title: Batch RL (Offline RL)
---
# Batch RL (Offline RL)

Si tes plans précédents se concentraient sur l'apprentissage "en ligne" (où l'agent interagit avec l'environnement), le Batch RL s'intéresse à : **"Comment apprendre la meilleure politique possible à partir d'un jeu de données fixe, sans aucune interaction supplémentaire ?"**

![[Pasted image 20260501174702.png]]

## Introduction

Offline RL, aussi appelé Batch Reinforcement Learning, est une variante du RL où l'agent doit apprendre à partir d'un batch de données **fixe, sans exploration**. On se concentre ici surtout sur le *safe* batch RL (avec garanties de sécurité).

**Exemple 1.** Expérience scientifique avec deux groupes A et B. Le groupe A résout d'abord des fractions en les réduisant au plus petit terme, puis fait de la multiplication croisée ; ils obtiennent en moyenne 95/100 à l'examen final. Le groupe B fait la même chose mais dans l'ordre inverse (multiplication croisée d'abord) ; ils obtiennent 92/100 en moyenne.

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im1.png]]

Pour un nouvel étudiant, que choisir ? On pourrait directement suivre la séquence du groupe A, mais si son échantillon était bien plus petit que celui de B, l'estimation du score serait biaisée. Si en plus les deux groupes diffèrent sur d'autres points, on retombe sur un problème de **données censurées** — on pourrait aussi raisonner en **contrefactuel** : qu'aurait-il obtenu dans l'autre groupe ?

**Exemple 2 (maintenance).** Trouver la séquence d'actions optimale pour qu'une machine tourne le plus longtemps possible.

**Exemple 3 (santé).** Trouver la séquence de soins qui maximise le résultat pour un patient. Difficulté supplémentaire : c'est **dépendant de l'état** — ce qui marche pour le patient A peut ne pas marcher pour B. C'est un scénario à **enjeu élevé** (*high stake*) : se tromper coûte cher, d'où le besoin de bonnes garanties avant de décider.

**Le problème.** Il faut des **bornes de confiance**, pour que le médecin sache à quel point il peut faire confiance à la politique avant de la déployer.

**Types de sécurité recherchés :**
- **(i) Safe exploration** — ne pas faire d'erreur pendant l'exploration en ligne.
- **(ii) Risk sensitivity** — chaque individu ne vit qu'**un seul** résultat (pas l'espérance) ; il faut en tenir compte.
- **(iii) Confiance relative à une politique de référence** — celle actuellement en place. On va montrer qu'on peut garantir une **amélioration monotone** par rapport à elle.

**Notation.**
- Politique : $\pi : \pi(a) = P(a_t = a \mid s_t = s)$
- Trajectoire : $h = T = (s_1, a_1, r_1, s_2, a_2, r_2, \ldots, s_L, a_L, r_L)$
- Données historiques : $D = \{T_1, T_2, \ldots, T_n\}$
- Données générées par une politique de comportement (*behavior policy*) $\pi_b$. Si elle n'est pas connue, on peut l'estimer par simulation Monte Carlo, en moyennant les retours observés : $\hat V^{\pi_b} = \frac{1}{n}\sum_{i=1}^n G_i$.
- Objectif :

$$V^\pi = \mathbb{E}\left[\sum_{t=1}^L \gamma^t R_t \,\middle|\, \pi\right]$$

$D$, ce sont les données auxquelles on a accès : dossiers médicaux électroniques, données d'une centrale électrique... Pour simplifier, on suppose connue la politique de comportement $\pi_b$ qui a généré ces données. Ce n'est pas toujours réaliste (données générées par des humains), mais pour des Google Ads par exemple on connaît exactement l'algorithme utilisé, ou les trajectoires ont pu être générées par un agent RL déjà connu.

**Objectif (Goal).**
- On dispose d'un algorithme de RL $\mathcal A$.
- De données historiques $D$ (une variable aléatoire).
- La politique produite par l'algorithme, $\mathcal A(D)$ (aussi une variable aléatoire).
- Un algorithme de *safe batch RL* $\mathcal A$ doit satisfaire :

> [!warning] Garantie de sécurité
> $$\Pr\big(V^{\mathcal A(D)} \geq V^{\pi_b}\big) \geq 1 - \delta$$
>
> ou, plus généralement :
>
> $$\Pr\big(V^{\mathcal A(D)} \geq V_{\min}\big) \geq 1 - \delta$$

En clair : la valeur retournée par notre algorithme est *au moins aussi bonne* que celle de la politique de comportement — et la version générale demande même d'être **substantiellement** meilleure (d'un facteur $V_{\min}$). C'est pertinent en production réelle : changer un protocole coûte du temps et de l'argent, donc autant s'assurer d'un gain net suffisant avant de le faire.

## Le plan (Model)

- **Off-policy Policy Evaluation (OPE).** Pour une politique d'évaluation $\pi_e$ donnée, transformer les données historiques $D$ en $n$ estimateurs indépendants et non biaisés de $V^{\pi_e}$.
- **High-Confidence OPE (HCOPE).** Utiliser une inégalité de concentration pour transformer ces $n$ estimateurs en une borne inférieure de confiance $1-\delta$ sur $V^{\pi_e}$.
- **Safe Policy Improvement (SPI).** Utiliser HCOPE pour construire un algorithme de safe batch RL complet.

### Off-policy Policy Evaluation (OPE)

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im2.png]]

On veut estimer la valeur $V^{\pi_1}(s)$ d'une politique $\pi_1$, à partir d'épisodes générés sous une politique de comportement $\pi_2$ :

$$D \sim \{s_1, a_1, r_1, s_2, a_2, r_2, \ldots\}$$

C'est un cadre **off-policy** : on veut que $\pi_1$ "ressemble" à $\pi_2$ dans l'estimation. On pourrait utiliser Q-learning (déjà off-policy) et bootstraper, mais le bootstrap introduit du biais — avec peu de données, on préfère une approche non biaisée : l'**importance sampling**.

> [!warning] Importance Sampling — dérivation
> Pour un épisode (historique) $h_j$, la probabilité de cette trajectoire sous une politique $\pi$ se factorise :
>
> $$\begin{aligned}
> p(h_j \mid \pi, s = s_{j,1}) &= p(a_{j,1}\mid s_{j,1})\,p(r_{j,1}\mid s_{j,1},a_{j,1})\,p(s_{j,2}\mid s_{j,1},a_{j,1}) \\
> &\quad \times\, p(a_{j,2}\mid s_{j,2})\,p(r_{j,2}\mid s_{j,2},a_{j,2})\,p(s_{j,3}\mid s_{j,2},a_{j,2}) \ldots \\
> &= \prod_{t=1}^{L_j - 1} p(a_{j,t}\mid s_{j,t})\,p(r_{j,t}\mid s_{j,t},a_{j,t})\,p(s_{j,t+1}\mid s_{j,t},a_{j,t}) \\
> &= \prod_{t=1}^{L_j - 1} \pi(a_{j,t}\mid s_{j,t})\,p(r_{j,t}\mid s_{j,t},a_{j,t})\,p(s_{j,t+1}\mid s_{j,t},a_{j,t})
> \end{aligned}$$
>
> Le ratio de vraisemblance entre $\pi_e$ et $\pi_b$ (les termes de dynamique/récompense s'annulent, ils ne dépendent pas de la politique) :
>
> $$\frac{p(h_j \mid \pi_e)}{p(h_j \mid \pi_b)} = \prod_{t=1}^{L_j-1} \frac{\pi_e(a_{j,t}\mid s_{j,t})}{\pi_b(a_{j,t}\mid s_{j,t})}$$
>
> D'où l'estimateur :
>
> $$V^{\pi_1}(s) \approx \frac{1}{n}\sum_{j=1}^n \frac{p(h_j\mid \pi_1, s)}{p(h_j \mid \pi_2, s)}\, G(h_j) = \frac{1}{n}\sum_{j=1}^n \left(\prod_{t=1}^{L}\frac{\pi_e(a_{j,t}\mid s_{j,t})}{\pi_b(a_{j,t}\mid s_{j,t})}\right) G(h_j)$$

Si une trajectoire est *plus probable* sous la politique d'évaluation $\pi_e$ que sous $\pi_b$, on **sur-pondère** sa récompense ; si elle est *moins probable*, on la **sous-pondère**. On a des paires $(h_j, G_j)$ (trajectoire, retour cumulé), et on les pondère par ce ratio.

C'est un estimateur **non biaisé**, qui ne nécessite ni de connaître la dynamique, ni la fonction de récompense, ni même la propriété de Markov. Le problème : si $\pi_b(a\mid s)$ est très petit, le ratio explose. $\pi_b$ ne peut pas être nul là où on a des données (sinon la trajectoire n'existerait pas), mais elle peut être *très* petite.

> [!info] Couverture (support)
> Si $\pi_b(a\mid s) = 0$ alors que $\pi_e(a\mid s) > 0$, on parle de trajectoire hors-couverture (*out of support*) — ça n'a pas de sens statistiquement. On exige : $\pi_b(a\mid s) > 0$ pour tout $(a,s)$ tel que $\pi_e(a\mid s) > 0$.

L'IS pour la Policy Evaluation est **consistant** : asymptotiquement (sous l'hypothèse de couverture), $\hat V^\pi(s) \xrightarrow{n \to \infty} V^\pi(s)$.

**Autres estimateurs.** En réécrivant l'IS dérivé ci-dessus :

$$IS(D) = \frac{1}{n}\sum_{i=1}^n \left(\prod_{t=1}^L \frac{\pi_e(a_t\mid s_t)}{\pi_b(a_t\mid s_t)}\right)\left(\sum_{t=1}^L \gamma^t R_t^i\right)$$

**Per-Decision Importance Sampling (PDIS)** exploite le fait que c'est un processus temporel — comme en policy gradient, le futur ne peut pas affecter les récompenses passées :

$$PDIS(D) = \sum_{t=1}^L \gamma^t \frac{1}{n}\sum_{i=1}^n \left(\prod_{\tau=1}^t \frac{\pi_e(a_\tau\mid s_\tau)}{\pi_b(a_\tau\mid s_\tau)}\right) R_t^i$$

**Weighted Importance Sampling (WIS)** :

$$WIS(D) = \frac{1}{\sum_{i=1}^n w_i}\sum_{i=1}^n w_i \left(\sum_{t=1}^L \gamma^t R_t^i\right)$$

En IS classique, les poids $w_i$ peuvent être minuscules si $\pi_b(a\mid s)$ l'est. WIS renormalise par la somme des poids — utile si les poids d'importance sont très petits pour certaines trajectoires. Résultat : **biaisé mais consistant**, et de variance plus faible — un compromis biais/variance classique.

**Autre alternative — control variates.** D'un point de vue statistique : soient $X$ et $Y$ deux variables aléatoires, où $Y$ est la *control variate*. On décale l'estimateur en soustrayant $Y$ et $\mathbb{E}[Y]$ (par exemple, $Y$ une Q-valeur estimée, $\mathbb{E}[Y]$ une $V$ estimée). Si $\text{Cov}(X,Y) > \text{Var}(Y)$, l'estimateur résultant a une variance réduite : $\text{Var}(\hat\mu) < \text{Var}(X)$.

**Doubly Robust estimator.** Combine control variate (une estimation de $Q$) et importance sampling — "doublement robuste" car l'estimation reste correcte si **l'un des deux** est faux (modèle de $Q$ imparfait, ou IS seul seraient tous deux dégradés indépendamment).

**Expérience empirique.** Petit gridworld $5\times5$ pour comparer ces techniques. Sur l'axe X la taille du dataset, sur l'axe Y la MSE :

$$MSE = \left(\hat V^{\pi_e} - V^{\pi_e}\right)^2$$

- **Model-based** (en noir) : plat, ne s'améliore pas avec plus de données (biais fixe du modèle approximatif).
- **IS** (en jaune) : non biaisé, tend vers 0 avec plus de données.
- **PDIS** : décalage vers le bas par rapport à IS.

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im6.png]]

- **Doubly Robust** combine modèle approximatif + IS : gain net important. Pour atteindre la même MSE, IS a besoin d'environ 2000 épisodes là où Doubly Robust n'en demande que ~200.
- **WIS et CWPDIS** aident aussi beaucoup.
- **WDR** (Weighted Doubly Robust) — meilleur compromis biais/variance : seulement ~5 épisodes suffisent ici. Uniquement un gridworld-jouet, mais le gain se confirme sur des jeux plus complexes.

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im7.png]]

**Blending.** Comment arbitrer le compromis biais/variance de façon systématique ?

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im8.png]]

L'estimateur **MAGIC** essaie de minimiser directement la MSE, par définition $MSE = f(\text{bias}, \text{var})$, avec $\text{bias} = \hat V^{\pi_e} - V^{\pi_e}$ — qu'on ne connaît pas directement. Idée : utiliser un intervalle de confiance (obtenu via importance sampling). Si un modèle donne une estimation en dehors de cet IC, on peut quantifier un biais minimal (ex : IC $[3,7]$, estimation du modèle en dehors → biais minimal de $1$).

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im9.png]]

### High-Confidence Off-policy Policy Evaluation (HCOPE)

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im3 (1).png]]

On veut :

$$\Pr\big(V^{\mathcal A(D)} - V^{\pi_b} > 0\big) \geq 1 - \delta$$

Comment faire ? Combiner IS et l'**inégalité de Hoeffding** — illustré sur Mountain Car (l'agent doit atteindre le drapeau, on veut une borne de confiance sur sa performance).

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im11.png]]

> [!warning] Inégalité de Hoeffding
> Soient $X_1, \ldots, X_n$ i.i.d. avec $X_i \in [0,b]$. Alors avec probabilité au moins $1-\delta$ :
>
> $$\mathbb{E}[X_i] \geq \frac{1}{n}\sum_{i=1}^n X_i - b\sqrt{\frac{\ln(1/\delta)}{2n}}$$
>
> où, dans notre cas, $X_i = \frac{1}{n}\sum_{i=1}^n \left(w_i \sum_{t=1}^L \gamma^t R_t^i\right)$.

Avec $100\,000$ trajectoires, la vraie performance de la politique d'évaluation est $V = 0.19 \in [0,1]$. Mais la borne inférieure à $95\%$ obtenue est de l'ordre de $-5\,831\,000$ — techniquement correcte (c'est bien une borne inférieure), mais totalement inexploitable : l'écart est énorme. Pourquoi ?

Le produit de poids d'importance $w_i = \prod_{t=1}^L \frac{\pi_e(a_t\mid s_t)}{\pi_b(a_t\mid s_t)}$ peut être minuscule à chaque pas (ex. $0.1$) — sur une séquence de $L$ pas, ça donne $(1/0.1)^L$ dans le pire cas. Sur Mountain Car il faut une séquence d'actions très spécifique pour réussir, donc la plupart des trajectoires de $D$ ne réussissent pas et le rare succès a un poids d'importance énorme. Or la borne de Hoeffding dépend de $b$, le maximum possible de la quantité pondérée :

$$b = \max\left(G \cdot \prod_{t=1}^L \frac{\pi_e(a_t\mid s_t)}{\pi_b(a_t\mid s_t)}\right)$$

— un maximum énorme puisque la plupart des trajectoires sont improbables sous $\pi_e$. Avec $L=200$ et un ratio de $0.1$ par pas, $b \sim (1/0.1)^{200}$ : la borne devient vide de sens (*vacuous*).

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im10 (1).png]]

**Solution.** Tronquer la queue de distribution : ignorer les retours pondérés extrêmement grands (peu probables) fait baisser l'estimation moyenne — on **sous-estime** volontairement la politique, en échange d'un intervalle de confiance exploitable.

> [!warning] Théorème (borne avec seuil de troncature)
> Soient $X_1, \ldots, X_n$ des variables aléatoires réelles indépendantes telles que pour chaque $i$, $\mathbb{P}[0 \leq X_i] = 1$, $\mathbb{E}[X_i] \leq \mu$, et un seuil $c_i > 0$. Soit $\delta > 0$ et $Y_i := \min\{X_i, c_i\}$. Alors avec probabilité au moins $1-\delta$ :
>
> $$\mu \geq \underbrace{\left(\sum_{i=1}^n \frac{1}{c_i}\right)^{-1}\sum_{i=1}^n \frac{Y_i}{c_i}}_{\text{moyenne empirique}} - \underbrace{\left(\sum_{i=1}^n \frac{1}{c_i}\right)^{-1}\frac{7n\ln(2/\delta)}{3(n-1)}}_{\to 0 \text{ en } 1/n} - \underbrace{\left(\sum_{i=1}^n \frac{1}{c_i}\right)^{-1}\sqrt{\frac{\ln(2/\delta)}{n-1}\sum_{i,j=1}^n \left(\frac{Y_i}{c_i} - \frac{Y_j}{c_j}\right)^2}}_{\to 0 \text{ en } 1/\sqrt{n}}$$

**Résultats.** $20\%$ des données servent à optimiser le seuil $c$, les $80\%$ restants à calculer la borne inférieure avec ce $c$ optimisé. Sur Mountain Car, la borne obtenue est $0.145$ — à comparer aux autres inégalités de concentration, bien pires :

| | CUT | Chernoff-Hoeffding | Maurer | Anderson | Bubeck et al. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Borne inf. de confiance à $95\%$ sur la moyenne | $0.145$ | $-5\,831\,000$ | $-129\,703$ | $0.055$ | $-0.46$ |

**Application (marketing digital).** Avant de déployer une politique jugée plus efficace, on peut quantifier la confiance qu'on a dans un gain de revenu — et cette borne peut être suffisamment haute pour confirmer que la politique est bien meilleure.

**Alternative (t-test).** Le test t de Student marche aussi très bien empiriquement.

### Safe Policy Improvement (SPI)

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im4.png]]

SPI répond à la question : étant donné des données, peut-on trouver une politique **garantie meilleure**, ou l'algorithme doit-il reconnaître qu'il ne peut proposer aucune amélioration sûre ? Avec trop peu de données, il est parfois impossible de garantir une amélioration.

Le pipeline SPI complet :

![[images/3-Apprentissage automatique/07_Reinforcement learning/Offline/im5.png]]

On peut appliquer ça à plusieurs politiques candidates de façon sûre : une partie des données sert à optimiser, une autre à évaluer la politique résultante (split train/test, comme en ML classique — évite l'overfitting sur les mêmes données).

> [!note] À compléter
> - Résultats empiriques sur un cas marketing (non détaillé dans la source).
> - Autres travaux pertinents sur SPI (non détaillé dans la source).
