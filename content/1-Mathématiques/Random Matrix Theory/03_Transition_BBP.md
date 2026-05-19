---
title: Transition BBP et détection de facteurs
date: 2026-05-12
tags: [random-matrix-theory, bbp, detection, tracy-widom, pca]
---

> [!warning] Prérequis
> Lecture de [[01_Fondamentaux]] (en particulier la section sur Tracy-Widom) et de [[02_Marchenko_Pastur]] (densité MP, support $[\lambda_-, \lambda_+]$). On reprend la matrice de Wishart $W = \frac{1}{T} X X^\top$ et le ratio $q = N/T$.

## I. Le problème : qu'est-ce qu'un "facteur" et comment le voit-on ?

Dans la note 02, on a étudié le cas "bruit pur" : la vraie covariance vaut l'identité $\mathbb{I}_N$, tous les actifs sont indépendants. On a vu que **même dans ce cas trivial**, le spectre empirique s'étale sur $[\lambda_-, \lambda_+]$ — il y a du bruit d'estimation pur, décrit exactement par Marchenko-Pastur.

Mais la finance ne te sert pas du bruit pur. Il y a des **vrais facteurs** dans les rendements : le facteur marché global (tous les actifs montent ou descendent ensemble), des facteurs sectoriels (les banques bougent ensemble, les techs bougent ensemble), des facteurs de style (valeur, momentum, taille). Tous ces facteurs créent de la corrélation **réelle** entre actifs.

> [!example] Qu'est-ce qu'un facteur, concrètement ?
> Un **facteur** est une direction $u \in \mathbb{R}^N$ telle que tous les actifs réagissent à un signal commun selon leur coordonnée dans $u$. Si $u_i$ est grand, l'actif $i$ est très exposé au facteur ; si $u_i$ est petit, il l'est peu.
>
> **Le facteur marché** est l'exemple classique : $u_{\rm marché} = (1, 1, \dots, 1)/\sqrt{N}$. Tous les actifs ont la même exposition, et quand le marché bouge, ils bougent tous ensemble. Empiriquement, ce facteur explique typiquement 30-50% de la variance des rendements actions.
>
> Mathématiquement, un facteur de **force** $\theta$ ajoute le terme $\theta \, u u^\top$ à la matrice de covariance vraie. Plus $\theta$ est grand, plus la corrélation induite par ce facteur est forte.

**La question centrale.** Si la "vraie" covariance contient un facteur (donc *vraiment* du signal), peut-on le détecter dans le spectre empirique ? Sous quelles conditions ? Et surtout : y a-t-il des facteurs réels qui resteraient *invisibles* malgré leur existence ?

## II. Modélisation : la vraie covariance avec spike

### Construction pas à pas

On part de la covariance "baseline" où il n'y a aucune structure :
$$\Sigma_{\rm baseline} = \mathbb{I}_N$$
Tous les actifs sont indépendants, chacun de variance 1. C'est le cas étudié en note 02.

On ajoute maintenant **un facteur** de direction $u$ (unitaire, $\|u\| = 1$) et de force $\theta > 0$ :
$$\boxed{\;\Sigma_{\rm vrai} = \mathbb{I}_N + \theta \, u u^\top\;}$$

Le terme $\theta u u^\top$ est ce qu'on appelle une **perturbation de rang 1** — c'est une matrice qui ne perturbe l'identité que dans *une seule direction*, à savoir $u$.

> [!note]- Pourquoi rang 1 ?
> Une matrice $u u^\top$ avec $u \in \mathbb{R}^N$ a la propriété suivante : son image est la droite engendrée par $u$. Donc cette matrice a un seul vecteur propre non trivial (qui est $u$ lui-même, avec valeur propre $\|u\|^2 = 1$), et toutes les autres directions sont dans son noyau.
>
> On dit qu'elle est de **rang 1** car elle peut s'écrire comme produit d'une colonne par une ligne : $u u^\top = u \cdot u^\top$.
>
> C'est l'objet jouet de toute la théorie de la détection de signal : on perturbe l'identité dans *une seule direction* et on regarde ce qui se passe. Si on veut plusieurs facteurs, on additionne plusieurs termes rang 1 (comme dans la figure 2 de la note 02).

### Spectre vrai

Quel est le spectre de $\Sigma_{\rm vrai} = \mathbb{I}_N + \theta u u^\top$ ?

- Le vecteur $u$ est vecteur propre : $\Sigma_{\rm vrai} u = u + \theta (u u^\top) u = u + \theta u = (1 + \theta) u$. Valeur propre $1 + \theta$.
- N'importe quel vecteur $v$ orthogonal à $u$ vérifie $u u^\top v = u (u^\top v) = 0$, donc $\Sigma_{\rm vrai} v = v$. Valeur propre $1$ (avec multiplicité $N - 1$).

> [!warning] Spectre vrai de $\Sigma_{\rm vrai} = \mathbb{I}_N + \theta u u^\top$
> $$\text{spectre}(\Sigma_{\rm vrai}) = \{ \underbrace{1, 1, \dots, 1}_{N-1\ \text{fois}},\ 1 + \theta \}$$
>
> Autrement dit : dans la "vraie" théorie, ce facteur est **trivialement** visible — il a sa propre valeur propre $1 + \theta$, parfaitement isolée de toutes les autres qui sont à 1.

![[fig_bbp_spectre_vrai.png]]
*Figure 1. Spectre vrai de $\Sigma_{\rm vrai} = \mathbb{I}_N + \theta u u^\top$ pour $N = 20, \theta = 3$. $N-1$ valeurs propres à $\lambda = 1$ (toutes les directions "baseline") et une seule valeur propre à $1 + \theta = 4$ (la direction perturbée $u$). Si on connaissait $\Sigma_{\rm vrai}$, le facteur serait identifié sans ambiguïté.*

## III. La question centrale

On n'observe **pas** $\Sigma_{\rm vrai}$ — on observe seulement $\hat\Sigma = \frac{1}{T} X X^\top$ calculée sur $T$ jours de rendements. Comment le spectre vrai $\{1+\theta, 1, 1, \dots, 1\}$ se transforme-t-il en spectre empirique ?

Trois scénarios *a priori* possibles :

1. **Détection nette** : le pic vrai à $1+\theta$ reste isolé dans l'empirique, et les $N-1$ autres valeurs propres s'étalent autour de 1 selon MP. Le facteur est clairement visible.
2. **Disparition** : le pic vrai est *noyé* dans le bulk MP, indistinguable du bruit d'estimation. Le facteur est invisible malgré son existence réelle.
3. **Cas intermédiaire** : ça dépend de la force $\theta$ et du ratio $q$.

C'est le scénario 3 qui est correct, et BBP nous dit *exactement* à partir de quel seuil on bascule de l'un à l'autre.

## IV. Expérience numérique : varions $\theta$ et observons

Avant d'énoncer le théorème, déroulons l'expérience. On fixe $N = 300, T = 600$ (donc $q = 0.5$), et on fait varier $\theta$ de 0.3 à 3. Pour chaque $\theta$, on simule la covariance empirique et on regarde son spectre.

![[fig_bbp_panneaux.png]]
*Figure 2. Spectre empirique pour 4 valeurs croissantes de $\theta$ à $q = 0.5$ fixé. La courbe rouge est la densité MP théorique. La tige verticale marque la plus grande valeur propre empirique $\lambda_1$ : en rouge si elle reste collée à l'edge MP (facteur invisible), en vert si elle sort.*

**Lecture panneau par panneau :**

- **$\theta = 0.3$ (panneau 1).** Il y a un vrai facteur de force 0.3 dans la covariance, mais le spectre empirique ressemble *exactement* à du bruit pur. La plus grande valeur propre est $\lambda_1 = 2.86$, soit pile à l'edge MP $\lambda_+ = 2.91$. Le facteur est **invisible**.

- **$\theta = 0.7$ (panneau 2).** On est juste sous le seuil critique $\theta_c = \sqrt{q} \approx 0.71$. Le spike est encore invisible : $\lambda_1 = 2.88$, toujours collé à l'edge. C'est dérangeant : un facteur réel de force 0.7 n'est *pas* détecté.

- **$\theta = 1.5$ (panneau 3).** Maintenant on est au-dessus du seuil. Une valeur propre $\lambda_1 = 3.26$ sort clairement du bulk. La prédiction théorique BBP (qu'on va énoncer dans une minute) vaut $(1+\theta)(1+q/\theta) = 3.33$ — l'écart est minime.

- **$\theta = 3.0$ (panneau 4).** Facteur fort : $\lambda_1 = 4.36$, bien au-dessus du bulk. BBP prédit 4.67, encore très bon.

**Conclusion empirique.** Il existe un *seuil* sous lequel le facteur est invisible et au-dessus duquel il est détectable. La position de ce seuil n'a rien d'évident *a priori* — c'est BBP qui la donne.

## V. Le théorème BBP

> [!warning] Théorème (Baik–Ben Arous–Péché, 2005)
> Soit $\hat\Sigma = \frac{1}{T} X X^\top$ la covariance empirique d'un échantillon de taille $T$ tiré selon $\mathcal{N}(0, \Sigma_{\rm vrai})$ avec $\Sigma_{\rm vrai} = \mathbb{I}_N + \theta u u^\top$. Dans la limite $N, T \to \infty$ à ratio $q = N/T$ fixé :
>
> **Régime sous-critique** : si $\theta \leq \sqrt{q}$, alors
> $$\lambda_1^{\rm emp} \xrightarrow{p.s.} \lambda_+ = (1 + \sqrt{q})^2$$
> La plus grande valeur propre empirique **reste collée à l'edge MP**. Le spike est indistinguable du bulk.
>
> **Régime sur-critique** : si $\theta > \sqrt{q}$, alors
> $$\lambda_1^{\rm emp} \xrightarrow{p.s.} (1 + \theta)\left(1 + \frac{q}{\theta}\right) > \lambda_+$$
> La plus grande valeur propre empirique **sort du bulk**, à une position prédite explicitement.

Le **seuil critique** est :
$$\boxed{\;\theta_c = \sqrt{q}\;}$$

> [!example] Pourquoi $\sqrt{q}$ ?
> Intuition rapide : la force du vrai facteur doit dépasser le "niveau de bruit d'estimation", qui scale comme $\sqrt{q}$. Plus précisément, l'edge MP est à $(1+\sqrt{q})^2 = 1 + 2\sqrt{q} + q$. Pour qu'un facteur de force $\theta$ soit détectable, il faut grosso modo qu'il pousse le pic au-dessus de cet edge — ce qui correspond à $\theta > \sqrt{q}$ (la preuve rigoureuse passe par l'équation auto-cohérente sur la résolvante, mais le scaling est celui-ci).
>
> **Asymétrie cruciale.** Le seuil n'est *pas* à $\theta_c = 0$, contrairement à ce qu'on pourrait naïvement attendre. Même un facteur réel non nul peut être invisible si sa force ne dépasse pas $\sqrt{q}$.

## VI. Vérification numérique : la bifurcation

Pour visualiser le théorème, on trace $\lambda_1^{\rm emp}$ en fonction de $\theta$ sur une grille fine, avec plusieurs tirages par valeur de $\theta$.

![[fig_bbp_bifurcation.png]]
*Figure 3. Diagramme de bifurcation BBP. Chaque point bleu = un tirage de matrice. Courbe rouge = prédiction théorique BBP. Courbe verte pointillée = spectre vrai $1 + \theta$. À gauche du seuil $\theta_c = \sqrt{q}$ : $\lambda_1^{\rm emp}$ reste plat à $\lambda_+$, peu importe la force du spike. À droite : $\lambda_1^{\rm emp}$ décolle et suit BBP.*

Trois observations importantes :

1. **La bifurcation est nette.** À gauche du seuil, $\lambda_1^{\rm emp} \approx \lambda_+$ avec des fluctuations TW de taille $N^{-2/3}$ (voir note 01). À droite, $\lambda_1^{\rm emp}$ suit BBP exactement.

2. **L'estimation sur-estime systématiquement.** La courbe rouge BBP est **au-dessus** de la verte pointillée $1+\theta$. Autrement dit, même quand on détecte un facteur de force vraie $\theta$, on observe une valeur propre empirique *plus grande* que $1+\theta$ à cause du bruit d'estimation. Ce **biais positif** est fondamental, et c'est précisément ce que le RIE (note 04) va corriger.

3. **Asymptote.** Quand $\theta \to \infty$, $(1+\theta)(1 + q/\theta) \to 1 + \theta + q$, donc l'écart à la vraie valeur tend vers $q$ (constant). Le biais d'estimation ne disparaît jamais — il devient juste petit relativement à la valeur observée.

## VII. Conséquence dure : les facteurs invisibles

Le théorème BBP a une conséquence pratique qu'il faut digérer : **il existe des facteurs réels que la PCA empirique ne peut pas détecter, peu importe la qualité du calcul.** Ce n'est pas un problème d'algorithme, c'est une limite fondamentale.

> [!warning] Tout facteur de force $\theta < \sqrt{q}$ est invisible
> Peu importe le nombre de simulations, l'algorithme, ou la finesse de l'analyse : si le ratio $q = N/T$ est trop grand par rapport à $\theta$, le facteur reste noyé dans le bulk MP. La seule manière de le détecter est :
> - **Plus de données** : faire baisser $q$ en augmentant $T$ (mais on perd la pertinence temporelle des observations anciennes).
> - **Moins d'actifs** : faire baisser $q$ en réduisant $N$ (sélection a priori).
> - **Structure additionnelle** : utiliser des facteurs *explicites* construits a priori (Barra, Fama-French, etc.) plutôt que de chercher des facteurs dans les données.

### Quantification pour des univers réels

| Univers | $N$ | $T$ (1 an) | $q = N/T$ | Seuil $\theta_c = \sqrt{q}$ |
|---|---|---|---|---|
| CAC 40 | 40 | 250 | 0.16 | 0.40 |
| EuroStoxx 50 | 50 | 250 | 0.20 | 0.45 |
| S&P 500 | 500 | 250 | 2.0 | 1.41 |
| Russell 3000 | 3000 | 250 | 12.0 | 3.46 |

**Lecture.** Sur le CAC 40, il suffit d'un facteur de force 0.4 pour qu'il soit détectable — assez clément. Sur le Russell 3000, il faut un facteur **énorme** ($\theta > 3.46$) pour être visible. Cela explique pourquoi les modèles à facteurs explicites (type Barra) sont la norme sur les grands univers : la détection empirique pure est mathématiquement trop bruitée.

> [!note]- Lien avec ton expérience MS / Barra
> Le Barra factor model construit explicitement des facteurs (style, industrie, taille) à partir de connaissances extérieures aux rendements eux-mêmes. C'est précisément la stratégie "structure additionnelle" mentionnée plus haut — on contourne la limite BBP en injectant de l'information *a priori*. RMT et Barra ne sont pas concurrents : ils se complètent, l'un (Barra) pour la partie structurée, l'autre (RIE, note 04) pour le nettoyage des résidus idiosyncratiques.

## VIII. Test statistique rigoureux avec Tracy-Widom

BBP nous dit ce qui se passe **asymptotiquement** ($N, T \to \infty$). À $N$ fini, $\lambda_1^{\rm emp}$ fluctue autour de sa valeur limite. Pour décider rigoureusement *"cette valeur propre observée est-elle significativement au-dessus du bulk ?"*, il faut connaître la distribution de $\lambda_{\max}$ sous l'hypothèse nulle H₀ = "pas de signal" (bruit pur, $\theta = 0$).

C'est exactement ce que donne la **loi de Tracy-Widom** (vue en section C de la note 01) : la fluctuation de $\lambda_{\max}$ autour de $\lambda_+$ a une distribution déterministe, à l'échelle $N^{-2/3}$.

### Procédure de test

**Hypothèse nulle H₀** : la vraie covariance est l'identité (bruit pur).

**Statistique du test** : on calcule $\lambda_1^{\rm emp}$ observé, et on regarde
$$p\text{-value} = P_{H_0}\!\left( \lambda_{\max} \geq \lambda_1^{\rm emp} \right)$$
où la probabilité est calculée sous H₀.

**Décision** : si $p < 0.05$ (ou plus strict, $p < 0.01$), on rejette H₀ — il y a un signal réel. Sinon, $\lambda_1^{\rm emp}$ est compatible avec une fluctuation TW de bruit pur, on ne peut pas conclure à l'existence d'un facteur.

![[fig_bbp_tw_test.png]]
*Figure 4. Test de détection en pratique. Distribution simulée de $\lambda_{\max}$ sous H₀ (4000 tirages de bruit pur). Deux observations : (gris) un cas sous-critique $\theta = 0.4$ où $\lambda_{\rm obs} = 2.81$ tombe au cœur de la distribution H₀ (p ≈ 0.83) — non significatif. (vert) un cas sur-critique $\theta = 1.5$ où $\lambda_{\rm obs} = 3.45$ est tellement au-delà du seuil 1% que p ≈ 0 — signal clairement détecté.*

### Calcul pratique de la p-value

Deux options :

1. **Simulation Monte-Carlo** (utilisée pour la figure 4) : on simule plusieurs milliers de matrices sous H₀ et on calcule la p-value empirique. Robuste, mais coûteux et peu élégant.

2. **Formule analytique TW** : on utilise la queue droite de la loi Tracy-Widom $\mathrm{TW}_1$ tabulée. La p-value est
$$p = P\!\left( \xi \geq c_{N,T} \, (\lambda_1^{\rm emp} - \lambda_+) \right)$$
où $\xi \sim \mathrm{TW}_1$ et $c_{N,T}$ est un facteur de rescaling explicite dépendant de $q$. Disponible dans `scipy.stats` via une approximation ou dans le package `wishart` dédié.

## IX. Application : combien de facteurs garder en PCA ?

Voici la procédure complète, rigoureuse, qui généralise BBP au cas de plusieurs spikes potentiels :

> [!warning] Procédure "PCA + BBP" pour la sélection de facteurs
> 1. Calculer la covariance empirique $\hat\Sigma$ et son spectre $\{\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_N\}$.
> 2. Calculer $q = N/T$ et $\lambda_+ = (1+\sqrt{q})^2$.
> 3. Pour chaque valeur propre $\lambda_i$ supérieure à $\lambda_+$, appliquer le test Tracy-Widom et calculer sa p-value.
> 4. Le **nombre de facteurs significatifs** = le nombre de tests qui rejettent H₀ au seuil choisi (typiquement 1% ou 5%).
> 5. Les autres valeurs propres (sous $\lambda_+$ ou non significatives) sont du bruit — on les traitera dans la note 04 (nettoyage RIE).

### Comparaison avec les règles ad hoc

| Méthode | Critère | Justification |
|---|---|---|
| **Kaiser** | garder $\lambda_i > 1$ | aucune (arbitraire) |
| **Scree plot** | inspection visuelle du "coude" | subjective |
| **% variance expliquée** | garder jusqu'à 80% (ou autre) | arbitraire |
| **BBP + Tracy-Widom** | tests statistiques rigoureux | **théorie RMT** |

Les trois premières règles sont des heuristiques sans fondement théorique sur ce que représente vraiment "du signal". BBP + TW donne enfin un critère *justifié* — c'est le test statistique optimal sous l'hypothèse RMT.

> [!note]- Limites du modèle à spike pur
> Le modèle $\Sigma_{\rm vrai} = \mathbb{I}_N + \sum_k \theta_k u_k u_k^\top$ suppose que la "vraie" covariance est l'identité plus quelques perturbations rang 1. C'est une approximation utile mais simplifiée. La réalité financière a une structure plus complexe : volatilité variable par actif, corrélations résiduelles entre secteurs, structure temporelle non stationnaire. Des extensions de BBP existent pour des covariances vraies plus générales (Bun-Bouchaud-Potters 2017 développent un cadre unifié), mais le seuil $\theta_c = \sqrt{q}$ reste le bon ordre de grandeur dans la plupart des cas pratiques.

## Ce qui suit

On sait maintenant **identifier** les facteurs réels parmi les valeurs propres empiriques. La question naturelle suivante : **comment construire un estimateur amélioré de la covariance**, en réduisant le bruit du bulk et en corrigeant le biais positif visible en figure 3 ? C'est l'objet de la prochaine note.

[[04_Nettoyage_RIE|Note 04 — Nettoyage de covariance : clipping, Ledoit-Wolf, RIE]]
