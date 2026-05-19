---
title: Free probability et extensions avancées — vue panoramique
date: 2026-05-12
tags: [random-matrix-theory, free-probability, deep-learning, graph-signal-processing, advanced]
---

> [!warning] Statut et positionnement de cette note
> Les notes [[01_Fondamentaux]] à [[04_Nettoyage_RIE]] forment le **cœur opérationnel** de RMT pour le quant : tout ce qui est *directement utilisable* en finance. Cette note 05 est une **carte mentale des extensions théoriques** — savoir qu'elles existent, comprendre leur logique, et savoir où aller les creuser si un problème spécifique le nécessite.
>
> *Aucun de ces sujets n'est requis en pratique pour appliquer RMT à un problème de portefeuille.* On les présente parce qu'ils dessinent le cadre plus large dans lequel RMT s'inscrit, et parce que les connexions sont intellectuellement éclairantes.

## I. Pourquoi étendre MP ?

La loi de Marchenko-Pastur (note 02) suppose que la **vraie covariance vaut l'identité** $\mathbb{I}_N$. Dans la note 03, on l'a un peu étendue en ajoutant des spikes ($\mathbb{I}_N + \sum_k \theta_k u_k u_k^\top$), mais la baseline reste l'identité.

Trois questions naturelles se posent au-delà :

1. **Et si la vraie covariance a un spectre arbitraire ?** Par exemple, $\Sigma_{\rm vrai}$ a deux blocs (deux marchés indépendants), ou des valeurs propres continûment distribuées. Quelle est la loi limite du spectre empirique ?

2. **Et si on multiplie ou compose plusieurs matrices aléatoires ?** Par exemple, un réseau de neurones profond calcule $W_L W_{L-1} \cdots W_1$ — un *produit* de matrices aléatoires. Quel est son spectre singulier ?

3. **Comment relier RMT à d'autres outils spectraux ?** La diagonalisation comme méthode de débruitage (note 04) ressemble à Fourier — y a-t-il un cadre commun qui englobe les deux ?

La réponse aux questions 1 et 2 passe par la **free probability**. La réponse à la question 3 passe par le **graph signal processing**.

## II. Free probability — la généralisation de l'indépendance

### Pourquoi un nouveau concept ?

En probabilité classique, deux variables aléatoires $X, Y$ sont indépendantes si leur loi jointe se factorise. Sous indépendance, on a des outils puissants : la fonction caractéristique linéarise la somme ($\phi_{X+Y} = \phi_X \cdot \phi_Y$), la transformée de Mellin linéarise le produit pour les variables positives.

**Le problème avec les matrices aléatoires** : si $A, B$ sont deux matrices aléatoires, la notion d'indépendance classique des entrées ne suffit pas à décrire leur interaction spectrale, parce que **les matrices ne commutent pas** ($AB \neq BA$ en général). On a besoin d'un concept plus subtil.

> [!warning] Définition (intuitive) — Indépendance libre
> Deux matrices aléatoires $A, B$ de grande taille sont dites **libres** (au sens de Voiculescu) si leurs vecteurs propres sont "tournés aléatoirement" l'un par rapport à l'autre — typiquement quand $A$ est donné et $B = U^\top D U$ avec $U$ uniformément distribuée sur le groupe orthogonal/unitaire.
>
> C'est l'analogue non-commutatif de l'indépendance classique : "aucune relation spéciale entre les directions propres".

### Voir la freeness à l'œuvre

L'idée la plus difficile à digérer : deux matrices peuvent avoir **les mêmes spectres marginaux** mais donner des **spectres de somme différents**, selon que leurs bases propres sont alignées ou tournées aléatoirement.

![[fig_freeness.png]]
*Figure 1. Deux matrices $A$ et $B$ ont les mêmes spectres marginaux (valeurs propres uniformes sur $[0, 2]$). **(a)** Si elles partagent la même base propre (non libres), le spectre de $A+B$ suit la convolution classique des spectres marginaux — une distribution triangulaire. **(b)** Si $B$ est obtenue par une rotation aléatoire de $A$ (libres), le spectre de $A+B$ suit la **convolution libre** $\mu_A \boxplus \mu_B$ — une distribution plus aplatie, étalée différemment. Mêmes ingrédients, résultats différents.*

Cette différence se quantifie analytiquement par les **transformées R et S**, qui sont à la free probability ce que la fonction caractéristique et la transformée de Mellin sont à la probabilité classique.

### Les deux transformées clés

| Opération | Probabilité classique | Free probability |
|---|---|---|
| Addition de variables / matrices | fonction caractéristique $\phi$ | **transformée R** $R_\mu$ |
| Loi de $X + Y$ | $\phi_{X+Y} = \phi_X \cdot \phi_Y$ | $R_{\mu \boxplus \nu} = R_\mu + R_\nu$ |
| Multiplication (variables > 0) | transformée de Mellin | **transformée S** $S_\mu$ |
| Loi de $X \cdot Y$ | produit des Mellin | $S_{\mu \boxtimes \nu} = S_\mu \cdot S_\nu$ |

L'idée est la même qu'en probabilité classique : on a une opération difficile (calculer le spectre de $A+B$ ou $AB$), on la transforme en opération simple (somme ou produit de fonctions), on résout, on revient.

### Application 1 — Convolution libre additive sur GOE

Cas d'école parfait : on prend deux GOE indépendantes $A, B$ (donc libres asymptotiquement), chacune de spectre limite le demi-cercle sur $[-2, 2]$. Quel est le spectre de $A + B$ ?

La convolution libre des deux demi-cercles donne... **un demi-cercle plus large**, de rayon $\sqrt{2}$ fois plus grand. C'est l'analogue parfait de "la somme de deux gaussiennes indépendantes est une gaussienne plus large", mais dans le monde libre.

![[fig_free_addition.png]]
*Figure 2. **(a)** Deux GOE indépendantes $A$ et $B$, chacune avec spectre suivant le demi-cercle sur $[-2, 2]$. **(b)** Le spectre de $A+B$ est un demi-cercle élargi sur $[-2\sqrt{2}, 2\sqrt{2}]$ — c'est la convolution libre $\mu_A \boxplus \mu_B$. La convolution classique (pointillé rouge) prédirait un pic central et des queues plus longues, qui ne correspondent pas du tout au phénomène observé.*

### Application 2 — Covariances vraies non-identité

Si la vraie covariance $\Sigma_{\rm vrai}$ n'est plus l'identité mais a un spectre arbitraire de loi $\mu_{\Sigma}$, alors la loi limite du spectre empirique $\hat\mu_N$ est la **convolution libre multiplicative** de $\mu_{\Sigma}$ avec une MP standard :
$$\mu_{\hat\Sigma} = \mu_{\Sigma} \boxtimes \mu_{\rm MP}(q)$$

C'est ce qui permet, en théorie, d'**inverser le bruit d'estimation pour des structures arbitraires** : connaissant $\hat\mu_N$ (mesuré sur les données) et $\mu_{\rm MP}(q)$ (connue analytiquement), on déconvolue pour récupérer $\mu_{\Sigma}$. C'est l'idée derrière le RIE généralisé (Bun-Bouchaud-Potters 2017) qui étend la formule simple de la note 04.

### Application 3 — Produits de matrices et deep learning

Pour un réseau de neurones profond, la Jacobienne d'entrée-sortie est un produit $J = W_L \cdots W_1$ où chaque $W_\ell$ est aléatoire à l'initialisation. Le spectre singulier de $J$ contrôle si le réseau peut être entraîné :
- $\sigma_{\max}(J) \gg 1$ ⟹ gradients qui explosent en backward pass.
- $\sigma_{\min}(J) \ll 1$ ⟹ gradients qui s'évanouissent.

> [!example] Dynamical isotropy (Pennington, Schoenholz, Ganguli 2017–2018)
> Une bonne initialisation rend le spectre singulier de $J$ **concentré autour de 1**, à toute profondeur $L$. Le spectre singulier d'un produit de matrices aléatoires libres se calcule par convolution libre multiplicative de leurs lois spectrales.
>
> Résultat surprenant : l'**initialisation orthogonale** (tirer $W_\ell$ uniformément sur le groupe orthogonal) donne $\sigma_i(J) = 1$ exactement, à toute profondeur — c'est mieux que Xavier qui donne $\sigma_i \approx 1$ en moyenne mais avec une dispersion qui **croît exponentiellement** avec $L$.

![[fig_deep_init.png]]
*Figure 3. Spectre singulier du produit $J = W_L \cdots W_1$ pour différentes profondeurs $L$. **(a)** Initialisation Xavier : la dispersion explose littéralement avec la profondeur. À $L=50$, $\sigma_{\max}/\sigma_{\min} \approx 10^{18}$ — le réseau est mathématiquement non entraînable sans BatchNorm ou ResNet. **(b)** Initialisation orthogonale : toutes les valeurs singulières restent à $\sigma_i = 1$ exactement, peu importe la profondeur. C'est la dynamical isotropy parfaite.*

C'est exactement la même logique que la normalisation $1/\sqrt{N}$ vue en note 01 — calibrer une opération matricielle pour préserver l'échelle — mais cette fois généralisée aux produits via la convolution libre.

### Application 4 — Universalité du sine kernel

L'universalité du sine kernel (les espacements entre valeurs propres au cœur du spectre, voir note 01) se prouve via free probability : la même structure algébrique émerge quelle que soit la distribution précise des entrées de la matrice. C'est ce qui explique pourquoi le sine kernel apparaît dans des contextes complètement déconnectés (zéros de la fonction zêta de Riemann, espacements d'arrivée des bus à Cuernavaca).

## III. Graph Signal Processing — le cadre unificateur

### Le pattern commun à trois domaines

Dans la note 04, on a vu que RMT et Fourier partagent le même pipeline : *diagonaliser un opérateur → identifier signal/bruit dans la base propre → filtrer → reconstruire*. Ce pattern est en réalité partagé par **trois domaines distincts**, qui forment ensemble une famille cohérente :

| Domaine | Opérateur diagonalisé | Base "naturelle" |
|---|---|---|
| Signaux temporels | Laplacien continu / translation | exponentielles complexes (TF) |
| Signaux sur graphes | Laplacien du graphe $L = D - A$ | vecteurs propres de $L$ |
| Variables corrélées | Covariance $\Sigma$ | vecteurs propres de $\Sigma$ |

Ces trois ne sont **pas** strictement emboîtés — RMT n'est pas un cas particulier de GSP, et inversement. Ce sont **trois instances parallèles d'un même geste mathématique** : la décomposition spectrale d'un opérateur self-adjoint comme outil universel d'analyse.

### Graph Signal Processing en deux mots

Depuis ~2010 (papiers fondateurs : Shuman et al. 2013, Ortega et al. 2018), un cadre unifié a émergé pour faire du traitement du signal sur des **données structurées par un graphe**. L'idée est simple : si tu as un graphe $G$ avec laplacien $L$, alors la base propre de $L$ joue le rôle des "fréquences" pour les signaux définis sur ce graphe.

Tu peux donc :
- Définir une **transformée de Fourier sur graphe** : $\hat{f}(k) = \langle f, u_k \rangle$ avec $u_k$ vecteur propre de $L$.
- Définir des **filtres spectraux** : multiplier $\hat{f}(k)$ par une fonction $h(\lambda_k)$ avant retransformation.
- Définir des **convolutions sur graphe** : opération qui devient multiplication dans la base spectrale.

### Lien avec RMT en finance

Le **graphe de corrélations entre actifs** est essentiellement la matrice de covariance vue comme un objet de graphe (nœuds = actifs, arêtes pondérées par les corrélations). RMT et GSP attaquent donc souvent le même objet sous deux angles complémentaires :

- **RMT** : on étudie le **spectre lui-même** comme objet aléatoire — distinguer bulk/edge, identifier les facteurs, nettoyer via RIE.
- **GSP** : on étudie les **signaux définis sur ce graphe** (rendements, volumes, sentiment...) — on filtre, on diffuse, on détecte des communautés.

Les deux se complètent naturellement : RMT donne la "bonne" base spectrale (les vecteurs propres nettoyés), GSP exploite cette base pour analyser ce qui se passe dessus.

### Lien avec ton expérience Graph ML

Les premières architectures de **Graph Neural Networks** (Bruna, Henaff, LeCun 2014 — *Spectral Networks*) sont littéralement des **convolutions définies via le laplacien spectral d'un graphe**. Toute la lignée spectrale des GNN (ChebNet, GCN de Kipf-Welling) repose sur ce cadre. Tu as déjà des notes [[Graph ML]] dans ton vault qui touchent ce sujet — c'est exactement la passerelle entre RMT (cette série), GSP (le cadre général), et GNN (l'application deep learning).

## IV. Pour aller plus loin

> [!note]- Ressources principales si tu veux creuser
> **Free probability et RMT :**
> - *Mingo & Speicher (2017), "Free Probability and Random Matrices"* — référence rigoureuse.
> - *Bun, Bouchaud, Potters (2017), "Cleaning large correlation matrices: tools from random matrix theory"*, Physics Reports — review accessible avec applications finance complètes.
>
> **RMT et deep learning :**
> - *Pennington, Schoenholz, Ganguli (2017–2018)* — série de papers sur dynamical isotropy et initialisation orthogonale.
> - *Pennington & Worah (2017), "Nonlinear random matrix theory for deep learning"* — extensions aux activations non-linéaires.
>
> **Graph Signal Processing :**
> - *Shuman et al. (2013), "The emerging field of signal processing on graphs"*, IEEE Signal Processing Magazine — vue d'ensemble fondatrice.
> - *Ortega et al. (2018), "Graph signal processing: overview, challenges and applications"*, Proceedings of the IEEE — review moderne avec applications.

## Statut final de la série RMT

| Note | Rôle | Statut |
|---|---|---|
| 01 | Fondamentaux théoriques (GOE, demi-cercle, Tracy-Widom, sine kernel) | ✅ |
| 02 | Loi de Marchenko-Pastur (le résultat clé) | ✅ |
| 03 | Transition BBP et détection rigoureuse de facteurs | ✅ |
| 04 | Nettoyage de covariance — production-ready avec RIE | ✅ |
| 05 | Extensions théoriques — carte mentale | ✅ |

**Les notes 01-04 forment le cœur opérationnel** : tout ce qu'il faut pour appliquer RMT en quant en pratique. **La note 05 est une carte mentale** : tu sais que ces extensions existent, tu sais où aller les creuser si un problème spécifique le nécessite, et tu vois les ponts vers tes autres domaines (Graph ML, deep learning).
