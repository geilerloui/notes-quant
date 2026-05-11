---
title: Régularisation entropique et algorithme de Sinkhorn
date: 2026-05-11
tags: [mathématiques, transport-optimal, sinkhorn, entropie, algorithme]
---

## L'idée fondatrice

Les notes précédentes ont posé la théorie du transport optimal : Monge, Kantorovich, Brenier, dualité, Wasserstein. Tout ça est mathématiquement élégant — mais en pratique, **comment calcule-t-on $W_p(\mu, \nu)$ pour deux datasets** ?

Le problème de Kantorovich discret est un **programme linéaire** (LP) : il se résout exactement par simplexe ou méthodes intérieures, mais avec une complexité **$O(n^3 \log n)$** pour $n$ points par distribution. Pour des datasets de taille $n = 10^4$ ou $10^5$ (cas typique en ML), c'est **inutilisable**.

L'idée de **Marco Cuturi (2013)** : **régulariser** le problème en ajoutant un terme d'**entropie**. Le problème devient :

1. **Strictement convexe** : solution unique, garanties de convergence
2. **Différentiable en les marginales** : on peut backpropager à travers
3. **Soluble par un algorithme itératif extrêmement simple** (Sinkhorn) en **$O(n^2)$ par itération**

C'est ce qui a fait du transport optimal un **outil pratique en ML moderne**. Avant 2013, OT était une curiosité mathématique ; après 2013, c'est dans presque tous les frameworks de comparaison de distributions.

Cette note construit la régularisation entropique, dérive l'algorithme de Sinkhorn, et explique pourquoi il fonctionne.

## I. Le problème régularisé

### Forme primale

Rappel : le problème de Kantorovich discret s'écrit (avec $\mathbf{P} \in \mathbb{R}^{n \times m}_+$, $\mathbf{a} \in \mathbb{R}^n_+$, $\mathbf{b} \in \mathbb{R}^m_+$, $\mathbf{C} \in \mathbb{R}^{n \times m}_+$) :

$$\min_{\mathbf{P}} \langle \mathbf{C}, \mathbf{P} \rangle \quad \text{s.c.} \quad \mathbf{P} \mathbf{1}_m = \mathbf{a}, \; \mathbf{P}^\top \mathbf{1}_n = \mathbf{b}, \; \mathbf{P} \geq 0$$

où $\langle \mathbf{C}, \mathbf{P} \rangle = \sum_{i,j} C_{ij} P_{ij}$.

La **version régularisée** ajoute un terme d'entropie sur $\mathbf{P}$ :

$$\boxed{\min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon \, H(\mathbf{P})}$$

où $H(\mathbf{P}) = -\sum_{i,j} P_{ij} (\log P_{ij} - 1)$ est l'**entropie de Shannon** de la matrice de transport, et $\varepsilon > 0$ est un **paramètre de régularisation**.

> [!note]- Pourquoi cette forme d'entropie ?
> L'entropie de Shannon "classique" est $H(\mathbf{P}) = -\sum P_{ij} \log P_{ij}$. La version utilisée ici, $-\sum P_{ij}(\log P_{ij} - 1)$, diffère d'un terme $\sum P_{ij}$ qui vaut 1 (puisque $\mathbf{P}$ est une matrice de couplage de masse totale 1).
> 
> Ce "$-1$" est purement cosmétique mais fait que la dérivée donne $\nabla_P H = -\log \mathbf{P}$ au lieu de $-\log \mathbf{P} - 1$. Cela simplifie les équations qu'on va dériver.

### Interprétation comme KL-divergence

Posons $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$ (la matrice de Gibbs/kernel). Alors :

$$\langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon H(\mathbf{P}) = \varepsilon \cdot \text{KL}(\mathbf{P} \| \mathbf{K}) + \text{const}$$

où $\text{KL}(\mathbf{P} \| \mathbf{K}) = \sum_{i,j} P_{ij} \log(P_{ij} / K_{ij}) - P_{ij} + K_{ij}$ est la **divergence de Kullback-Leibler généralisée** (étendue aux mesures non normalisées).

> [!note]- Dérivation
> $$\langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon H(\mathbf{P}) = \sum_{ij} C_{ij} P_{ij} + \varepsilon \sum_{ij} P_{ij}(\log P_{ij} - 1)$$
> 
> $$= \varepsilon \sum_{ij} P_{ij} \left[ \frac{C_{ij}}{\varepsilon} + \log P_{ij} - 1 \right]$$
> 
> $$= \varepsilon \sum_{ij} P_{ij} \left[ \log P_{ij} - \log e^{-C_{ij}/\varepsilon} - 1 \right] = \varepsilon \sum_{ij} P_{ij} \log \frac{P_{ij}}{K_{ij}} - \varepsilon \sum_{ij} P_{ij}$$
> 
> Et $\sum P_{ij} = 1$ (couplage normalisé), donc le dernier terme est une constante. On peut aussi ajouter et soustraire $\sum K_{ij}$ pour obtenir la forme symétrique de la KL généralisée.

Le problème régularisé devient donc :

$$\min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \text{KL}(\mathbf{P} \| \mathbf{K})$$

**Interprétation** : on cherche le couplage $\mathbf{P}$ qui satisfait les contraintes marginales **et** qui est le **plus proche possible** (au sens KL) de la matrice $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$.

### Que fait $\varepsilon$ ?

Le paramètre $\varepsilon$ contrôle un **trade-off** :

- **$\varepsilon \to 0$** : on retrouve le problème original (LP). Solution **sparse** (peu de $P_{ij}$ non nuls).
- **$\varepsilon \to \infty$** : l'entropie domine, $\mathbf{P}^* \to \mathbf{a} \mathbf{b}^\top$ (le couplage indépendant, sans information de coût). Solution **dense**.
- **$\varepsilon$ modéré** : solution **lisse**, différentiable, qui interpole entre les deux.

C'est ce **lissage** qui rend le problème numériquement tractable et qui permet la backprop (la solution dépend de $\mathbf{a}, \mathbf{b}, \mathbf{C}$ de manière différentiable).

![[sinkhorn_epsilon_tradeoff.png]]
*Effet du paramètre $\varepsilon$ sur la matrice de transport $\mathbf{P}^*$, pour la même paire de distributions. **À gauche** ($\varepsilon = 0.05$, petit) : $\mathbf{P}^*$ est quasi-diagonale et sparse, on retrouve le LP exact. **Au milieu** ($\varepsilon = 0.5$, modéré) : solution lisse et différentiable — c'est le régime utilisé en pratique. **À droite** ($\varepsilon = 5$, grand) : la régularisation domine et $\mathbf{P}^*$ tend vers le couplage indépendant $\mathbf{a}\mathbf{b}^\top$ (la matrice de produit tensoriel), qui n'utilise plus l'information de coût.*

## II. La forme de la solution

### Dérivation via Lagrangien

On veut résoudre :

$$\min_{\mathbf{P} \geq 0} \langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon H(\mathbf{P}) \quad \text{s.c.} \quad \mathbf{P} \mathbf{1} = \mathbf{a}, \; \mathbf{P}^\top \mathbf{1} = \mathbf{b}$$

> [!note]- Dérivation Lagrangienne complète
> Introduisons des multiplicateurs $\mathbf{f} \in \mathbb{R}^n$ et $\mathbf{g} \in \mathbb{R}^m$ pour les deux contraintes de marginales. Le Lagrangien est :
> 
> $$\mathcal{L}(\mathbf{P}, \mathbf{f}, \mathbf{g}) = \sum_{ij} \left[ C_{ij} P_{ij} + \varepsilon P_{ij}(\log P_{ij} - 1) \right] - \mathbf{f}^\top (\mathbf{P}\mathbf{1} - \mathbf{a}) - \mathbf{g}^\top (\mathbf{P}^\top \mathbf{1} - \mathbf{b})$$
> 
> $$= \sum_{ij} \left[ C_{ij} P_{ij} + \varepsilon P_{ij} \log P_{ij} - \varepsilon P_{ij} - f_i P_{ij} - g_j P_{ij} \right] + \mathbf{f}^\top \mathbf{a} + \mathbf{g}^\top \mathbf{b}$$
> 
> Condition d'optimalité $\partial \mathcal{L} / \partial P_{ij} = 0$ :
> 
> $$C_{ij} + \varepsilon \log P_{ij} - f_i - g_j = 0$$
> 
> $$\Rightarrow \log P_{ij} = \frac{f_i + g_j - C_{ij}}{\varepsilon}$$
> 
> $$\Rightarrow \boxed{P_{ij} = e^{f_i/\varepsilon} \cdot e^{-C_{ij}/\varepsilon} \cdot e^{g_j/\varepsilon} = u_i \, K_{ij} \, v_j}$$
> 
> où on a posé $u_i = e^{f_i/\varepsilon}$, $v_j = e^{g_j/\varepsilon}$, et $K_{ij} = e^{-C_{ij}/\varepsilon}$.

### La structure produit

On obtient la **forme remarquable** :

$$\boxed{P_{ij}^* = u_i \, K_{ij} \, v_j}$$

ou, en notation matricielle :

$$\mathbf{P}^* = \text{diag}(\mathbf{u}) \, \mathbf{K} \, \text{diag}(\mathbf{v})$$

**Tout l'optimum est encodé dans deux vecteurs $\mathbf{u}, \mathbf{v}$**. La matrice $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$ est fixée (elle ne dépend que des données), et la solution $\mathbf{P}^*$ se factorise par **pondération en ligne** ($\mathbf{u}$) et **en colonne** ($\mathbf{v}$).

![[sinkhorn_factorization.png]]
*Exemple concret de factorisation. À partir des marginales $\mathbf{a}$ (gauche, rouge) et $\mathbf{b}$ (haut, bleu), on construit la matrice de coût $\mathbf{C}$ (gris) puis le noyau $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$ (orange). Sinkhorn calcule les facteurs $\mathbf{u}, \mathbf{v}$ et la matrice optimale $\mathbf{P}^*$ (vert) qui est concentrée sur la "diagonale" — le transport optimal apparie les percentiles entre source et cible.*

### Les contraintes de marginales redeviennent simples

Les contraintes $\mathbf{P} \mathbf{1} = \mathbf{a}$ et $\mathbf{P}^\top \mathbf{1} = \mathbf{b}$ s'écrivent maintenant comme **deux équations simples** sur $\mathbf{u}, \mathbf{v}$ :

$$u_i \sum_j K_{ij} v_j = a_i \quad \forall i$$

$$v_j \sum_i K_{ij} u_i = b_j \quad \forall j$$

Ou, en notation matricielle :

$$\boxed{\mathbf{u} \odot (\mathbf{K} \mathbf{v}) = \mathbf{a} \quad \text{et} \quad \mathbf{v} \odot (\mathbf{K}^\top \mathbf{u}) = \mathbf{b}}$$

où $\odot$ est le produit terme à terme (Hadamard). C'est ce système qu'on va résoudre itérativement.

## III. L'algorithme de Sinkhorn

### Idée : alternance

Le système $\mathbf{u} \odot (\mathbf{K} \mathbf{v}) = \mathbf{a}$ et $\mathbf{v} \odot (\mathbf{K}^\top \mathbf{u}) = \mathbf{b}$ a une structure très particulière : si on **fixe** $\mathbf{v}$, on peut résoudre **trivialement** pour $\mathbf{u}$ (et inversement).

En effet, si $\mathbf{v}$ est connu, alors :

$$u_i = \frac{a_i}{\sum_j K_{ij} v_j} = \frac{a_i}{(\mathbf{K} \mathbf{v})_i}$$

Et de même, si $\mathbf{u}$ est connu :

$$v_j = \frac{b_j}{\sum_i K_{ij} u_i} = \frac{b_j}{(\mathbf{K}^\top \mathbf{u})_j}$$

D'où l'algorithme : **alterner** les deux mises à jour jusqu'à convergence.

### Algorithme

> [!quote] Algorithme (Sinkhorn)
> **Entrées** : matrice de coût $\mathbf{C}$, marginales $\mathbf{a}, \mathbf{b}$, paramètre $\varepsilon > 0$, tolérance $\delta$
> 
> 1. Calculer le noyau $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$
> 2. Initialiser $\mathbf{v} = \mathbf{1}_m$
> 3. Répéter :
>    - $\mathbf{u} \leftarrow \mathbf{a} \oslash (\mathbf{K} \mathbf{v})$
>    - $\mathbf{v} \leftarrow \mathbf{b} \oslash (\mathbf{K}^\top \mathbf{u})$
> 4. Jusqu'à $\| \mathbf{u} \odot (\mathbf{K} \mathbf{v}) - \mathbf{a} \|_1 < \delta$
> 5. Retourner $\mathbf{P}^* = \text{diag}(\mathbf{u}) \mathbf{K} \text{diag}(\mathbf{v})$
> 
> où $\oslash$ est la division terme à terme.

> [!note]- Implémentation Python (5 lignes)
> ```python
> import numpy as np
> 
> def sinkhorn(C, a, b, eps=0.1, n_iter=100):
>     K = np.exp(-C / eps)
>     v = np.ones_like(b)
>     for _ in range(n_iter):
>         u = a / (K @ v)
>         v = b / (K.T @ u)
>     return u[:, None] * K * v[None, :]  # P optimale
> ```
> En pratique on utilise une version **log-space** pour la stabilité numérique (les exponentielles débordent quand $\varepsilon$ est petit), mais l'algorithme reste essentiellement le même.

### Coût par itération

Chaque itération est dominée par les produits matrice-vecteur $\mathbf{K} \mathbf{v}$ et $\mathbf{K}^\top \mathbf{u}$, qui coûtent **$O(nm)$** flops. C'est $O(n^2)$ pour $n = m$, contre $O(n^3 \log n)$ pour le LP original. **Énorme gain** en pratique.

Avantage supplémentaire : ces opérations sont **trivialement parallélisables** sur GPU. C'est pour ça que Sinkhorn est l'algorithme de référence en deep learning quand on a besoin d'OT.

## IV. Pourquoi ça converge

### Idée intuitive : projections alternées sur deux convexes

Sinkhorn peut se voir comme un algorithme de **projections KL alternées** :

- L'itération $\mathbf{u} \leftarrow \mathbf{a} / (\mathbf{K}\mathbf{v})$ projette $\mathbf{P}$ (au sens KL) sur l'ensemble $\{\mathbf{P} : \mathbf{P}\mathbf{1} = \mathbf{a}\}$ — c'est-à-dire qu'on rescale les **lignes** pour qu'elles aient les bonnes sommes
- L'itération $\mathbf{v} \leftarrow \mathbf{b} / (\mathbf{K}^\top \mathbf{u})$ projette sur $\{\mathbf{P} : \mathbf{P}^\top \mathbf{1} = \mathbf{b}\}$ — on rescale les **colonnes**

L'algorithme alterne entre les deux projections jusqu'à ce que les deux contraintes soient simultanément satisfaites. C'est une version du **Bregman alternating projection**.

### Convergence : la métrique de Hilbert

Le résultat de convergence rigoureux passe par la **métrique projective de Hilbert** sur les vecteurs strictement positifs :

$$d_H(\mathbf{u}, \mathbf{u}') = \log \frac{\max_i (u_i / u'_i)}{\min_i (u_i / u'_i)}$$

> [!note]- Pourquoi Hilbert et pas euclidien ?
> Sinkhorn est invariant sous rescaling $(\mathbf{u}, \mathbf{v}) \to (\lambda \mathbf{u}, \mathbf{v}/\lambda)$ (la solution $\mathbf{P}$ reste la même). La métrique euclidienne ne respecte pas cette invariance, mais $d_H$ oui — c'est une métrique sur l'**espace projectif** des rayons positifs.

**Théorème (Franklin-Lorenz, 1989)** : si $\mathbf{K}$ a tous ses coefficients strictement positifs (vrai dès que $\varepsilon > 0$), alors l'itération de Sinkhorn est une **contraction stricte** pour $d_H$ :

$$d_H(\mathbf{u}^{(k+1)}, \mathbf{u}^*) \leq \lambda \cdot d_H(\mathbf{u}^{(k)}, \mathbf{u}^*)$$

avec $\lambda < 1$ qui dépend de $\mathbf{K}$. La convergence est donc **géométrique** (linéaire en log).

**Le prix à payer** : $\lambda$ est proche de 1 quand $\varepsilon$ est petit. Pour $\varepsilon$ très petit, on a beaucoup d'itérations à faire — ce qui correspond à l'intuition : plus on s'approche du LP original, plus c'est dur.

## V. Lien avec la dualité de Kantorovich

Les multiplicateurs $\mathbf{f} = \varepsilon \log \mathbf{u}$ et $\mathbf{g} = \varepsilon \log \mathbf{v}$ obtenus par Sinkhorn ne sont pas anodins : **ce sont les potentiels duaux** de Kantorovich (note 04), version régularisée.

### Le dual régularisé

Le problème dual régularisé s'écrit :

$$\max_{\mathbf{f}, \mathbf{g}} \left\{ \mathbf{f}^\top \mathbf{a} + \mathbf{g}^\top \mathbf{b} - \varepsilon \sum_{ij} e^{(f_i + g_j - C_{ij})/\varepsilon} \right\}$$

Comparons au dual non-régularisé :

$$\max_{f_i + g_j \leq C_{ij}} \mathbf{f}^\top \mathbf{a} + \mathbf{g}^\top \mathbf{b}$$

La régularisation **remplace la contrainte dure** $f_i + g_j \leq C_{ij}$ par une **pénalité exponentielle douce** : quand $f_i + g_j > C_{ij}$, on paie une pénalité $\propto e^{(f_i+g_j-C_{ij})/\varepsilon}$ qui explose vite. Quand $\varepsilon \to 0$, on retrouve la contrainte dure.

### Soft-min

Pour simplifier, on peut **éliminer $\mathbf{g}$** en résolvant analytiquement (condition d'optimalité par rapport à $g_j$) :

$$g_j = -\varepsilon \log \sum_i e^{(f_i - C_{ij})/\varepsilon} + \varepsilon \log b_j$$

Cette opération $-\varepsilon \log \sum e^{\cdot/\varepsilon}$ est la **soft-min** (limite vers min quand $\varepsilon \to 0$). Sinkhorn peut s'écrire comme un algorithme de **soft-min alterné** sur les potentiels — une forme très utilisée en pratique dans les implémentations log-space.

### Wasserstein régularisé

La **distance de Sinkhorn** $W_\varepsilon$ est définie comme la valeur optimale du problème régularisé. Elle vérifie :

$$W_\varepsilon(\mu, \nu) \xrightarrow{\varepsilon \to 0} W_2(\mu, \nu)^2$$

(on récupère bien Wasserstein quand on enlève la régularisation)

**Attention** : $W_\varepsilon$ n'est **pas une vraie distance** ($W_\varepsilon(\mu, \mu) \neq 0$ en général). On utilise souvent la **Sinkhorn divergence** :

$$S_\varepsilon(\mu, \nu) = W_\varepsilon(\mu, \nu) - \frac{1}{2}\left[ W_\varepsilon(\mu, \mu) + W_\varepsilon(\nu, \nu) \right]$$

qui corrige ce biais et redonne une vraie divergence (positive, séparante).

## VI. Trois idées à retenir

1. **La régularisation entropique** transforme le LP du transport optimal en un problème **strictement convexe** et **différentiable**. La solution se factorise sous la forme $P_{ij} = u_i K_{ij} v_j$ avec $\mathbf{K} = e^{-\mathbf{C}/\varepsilon}$.

2. **Sinkhorn** est l'algorithme qui calcule $\mathbf{u}, \mathbf{v}$ par mises à jour alternées : $\mathbf{u} \leftarrow \mathbf{a} / (\mathbf{K}\mathbf{v})$ puis $\mathbf{v} \leftarrow \mathbf{b} / (\mathbf{K}^\top \mathbf{u})$. Chaque itération coûte **$O(n^2)$** (contre $O(n^3 \log n)$ pour le LP exact). Convergence géométrique garantie par contraction Hilbert.

3. **Lien avec la dualité** : les facteurs $\mathbf{u}, \mathbf{v}$ encodent les **potentiels duaux** $\mathbf{f} = \varepsilon \log \mathbf{u}, \, \mathbf{g} = \varepsilon \log \mathbf{v}$. La régularisation remplace la contrainte dure $f_i + g_j \leq C_{ij}$ par une pénalité exponentielle. On utilise la **Sinkhorn divergence** $S_\varepsilon$ (corrigée du biais) plutôt que $W_\varepsilon$ brute.

## VII. Vers la suite

- **Note 06 — Applications ML** : Wasserstein barycenters (qui se calculent essentiellement par Sinkhorn itéré), domain adaptation, image morphing. Sinkhorn est le moteur de toutes ces applications.
- **Note 07 — Géométrie de Wasserstein** : la structure riemannienne de $\mathcal{P}_2$, où la map de Brenier joue le rôle de l'exponentielle riemannienne.
