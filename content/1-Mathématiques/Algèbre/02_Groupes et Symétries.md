---
title: Groupes et Symétries
description: Structures algébriques, groupes, action de groupe, invariance et équivariance — avec applications au ML
---

# Groupes et Symétries

> Cette note part des structures algébriques de base pour arriver à la notion de symétrie d'une fonction — et pourquoi c'est central en ML. Le fil rouge : un réseau de neurones génère une fonction $f_w$, et la question "est-ce qu'elle généralise bien" se ramène à "est-ce qu'elle respecte les symétries des données".

## I. Hiérarchie des structures algébriques

On part d'un ensemble $G$ muni d'une loi de composition $\cdot$ (une façon de combiner deux éléments). On ajoute des axiomes progressivement :

![[group_hierarchy.png]]
**Figure.** *Hiérarchie des structures algébriques — chaque niveau ajoute un axiome. Le groupe est la structure centrale pour étudier les symétries.*

| Structure | Axiomes | Exemple |
|---|---|---|
| **Magma** | Loi interne : $a \cdot b \in G$ | $(\mathbb{Z}, -)$ |
| **Semi-groupe** | + Associativité | $(\mathbb{N}, +)$ |
| **Monoïde** | + Élément neutre $e$ | $(\mathbb{N} \cup \{0\}, +)$ |
| **Groupe** | + Inverse $a^{-1}$ | Translations, rotations, $(\mathbb{Z}, +)$ |
| **Groupe abélien** | + Commutativité | $(\mathbb{R}, +)$, $(\mathbb{Z}/n\mathbb{Z}, +)$ |

La structure qui nous intéresse pour les symétries c'est le **groupe**.

> [!warning] Définition — Groupe
> Un groupe $(G, \cdot)$ est un ensemble $G$ muni d'une loi $\cdot$ vérifiant :
> 1. **Fermeture** : $\forall a, b \in G,\ a \cdot b \in G$
> 2. **Associativité** : $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
> 3. **Élément neutre** : $\exists\, e \in G$ tel que $a \cdot e = e \cdot a = a$
> 4. **Inverse** : $\forall a \in G,\ \exists\, a^{-1}$ tel que $a \cdot a^{-1} = a^{-1} \cdot a = e$

**Exemple concret — les translations de $\mathbb{R}^2$.**

Soit $G = \{t_v : x \mapsto x + v,\ v \in \mathbb{R}^2\}$ l'ensemble de toutes les translations du plan :
- Composer deux translations : $t_{v_1} \circ t_{v_2} = t_{v_1 + v_2}$ ✓ (fermeture)
- Élément neutre : $t_0$ (ne rien bouger) ✓
- Inverse : $t_v^{-1} = t_{-v}$ (translation opposée) ✓

C'est un groupe — et un groupe abélien car $t_{v_1} \circ t_{v_2} = t_{v_2} \circ t_{v_1}$.

## II. Action de groupe

Un groupe seul c'est abstrait. Ce qui est utile en ML c'est de voir comment un groupe **agit** sur un espace de données.

> [!warning] Définition — Action de groupe
> Une action du groupe $G$ sur un ensemble $X$ est une application
> $$G \times X \to X, \quad (g, x) \mapsto g \cdot x$$
> telle que :
> - $e \cdot x = x$ (l'élément neutre ne fait rien)
> - $(g_1 \cdot g_2) \cdot x = g_1 \cdot (g_2 \cdot x)$ (compatible avec la composition)

**En pratique pour les images.** Une image $x$ est un champ de pixels $x(u)$ où $u \in \mathbb{R}^2$ indexe la position. Le groupe des translations agit sur les images par :

$$g \cdot x(u) = x(u - g)$$

C'est-à-dire : décaler l'image de $g$ pixels. Les rotations, réflexions, déformations locales agissent de façon similaire.

![[group_action_image.png]]
**Figure.** *Le groupe $G = \{\text{translations, rotations, réflexions}\}$ agit sur l'espace des images. Chaque $g \in G$ transforme l'image $x$ en $g \cdot x$. On utilise la lettre F — asymétrique — pour que chaque transformation soit visible.*

## III. Invariance et Équivariance

On a une fonction $f : X \to Y$ (par exemple un réseau de neurones). Deux propriétés clés par rapport à un groupe $G$ :

> [!warning] Invariance et Équivariance
> Soit $G$ un groupe agissant sur $X$ (et éventuellement sur $Y$).
>
> - $f$ est **invariante** par $G$ si : $\forall g \in G,\ f(g \cdot x) = f(x)$
>   → la sortie ne change pas quand on transforme l'input
>
> - $f$ est **équivariante** par $G$ si : $\forall g \in G,\ f(g \cdot x) = g \cdot f(x)$
>   → la sortie se transforme de la même façon que l'input

![[invariance_equivariance.png]]
**Figure.** *Gauche : équivariance — si on décale l'image, la feature map est décalée pareil. Droite : invariance — si on décale l'image, la prédiction finale reste « Chat ».*

**Pourquoi la distinction compte.** Dans un CNN :
- Les **couches de convolution** sont équivariantes par translation — elles détectent les mêmes motifs partout dans l'image, mais leur position dans la feature map suit l'image
- Le **pooling global** + classifieur final est invariant — peu importe où est le chat dans l'image, la sortie est « chat »

Un MLP ordinaire n'est **ni invariant ni équivariant** par translation — si tu décales l'image de 2 pixels, les pixels arrivent dans des neurones différents et la sortie change complètement.

## IV. Symétries d'une fonction

On peut maintenant définir proprement ce que Mallat appelle "symétrie de $f$".

> [!warning] Symétrie d'une fonction
> $g \in G$ est une symétrie de $f : X \to Y$ si $f(g \cdot x) = f(x)$ pour tout $x \in X$.
>
> L'ensemble de toutes les symétries de $f$ forme un groupe :
> $$G_f = \{g \in G \mid f(g \cdot x) = f(x)\ \forall x\}$$

Note que c'est bien une généralisation de la symétrie classique : $f(x) = x^2$ est symétrique par rapport à 0 car $f(-x) = f(x)$ — le groupe $G_f = \{id, x \mapsto -x\}$.

**Le lien avec la régularité.** Plus $G_f$ est grand, plus $f$ est "régulière" au sens global — pas seulement localement comme Lipschitz. Et plus $G_f$ est grand, plus l'espace d'étude se réduit : si $G_f$ agit sur $X$, on peut quotienter et travailler sur $X/G_f$ de dimension bien plus petite.

## V. Application ML — pourquoi les symétries cassent la malédiction

La borne de Mallat (cf. [[01_Fondation#E. Curse of dimensionality]]) dit : sous hypothèse Lipschitz seule, il faut $n \geq \epsilon^{-d}$ exemples. Les réseaux bien conçus échappent à cette borne parce qu'ils exploitent les symétries des données.

**Fil rouge CNN.** Les images naturelles sont invariantes par translation (un chat décalé reste un chat), par déformation locale (un chat tordu reste un chat), partiellement par rotation. En encodant la symétrie de translation dans l'architecture via le partage de poids des filtres convolutifs, le CNN réduit massivement la dimension effective du problème — chaque exemple d'entraînement en vaut plusieurs (l'image + toutes ses versions traduites).

> [!note]- Tableau récap — symétrie par architecture
> | Architecture | Symétrie encodée | Mécanisme |
> |---|---|---|
> | CNN | Translation spatiale | Partage de poids des filtres |
> | RNN | Translation temporelle | Même poids $W_h$ à chaque pas |
> | Transformer | Permutation (partielle) | Attention + positional encoding |
> | GNN | Permutation des nœuds | Agrégation par voisinage |
> | MLP | Aucune | — doit tout apprendre des données |

> [!note]- Lien avec les autres notes
> - [[01_Fondation#E. Curse of dimensionality]] — la borne $\epsilon^{-d}$ que les symétries permettent de casser
> - [[01_Fondation#F. Manifold Hypothesis]] — les données vivent sur une variété = elles ont une structure de groupe implicite
> - [[02_CNN#IV — Pourquoi les CNN fonctionnent : symétries et hiérarchie (Mallat)]] — développement complet pour les CNN
> - [[00_Perceptron Multi-Couches#II.7 — Pourquoi les réseaux marchent : géométrie et symétries (Mallat)]] — programme de recherche de Mallat
