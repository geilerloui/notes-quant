---
title: Probabilistic Graphical Models
---
# Modèles Graphiques Probabilistes

> Cette note pose les bases des **modèles graphiques probabilistes** (PGM) : comment représenter des distributions de probabilité complexes à l'aide de graphes, comment poser des questions à ces modèles (inférence), et comment les ajuster aux données (apprentissage). Le fil conducteur sera l'**exemple de l'étudiant** (intelligence / difficulté / note), un classique de Koller & Friedman.

---

## I. Introduction

### A. Vue d'ensemble du cours

**Pourquoi des graphes pour les probabilités ?** Deux motivations centrales :

- **Requêtes** : on veut répondre à des questions du type *"quelle est la probabilité que ce soit un spam sachant que je vois le mot pill ?"*. Les modèles graphiques rendent ces requêtes systématiques.
- **Complexité** : la théorie des graphes va nous permettre d'analyser la **vitesse des algorithmes d'apprentissage** et de quantifier la **complexité computationnelle** (NP-difficulté, par ex.) de différentes tâches.

L'étude des PGM se divise en **trois grandes parties**, qu'il faut toujours garder ensemble parce qu'elles s'imbriquent :

> [!warning] Les trois piliers
> 1. **Représentation** — comment spécifier un modèle ?
> 2. **Inférence** — comment poser des questions au modèle ?
> 3. **Apprentissage** — comment ajuster un modèle à des données réelles ?
>
> Ces trois thèmes sont **étroitement liés** : pour avoir des algorithmes d'inférence et d'apprentissage efficaces, il faut représenter le modèle adéquatement ; et apprendre nécessitera l'inférence comme sous-routine.

#### A.1 Représentation

Comment exprimer une distribution de probabilité qui modélise un phénomène réel ? Ce n'est pas trivial : on a vu qu'un modèle naïf pour classifier des spams avec $n$ mots possibles requiert en général $O(2^n)$ paramètres. On va attaquer cette difficulté avec des **techniques générales pour construire des modèles tractables**.

Ces recettes feront un usage intensif de la théorie des graphes : les probabilités seront décrites par des graphes dont les propriétés (connexité, **tree-width**, etc.) révèleront des propriétés probabilistes et algorithmiques du modèle (indépendance, complexité d'apprentissage).

#### A.2 Inférence

Étant donné un modèle probabiliste, comment obtenir des réponses à des questions pertinentes sur le monde ? Ces questions se ramènent souvent à **calculer des probabilités marginales ou conditionnelles** d'événements d'intérêt. Concrètement, on s'intéresse à deux types de questions :

> [!warning] Deux types d'inférence
> **Inférence marginale (marginal inference)** — quelle est la probabilité d'une variable donnée après qu'on ait sommé sur tout le reste ?
>
> $$p(x_1) \;=\; \sum_{x_2} \sum_{x_3} \cdots \sum_{x_n} p(x_1, x_2, \ldots, x_n)$$
>
> Exemple : déterminer la probabilité qu'une maison tirée au hasard ait plus de trois chambres.
>
> **Maximum a posteriori (MAP)** — quelle est l'assignation **la plus probable** des variables ?
>
> $$\underset{x_1, \ldots, x_n}{\arg\max} \; p(x_1, \ldots, x_n, y = 1)$$
>
> Exemple : déterminer le message spam le plus probable.

Souvent, ces requêtes incluent des **évidences** (comme dans l'exemple MAP ci-dessus avec $y = 1$), c'est-à-dire qu'on fixe l'assignation d'un sous-ensemble de variables.

> 💡 **L'inférence est difficile.** Pour beaucoup de probabilités d'intérêt, répondre à ces questions est **NP-difficile**. Crucialement, la tractabilité de l'inférence dépend de la **structure du graphe** qui décrit la distribution. Quand le problème est intractable, on se rabattra sur des **méthodes d'inférence approchée** — beaucoup d'algorithmes de cette partie viennent en fait de la physique statistique du milieu du 20e siècle.

#### A.3 Apprentissage

Notre dernière tâche est d'**ajuster un modèle à un dataset** — par exemple, un grand nombre d'exemples de mails labellisés spam/non-spam. À partir des données, on infère des patterns utiles (ex : quels mots se trouvent plus fréquemment dans les spams), qu'on utilisera ensuite pour prédire le futur.

> 💡 **Apprentissage et inférence sont indissociables.** L'inférence sera une **sous-routine clé** appelée de manière répétée à l'intérieur des algorithmes d'apprentissage. Le sujet de l'apprentissage fera aussi des ponts importants avec la **théorie de l'apprentissage statistique** (généralisation, overfitting) et la **statistique bayésienne** (combinaison de connaissances a priori et d'évidence observée).

---

### B. Distributions

On définit le dataset avec les paramètres suivants — c'est ce qu'on appellera **l'exemple de l'étudiant** (the student example) tout au long de la note :

> [!example] Fil rouge — l'exemple de l'étudiant
> - **Intelligence (I)** : $i^0$ (faible), $i^1$ (élevée)
> - **Difficulté (D)** du cours : $d^0$ (facile), $d^1$ (difficile)
> - **Note (G)** : $g^1$ (A), $g^2$ (B), $g^3$ (C)
>
> Avec sa loi jointe $P(I, D, G)$.

![[im1 (3).png|263]]
**Figure 1.** Loi jointe $P(I, D, G)$ de l'exemple de l'étudiant.

**Opération de conditionnement** sur $g^1$ : c'est un processus en **deux étapes** : (i) **réduction** et (ii) **renormalisation**, où l'on part d'une mesure non normalisée $p(I, D, g^1)$ pour aboutir à la probabilité conditionnelle $p(I, D \mid g^1)$.

![[im2 (3).png]]
**Figure 2.** Conditionnement sur $g^1$ : on extrait la tranche $g = g^1$ (réduction), puis on renormalise pour que la somme fasse 1.

**Opération de marginalisation** par rapport à $I$ :

$$p(D) \;=\; \sum_I p(I, D)$$

![[im3 (3).png|490]]
**Figure 3.** Marginalisation : on somme sur les valeurs de la variable qu'on veut éliminer.

---

### C. Facteurs

> [!warning] Définition (Facteur)
> Un **facteur** prend un ensemble de variables aléatoires comme argument et renvoie un nombre réel :
>
> $$\phi : \text{Val}(X_1, \ldots, X_k) \mapsto \mathbb{R}$$
>
> Et on définit le **scope** (portée) comme l'ensemble des variables aléatoires :
>
> $$\text{Scope} = \{X_1, \ldots, X_k\}$$

> [!example] Exemples de facteurs
> 1. Une loi jointe comme $p(I, D, G)$ est un facteur : elle prend un ensemble de variables aléatoires en entrée et sort un nombre.
> 2. La mesure non normalisée $p(I, D, g^1)$ est aussi un facteur — elle ne renvoie pas une probabilité mais un réel, ce qui est tout ce qui compte. **Important** : son scope est $\{I, D\}$ parce que $g$ est constante.

#### C.1 Distribution conditionnelle (CPD)

Une **distribution conditionnelle** (Conditional Probability Distribution, CPD) est un type particulier de facteur.

![[im4 (2).png|300]]
**Figure 4.** Une CPD pour la note $G$ sachant l'intelligence $I$ et la difficulté $D$.

Lecture : si j'ai un étudiant intelligent $i^1$ dans un cours difficile $d^1$, sa probabilité d'obtenir une bonne note est de $0.5$.

#### C.2 Facteurs généraux

![[im5 (2).png|224]]
**Figure 5.** Un facteur général sur trois variables — toutes les valeurs sont des réels positifs, mais la table ne somme pas nécessairement à 1.

#### C.3 Opérations sur les facteurs

**(a) Produit de facteurs.**

![[im6 (2).png|549]]
**Figure 6.** Produit de facteurs.

> [!example] Scope du produit de facteurs
> Le scope du produit de facteurs $\phi(A, B, C) \times \phi(C, D)$ est son domaine. Soit $f$ le produit de facteurs, c'est-à-dire $f(A, B, C, D) = \phi(A, B, C) \times \phi(C, D)$. Comme $f$ est une fonction sur $A, B, C, D$, son scope est $\{A, B, C, D\}$.

**(b) Marginalisation de facteurs.**

![[images/3-Apprentissage automatique/PGM/Introduction/im7 (1).png|342]]
**Figure 7.** Marginalisation d'un facteur — on somme sur les valeurs de la variable qu'on veut éliminer, exactement comme pour une distribution.

**(c) Réduction de facteurs.** *(Exactement la même opération que celle utilisée pour le conditionnement.)*

![[im8 (1).png|388]]
**Figure 8.** Réduction d'un facteur sur une assignation : on ne garde que les lignes compatibles avec l'assignation choisie.
