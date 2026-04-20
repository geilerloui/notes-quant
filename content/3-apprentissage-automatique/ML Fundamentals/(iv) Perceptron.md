---
title: Réseaux de Neurones
description: Introduction aux réseaux de neurones artificiels
weight: 3
---

# Perceptron Multi-Couches

## Architecture du Perceptron Multi-Couches (MILP)

Un réseau de neurones artificiels est composé de neurones organisés en couches. Le cas le plus simple est le perceptron multi-couches (MLP) avec une couche cachée :

![[Pasted image 20260418192240.png|397]]

![[Pasted image 20260418192459.png|317]]

## I - Forward Pass

### Étape 1: Initialisation des paramètres

$[X]_{3 \times 2}$ : matrice des données d'entrée  
$[W^{(1)}]_{2 \times 3} \sim \mathcal{N}(0,1)$ : matrice de poids entre couche d'entrée et couche cachée  
$[b^{(1)}]_{1 \times 3}$ : vecteur de biais pour la couche cachée

### Étape 2: Passage de l'entrée vers la couche cachée

Cette étape transforme nos données d'entrée 2D en représentations 3D dans l'espace latent. Le calcul se décompose en deux phases : combinaisons linéaires puis activations non-linéaires.

**Phase 1 : Combinaisons linéaires avec biais**
$$Z^{(2)} = X W^{(1)} + b^{(1)}$$

Avec nos données d'exemple :
$$Z^{(2)} = \begin{bmatrix} 
\textcolor{teal}{\mathbf{3}} & \textcolor{teal}{\mathbf{5}} \\
\textcolor{orange}{\mathbf{5}} & \textcolor{orange}{\mathbf{1}} \\
\textcolor{violet}{\mathbf{10}} & \textcolor{violet}{\mathbf{2}}
\end{bmatrix} 
\begin{bmatrix} 
w_{11}^{(1)} & w_{12}^{(1)} & w_{13}^{(1)} \\
w_{21}^{(1)} & w_{22}^{(1)} & w_{23}^{(1)}
\end{bmatrix}
+ \begin{bmatrix} b_1^{(1)} & b_2^{(1)} & b_3^{(1)} \end{bmatrix}$$

**Phase 2 : Activations non-linéaires**  
$$A^{(2)} = g_1(Z^{(2)})$$

Chaque élément $z_{ij}^{(2)}$ est transformé par la fonction d'activation $g_1()$ (ex: sigmoid, ReLU, tanh). Le résultat final est :

$$A^{(2)} = \begin{bmatrix} 
\textcolor{teal}{a_{11}^{(2)}} & \textcolor{teal}{a_{12}^{(2)}} & \textcolor{teal}{a_{13}^{(2)}} \\
\textcolor{orange}{a_{21}^{(2)}} & \textcolor{orange}{a_{22}^{(2)}} & \textcolor{orange}{a_{23}^{(2)}} \\
\textcolor{violet}{a_{31}^{(2)}} & \textcolor{violet}{a_{32}^{(2)}} & \textcolor{violet}{a_{33}^{(2)}}
\end{bmatrix}$$


> [!note]- 💡 Représentation Learning
> **Transformation de l'espace** : La fonction d'activation $g()$ effectue une projection non linéaire dans un nouvel espace en 3D. 
> 
> **Espaces de représentation** :
> - $X^{(1)}$ : espace de représentation initiale (2D)
> - $A^{(2)}$ : projection dans un espace latent de dimension supérieure (3D)
> 
> **Objectif** : Ce "representation learning" permet de rendre les structures linéairement séparables dans le nouvel espace.


### Étape 3: Passage vers la couche de sortie

Cette étape agrège l'information des 3 neurones cachés en une seule prédiction finale. C'est une **combinaison linéaire pondérée** des activations de la couche cachée.

**Combinaison linéaire avec biais**
$$Z^{(3)} = A^{(2)} W^{(2)} + b^{(2)}$$

Avec les dimensions explicites :
$$[Z^{(3)}]_{3 \times 1} = \begin{bmatrix}
\textcolor{teal}{a_{11}^{(2)}} & \textcolor{teal}{a_{12}^{(2)}} & \textcolor{teal}{a_{13}^{(2)}}\\
\textcolor{orange}{a_{21}^{(2)}} & \textcolor{orange}{a_{22}^{(2)}} & \textcolor{orange}{a_{23}^{(2)}} \\
\textcolor{violet}{a_{31}^{(2)}} & \textcolor{violet}{a_{32}^{(2)}} & \textcolor{violet}{a_{33}^{(2)}}
\end{bmatrix}_{3 \times 3}
\begin{bmatrix}
w_1^{(2)} \\ w_2^{(2)} \\ w_3^{(2)}
\end{bmatrix}_{3 \times 1}
+ b^{(2)}$$

**Interprétation** : Chaque poids $w_i^{(2)}$ détermine l'importance du neurone caché $i$ dans la prédiction finale. Le biais $b^{(2)}$ permet de décaler la sortie. Cette couche agit comme un **classifieur linéaire** dans l'espace latent appris par la couche cachée.

### Étape 4: Prédiction finale et évaluation

Cette étape finale transforme les scores bruts en prédictions et quantifie l'erreur entre prédictions et vraies valeurs.

**Activation de sortie**
$$\hat{y} = g_2(Z^{(3)})$$

La fonction d'activation de sortie $g_2()$ dépend du type de problème :
- **Régression** : identité $g_2(z) = z$ (sortie linéaire)
- **Classification binaire** : sigmoïde $g_2(z) = \frac{1}{1+e^{-z}}$ (sortie entre 0 et 1)
- **Classification multi-classe** : softmax $g_2(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ (probabilités)

**Fonction de coût**

Le réseau apprend en minimisant une fonction de coût qui mesure l'écart entre prédictions $\hat{y}$ et vraies valeurs $y$. Pour la régression, on utilise l'erreur quadratique moyenne (MSE) :

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2=\frac{1}{2n}\sum_{i=1}^{n}(y^{(i)} - g_2(g_1(XW^{(1)})W^{(2)} ))^2
$$

où :
- $n$ est le nombre d'exemples d'entraînement
- $\theta = \{W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}\}$ représente tous les paramètres du réseau
- Le facteur $\frac{1}{2}$ simplifie la dérivation lors de la backpropagation

**Interprétation** : Cette fonction de coût est **différentiable** partout, permettant l'utilisation du gradient descent. L'objectif de l'entraînement est de trouver les paramètres $\theta^*$ qui minimisent $J(\theta)$ :

$$\theta^* = \arg\min_{\theta} J(\theta)$$

## II - Algorithme de Backpropagation

L'algorithme de rétropropagation utilise la règle de dérivation en chaîne (chain rule) pour calculer les gradients de la fonction de coût par rapport aux paramètres du réseau.

### Calcul des gradients

L'algorithme calcule les gradients de $J(\theta)$ par rapport à tous les paramètres et met à jour les poids :

$$\frac{\partial J}{\partial W^{(2)}} = (A^{(2)})^T \delta^{(3)}, \quad \frac{\partial J}{\partial b^{(2)}} = \sum \delta^{(3)}$$

$$\frac{\partial J}{\partial W^{(1)}} = X^T \delta^{(2)}, \quad \frac{\partial J}{\partial b^{(1)}} = \sum \delta^{(2)}$$

où $\delta^{(3)}$ représente les erreurs pour la couche de sortie et $\delta^{(2)} = \delta^{(3)}(W^{(2)})^T \odot g'(Z^{(2)})$ les erreurs propagées vers la couche cachée. La mise à jour s'effectue par : $W := W - \alpha \nabla W$ et $b := b - \alpha \nabla b$.

### Parallèle avec la régression linéaire

La backpropagation suit exactement le même schéma conceptuel que le gradient descent en régression linéaire, mais avec des compositions de fonctions plus complexes.

**Cadre de la régression linéaire :** $\theta = (\theta_0, \theta_1)$, $n$ individus, $h_\theta(x_i) = \theta_0 + \theta_1 x_i$. Chaque terme de la somme :

$$f_i(\theta) = \frac{1}{2}\left(y_i - (\theta_0 + \theta_1 x_i)\right)^2$$

Le gradient individuel :

$$g_i = \nabla f_i(\theta) = \begin{pmatrix} \partial f_i / \partial \theta_0 \\ \partial f_i / \partial \theta_1 \end{pmatrix} = \begin{pmatrix} -(y_i - \hat{y}_i) \\ -x_i(y_i - \hat{y}_i) \end{pmatrix}$$

Mise à jour gradient descent full-batch :

$$\theta^{(k+1)} = \theta^{(k)} - \alpha \cdot \frac{1}{n} \sum_{i=1}^n \nabla f_i(\theta^{(k)})$$

**Analogie conceptuelle :** c'est comme si on avait forward pass je multiplie par $x_i$ j'ai mon vecteur initial $\theta$ je backward pass je calcule mon nouveau $\theta$ puis je multiplie par $x_i$ je multiplie par la différence des $y_i$ qui a changé à cause du nouveau $\theta$ etc 

En comparaison dans le réseau de neurone : j'initialise mes matrices $W^{(i)}$ je fais une forward passe j'obtiens mes $\hat{y}$ puis je fais une rétropagation (ie un chaine rule en arrière) pour mettre à jours mes poids $W^{(i)}$, je refais une forward pass avec ces nouveaux poids je refais un chaine rule en arrière pour mettre à jours mes poids etc 

C'est étonnamment plutôt rapide alors qu'il y a beaucoup d'étapes de calcul !

## III - Algorithme d'Entraînement

L'algorithme d'entraînement alterne entre les phases forward (Section I) et backward (Section II) :

**Différence entre itération et epoch** : Une itération correspond à un passage forward-backward sur un batch, une époque correspond à un passage sur tout le dataset.

```python
for epoch in range(num_epochs):
    for batch in dataset:
        # 1. FORWARD PASS
        y_pred = forward_pass(X, W)
        loss = compute_loss(y_true, y_pred)
        
        # 2. BACKWARD PASS (backpropagation)
        gradients = backward_pass(loss, W)
        
        # 3. MISE À JOUR DES POIDS
        W = W - alpha * gradients
```

## IV - Visualisations 3D


en gros une fois que le modèle a été appris on a les matrice W^(i) qui ont été updaté et donc on a une fonction qui permet d'écrire une fonction pour faire de l'interpolation quoi en gros 


$J(\theta)=\frac{1}{2n}\sum_{i=1}^{n}(y^{(i)} - g_2(g_1(XW^{(1)})W^{(2)} ))^2$

![[Pasted image 20260419141718.png|332]]

## V - Choix d'Architectures

### A. Architecture de Base

#### Fonctions d'activation
![[im5.png]]


Le choix de la fonction d'activation dans les couches cachées est crucial pour les performances du réseau. Chaque fonction a ses avantages et inconvénients :

- **Sigmoid** : $g(z) = \frac{1}{1 + e^{-z}}$ (sortie entre 0 et 1)
- **Tanh** : $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ (sortie entre -1 et 1)
- **ReLU** : $g(z) = \max(0, z)$ (supprime les valeurs négatives)
- **Leaky ReLU** : $g(z) = \max(0.01z, z)$ (pente légère pour les valeurs négatives)

**Problème du vanishing gradient avec Sigmoid/Tanh**

Les fonctions sigmoid et tanh souffrent de deux problèmes majeurs :

*1. Saturation et gradients nuls*

Lorsque la matrice de poids $W$ est initialisée avec des valeurs trop importantes, les sorties du produit matriciel peuvent devenir très grandes (ex: valeurs entre -400 et 400). Dans ce cas, les sorties du vecteur $z$ deviennent quasi-binaires (0 ou 1). 

Le problème est que le gradient local de la sigmoid $z \times (1-z)$ devient alors **proche de zéro** ("vanish"), rendant les gradients pour $x$ et $W$ également nuls. Le reste de la rétropropagation sera nul à cause de la multiplication dans la chain rule.

![[im6.png|494]]

*2. Diminution systématique du gradient*

Un fait moins évident : le gradient local de la sigmoid $(z \times (1 - z))$ atteint un maximum de 0.25 quand $z = 0.5$. Cela signifie qu'**à chaque passage** à travers une porte sigmoid, l'amplitude du signal de gradient diminue d'au moins 75%. Avec SGD classique, cela rend l'entraînement des couches inférieures beaucoup plus lent que les couches supérieures.

**Problème des ReLU "mortes"**

ReLU résout le problème du vanishing gradient, mais introduit un nouveau défi : les "dead ReLUs".

Si un neurone est "clampé" à zéro lors du forward pass (i.e., $z = 0$, il ne "fire" pas), ses poids recevront un gradient de zéro. Cela peut mener au problème du "dead ReLU" : si un neurone ReLU est malheureusement initialisé de façon à ne jamais s'activer, ou si ses poids sont "assommés" par une grande mise à jour pendant l'entraînement, ce neurone restera **définitivement mort**. C'est comme des dommages cérébraux permanents et irréversibles.

Il arrive qu'en passant tout le dataset d'entraînement dans un réseau entraîné, une large fraction (ex: 40%) des neurones soient restés à zéro tout le temps.

![[im7.png|598]]

**Solutions possibles** :
- **Leaky ReLU** : pente légère (0.01) pour les valeurs négatives
- **Initialization soignée** : Xavier/He initialization
- **Learning rate adaptatif** : éviter les mises à jour trop agressives

### B. Techniques de Regularization

Les techniques de regularization visent à **prévenir l'overfitting** en contraignant le modèle à ne pas trop s'ajuster aux données d'entraînement.

#### L2 Regularization (Weight Decay)

*Section à développer*

#### Dropout

*Section à développer*

#### Data Augmentation

*Section à développer*

#### Early Stopping

*Section à développer*

### C. Optimisation & Stabilité

Ces techniques améliorent la **stabilité** et **l'efficacité** du processus d'entraînement.

#### Batch Normalization

*Section à développer*

#### Learning Rate Decay

*Section à développer*

### D. Problèmes Transversaux

#### Problème du vanishing gradient

Quand on remonte en arrière lors de la rétropropagation, si les gradients sont proches de 0, on multiplie de nombreux petits nombres, ce qui peut faire disparaître le signal de gradient dans les couches profondes. Ce phénomène affecte particulièrement les réseaux profonds et constitue l'un des défis majeurs de l'entraînement des réseaux de neurones.
