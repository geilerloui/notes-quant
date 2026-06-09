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

### A. Initialisation des poids

Le choix des valeurs initiales des matrices $W^{(l)}$ avant l'entraînement n'est pas anodin — un mauvais choix peut **empêcher complètement l'apprentissage**, même avec une architecture par ailleurs correcte. Avant de présenter les schémas standards (Xavier, He), il faut comprendre pourquoi naïvement initialiser à zéro ou à grandes valeurs ne marche pas.

#### Pourquoi pas zéro ?

Si on pose $W^{(l)} = 0$ pour toutes les couches, **tous les neurones d'une même couche reçoivent le même signal en entrée**. Lors du forward pass, ils produisent tous la même activation. Et lors du backward pass, ils reçoivent tous le même gradient — donc ils sont mis à jour de manière identique.

Le réseau se comporte alors comme s'il n'avait **qu'un seul neurone par couche** : la "symétrie" entre neurones n'est jamais brisée. C'est le **problème de la symétrie d'initialisation** — et il est rédhibitoire.

> [!warning] Conclusion
> Initialiser à zéro est **interdit** pour les matrices de poids. Pour les biais $b^{(l)}$, c'est en revanche acceptable (la symétrie est cassée par les poids).

On utilise donc nécessairement une initialisation **aléatoire**, typiquement gaussienne $W_{ij} \sim \mathcal{N}(0, \sigma^2)$. Reste à choisir $\sigma^2$.

#### Pourquoi pas n'importe quel $\sigma^2$ ?

Considérons une couche cachée avec $n_{\text{in}}$ entrées et activation sigmoïde. Pour un neurone donné, la pré-activation $z = \sum_{i=1}^{n_{\text{in}}} w_i x_i$ a une variance qui dépend de $\sigma^2$ et de $n_{\text{in}}$ :

$$\text{Var}(z) = n_{\text{in}} \cdot \sigma^2 \cdot \text{Var}(x)$$

(on suppose les $w_i$ et $x_i$ indépendants, centrés). Deux pièges symétriques en découlent :

- **$\sigma$ trop grand** : $\text{Var}(z)$ explose, la sigmoïde sature ($g(z) \approx 0$ ou $\approx 1$). Le gradient local $g'(z) = g(z)(1-g(z))$ devient nul ⇒ **vanishing gradient à l'initialisation**. L'apprentissage ne démarre jamais.
- **$\sigma$ trop petit** : $\text{Var}(z)$ s'effondre, toutes les pré-activations sont concentrées autour de 0. Le réseau se comporte comme une longue chaîne d'applications quasi-linéaires (sigmoïde près de 0 ≈ identité), perdant toute capacité expressive.

L'enjeu est de **préserver la variance du signal** quand il traverse les couches — ni explosion, ni effondrement. C'est exactement ce que Xavier et He optimisent.

#### Xavier / Glorot (2010)

[Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) cherchent à préserver la variance **dans les deux sens** : forward (activations) et backward (gradients). En faisant les calculs sous hypothèse d'activation linéaire et de symétrie, on tombe sur la condition

$$\boxed{\;\sigma^2 = \text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}\;}$$

où $n_{\text{in}}$ et $n_{\text{out}}$ sont les nombres de neurones en entrée et en sortie de la couche.

**Variantes équivalentes** :
- **Xavier uniforme** : $W_{ij} \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$ (même variance que la version gaussienne).
- **Variante simplifiée** : $\sigma^2 = 1/n_{\text{in}}$ — souvent suffisante en pratique.

Xavier est conçu pour des activations **symétriques autour de zéro** (tanh, sigmoïde rescalée). Avec ReLU, il sous-estime la variance nécessaire.

#### He (2015) — la version ReLU

[He et al. (2015)](https://arxiv.org/abs/1502.01852) refont la dérivation pour ReLU. La différence-clé : ReLU "tue" environ **la moitié** des activations (toutes celles avec $z < 0$ donnent $\text{ReLU}(z) = 0$). Pour compenser cette perte, il faut **doubler la variance des poids** :

$$\boxed{\;\sigma^2 = \frac{2}{n_{\text{in}}}\;}$$

En pratique, c'est l'initialisation **par défaut** dans PyTorch et TensorFlow pour les couches suivies d'une ReLU (ou Leaky ReLU).

#### Récapitulatif

| Activation | Initialisation | Variance |
|:---:|:---:|:---:|
| tanh, sigmoïde | **Xavier / Glorot** | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| ReLU, Leaky ReLU | **He** | $\frac{2}{n_{\text{in}}}$ |
| SELU, autres | Variantes spécifiques | — |

**En pratique** : ne pas y penser, les frameworks utilisent les bons défauts. Mais comprendre l'enjeu permet de diagnostiquer un réseau qui ne s'entraîne pas — un mauvais init (ex : He sur tanh, ou Xavier sur ReLU profond) peut donner l'illusion d'un problème d'architecture alors que c'est juste l'initialisation.

### B. Architecture de Base

#### Fonctions d'activation
![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im5.png]]


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

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im6.png|494]]

*2. Diminution systématique du gradient*

Un fait moins évident : le gradient local de la sigmoid $(z \times (1 - z))$ atteint un maximum de 0.25 quand $z = 0.5$. Cela signifie qu'**à chaque passage** à travers une porte sigmoid, l'amplitude du signal de gradient diminue d'au moins 75%. Avec SGD classique, cela rend l'entraînement des couches inférieures beaucoup plus lent que les couches supérieures.

**Problème des ReLU "mortes"**

ReLU résout le problème du vanishing gradient, mais introduit un nouveau défi : les "dead ReLUs".

Si un neurone est "clampé" à zéro lors du forward pass (i.e., $z = 0$, il ne "fire" pas), ses poids recevront un gradient de zéro. Cela peut mener au problème du "dead ReLU" : si un neurone ReLU est malheureusement initialisé de façon à ne jamais s'activer, ou si ses poids sont "assommés" par une grande mise à jour pendant l'entraînement, ce neurone restera **définitivement mort**. C'est comme des dommages cérébraux permanents et irréversibles.

Il arrive qu'en passant tout le dataset d'entraînement dans un réseau entraîné, une large fraction (ex: 40%) des neurones soient restés à zéro tout le temps.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im7.png|598]]

**Solutions possibles** :
- **Leaky ReLU** : pente légère (0.01) pour les valeurs négatives
- **Initialization soignée** : Xavier/He initialization
- **Learning rate adaptatif** : éviter les mises à jour trop agressives

### C. Techniques de Régularisation

Les techniques de régularisation visent à **prévenir l'overfitting** en contraignant le modèle à ne pas trop s'ajuster aux données d'entraînement. Quatre techniques standards : L2 (weight decay), Dropout, Data Augmentation, Early Stopping.

#### L2 Regularization (Weight Decay)

**Idée.** Quand on observe de l'overfitting, la première solution est d'ajouter de la régularisation ou plus de données. Si avoir plus de données n'est pas faisable, la régularisation L2 est l'outil le plus couramment utilisé (bien plus que L1 sur les réseaux de neurones).

**Formulation.** On ajoute un terme de pénalité sur la **norme de Frobenius** des matrices de poids à la fonction de coût :

$$J(W_1, b_1, \ldots, W_L, b_L) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}_i, y_i) + \frac{\lambda}{2m} \sum_{l=1}^L \|W_l\|_F^2$$

avec

$$\|W_l\|_F^2 = \sum_{i=1}^{n_{l+1}} \sum_{j=1}^{n_l} (w_{ij}^{(l)})^2$$

$\lambda$ est le **paramètre de régularisation** (hyperparamètre à régler).

**Pseudocode de mise à jour** :

$$\begin{aligned}
dW_l &= (\text{gradient de la backprop classique}) + \frac{\lambda}{m} W_l \\
W_l &:= W_l - \alpha \cdot dW_l
\end{aligned}$$

**Pourquoi "Weight Decay" ?** En réorganisant la mise à jour :

$$W_l := W_l \left(1 - \frac{\alpha \lambda}{m}\right) - \alpha \cdot (\text{gradient backprop})$$

À chaque pas, **indépendamment du gradient de la loss**, les poids sont multipliés par $(1 - \alpha\lambda/m) < 1$ — ils décroissent ("decay") vers zéro. D'où le nom alternatif.

**Pourquoi ça réduit l'overfitting ?** Si on augmente $\lambda \gg 0$, les poids sont fortement tirés vers zéro $W_l \approx 0$. Le réseau garde tous ses nœuds en apparence, mais l'**influence effective** de chaque neurone caché est très réduite — on travaille en pratique sur un réseau "plus petit", avec plus de biais et moins de variance.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im24.png]]

Plus concrètement, sur un neurone avec activation tanh et $z_l = W_l a_{l-1} + b_l$ : avec $\lambda$ grand, $W_l$ devient petit, donc $z_l$ reste autour de 0, et la tanh est dans sa **zone quasi-linéaire**. Le réseau devient effectivement plus linéaire — réduisant sa capacité à overfitter.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im25.png]]

> [!warning] Astuce de debugging
> Quand on plot $J$ pendant l'entraînement pour vérifier la convergence, il faut bien plotter $J$ **incluant le terme de régularisation** — sinon on peut voir le terme de loss diminuer alors que le terme régularisé augmente, et passer à côté du diagnostic.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im23.png]]

#### Dropout

**Idée.** À chaque passage forward pendant l'entraînement, on **désactive aléatoirement** une fraction des neurones (on les met à zéro). On se retrouve à travailler à chaque itération sur un **sous-réseau** différent, plus petit.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im21.png]]

**Inverted Dropout — l'implémentation standard.** On illustre avec une couche $l=3$ et l'hyperparamètre $\text{keep\_prob} = 0.8$ (probabilité de **garder** un neurone) :

```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob   # masque booléen
a3 = np.multiply(a3, d3)                                    # masquer
a3 /= keep_prob                                             # rescaling (← "inverted")
```

- $d3$ est un **masque booléen** qui multiplie $a3$ pour annuler les neurones désactivés.
- La division par $\text{keep\_prob}$ est le truc clé de la variante **inverted** : avec 80% de neurones gardés, l'activation a3 totale est en moyenne 20% plus faible — la division compense pour préserver l'**espérance** de la sortie. Sans ce rescaling, la statistique du signal change entre train et test.

**Prédiction au test time.** On **n'utilise PAS Dropout au test time**. Le forward pass est complet, sans aucun masquage. Si on activait Dropout au test, on ajouterait du bruit aléatoire aux prédictions (et il faudrait moyenner sur plusieurs runs pour stabiliser).

**Pourquoi Dropout fonctionne ?** Deux intuitions complémentaires :

1. **Sous-réseaux multiples** : à chaque itération, on entraîne un sous-réseau aléatoire. C'est un peu comme entraîner un ensemble de modèles et moyenner — un effet similaire au bagging.
2. **Pas de dépendance excessive à une feature** : un neurone ne peut pas trop "miser" sur une entrée précise, puisque celle-ci peut être désactivée. Il est donc forcé de **répartir** son poids sur plusieurs entrées — effet similaire à L2 (shrinkage).

**Réglage du `keep_prob` par couche.** On peut varier `keep_prob` selon les couches. Sur les couches très larges (plus susceptibles d'overfitter), on prend `keep_prob` faible (0.5–0.7). Sur les couches plus petites ou les couches de sortie, on monte à 0.9 ou même 1.0 (pas de Dropout).

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im22.png|503]]

> [!warning] Le défaut principal de Dropout
> Avec Dropout actif, la loss $J$ **n'est plus bien définie** — elle change à chaque itération en fonction du masque tiré. On perd donc l'outil de debugging "vérifier que la loss décroît monotonement". L'astuce pratique : **entraîner d'abord sans Dropout** jusqu'à confirmer que ça décroît bien, puis activer Dropout pour la phase finale d'entraînement.

#### Data Augmentation

Augmenter artificiellement la taille du training set en transformant les exemples existants. Les versions augmentées n'apportent pas autant d'information que de vraies nouvelles données, mais c'est un moyen **bon marché** d'enrichir le dataset.

**Pour les images** : retournements horizontaux/verticaux, crops aléatoires, rotations légères, zoom, changements de luminosité/contraste. On apprend implicitement au réseau que *"un chat zoomé/retourné est toujours un chat"*.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im26.png|554]]

**Pour des digits** (MNIST-like) : déformations élastiques, distorsions locales.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im27.png|543]]

**Modernement** : la data augmentation est devenue centrale en deep learning, avec des techniques sophistiquées (Mixup, CutMix, AugMix, RandAugment…) qui combinent plusieurs transformations stochastiques. Pour le texte (LLM, NLP) : back-translation, paraphrasing. Pour l'audio : time stretching, pitch shifting, ajout de bruit.

#### Early Stopping

**Idée.** On suit la performance sur un **set de validation** (séparé du training set) en parallèle de l'entraînement, et on **arrête** quand cette performance commence à se dégrader — même si la loss d'entraînement continue de diminuer.

À l'époque 0 le modèle est aléatoire : il sous-fit aussi bien le train que la validation. Au fil des époques, la training error décroît continûment — c'est attendu. La validation error décroît au début (le modèle apprend de vrais patterns), atteint un **minimum**, puis remonte (le modèle commence à mémoriser le bruit du training set).

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im2 (3).png|436]]

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im3 (4).png|430]]

**Recette pratique** :
1. Réserver un **set de validation** (typiquement 10-20% du training).
2. Évaluer la perf à chaque époque (ou à intervalles réguliers).
3. **Sauvegarder le modèle** à chaque amélioration de la validation loss.
4. Arrêter si la validation loss n'a pas diminué depuis $p$ époques (le "patience" — typiquement 5-20).
5. Restaurer le meilleur modèle sauvegardé pour la prédiction finale.

> [!note]- Le piège de l'overfitting suivi du double descent
> La courbe classique "validation loss en U" décrite ci-dessus est l'histoire racontée par la théorie classique du biais-variance. Mais sur des modèles très sur-paramétrés (réseaux modernes, $> 10^8$ paramètres), on observe le phénomène **double descent** ([Belkin et al. 2018](https://arxiv.org/pdf/1812.11118.pdf)) : la validation loss peut **redescendre** après le pic d'overfitting, pour atteindre des perfs encore meilleures. Early stopping reste utile en pratique, mais il faut être conscient de cette subtilité — sur des modèles très larges, attendre plus longtemps peut payer.

### D. Optimisation & Stabilité

Ces techniques améliorent la **stabilité** et **l'efficacité** du processus d'entraînement, mais ne sont pas à proprement parler de la régularisation (elles modifient l'optimisation, pas la fonction objectif).

#### Batch Normalization

**Idée centrale.** Introduite par [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167). La motivation : tout comme normaliser les inputs $X$ accélère l'entraînement, **normaliser les activations intermédiaires** $a^{(l)}$ devrait accélérer l'entraînement des couches suivantes. Techniquement, on normalise les **pré-activations** $z^{(l)}$ plutôt que les $a^{(l)}$ — la communauté débat encore sur lequel est meilleur, mais l'effet est similaire.

**Mécanique en 4 étapes.** Pour un mini-batch de pré-activations $\{z_1, \ldots, z_n\}$ à une couche donnée :

1. **Moyenne du batch** : $\mu = \frac{1}{n}\sum_i z_i$
2. **Variance du batch** : $\sigma^2 = \frac{1}{n}\sum_i (z_i - \mu)^2$
3. **Normalisation** (avec $\varepsilon$ pour stabilité numérique) :
$$z_i^{\text{norm}} = \frac{z_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$
4. **Rescaling apprenable** :
$$\tilde{z}_i = \gamma\, z_i^{\text{norm}} + \beta$$

où $\gamma$ et $\beta$ sont des **paramètres appris**, propres à chaque couche.

**Pourquoi $\gamma$ et $\beta$ ?** Forcer toutes les pré-activations à avoir moyenne 0 et variance 1 est peut-être trop restrictif (la distribution optimale dépend de la non-linéarité qui suit). On laisse donc le réseau **choisir librement** la moyenne et la variance désirées via $\gamma$ et $\beta$. Si $\gamma = \sqrt{\sigma^2 + \varepsilon}$ et $\beta = \mu$, on retombe sur $\tilde{z}_i = z_i$ — le réseau peut "défaire" la normalisation s'il en a besoin. Mais en pratique, il converge généralement vers des $(\gamma, \beta)$ différents qui facilitent l'apprentissage.

**Batch Norm au test time.** Petit souci : au test time, on n'a pas de mini-batch — on prédit souvent **une seule observation** à la fois. Comment calculer $\mu$ et $\sigma^2$ ?

Solution : pendant l'entraînement, on maintient une **moyenne mobile exponentielle** des $\mu$ et $\sigma^2$ observés sur les mini-batches. Pour chaque couche $l$, on a une séquence

$$\mu_{\{1\}}^{(l)}, \mu_{\{2\}}^{(l)}, \mu_{\{3\}}^{(l)}, \ldots$$

dont on calcule la moyenne mobile pour estimer les vraies $\mu$ et $\sigma^2$ de la couche. Au test time, on utilise ces estimations fixes (pas celles du "batch" courant).

##### Pourquoi Batch Norm fonctionne

**Raison 1 — accélération de l'apprentissage.** Comme normaliser $X$ accélère la régression linéaire, normaliser les pré-activations accélère l'apprentissage des couches profondes.

**Raison 2 — robustesse au covariate shift interne.** Imaginons qu'on entraîne un classifieur de chats sur des chats noirs uniquement, et qu'on teste sur des chats de toutes couleurs : la distribution des entrées change entre train et test, c'est le **covariate shift**.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im29.png]]

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im30.png]]

Le même phénomène se produit **à l'intérieur du réseau** entre couches. Du point de vue de la couche $l = 3$, ses entrées sont les sorties des couches précédentes — et celles-ci **changent en permanence** pendant l'entraînement (les poids des couches 1 et 2 sont en train d'être mis à jour). C'est le **covariate shift interne**. Batch Norm fige les statistiques (moyenne, variance) des entrées de chaque couche, ce qui rend les couches profondes plus robustes aux mises à jour des couches superficielles. Résultat : on peut utiliser des learning rates plus agressifs sans diverger.

**Raison 3 — effet régularisateur.** Chaque mini-batch a sa propre moyenne et variance, qui **fluctuent** selon les observations échantillonnées. Ces fluctuations ajoutent du bruit à $z^{(l)}$ — un peu comme Dropout qui ajoute du bruit en multipliant par un masque aléatoire. Cet effet de régularisation est un **bonus** de Batch Norm, pas son intention première.

> [!warning] Conséquences pratiques
> - Plus le mini-batch est grand, plus les statistiques sont stables ⇒ moins de bruit ⇒ moins d'effet régularisateur. Avec `batch_size = 512` au lieu de `64`, l'effet régularisateur de Batch Norm diminue significativement.
> - Au test time, on traite **un seul exemple à la fois** : impossible d'utiliser les stats du "batch" (variance non définie sur 1 exemple). D'où l'astuce de la moyenne mobile expliquée plus haut.

#### Learning Rate Decay

**Idée.** Avec un `mini-batch_size = 64` (par exemple), les pas de SGD sont un peu bruités à cause de la variance d'estimation du gradient. Vers la fin de l'entraînement, on approche du minimum mais avec un learning rate $\alpha$ **fixe**, on "tourne autour" sans converger. Solution : **réduire $\alpha$** progressivement ⇒ on oscille dans une zone de plus en plus serrée autour du minimum.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im32.png|397]]

**Schéma classique** : décroissance hyperbolique avec le numéro d'époque

$$\alpha = \frac{1}{1 + \text{decay\_rate} \times \text{epoch\_num}} \cdot \alpha_0$$

Avec $\alpha_0 = 0.2$ et $\text{decay\_rate} = 1$ :

| Époque | 1 | 2 | 3 | 4 | … |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\alpha$ | $0.10$ | $0.067$ | $0.05$ | $0.04$ | … |

**Variantes courantes** :

- **Décroissance exponentielle** : $\alpha = 0.95^{\text{epoch\_num}} \cdot \alpha_0$
- **Décroissance en $1/\sqrt{\text{epoch}}$** : $\alpha = \frac{k}{\sqrt{\text{epoch\_num}}} \cdot \alpha_0$
- **Décroissance en $1/\sqrt{t}$** (avec $t$ = numéro d'itération, pas d'époque) : $\alpha = \frac{k}{\sqrt{t}} \cdot \alpha_0$
- **Discrete staircase** : on divise $\alpha$ par 2 (ou 10) à des époques précises (ex : après 30, 60, 90 époques).
- **Manual decay** : on observe la training/validation loss et on réduit $\alpha$ à la main quand ça plafonne.

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im33.png]]

**Modernement** : avec Adam (et ses dérivés AdamW, Lion…), le LR decay est moins critique qu'avec SGD vanilla — Adam adapte déjà son pas par paramètre. Mais des techniques comme **cosine annealing** (décroissance en cosinus) ou **warmup linéaire** (montée puis descente) restent standards dans les setups SOTA modernes (transformers, ViTs).

### E. Problèmes Transversaux

#### Problème du vanishing gradient

Quand on remonte en arrière lors de la rétropropagation, si les gradients sont proches de 0, on multiplie de nombreux petits nombres, ce qui peut faire disparaître le signal de gradient dans les couches profondes. Ce phénomène affecte particulièrement les réseaux profonds et constitue l'un des défis majeurs de l'entraînement des réseaux de neurones.
