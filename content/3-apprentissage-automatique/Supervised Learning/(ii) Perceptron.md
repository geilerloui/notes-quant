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
### A. Initialisation avec Xavier

### B. Architecture de Base

#### Fonctions d'activation
![[images/3-Apprentissage automatique/Supervised learning/im5.png]]


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

![[images/3-Apprentissage automatique/Supervised learning/im6.png|494]]

*2. Diminution systématique du gradient*

Un fait moins évident : le gradient local de la sigmoid $(z \times (1 - z))$ atteint un maximum de 0.25 quand $z = 0.5$. Cela signifie qu'**à chaque passage** à travers une porte sigmoid, l'amplitude du signal de gradient diminue d'au moins 75%. Avec SGD classique, cela rend l'entraînement des couches inférieures beaucoup plus lent que les couches supérieures.

**Problème des ReLU "mortes"**

ReLU résout le problème du vanishing gradient, mais introduit un nouveau défi : les "dead ReLUs".

Si un neurone est "clampé" à zéro lors du forward pass (i.e., $z = 0$, il ne "fire" pas), ses poids recevront un gradient de zéro. Cela peut mener au problème du "dead ReLU" : si un neurone ReLU est malheureusement initialisé de façon à ne jamais s'activer, ou si ses poids sont "assommés" par une grande mise à jour pendant l'entraînement, ce neurone restera **définitivement mort**. C'est comme des dommages cérébraux permanents et irréversibles.

Il arrive qu'en passant tout le dataset d'entraînement dans un réseau entraîné, une large fraction (ex: 40%) des neurones soient restés à zéro tout le temps.

![[images/3-Apprentissage automatique/Supervised learning/im7.png|598]]

**Solutions possibles** :
- **Leaky ReLU** : pente légère (0.01) pour les valeurs négatives
- **Initialization soignée** : Xavier/He initialization
- **Learning rate adaptatif** : éviter les mises à jour trop agressives

### C. Techniques de Regularization

Les techniques de regularization visent à **prévenir l'overfitting** en contraignant le modèle à ne pas trop s'ajuster aux données d'entraînement.

#### L2 Regularization (Weight Decay)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection{L2 Regularization: Weight Decay}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\textbf{Idea:} When we are encountering overfitting the first solution is to apply regularization or adding more data but this technique can be not available.\\

L2 regularization is used much more often than L1
$$
\boxed{\lambda = \text{regularization parameter}}
$$
That we add to the cost function of the Neural Network
$$
J(w_1, b_1, ..., w_l, b_l) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}_i, y_i) + \frac{\lambda}{2m} \sum_{l=1}^L || W_l||_F^2
$$
Where
$$
||W_l||_F^2 = \sum_{i=1}^{n_{l+1}} \sum_{j=1}^{n_{l}} (w_{ij}^l)^2
$$
Note that this is the Frobenius norm that is used here.\\

\textbf{Pseudocode:}
$$
\begin{aligned}
dW_l &= (from~backprop) + \frac{\lambda}{m} W_l \\
W_l &= W_l - \alpha \cdot dW_l
\end{aligned}
$$
Where $dW_l = \frac{\partial J}{\partial W_l}$.\\

\textbf{Why is also called "Weight Decay" ?} We just need to rewrite the update equations:
$$
\begin{aligned}
W_l &= W_l - \alpha\Big[ (from~backprop) + \frac{\lambda}{m}W_l    \Big] \\
&= W_l (1- \frac{\alpha \lambda}{m}) - \alpha(from~backprop)
\end{aligned}
$$
We see that according to this equation the weight matrix independently of backprop will be reduce by the fraction coefficient.\\

\textbf{Why Regularization reduces overfitting ?} If we increase the $\lambda>>0$ regularization parameter well enough, the weights will be close to zero $W_l \approx 0$. Thus, not exactly on the image we still keep all the nodes, but their influence will be reduced. As such, we will be working on a smaller neural nets and we will get high bias:

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im24.png]]


To be more concrete on a specific cell, if we consider the tanh activation function as below and the equation:
$$
z_l = W_l a_{l-1} + b_l
$$
As well as a large $\lambda$ which entails a reduction in $W_l$, we notice that the equation of $z_l$ becomes linear. Which entails a linear neural network.


![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im25.png]]



\textbf{Debugging Tool:} Note that when plotting the cost function of $J$ you should keep into account the cost function and its regularizer:

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im23.png]]

#### Dropout


\textbf{Core idea:} For each node we will have a certain probability to cancel some nodes, you end up with a diminished network as depicted on the right:

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im21.png]]


\textbf{Implementation of Dropout:} There are few ways to coding dropout, let's focus on the \textcolor{cornellred}{"Inverted dropout"} technique. We will illustrate this with a \textcolor{officegreen}{layer $l=3$} and with an hyperparameter that we call \textcolor{officegreen}{$keep\_prob=0.8$} which is the probability that a given hidden unit we'll be kept:
$$
\begin{aligned}
d3 &= np.random.rand(a3.shape[0], a3.shape[1]) < keep\_prob \\
a3 &= np.multiply(a3, d3) \\
a3 &/= keep\_prob \rightarrow \text{this is where the name "Inverted" comes from}
\end{aligned}
$$
\textit{Important remark:}
\begin{itemize}
    \item $d3$ is a Boolean matrix that we multiply with $a3$ to cancel some of the nodes
    \item The last step of the pseudo-code
    $$
    a3 /= keep\_prob
    $$
    Means that with a \textcolor{officegreen}{$keep\_prob=0.8$} and if we have 50 units. We will have 10 units set to zero. And in order to not change the expected value of $a_4$ we correct this change with the keep\_prob.
    $$
    z_4 = W_4 \cdot a_3 + b_4
    $$
\end{itemize}



\textbf{Making prediction at test time:} \textcolor{oceanblue}{We won't use drop out at all} it is only for the training, at test time, rather we will 
$$
\begin{aligned}
z_1 &= W_1 a_0 + b_1 \\
a_1 &= g_1(z_1) \\
z_2 &= W_2 a_1 + b_2 \\
a_2 &= \hdots \\
\hat{y}&= 
\end{aligned}
$$
If we were to use It would add noise to the prediction at test time. \\

\textbf{But why dropout is working ?}

Intuition: At each iteration we knock off some of the nodes, as such, we will be working with a smaller neural network which yields a regularizing effect.\\

Intuition 2: Let's look at the perspective at a single unit. By considering dropout the node knows that it cannot rely on any feature, a feature could be set to zero randomly. So it would be reluctant to put too much weights to one input because it could go away. Spreading out the weights will have the effect of shrinking weights, similarly to L2 regularization.\\

\textbf{Another detail:} We had to choose the keep\_prob parameter, here $W_2$ is the biggest set of matrix, to reduce overfitting of that matrix you might have a \textcolor{officegreen}{$keep\_prob$} that is around 0.5, and for the other you can have a higher \textcolor{officegreen}{$keep\_prob$} . And also 1.0 for those you don't want to loose any connection like at the end.

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im22.png|503]]


\textbf{Downside of dropout:} the cost fct J is not well defined, at each iteration you cut off some of the nodes, it is here harder to verify that the cost function is always decreasing (like on the plot below). Because it less well defined, you loose this debugging tool. What he usually do is adding dropout only at the end once you are sure everything is working well.

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im23.png]]




#### Data Augmentation


You can augment your train set, by flipping horizontally your images, you can also takes random crops of the image, ; Those fakes images won't bring as much information as a new image but this can be an inexpensive way to your algorithm more data. And basically you tell your algorithm that a zoom or a vertical transformation is still a cat. 

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im26.png|554]]


On digits you can apply distortion on the image

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im27.png|543]]


#### Early Stopping

We start with random weights in our first epoch and we get model like this one which is underfitting. As we train, let's say for 20 epochs we get a pretty good model. But then let's say we keep going for a 100 epochs, we'll get something that fits the data much better, but we can see that this is starting to overfit. If we go for even more the model heavy overfits. 

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im2 (3).png|436]]


Let's try to evaluate these models by adding a testing set such a the gray points. We make a plot of the error in the training set and the testing set with respect to each epoch. For the first epoch, since the model is completely random, then it badly missclassifies both the training and the testing sets.

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im3 (4).png|430]]


Based on the model complexity graph we can determine the number of epochs we'll be using. Where on the right, we have a high testing error and low training error, so we're overfitting. On the left we have high testing error and training error so we're underfitting.\\

So in summary what we do is we degrade in descent until the testing error stops decreasing and start to increase. At that moment, we stop. This algorithm is called \textbf{Early Stopping}.
\begin{itemize}
    \item To check: early stopping vs double slope \url{https://arxiv.org/pdf/1812.11118.pdf}
\end{itemize}
### D. Optimisation & Stabilité

Ces techniques améliorent la **stabilité** et **l'efficacité** du processus d'entraînement.

#### Batch Normalization

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsubsection{Normalizing activations in a network}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\textbf{Core idea:} Created by two researchers \href{https://arxiv.org/abs/1502.03167}{Loffe and Szegedy in Mar 2015} The motivation is that normalizing the value of $a_2$ will make the training of $W_3, b_3$ faster. Though technically we will normalize the value of $z_2$ and not $a_2$. Even if there is debta over this in the community.\\


\textbf{Implementing Batch Norm:} Given some intermediate values in your neural nets, let's say we have $z_1, ..., z_m$ that are elements of $z_l$. We compute their means
$$
\mu = \frac{1}{n} \sum_i z_i
$$
then
$$
\sigma^2 = \frac{1}{n}\sum_i (z_i - \mu)^2
$$
Also for numerical stability
$$
z_{i}^{norm} = \frac{z_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$
Every component of z has mean 0 and std 1. But we don't want that instead we compute
$$
\Tilde{z}_i = \gamma z_i^{norm} + \beta
$$
where $\gamma$ and $\beta$ are learnable parameters of our model. basically it enables us to set the mean to be whatever we want.\\

\textbf{Notice:} that we can go back to our original equation as:
$$
\gamma = \sqrt{\sigma^2 + \varepsilon}
$$
and 
$$
\beta = \mu
$$
Then we see that with the parameters we picked we got back the previous equation:
$$
\Tilde{z}_i = z_i
$$
Thus now your neural nets you use $\Tilde{z}$ rather than $z$.\\

\textbf{Batch Norm at Test Time:} Recall that $\mu$ and $\sigma$ comes up in the mini-batch; that we might not have in the test set. Thus we must come up with, we use exponentially weighted average (across mini-batch).\\

To be more concrete Let consider a layer $l$ consider the mini-batches, $X_{\{ 1\}}, X_{\{ 2\}}, X_{\{ 3\}}$. When training each mini-batch we will get a sequence of mean
$$
\mu_{ \{1 \} [l]}, \mu_{ \{2 \} [l]}, \mu_{ \{3 \} [l]} ... \sim \mu
$$
And apply the running average formula for each layer as you train the neural network. You also repeat the operation for $\sigma$.\\

Finally you apply $z_{norm}$ ..

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsubsection{Why batch Norm works ?}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\textbf{One reason.} We have seen that normalizing the input feature X, can speed up learning. Batch Norm is doing a similar things but to values in the hidden units. But there are also other reason.\\

\textbf{2nd reason.} it makes weights of later layers like 10 more robust to change than earlier layers like the first. To explain let's look at a logistic regression network on a cat detection task. \\

Covariate shift: Imagine we train our dataset on black cats, and in our test we have all sort of cats. 

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im29.png]]


Then our training set is on the left with positive example in red; and we want to generalize it to a dataset on the right:

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im30.png]]

We might not expect a model trained with the data on the left to generalize to the one on the right. This idea of the data distribution changing comes by the name "Covariate Shift". It means if we learn some $x \rightarrow y$ , if $x$ change we have to relearn our model.\\

\textbf{Why this is a problem with neural nets ?} We look at the learning process from the third hidden layer. We are at $W^3, b^3$, from its perspectives it gets some set of layers from the earlier layers.\\ 


\textbf{Batch Norm as a regularization:} 

\begin{itemize}
    \item Each mini-batch is scaled by the mean/variance computed on just that mini-batch. 
    \item This adds some noise to the values $z^{(l)}$ within that minibatch. Cause until we have done all the batch we are always learning on a different dataset which implies a different $\mu$ and $\sigma$ at each batch
    \item Similarly Dropout also adds noise to each hidden layer's activations by multiplying them with a vector of 0 and 1, where 0 turns off the neuron.
    \item Note that if you increase the size of the mini-batch you reduce the noise and also reduce this regularization effect.
\end{itemize}




variance has some noise because it is not trained on the full dataset.\\

3. if you use a larger mini batch size like 512 rather than 64 you reduce the noise effect thus the regularization effect.\\

conclusion: one more detail next time, batch norm handles data one mini batch at a time; so at test time, we can process one single example at a time, we have to do something different so that our prediction make sense.

#### Learning Rate Decay


\textbf{Core idea:} Suppose we apply mini-batch 64, as you iterate the steps will be a bit noisy it will tends toward the minimum but it might not convergence and rather wandering around. Because you use some fixed values for $\alpha$. But if you were to reduce the $\alpha$ parameters as we get closer to the minimum we start oscillating in a tighter region instead.

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im32.png|397]]




\textbf{Implementation:} Remember that one epoch, is one pass through the data, we can set our learning rate alpha to be
$$
\alpha = \frac{1}{1+ decay\_rate \times epoch\_num} \alpha_0
$$
Note that the $decay\_rate$ is another hyperpaaramater we have to tuned.\\

\textbf{Example:} If $\alpha_0=0.2$ and $decay\_rate=1$ it decays as follow:
\begin{table}[H]
\begin{tabular}{|l|l|l|l|l|l|}
\hline
Epoch    & 1     & 2       & 3      & 4      & $\hdots$ \\ \hline
$\alpha$ & $0.1$ & $0.067$ & $0.05$ & $0.04$ & $\hdots$ \\ \hline
\end{tabular}
\end{table}


\textbf{Alternative techniques:}
\begin{itemize}
    \item Exponential Decay:
    $$
\alpha = 0.95^{epoch\_num} \alpha_0
$$
\item Or
$$
\alpha = \frac{k}{\sqrt{epoch\_num}} \alpha_0
$$
\item Or
$$
\frac{k}{\sqrt{t}} \alpha_0
$$
\item Discrete staircase

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im33.png]]

\item Manual Decay
\end{itemize}



### E. Problèmes Transversaux

#### Problème du vanishing gradient

Quand on remonte en arrière lors de la rétropropagation, si les gradients sont proches de 0, on multiplie de nombreux petits nombres, ce qui peut faire disparaître le signal de gradient dans les couches profondes. Ce phénomène affecte particulièrement les réseaux profonds et constitue l'un des défis majeurs de l'entraînement des réseaux de neurones.
