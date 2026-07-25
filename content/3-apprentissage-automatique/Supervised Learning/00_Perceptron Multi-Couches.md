---
title: Réseaux de Neurones
description: Introduction aux réseaux de neurones artificiels
weight: 3
---

# Perceptron Multi-Couches

## I — Origine et Architecture

### Fil historique

Les réseaux de neurones artificiels s'inspirent du cerveau biologique, mais leur développement est avant tout mathématique. Les grandes étapes :

- **1943** — McCulloch & Pitts : premier modèle formel du neurone artificiel (porte logique binaire)
- **1957** — Rosenblatt : le **perceptron**, premier algorithme d'apprentissage supervisé. Preuve de convergence si les données sont linéairement séparables.
- **1969** — Minsky & Papert : démontrent que le perceptron ne peut pas apprendre XOR → premier "hiver de l'IA"
- **1986** — Rumelhart, Hinton, Williams : **backpropagation** généralisée aux réseaux multicouches. Débloque les MLP.
- **2012** — AlexNet (Hinton/Krizhevsky) : deep learning explose sur ImageNet. Le reste est de l'histoire.

> [!note]- Cybernétique et Mallat
> Mallat (cours Collège de France) replace les réseaux dans un cadre plus large : la **cybernétique** (Wiener, 1948) — l'étude des systèmes de contrôle et de communication — est l'ancêtre conceptuel. L'idée que des boucles de feedback peuvent produire un comportement intelligent est au cœur des deux domaines.

### Architecture MLP

Un réseau de neurones artificiels est composé de neurones organisés en couches. Le cas le plus simple est le perceptron multi-couches (MLP) avec une couche cachée :

![[nn_architecture.png|397]]

![[input_dataset.png|317]]

### Forward Pass

#### Étape 1 — Initialisation des paramètres

$[X]_{3 \times 2}$ : matrice des données d'entrée
$[W^{(1)}]_{2 \times 3} \sim \mathcal{N}(0,1)$ : matrice de poids entre couche d'entrée et couche cachée
$[b^{(1)}]_{1 \times 3}$ : vecteur de biais pour la couche cachée

#### Étape 2 — Passage vers la couche cachée

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

> [!note]- Représentation Learning
> **Transformation de l'espace** : La fonction d'activation $g()$ effectue une projection non linéaire dans un nouvel espace en 3D.
>
> **Espaces de représentation** :
> - $X^{(1)}$ : espace de représentation initiale (2D)
> - $A^{(2)}$ : projection dans un espace latent de dimension supérieure (3D)
>
> **Objectif** : Ce "representation learning" permet de rendre les structures linéairement séparables dans le nouvel espace.

#### Étape 3 — Passage vers la couche de sortie

$$Z^{(3)} = A^{(2)} W^{(2)} + b^{(2)}$$

$$[Z^{(3)}]_{3 \times 1} = \begin{bmatrix}
\textcolor{teal}{a_{11}^{(2)}} & \textcolor{teal}{a_{12}^{(2)}} & \textcolor{teal}{a_{13}^{(2)}}\\
\textcolor{orange}{a_{21}^{(2)}} & \textcolor{orange}{a_{22}^{(2)}} & \textcolor{orange}{a_{23}^{(2)}} \\
\textcolor{violet}{a_{31}^{(2)}} & \textcolor{violet}{a_{32}^{(2)}} & \textcolor{violet}{a_{33}^{(2)}}
\end{bmatrix}_{3 \times 3}
\begin{bmatrix}
w_1^{(2)} \\ w_2^{(2)} \\ w_3^{(2)}
\end{bmatrix}_{3 \times 1}
+ b^{(2)}$$

#### Étape 4 — Prédiction finale et fonction de coût

$$\hat{y} = g_2(Z^{(3)})$$

La fonction d'activation de sortie $g_2()$ dépend du type de problème :
- **Régression** : identité $g_2(z) = z$
- **Classification binaire** : sigmoïde $g_2(z) = \frac{1}{1+e^{-z}}$
- **Classification multi-classe** : softmax $g_2(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

**Fonction de coût** (MSE pour la régression) :

$$J(\theta) = \frac{1}{2n}\sum_{i=1}^{n}(y^{(i)} - g_2(g_1(XW^{(1)})W^{(2)} ))^2$$

$$\theta^* = \arg\min_{\theta} J(\theta)$$

---

## II — Backpropagation

L'algorithme de rétropropagation utilise la règle de dérivation en chaîne (chain rule) pour calculer les gradients de la fonction de coût par rapport aux paramètres du réseau.

### Calcul des gradients

$$\frac{\partial J}{\partial W^{(2)}} = (A^{(2)})^T \delta^{(3)}, \quad \frac{\partial J}{\partial b^{(2)}} = \sum \delta^{(3)}$$

$$\frac{\partial J}{\partial W^{(1)}} = X^T \delta^{(2)}, \quad \frac{\partial J}{\partial b^{(1)}} = \sum \delta^{(2)}$$

où $\delta^{(3)}$ représente les erreurs pour la couche de sortie et $\delta^{(2)} = \delta^{(3)}(W^{(2)})^T \odot g'(Z^{(2)})$ les erreurs propagées vers la couche cachée. La mise à jour s'effectue par : $W := W - \alpha \nabla W$ et $b := b - \alpha \nabla b$.

### Parallèle avec la régression linéaire

La backpropagation suit exactement le même schéma conceptuel que le gradient descent en régression linéaire, mais avec des compositions de fonctions plus complexes.

**Cadre de la régression linéaire :** $\theta = (\theta_0, \theta_1)$, $n$ individus, $h_\theta(x_i) = \theta_0 + \theta_1 x_i$. Le gradient individuel :

$$g_i = \nabla f_i(\theta) = \begin{pmatrix} -(y_i - \hat{y}_i) \\ -x_i(y_i - \hat{y}_i) \end{pmatrix}$$

En comparaison dans le réseau de neurone : j'initialise mes matrices $W^{(i)}$ je fais une forward passe j'obtiens mes $\hat{y}$ puis je fais une rétropagation (chain rule en arrière) pour mettre à jours mes poids $W^{(i)}$, je refais une forward pass etc.

### Boucle d'entraînement

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

---

## III — Entraînement pratique

### A. Initialisation des poids

Le choix des valeurs initiales des matrices $W^{(l)}$ avant l'entraînement n'est pas anodin — un mauvais choix peut **empêcher complètement l'apprentissage**, même avec une architecture par ailleurs correcte.

#### Pourquoi pas zéro ?

Si on pose $W^{(l)} = 0$ pour toutes les couches, **tous les neurones d'une même couche reçoivent le même signal en entrée**. Lors du forward pass, ils produisent tous la même activation. Et lors du backward pass, ils reçoivent tous le même gradient — donc ils sont mis à jour de manière identique.

Le réseau se comporte alors comme s'il n'avait **qu'un seul neurone par couche** : la "symétrie" entre neurones n'est jamais brisée. C'est le **problème de la symétrie d'initialisation** — et il est rédhibitoire.

> [!warning] Conclusion
> Initialiser à zéro est **interdit** pour les matrices de poids. Pour les biais $b^{(l)}$, c'est en revanche acceptable (la symétrie est cassée par les poids).

On utilise donc nécessairement une initialisation **aléatoire**, typiquement gaussienne $W_{ij} \sim \mathcal{N}(0, \sigma^2)$. Reste à choisir $\sigma^2$.

#### Xavier / Glorot (2010)

[Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) cherchent à préserver la variance **dans les deux sens** : forward (activations) et backward (gradients). En faisant les calculs sous hypothèse d'activation linéaire et de symétrie, on tombe sur la condition

$$\boxed{\;\sigma^2 = \text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}\;}$$

Xavier est conçu pour des activations **symétriques autour de zéro** (tanh, sigmoïde rescalée). Avec ReLU, il sous-estime la variance nécessaire.

#### He (2015) — la version ReLU

[He et al. (2015)](https://arxiv.org/abs/1502.01852) refont la dérivation pour ReLU. ReLU "tue" environ **la moitié** des activations — pour compenser cette perte, il faut **doubler la variance des poids** :

$$\boxed{\;\sigma^2 = \frac{2}{n_{\text{in}}}\;}$$

#### Regard RMT sur l'initialisation

Xavier et He sont des règles empiriques bien justifiées, mais la **Random Matrix Theory** (RMT) donne un cadre théorique plus profond. Une matrice de poids $W^{(l)} \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$ avec des entrées i.i.d. $\mathcal{N}(0, \sigma^2)$ est une **matrice rectangulaire aléatoire**. Quand $n_{\text{in}}, n_{\text{out}} \to \infty$ avec ratio $q = n_{\text{out}}/n_{\text{in}}$ fixe, ses valeurs propres de $\frac{1}{n_{in}}WW^T$ convergent vers la **loi de Marchenko-Pastur** :

$$\rho_{\text{MP}}(\lambda) = \frac{1}{2\pi q} \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\ \lambda}, \qquad \lambda \in [\lambda_-, \lambda_+]$$

avec $\lambda_{\pm} = (1 \pm \sqrt{q})^2$.

![[marchenko_pastur_init.png]]
**Figure.** *Spectre empirique d'une matrice de poids $W \sim \mathcal{N}(0,1)$ pour différents ratios $q$, superposé à la densité Marchenko-Pastur théorique. Xavier init ($\sigma^2 = 1/n_{in}$) maintient les valeurs propres d'ordre 1 — ni explosion ni effondrement.*

Pour que la valeur singulière typique soit d'ordre 1, il faut $\sigma^2 \sim 1/n_{\text{in}}$ — exactement la prescription Xavier. RMT justifie donc Xavier depuis la **géométrie spectrale** de la matrice entière, pas juste un argument de variance scalaire.

> [!note]- Lien avec le vanishing gradient
> Le vanishing gradient a aussi une lecture RMT. Si les valeurs singulières de $W^{(l)}$ sont toutes $< 1$, le produit de $L$ telles matrices fait tendre le gradient vers 0 exponentiellement vite. La condition Xavier/He revient à maintenir les valeurs singulières **d'ordre 1** — ce que RMT garantit.

#### Récapitulatif

| Activation | Initialisation | Variance |
|:---:|:---:|:---:|
| tanh, sigmoïde | **Xavier / Glorot** | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ |
| ReLU, Leaky ReLU | **He** | $\frac{2}{n_{\text{in}}}$ |

### B. Fonctions d'activation

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im5.png]]

- **Sigmoid** : $g(z) = \frac{1}{1 + e^{-z}}$ (sortie entre 0 et 1)
- **Tanh** : $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ (sortie entre -1 et 1)
- **ReLU** : $g(z) = \max(0, z)$
- **Leaky ReLU** : $g(z) = \max(0.01z, z)$

**Vanishing gradient avec Sigmoid/Tanh.** Le gradient local de la sigmoid $z \times (1-z)$ atteint un maximum de 0.25 — à chaque passage, l'amplitude du gradient diminue d'au moins 75%. Sur des réseaux profonds, le signal disparaît dans les couches inférieures.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im6.png|494]]

**Dead ReLUs.** ReLU résout le vanishing gradient mais introduit les "dead ReLUs" : si un neurone ne s'active jamais ($z < 0$ toujours), son gradient est nul et il ne se met plus jamais à jour — mort permanent.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im7.png|598]]

### C. Techniques de Régularisation

#### L2 Regularization (Weight Decay)

$$J(W_1, b_1, \ldots, W_L, b_L) = \frac{1}{m} \sum_{i=1}^m L(\hat{y}_i, y_i) + \frac{\lambda}{2m} \sum_{l=1}^L \|W_l\|_F^2$$

La mise à jour devient $W_l := W_l(1 - \frac{\alpha\lambda}{m}) - \alpha \cdot \nabla_{\text{backprop}}$ — les poids décroissent ("decay") vers zéro à chaque pas.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im24.png]]

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im25.png]]

> [!warning] Astuce de debugging
> Plotter $J$ **incluant le terme de régularisation** — sinon on peut voir le terme de loss diminuer alors que le terme régularisé augmente.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im23.png]]

#### Dropout

À chaque passage forward pendant l'entraînement, on **désactive aléatoirement** une fraction des neurones.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im21.png]]

**Inverted Dropout** avec `keep_prob = 0.8` :

```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 /= keep_prob  # rescaling pour préserver l'espérance
```

On **n'utilise PAS Dropout au test time**.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im22.png|503]]

> [!warning] Le défaut principal de Dropout
> Avec Dropout actif, la loss $J$ n'est plus bien définie — elle change à chaque itération. Entraîner d'abord sans Dropout pour vérifier la convergence, puis activer.

#### Data Augmentation

Augmenter artificiellement le training set en transformant les exemples existants (retournements, crops, rotations, déformations élastiques…).

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im26.png|554]]

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im27.png|543]]

#### Early Stopping

On suit la validation loss en parallèle et on arrête quand elle commence à remonter — même si la training loss continue de descendre.

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im2 (3).png|436]]

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im3 (4).png|430]]

> [!note]- Le piège du double descent
> Sur des modèles très surparamétrés ($> 10^8$ paramètres), la validation loss peut **redescendre** après le pic d'overfitting (Belkin et al. 2018). Early stopping reste utile mais sur des modèles très larges, attendre plus longtemps peut payer.

### D. Optimisation & Stabilité

#### Batch Normalization

Introduite par [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167). Pour un mini-batch de pré-activations $\{z_1, \ldots, z_n\}$ :

$$z_i^{\text{norm}} = \frac{z_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad \tilde{z}_i = \gamma\, z_i^{\text{norm}} + \beta$$

où $\gamma$ et $\beta$ sont appris — le réseau peut "défaire" la normalisation si besoin.

Trois effets : accélération de l'apprentissage, robustesse au covariate shift interne, effet régularisateur (bruit des stats par batch).

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im29.png]]

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im30.png]]

> [!warning] Au test time
> Pas de mini-batch → on utilise une **moyenne mobile** des $\mu$ et $\sigma^2$ accumulés pendant l'entraînement.

#### Learning Rate Decay

$$\alpha = \frac{1}{1 + \text{decay\_rate} \times \text{epoch\_num}} \cdot \alpha_0$$

![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im32.png|397]]

![[images/3-Apprentissage automatique/00_Neural_nets_MLP/im33.png]]

Modernement : avec Adam, le LR decay est moins critique. Cosine annealing et warmup linéaire restent standards pour les transformers.

---

## IV — Ce que génère un réseau : approximation universelle

> Un réseau de neurones génère une **fonction** $f_w : \mathbb{R}^d \to \mathbb{R}$. La question centrale est : quelle fonction peut-il générer ? La réponse : n'importe laquelle.

### Théorème d'approximation universelle

> [!warning] Théorème (Cybenko 1989, Hornik 1991)
> Soit $g : \mathbb{R} \to \mathbb{R}$ une fonction d'activation non polynomiale. Alors pour toute fonction continue $f : [0,1]^d \to \mathbb{R}$ et tout $\varepsilon > 0$, il existe $N \in \mathbb{N}$ et des paramètres $w_i, b_i, c_i$ tels que :
> $$\sup_{x \in [0,1]^d} \left| f(x) - \sum_{i=1}^N c_i\, g(w_i^T x + b_i) \right| < \varepsilon$$

Chaque neurone $g(w^T x + b)$ découpe l'espace input avec un hyperplan et applique une non-linéarité de part et d'autre. En combinant linéairement assez de ces "tranchées", on peut approximer n'importe quelle forme.

![[universal_approximation.png]]
**Figure.** *Ligne du haut : approximation d'une fonction 1D avec $N = 3, 10, 50$ neurones — erreur max de 0.81 à 0.001. Ligne du bas : chaque neurone découpe $\mathbb{R}^2$ avec un hyperplan ; la combinaison de 30 neurones crée une frontière arbitrairement complexe.*

**Ce que le théorème ne dit PAS** : il garantit l'existence des paramètres, pas qu'on sait les trouver. Il ne dit rien sur la taille $N$ nécessaire, ni sur la généralisation.

**Pourquoi la profondeur plutôt que la largeur.** Un réseau profond peut représenter certaines fonctions avec exponentiellement moins de neurones qu'un réseau plat — chaque couche compose les représentations de la couche précédente.

### Visualisation : un réseau = une fonction

![[surface_3D_NN.png|332]]

![[Pasted image 20260617213937.png]]

Un réseau entraîné définit une surface dans l'espace input → output. C'est cette surface qu'on optimise pendant l'entraînement — on "sculpte" la fonction pour qu'elle colle aux données.

### Ce qu'apprend un réseau : déformation de l'espace

Le théorème dit qu'on peut approcher n'importe quelle fonction. Mais concrètement, comment le réseau y arrive-t-il ? En **déformant l'espace** couche par couche pour rendre le problème linéairement soluble.

Exemple : deux spirales entrelacées dans $\mathbb{R}^2$ — non linéairement séparables. On entraîne un réseau $2 \to 16 \to 16 \to \mathbf{2} \to 1$ (tanh, Adam). L'avant-dernière couche $h_3 \in \mathbb{R}^2$ est directement visualisable.

![[nn_learning.png]]
**Figure.** *Ligne du haut : frontière de décision dans l'espace input au fil de l'entraînement. Ligne du bas : représentation cachée $h_3 \in \mathbb{R}^2$. À epoch 6000, les deux classes sont linéairement séparables dans $h_3$ — le réseau a "déplié" la variété.*

> [!warning] Le message fondamental
> Le réseau n'apprend pas à mémoriser les points. Il apprend une **transformation de l'espace** $\mathbb{R}^2 \to \mathbb{R}^2$ telle que le problème devient linéairement soluble dans l'espace de représentation finale. La dernière couche n'est qu'un classifieur linéaire — toute la "magie" est dans les couches cachées.

> [!note]- Espace latent et Manifold Hypothesis
> Les couches cachées définissent un **espace latent** — la représentation intermédiaire apprise. Si les données vivent sur une sous-variété de dimension $k \ll d$ dans l'espace input (Manifold Hypothesis, cf. [[01_Fondation#F. Manifold Hypothesis]]), le réseau apprend à "déplier" cette variété dans l'espace latent. Les coordonnées latentes correspondent aux paramètres intrinsèques des données (dans l'exemple de Bishop : orientation, position, distance caméra — pas les pixels).

---

## V — Géométrie de la loss

> Le gradient descent descend une surface dans l'espace des paramètres $\theta \in \mathbb{R}^P$ — mais $P$ est typiquement de l'ordre de $10^6$ à $10^{12}$, complètement invisible. La **loss landscape** est une façon de projeter cette surface en 2D ou 3D.

**La méthode (Li et al. 2018).** On fixe $\theta^*$, on tire deux directions aléatoires filter-normalisées $\delta_1, \delta_2$, et on trace $L(\alpha, \beta) = J(\theta^* + \alpha\delta_1 + \beta\delta_2)$.

![[loss_landscape.png]]
**Figure.** *Loss landscape d'un réseau 2→16→16→2→1. Gauche : surface 3D. Milieu : contours. Droite : coupe 1D. Le minimum est dans une vallée large et plate — caractéristique des flat minima.*

- **Flat vs sharp minima** : un minimum plat généralise mieux. Les petits mini-batches convergent vers des flat minima, les gros vers des sharp minima (Keskar et al. 2017).
- **Pas de vrais minima locaux en haute dim** : les points critiques sont presque tous des points-selle (Dauphin et al. 2014). C'est pourquoi le gradient descent fonctionne malgré la non-convexité.

> [!note]- Lien avec Adam
> Adam adapte son pas par paramètre en divisant par la racine carrée du moment du second ordre — ce qui revient à préconditionner par une estimation diagonale de la Hessienne. Dans les directions de forte courbure, Adam fait de petits pas ; dans les directions plates, de grands pas. Réponse optimale à la géométrie anisotrope de la loss landscape.

> [!note]- TODO — gradient descent comme système dynamique
> Le gradient descent peut se voir comme la discrétisation d'une ODE continue : $\frac{d\theta}{dt} = -\nabla_\theta J(\theta)$. La trajectoire d'entraînement est un écoulement sur la loss landscape ci-dessus — les flat minima sont des attracteurs stables, les sharp minima des attracteurs instables. À creuser : pourquoi SGD avec petit batch "saute" hors des sharp minima (bruit stochastique = perturbation de la trajectoire), et comment Adam déforme la courbure effective de l'écoulement. Lien possible avec les Neural ODEs (Chen et al. 2018).

---

## VI — Pourquoi ça marche : Mallat

> Source : [Cours Collège de France — S. Mallat](https://www.di.ens.fr/~mallat/CoursCollege.html)

Mallat identifie trois angles pour comprendre pourquoi les réseaux profonds fonctionnent malgré la malédiction de la dimension.

**1. Non-convexité de $J$.** Développé en V — les points critiques sont des points-selle, pas des minima locaux. Le gradient descent fonctionne donc malgré la non-convexité.

**2. Systèmes dynamiques.** Les poids évoluent pendant l'entraînement comme une ODE. Formalisé par les Neural ODEs (Chen et al. 2018) : $\frac{dh}{dt} = f_\theta(h(t), t)$. Programme de recherche ouvert, hors scope ici.

**3. Géométrie et symétries.** La borne $n \geq \epsilon^{-d}$ (cf. [[01_Fondation#E. Curse of dimensionality]]) suppose une régularité Lipschitz seule. Les vraies fonctions $f$ ont une structure plus riche — des **symétries** — qui réduisent massivement la dimension effective. Une symétrie de $f$ c'est une transformation $g$ telle que $f(g \cdot x) = f(x)$ — une régularité globale, pas locale. Si $f$ possède un groupe de symétrie $G$, l'espace d'étude se réduit de $\Omega$ à $\Omega/G$. La hiérarchie multi-échelles réduit $d$ à $O(\log d)$.

> [!warning] Limite importante
> Ces arguments ne s'appliquent que si l'architecture **encode explicitement les symétries**. Un MLP sur des images ne "voit" pas la translation — les pixels changent entièrement, le réseau repart de zéro. Les symétries existent dans les données, mais le MLP ne peut pas en bénéficier structurellement. C'est pour ça que les CNN ont été inventés. Le développement complet est dans [[02_CNN#IV — Pourquoi les CNN fonctionnent : symétries et hiérarchie (Mallat)]]. Les structures algébriques sous-jacentes (groupes, invariance, équivariance) sont dans [[02_Groupes et Symétries]].
