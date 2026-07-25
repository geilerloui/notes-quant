---
title: Self-supervised Learning
description: Pretext tasks, self-labeling (DeepCluster) et contrastive learning (InfoNCE, SimCLR, PIRL)
weight: 6
---

# Self-supervised Learning

> [!warning] Brouillon
> Nettoyé depuis les notes Overleaf/LaTeX d'origine (thèse), pas encore condensé. Une image (`paper-3-2.png`, section SimCLR) référence encore l'ancien chemin Overleaf non migré — à vérifier/re-exporter si elle ne s'affiche pas.

## I - Introduction

**Related works.** Les frameworks de self-supervised learning sont des méthodes de representation learning utilisant des données non labellisées. On distingue deux catégories : via des **pretext task(s)**, et via le **contrastive learning**.

**Qu'est-ce que le self-supervised learning ?** C'est une méthode qui pose la question suivante pour reformuler un problème non supervisé en problème supervisé : peut-on concevoir la tâche de façon à générer virtuellement un nombre illimité de labels à partir des images existantes, et utiliser ça pour apprendre des représentations ?

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/0.Introduction/im1.png|400]]

On remplace le bloc d'annotation humaine en exploitant de façon créative une propriété des données pour construire une tâche pseudo-supervisée. Par exemple, au lieu de labelliser des images chat/chien, on peut les faire pivoter de 0/90/180/270 degrés et entraîner un modèle à prédire la rotation. On génère ainsi un volume d'entraînement virtuellement illimité à partir des millions d'images disponibles sur internet.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/0.Introduction/im2.png|400]]

Une fois les représentations apprises sur ces millions d'images, on peut faire du transfer learning pour fine-tuner sur une tâche supervisée (ex classification chat vs chien) avec très peu d'exemples.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/0.Introduction/im3.png|400]]

## II - Noise Contrastive Estimation (NCE)

**Introduction.** Local NCE et Global NCE sont deux méthodes peu coûteuses en calcul pour apprendre une vraisemblance conditionnelle $p_{\theta}(x \mid c)$. NCE est utile quand le nombre de $x$ possibles est très grand, comme en language modeling où $|X|$ est la taille du vocabulaire.

### 1. Local NCE (aka Binary NCE)

**(i) Objectif.** Comment estimer $p_{\theta}(x \mid c)$ où $c$ est un contexte et $x$ une classe parmi un grand nombre ? On réécrit l'objectif :

$$
\boxed{p_{\theta}(x \mid c)=\frac{f_{\theta}(x, c)}{\sum_{x^{\prime}} f_{\theta}\left(x^{\prime}, c\right)}=\frac{f_{\theta}(x, c)}{Z_{\theta}(c)}}
$$

$f_{\theta}(x, c)$ assigne un score à $x$ dans le contexte $c$, $Z_{\theta}(c)$ est la constante de normalisation (partition function). $Z$ est difficile à calculer avec beaucoup de classes, car il faut sommer sur tous les $x'$ possibles.

**(ii) Solution.** Local NCE simplifie le problème d'apprentissage de $p_{\theta}(x \mid c)$ en un problème de classification binaire $p(d \mid x,c)$ où :
- $d=0 \Rightarrow x \sim q(x)$, c'est-à-dire que les points sont du bruit
- $d=1 \Rightarrow x \sim p(x \mid c)$, c'est un vrai point de donnée

On échantillonne $k$ échantillons de $q(x)$ avec label $d=0$, et un vrai point de $p(x \mid c)$ :

$$
\begin{aligned}
&p(D=0 \mid x, c)= \frac{p(D=0, x, c)}{p(x, c)} = \frac{k \cdot q(x)}{p(x \mid c)+k \cdot q(x)} \\
&p(D=1 \mid x, c)= \frac{p(D=1, x, c)}{p(x, c)} =\frac{p(x \mid c)}{p(x \mid c)+k \cdot q(x)}
\end{aligned}
$$

**Hypothèse de self-normalisation.** On suppose $Z_{\theta}(c) \approx 1$ ; ça marche bien en pratique car les réseaux de neurones sont assez expressifs pour apprendre des fonctions de score auto-normalisées :

$$
\boxed{\Rightarrow p(x \mid c)=f_{\theta}(x, c)}
$$

D'où :

$$
\begin{aligned}
&p(D=0 \mid x, c)=\frac{k \cdot q(x)}{f_{\theta}(x, c)+k \cdot q(x)} \\
&p(D=1 \mid x, c)=\frac{f_{\theta}(x, c)}{f_{\theta}(x, c)+k \cdot q(x)}
\end{aligned}
$$

**NCE comme cross-entropy.** On vient de dériver un problème de classification binaire classique : apprendre $\theta$ qui maximise la log-vraisemblance conditionnelle :

$$
\mathcal{L}_{\text{LocalNCE}}=\sum_{(x, c) \in D}\left[\log p(D=1 \mid x, c)+k \mathbb{E}_{x^{\prime} \sim q} \log p\left(D=0 \mid x^{\prime}, c\right)\right]
$$

Simplification par Monte-Carlo :

$$
\boxed{\mathcal{L}_{\mathrm{LocalNCE}, \mathrm{MC}}=\sum_{(x, c) \in D}\left(\log p(D=1 \mid x, c)+\sum_{i=1, x^{\prime} \sim q}^{k} \log p\left(D=0 \mid x^{\prime}, c\right)\right)}
$$

**(iii) Code (illustration numérique).**
- **Initialisation.** $V=64$ la taille du vocabulaire, $n=32$ les contextes possibles, $k=16$ le nombre de négatifs.
- **Sample.** On échantillonne un contexte $C \sim Cat()$ : $[c]_{64} \sim p(c)$. On initialise aussi $p(x \mid c) \sim Cat()$, on échantillonne $[x]_{32} \sim p(x \mid c)$ qu'on subset en $[x[c]]_{64}$. Exemple : $x = [48, 44, 28, .., 31]_{32}$, $c = [23, 31, 58, ..]_{64}$ — $x[c]$ pour $48$ correspond à la 48e valeur de $c$.
- **Noise distribution.** $[x']_{64 \times 16} \sim q(x)$.
- **Loss function :**


$L_\text{LocalNCE}= \sum_{(x, c) \in D} \left[ [f(x, c)]_{64 \times 32} - \log{(\exp{[f(x, c)]_{64 \times 32}} + k * [q(x)]_{64 \times 16})} + \sum_{i = 1, x' \sim q(x)}^{k} \log (k) + \log q(x') - \log{(\exp{f(x', c)} + k * q(x'))} \right]$



- **Optimization.** On définit une fonction $[f]_{64 \times 32}$ optimisée par descente de gradient :

$$
f_{\theta}(x, c)^+ = f_{\theta}(x,c) + \eta \nabla L_{localNCE}
$$

### 2. Partition function estimation

**(i) Objectif.** Même problème que ci-dessus : $p_{\theta}(x \mid c)=\frac{f_{\theta}(x, c)}{Z_{\theta}(c)}$, mais cette fois $Z_{\theta}(c)$ est approximé directement (plutôt que supposé $\approx 1$).

**(ii) Solution — importance sampling.** On approxime $Z$ en échantillonnant $x \sim q(x)$, $k$ fois :

$$
\begin{aligned}
Z_{\theta}(c)=\sum_{j} f_{\theta}\left(x_{j}, c\right) &=\sum_{j} f_{\theta}\left(x_{j}, c\right) \frac{q\left(x_{j}\right)}{q\left(x_{j}\right)} \\
&=\mathbb{E}_{x_{j} \in X}\left[\frac{f_{\theta}\left(x_{j}, c\right)}{q(x)}\right] \\
& \approx \sum_{j=1, x_{j} \sim q(x)}^{k} \frac{f_{\theta}\left(x_{j}, c\right)}{q\left(x_{j}\right)}
\end{aligned}
$$

**Conditional.** On injecte cette approximation de $Z_{\theta}$ pour approximer la vraisemblance conditionnelle :

$$
\boxed{p(x \mid c) \approx \frac{f_{\theta}(x, c)}{\sum_{j=1, x_{j} \sim q(x)} f_{\theta}\left(x_{j}, c\right) / q\left(x_{j}\right)} \quad (2)}
$$

Même dérivation ensuite que Local NCE (classification binaire, cross-entropy, simplification Monte-Carlo).

### 3. InfoNCE

À partir de l'équation (2), une façon d'apprendre $f$ serait le MLE direct de $p(x \mid c)$. Mais comme pour Local NCE, on définit un classifieur de substitution.

Avec **Global NCE**, le classifieur identifie le vrai point de donnée parmi plusieurs exemples $x_1, ..., x_k$ où un seul $x_i \sim p(x \mid c)$ est réel et les autres $x_{j \neq i} \sim q(x)$ sont du bruit.

**La loss.** La grande fraction ci-dessous est la cross-entropy loss écrite en détail : probabilité que $x_i$ soit l'exemple positif, et tous les autres $x_l$ ($l \neq i$) soient négatifs.

$$
\begin{aligned}
p\left(d=i \mid x_{i}, c ; k\right)&=\frac{p\left(x_{i} \mid c\right) \prod_{l \neq i} p\left(x_{l}\right)}{\sum_{n=1}^{k} p\left(x_{n} \mid c\right) \prod_{l \neq k} p\left(x_{l}\right)}  \\
&=\frac{\frac{p\left(x_{i} \mid c\right)}{p\left(x_{i}\right)}}{\sum_{j=1}^{k} \frac{p\left(x_{j} \mid c\right)}{p\left(x_{j}\right)}} \quad (4)
\end{aligned}
$$

La deuxième ligne simplifie les produits qui s'annulent entre numérateur et dénominateur.

**Nouvelle fonction de score.** $g_{\theta}(x_i, c)$ donne un score de correspondance entre $x_i$ et le contexte $c$ :

$$
g_{\theta}(x, c) \propto \frac{p(x \mid c)}{q(x)}
$$

**Ratio de densité :**

$$
\boxed{p(d=i \mid X, c) \approx \frac{g_{\theta}\left(x_{i}, c\right)}{\sum_{j=1}^{k} g_{\theta}\left(x_{j}, c\right)}}
$$

**Différence Global NCE vs importance sampling local.** Pour Global NCE, la fonction de score $g$ approxime le **ratio de densité** $\frac{p(x \mid c)}{q(x)}$, alors qu'avant on utilisait une fonction $f$ pour approximer $p(x \mid c)$ directement.

Loss pour cet objectif :

$$
\mathcal{L}_{\text{GlobalNCE}}=\sum_{\left(x_{1}, \ldots, x_{k}, c\right) \in D}\left[\log \frac{g_{\theta}\left(x_{i}, c\right)}{\sum_{j} g_{\theta}\left(x_{j}, c\right)}\right]
$$

**Objectif d'InfoNCE.** Le but n'est pas d'estimer $p(x \mid c)$ du tout. InfoNCE montre qu'en apprenant le ratio de densité entre $p(x \mid c)$ et $q(x)$, la loss Global NCE maximise $I(x_i ; c)$, l'**information mutuelle** entre $x_i$ et son contexte $c$, et minimise $I(x_{j \neq i} ; c)$.

Information mutuelle entre deux v.a. $x$ et $c$ :

$$
\begin{aligned}
I(x ; c) &= \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c}) \log \frac{p(\mathbf{x}, \mathbf{c})}{p(\mathbf{x}) p(\mathbf{c})}\\
&=\sum_{(x, c)} p(x, c) \log \frac{p(x \mid c)}{p(x)} \\
&=\mathbb{KL}(p(x, c) \| p(x) p(c))
\end{aligned}
$$

Biblio : [1](https://lilianweng.github.io/posts/2021-05-31-contrastive/) [2](https://jxmo.io/posts/nce)

## III - Image Data

### 1. Pretext tasks — Generation-Based Methods

**1. Image Colorization.** Et si on préparait des paires (niveaux de gris, colorisée) en appliquant un filtre grayscale à des millions d'images disponibles librement ?

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-1-1.png|400]]

Architecture encoder-decoder (fully convolutional), loss L2 entre l'image prédite et l'image couleur réelle.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-1-2.png|400]]

Pour résoudre cette tâche, le modèle doit apprendre les différents objets présents dans l'image et leurs parties pour les colorier de façon cohérente. Les représentations apprises sont donc utiles pour des tâches downstream.

**2. Image Super-résolution.** Et si on préparait des paires (petite, upscalée) en sous-échantillonnant des millions d'images ?

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-2-1.png|400]]

Les modèles GAN type SRGAN sont populaires pour cette tâche. Un générateur prend une image basse résolution et sort une image haute résolution (réseau fully convolutional). Comparaison avec la vraie image via MSE + content loss. Un discriminateur classifie si l'image est une vraie haute résolution (1) ou une fausse générée (0). Cette interaction pousse le générateur à produire des détails fins.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-2-2.png|400]]

Générateur et discriminateur apprennent tous deux des features sémantiques utiles pour des tâches downstream.

**3. Image Inpainting.** Et si on préparait des paires (corrompue, réparée) en supprimant aléatoirement des parties d'image ?

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-3-1.png|400]]

Architecture GAN similaire : le générateur apprend à reconstruire l'image, le discriminateur sépare vrai/généré.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-3-2.png|400]]

Pathak et al. montrent que les features sémantiques apprises par un tel générateur donnent +10.2% par rapport à une initialisation aléatoire sur PASCAL VOC 2012 (segmentation sémantique), et <4% d'amélioration sur classification/détection.

### 2. Pretext tasks — Context-Based Methods

**1. Image Jigsaw Puzzle.** Et si on préparait des paires (mélangée, ordonnée) en mélangeant aléatoirement des patches d'image ?

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-4-1.png|400]]

Même avec seulement 9 patches, il existe 362 880 puzzles possibles. Pour limiter ça, on n'utilise qu'un sous-ensemble de permutations (ex 64 permutations avec la plus grande distance de Hamming entre elles).

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-4-2.png|400]]

Exemple avec la permutation n°64 parmi les 64 disponibles :

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-4-3.png|400]]

Pour retrouver les patches d'origine, Noroozi et al. proposent un réseau appelé **context-free network (CFN)** : chaque patch passe par les mêmes couches convolutionnelles siamoises (poids partagés), les features sont ensuite combinées dans une couche fully-connected. En sortie, le modèle prédit quelle permutation a été utilisée parmi les 64 classes possibles — connaître la permutation permet de résoudre le puzzle.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/1.pretext/paper-4-4.png|400]]

Pour résoudre le jigsaw, le modèle doit apprendre comment les parties s'assemblent, les positions relatives des différentes parties d'un objet, et sa forme. Représentations utiles pour classification et détection downstream.

**2. Geometric Transformation Recognition.** Et si on préparait des paires (image tournée, angle de rotation) en tournant aléatoirement des images par (0, 90, 180, 270) depuis une grande collection non labellisée ?

![[paper-6-1.png|400]]

Gidaris et al. : l'image tournée passe dans un ConvNet qui doit classifier en 4 classes (0/90/180/270°).

![[paper-6-2.png|400]]

Idée très simple mais le modèle doit comprendre la position, le type et la pose des objets dans l'image pour résoudre la tâche — représentations utiles downstream.

**3. Image Clustering.** Et si on préparait des paires (image, numéro de cluster) en clusterisant une grande collection non labellisée ?

![[paper-5-1.png|400]]

Caron et al. proposent l'architecture **deep clustering** : les images sont d'abord clusterisées, les clusters servent de classes, et la tâche du ConvNet est de prédire le label de cluster pour une image en entrée.

![[paper-5-2.png|400]]

### 3. Self-Labeling Methods

**Motivation.** Combiner clustering et representation learning pour apprendre simultanément features et labels.

**1. DeepCluster.**

**Motivation.** Beaucoup de méthodes self-supervised utilisent des pretext tasks pour générer des labels de substitution (rotation, colorization, jigsaw...). Mais ces pretext tasks dépendent du domaine et demandent de l'expertise à concevoir.

[DeepCluster](https://arxiv.org/abs/1807.05520) (Caron et al., Facebook AI Research) propose une approche différente, sans connaissance spécifique au domaine, utile quand les données annotées sont rares.

**Pipeline DeepCluster.** Combine clustering non supervisé et réseaux profonds : méthode end-to-end pour apprendre conjointement les paramètres du réseau et les assignations de clusters de ses représentations. Features générées et clusterisées itérativement pour obtenir à la fois un modèle entraîné et des labels.

**Algorithme.**

(i) On prend des images non labellisées d'ImageNet (1.3 million d'images, 1000 classes), en mini-batchs de 256, avec augmentation classique.

(ii) Choix du nombre de clusters : ImageNet a 1000 classes par défaut, mais le papier utilise 10 000 clusters pour un regroupement plus fin (ex sous-espèces d'animaux plutôt que juste chat/chien).

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/3.self-label/paper-1-2.png|400]]

(iii) Pour générer les labels initiaux : AlexNet initialisé aléatoirement, dernière couche FC3 retirée. Forward pass, on prend le vecteur de features de la couche FC2 (dimension 4096).

![[paper-1-3.png|400]]

Répété pour toutes les images du dataset → matrice image-feature de taille $[N, 4096]$.

![[paper-1-4.png|400]]

(iv) **Clustering.** Réduction de dimension avant clustering : 4096 → 256.

![[paper-1-5.png|400]]

K-means appliqué aux features réduites → images et clusters correspondants, qui servent de pseudo-labels pour l'entraînement.

![[paper-1-6.png|400]]

(v) **Representation Learning.** Une fois images et clusters obtenus, on entraîne le ConvNet comme en supervisé classique (batch size 256, cross-entropy loss vs le label de cluster). Le modèle apprend des représentations utiles.

![[paper-1-7.png|300]]

(vi) **Alternance training/clustering.** 500 epochs. Le clustering est relancé au début de chaque epoch pour régénérer les pseudo-labels sur tout le dataset, puis entraînement classique du ConvNet sur tous les batchs. SGD momentum 0.9, learning rate 0.05, weight decay $10^{-5}$, sur GPU Pascal P100.

**2. Self-Labeling with simultaneous clustering.**

**Motivation.** Utiliser un réseau initialisé aléatoirement pour bootstrap le premier jeu de labels — DeepCluster montre empiriquement que ça marche. Les auteurs de DeepCluster ont évalué un AlexNet initialisé aléatoirement sur ImageNet : baseline aléatoire = 1/1000 = 0.1%, mais un AlexNet random atteint 12% — un réseau initialisé aléatoirement possède déjà un signal faible dans ses poids, exploitable pour amorcer le processus.

**Algorithme.**

(i) Dataset de $N$ images non labellisées, batchs de 256 (ImageNet), augmentation classique. Nombre de clusters testé de 1k à 10k, la performance ImageNet s'améliore jusqu'à 3k.

(ii) **Architecture.** ConvNet (AlexNet ou ResNet-50) comme extracteur de features $\phi(I)$, mappant une image $I$ vers un vecteur $m \in \mathbb{R}^D$. Une tête de classification (couche linéaire simple) convertit les features en scores de classe, puis softmax :

$$
p\left(y=. \mid x_{i}\right)=\operatorname{softmax}\left(h \circ \phi\left(x_{i}\right)\right)
$$

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/3.self-label/paper-2-1.png|400]]

(iii) **Assignation initiale aléatoire.** Modèle initialisé aléatoirement, forward pass pour obtenir des prédictions de classe par image — utilisées comme labels initiaux.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/3.self-label/paper-2-2.png|400]]

(iv) **Self-labeling par Optimal Transport.** À partir des labels initiaux, on cherche une meilleure distribution des images en clusters — approche différente de K-means (DeepCluster), empruntée au **transport optimal** (recherche opérationnelle). On génère une matrice optimale $Q$ qui alloue les $N$ images non labellisées à $K$ clusters, sous contrainte : (a) répartition égale entre les $K$ clusters (**equipartition condition**), (b) coût d'allocation basé sur la performance du modèle entraîné avec ces clusters comme labels (vs les labels du modèle aléatoire). $Q$ optimale trouvée via une variante rapide de l'**algorithme de Sinkhorn-Knopp**.

![[paper-2-3.png|400]]

(v) **Fonction de coût.** Avec les labels $Q$ mis à jour, on compare les prédictions du modèle aux labels de cluster via cross-entropy. Le modèle est entraîné un nombre fixe d'epochs ; la loss décroît, les représentations internes s'améliorent :

$$
E\left(p \mid y_{1}, \ldots, y_{N}\right)=-\frac{1}{N} \sum_{i=1}^{N} \log p\left(y_{i} \mid x_{i}\right)
$$

![[paper-2-4.png|400]]

### 4. Contrastive Learning

**Méthodes génératives vs contrastives.** Les méthodes self-supervised contemporaines se répartissent grossièrement en deux familles :

![[intro-1.png|400]]

Les méthodes contrastives apprennent des représentations en contrastant exemples positifs et négatifs. Pas un paradigme nouveau, mais grand succès empirique en computer vision avec le pré-entraînement contrastif non supervisé :
- Entraînées sur ImageNet non labellisé et évaluées avec un classifieur linéaire, elles dépassent aujourd'hui la précision d'un AlexNet supervisé, avec une efficacité data notable par rapport au supervisé pur (Data-Efficient CPC, Hénaff et al. 2019).
- Le pré-entraînement contrastif sur ImageNet transfère bien sur d'autres tâches downstream et surpasse le pré-entraînement supervisé équivalent (MoCo, He et al. 2019).

Elles diffèrent des méthodes génératives plus traditionnelles, qui se concentrent sur l'erreur de reconstruction au niveau pixel :
- Les loss au niveau pixel poussent le modèle à se focaliser sur des détails bas niveau plutôt que sur des facteurs latents abstraits.
- Les objectifs pixel-based supposent souvent l'indépendance entre pixels, réduisant la capacité à modéliser des corrélations ou structures complexes.

**Qu'est-ce que le contrastive learning ?** Apprendre à une machine à distinguer des choses similaires et dissimilaires. *(Contraster = être en opposition avec quelque chose)*

![[sim-1.png|400]]

Ex : un enfant sait reconnaître un objet et le **contraster** à d'autres objets.

![[sim-6.png|400]]
![[sim-2.png|400]]

**(1) InfoNCE.** Formellement, pour un point $x$, les méthodes contrastives cherchent à apprendre un encodeur $f$ tel que :

$$
\operatorname{score}\left(f(x), f\left(x^{+}\right)\right)>>\operatorname{score}\left(f(x), f\left(x^{-}\right)\right)
$$

- $x^{+}$ : point similaire/congruent à $x$ (échantillon **positif**)
- $x^{-}$ : point dissimilaire à $x$ (échantillon **négatif**)
- la fonction de score mesure la similarité entre deux features

$x$ est appelé l'**anchor**. Pour optimiser cette propriété, on construit un classifieur softmax qui classifie correctement positifs et négatifs, poussant la fonction de score à assigner de grandes valeurs aux positifs et petites aux négatifs :

$$
\mathcal{L}_{N}=-\mathbb{E}_{X}\left[\log \frac{\exp \left(f(x)^{T} f\left(x^{+}\right)\right)}{\exp \left(f(x)^{T} f\left(x^{+}\right)\right)+\sum_{j=1}^{N-1} \exp \left(f(x)^{T} f\left(x_{j}\right)\right)}\right]
$$

Dénominateur = un positif + $N-1$ négatifs. Score = produit scalaire :

$$
\operatorname{score}\left(f(x), f\left(x^{+}\right)\right)=f(x)^{T} f\left(x^{+}\right)
$$

C'est la classique cross-entropy loss d'un classifieur softmax à $N$ classes, appelée **InfoNCE loss** dans la littérature contrastive.

**(2) Deep InfoMax.**

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-2-1.png|300]]

### 5. Contrastive Training Objectives

**1. Contrastive Loss.** ([Chopra et al. 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)) une des premières loss pour le metric learning contrastif.

Étant donné des échantillons $\{\mathbf{x}_i\}$ avec labels $y_i \in \{1,...,L\}$ parmi $L$ classes, on veut apprendre $f_\theta(.): \mathcal{X} \rightarrow \mathbb{R}^d$ qui encode $x_i$ tel que les exemples de même classe aient des embeddings proches, et ceux de classes différentes des embeddings éloignés. La contrastive loss prend une paire $(x_i, x_j)$, minimise la distance d'embedding si même classe, la maximise sinon :

$$
\mathcal{L}_{\text{cont}}\left(\mathbf{x}_{i}, \mathbf{x}_{j}, \theta\right)=\mathbb{1}\left[y_{i}=y_{j}\right]\left\|f_{\theta}\left(\mathbf{x}_{i}\right)-f_{\theta}\left(\mathbf{x}_{j}\right)\right\|_{2}^{2}+\mathbb{1}\left[y_{i} \neq y_{j}\right] \max \left(0, \epsilon-\left\|f_{\theta}\left(\mathbf{x}_{i}\right)-f_{\theta}\left(\mathbf{x}_{j}\right)\right\|_{2}\right)^{2}
$$

$\epsilon$ = hyperparamètre, borne inférieure de distance entre classes différentes.

Coût : $O(N^2)$ calculs de distance par paire pour un dataset de taille $N$ (ou $O(B^2)$ pour un batch de taille $B$), en s'assurant d'avoir assez de labels positifs/négatifs dans l'échantillon.

**2. Triplet Loss.** Proposée dans [FaceNet (Schroff et al. 2015)](https://arxiv.org/abs/1503.03832) pour la reconnaissance faciale multi-pose/angle.

Étant donné un anchor $\mathbf{x}$, un positif $\mathbf{x}^+$ (même classe) et un négatif $\mathbf{x}^-$ (classe différente), la triplet loss minimise la distance anchor-positif et maximise la distance anchor-négatif simultanément :

$$
\mathcal{L}_{\text{triplet}}\left(\mathbf{x}, \mathbf{x}^{+}, \mathbf{x}^{-}\right)=\sum_{\mathbf{x} \in \mathcal{X}} \max \left(0,\left\|f(\mathbf{x})-f\left(\mathbf{x}^{+}\right)\right\|_{2}^{2}-\left\|f(\mathbf{x})-f\left(\mathbf{x}^{-}\right)\right\|_{2}^{2}+\epsilon\right)
$$

La marge $\epsilon$ = offset minimum entre distances similaire/dissimilaire. Crucial de choisir des $\mathbf{x}^-$ difficiles pour vraiment améliorer le modèle.

La triplet loss règle certains problèmes de la contrastive loss, mais coûte cher en calcul : pour un mini-batch de $B$ triplets, il faudrait $O(N^3)$ combinaisons de triplets — infaisable en pratique.

![[intro-2.png|300]]

**3. Lifted Structured Loss.** ([Song et al. 2015](https://arxiv.org/abs/1511.06452)) utilise toutes les arêtes pairwise d'un batch pour plus d'efficacité de calcul. Comparaison visuelle contrastive loss / triplet loss / lifted structured loss : arêtes rouges/bleues connectent respectivement paires similaires/dissimilaires.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-1-2.png|400]]

Avec $D_{ij}=\|f(\mathbf{x}_i)-f(\mathbf{x}_j)\|_2$ :

$$
\begin{aligned}
\mathcal{L}_{\text{struct}} &=\frac{1}{2|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} \max \left(0, \mathcal{L}_{\text{struct}}^{(ij)}\right)^{2} \\
\text{où } \mathcal{L}_{\text{struct}}^{(ij)} &=D_{i j}+\max \left(\max _{(i, k) \in \mathcal{N}} \epsilon-D_{i k}, \max _{(j, l) \in \mathcal{N}} \epsilon-D_{j l}\right)
\end{aligned}
$$

$\mathcal{P}$ = paires positives, $\mathcal{N}$ = paires négatives. La matrice de distances pairwise se calcule facilement par batch.

Pour chaque paire positive, on ancre $z_i$ et on cherche les négatifs de distance minimale (en rouge), même chose pour l'ancre de droite.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-1-1.png|400]]

La partie rouge de $\mathcal{L}_{\text{struct}}^{(ij)}$ sert au **hard negative mining**, mais n'est pas lisse et peut mener à un mauvais optimum local. Version relaxée :

$$
\mathcal{L}_{\text{struct}}^{(ij)}=D_{i j}+\log \left(\sum_{(i, k) \in \mathcal{N}} \exp \left(\epsilon-D_{i k}\right)+\sum_{(j, l) \in \mathcal{N}} \exp \left(\epsilon-D_{j l}\right)\right)
$$

Le papier propose aussi d'enrichir la qualité des négatifs par batch en intégrant activement des négatifs difficiles à partir de quelques paires positives aléatoires.

### 6. Parallel Augmentation

**1. SimCLR.** [Chen et al., juillet 2020](https://arxiv.org/abs/2002.05709).

**Algorithme.**

(i) Batch de $N=8192$ images non labellisées. Pour chaque image (ex un chat), on construit une version augmentée (ex rotation). La paire passe dans l'encodeur ResNet-50, sortie = vecteur $h$ de dimension 2048.

(ii) Les représentations $h_i, h_j$ passent dans une **projection head** vers $z_i, z_j$.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-3-1.png|400]]

(iii) Similarité cosinus entre les embeddings $z$ :

$$
s_{i, j}=\frac{z_{i}^{T} z_{j}}{\tau\left\|z_{i}\right\|\left\|z_{j}\right\|}
$$

$\tau$ = paramètre de température ajustable, qui met à l'échelle et élargit la plage $[-1,1]$ de la similarité cosinus.

![[paper-3-2.png|400]]

SimCLR utilise ensuite une loss contrastive appelée **NT-Xent loss** (Normalized Temperature-scaled Cross-Entropy Loss).

D'abord, les paires augmentées du batch sont prises une par une.

![[paper-3-3.png|400]]

Puis softmax pour obtenir la probabilité que ces deux images soient similaires.

![[paper-3-4.png|400]]

Ce calcul softmax équivaut à la probabilité que la deuxième image augmentée du chat soit la plus similaire à la première image de la paire. Toutes les autres images du batch sont échantillonnées comme négatifs. Pas besoin d'architecture spécialisée, de memory bank ou de queue (contrairement à InstDisc, MoCo ou PIRL).

![[paper-3-5.png|400]]

La loss d'une paire est le négatif du log du calcul précédent — c'est la formulation **NCE Loss** :

$$
l(i, j)=-\log \frac{\exp \left(s_{i, j}\right)}{\sum_{k=1}^{2 N} \mathbb{1}_{[k \neq i]} \exp \left(s_{i, k}\right)}
$$

![[paper-3-6.png|400]]

On calcule aussi la loss pour la même paire en inversant les positions.

![[paper-3-7.png|400]]

Loss totale sur tout le batch de taille $N=2$, moyennée :

$$
L=\frac{1}{2 N} \sum_{k=1}^{N}[l(2 k-1,2 k)+l(2 k, 2 k-1)]
$$

![[paper-3-8.png|400]]

Les représentations de l'encodeur et de la projection head s'améliorent avec la loss, les images similaires se rapprochent dans l'espace des embeddings.

**NT-Xent loss — détail.**
- **(i) Input.** $N$ = batch size, ici $N=2$ ; avec deux images augmentées chacune → 4 images, donc 4 représentations $\{z_1, z_2, .., z_4\}$.

![[sim-5.png|300]]

- **(ii) Matrice de similarité.**

![[sim-3.png|400]]

- **(iii) Loss finale**, moyennée sur le batch $N=2$ :

$$
L=\frac{1}{2 N} \sum_{k=1}^{N}[l(2 k-1,2 k)+l(2 k, 2 k-1)]
$$

![[sim-4.png|400]]

**2. PIRL.** [Misra et al., déc. 2019](https://arxiv.org/abs/1912.01991) (se prononce "pearl").

La pretext task apprend des représentations pour une image transformée afin de prédire une propriété de la transformation (ex prédire l'angle de rotation après une rotation de 90°). Problème : les représentations apprises peuvent overfitter à cet objectif et mal généraliser — elles deviennent **covariantes** avec la transformation, n'encodant que l'info nécessaire à prédire l'angle, au détriment de l'info sémantique utile.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-4-1.png|400]]

PIRL propose de rendre les représentations de l'image originale et transformée **similaires**, avec deux objectifs :
- rendre la transformation d'une image similaire à l'originale
- rendre les représentations (originale + transformée) différentes des autres images aléatoires du dataset

**Algorithme.**

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-4-2.png|400]]

(i) **Memory Bank.** Pour de meilleures représentations, il vaut mieux comparer l'image courante à un grand nombre d'images négatives. Utiliser de plus gros batchs (tout le reste du batch = négatifs) pose des problèmes de ressources.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-4-3.png|400]]

PIRL utilise donc une **memory bank** qui cache les représentations de toutes les images, utilisable pendant l'entraînement — beaucoup de négatifs sans augmenter la taille du batch. Le modèle PIRL est initialisé aléatoirement, un forward pass sur toutes les images d'entraînement stocke $f(V_I)$ dans la memory bank.

![[images/3-Apprentissage automatique/04_Computer vision/05_Self supervised learning/2.contrastive/paper-4-4.png|400]]

(ii) On prépare des batchs, on applique une transformation pretext (ex rotation). Pour deux images (chat, chien) on obtient 4 résultats. Les images passent dans le réseau jusqu'à obtenir leurs embeddings $f(V_{ij})$.

(iii) **Amélioration du modèle (loss function).** Pour chaque image on a des représentations originale et transformée. Objectif : les rendre similaires entre elles, différentes des autres images.

![[paper-4-5.png|400]]

Calcul de la loss :

**(a) Similarité cosinus** entre deux représentations, notée $s()$ — ex chat vs chat tourné.

![[paper-4-10.png|300]]

**(b) Noise Contrastive Estimator.** Score de similarité normalisé par toutes les images négatives :

![[paper-4-6.png|400]]

Calculé sur les représentations des projection heads plutôt que sur celles de ResNet-50 :

$$
h(f(V_I), g(V_{I^T})) = \frac{ \exp\left(\frac{s(f(V_I),\ g(V_{I^t}))}{\tau}\right) }{ \exp\left(\frac{s(f(V_{I}),\ g(V_{I^t}))}{\tau}\right) + \sum_{ I' \in D_{N} } \exp\left(\frac{s(g(V_{I^t}),\ f(V_{I'}))}{\tau}\right) }
$$

![[paper-4-7.png|400]]

Loss d'une paire, cross-entropy :

$$
L_{NCE}\left(I, I^{t}\right)=-\log \left[h\left(f\left(V_{I}\right), g\left(V_{I^{t}}\right)\right)\right]-\sum_{I^{\prime} \in D_{N}} \log \left[1-h\left(g\left(V_{I^{t}}\right), f\left(V_{I^{\prime}}\right)\right)\right]
$$

En pratique, on utilise les représentations de la memory bank $m_I$ plutôt que recalculées :

$$
L_{NCE}\left(I, I^{t}\right)=-\log \left[h\left(m_{I}, g\left(V_{I^{t}}\right)\right)\right]-\sum_{I^{\prime} \in D_{N}} \log \left[1-h\left(g\left(V_{I^{t}}\right), m_{I^{\prime}}\right)\right]
$$

Cas idéal : similarité image/transformation = 1, similarité avec négatifs = 0 → loss nulle.

![[paper-4-8.png|400]]

$$
L_{NCE}\left(I, I^{t}\right)=-\log [1]-(\log [1-0]+\log [1-0])=0
$$

Cette loss compare $I$ à $I^t$ et $I^t$ aux négatifs $I'$, mais jamais $I$ directement aux négatifs. D'où l'ajout d'un second terme de loss, combiné :

$$
L\left(I, I^{t}\right)=\lambda L_{NCE}\left(m_{I}, g\left(V_{I^{t}}\right)\right)+(1-\lambda) L_{NCE}\left(m_{I}, f\left(V_{I}\right)\right)
$$

![[paper-4-9.png|400]]

Avec cette formulation, on compare image ↔ transformation, transformation ↔ négatif, et image originale ↔ négatif.

Ces losses améliorent progressivement l'encodeur et les projection heads. Les représentations dans la memory bank pour le batch courant sont mises à jour par exponential moving average (EMA).
