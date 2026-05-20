---
title: Modèles autorégressifs
---
# Modèles autorégressifs

> Cette note dévide la première famille concrète : les **modèles autorégressifs**. C'est l'application la plus directe du squelette posé dans `[[00_Fondations]]` (chain rule + paramétrisation par NN) ; c'est aussi la famille qui sous-tend **tous les LLM modernes** (GPT, Claude, Llama, Mistral, Gemini). On suit le cours CS236 d'Ermon en remettant les pièces dans le bon ordre conceptuel : d'abord l'idée centrale, ensuite les trois architectures historiques (FVSBN, NADE, MADE) qui construisent ensemble la machinerie utilisée encore aujourd'hui, puis les déclinaisons par modalité (images, texte, audio).

## I. L'idée centrale

### A. Chain rule avec ordre total

Un modèle autorégressif fait un seul choix de représentation : il fixe un **ordre total** sur les variables $x_1, x_2, \ldots, x_n$ et utilise la chain rule **sans aucune hypothèse d'indépendance conditionnelle** :

$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_1, x_2, \ldots, x_{i-1}) = \prod_{i=1}^{n} p(x_i \mid x_{<i}).$$

![[auto1.png]]

Le DAG associé est totalement connecté dans le sens de l'ordre : chaque nœud a pour parents tous les nœuds précédents. C'est ce qu'on appelle la **propriété autorégressive**. Le mot vient des séries temporelles : prédire l'instant $i$ à partir de tout l'historique.

> [!note] Le choix de l'ordre dépend de la modalité
> - **Texte** : ordre naturel gauche → droite (en langues occidentales). Pas de choix à faire, l'ordre est imposé par la nature séquentielle du langage.
> - **Images** : *raster scan* — pixel par pixel de haut en bas, de gauche à droite. Choix arbitraire mais standard.
> - **Audio** : ordre temporel.
> 
> Pour des données non séquentielles par nature (un graphe, un nuage de points), choisir un ordre est un problème délicat qu'on revisitera plus loin.

### B. Densité exacte et MLE direct

Le grand avantage de l'autorégressif tient en une ligne. La log-vraisemblance d'un échantillon $x$ s'écrit :

$$\log p_\theta(x) = \sum_{i=1}^{n} \log p_\theta(x_i \mid x_{<i}).$$

Tous les termes sont **calculables exactement** — pas de borne, pas d'approximation. On peut donc faire de la **MLE directe** via SGD comme expliqué dans `[[00_Fondations#V bis. Apprentissage par maximum de vraisemblance]]`. C'est ce qui distingue les autorégressifs des VAE (borne ELBO), des GAN (pas de densité) et des diffusion (objectif détourné via score matching).

Mieux encore : avec une architecture bien pensée (MADE, PixelCNN, Transformer causal), **les $n$ termes se calculent en un seul forward pass parallèle**. C'est *le* fait qui rend l'entraînement à l'échelle des LLM possible. On détaille pourquoi en §I.D.

### C. Le prix à payer : la génération séquentielle

Le revers : pour **échantillonner**, il faut tirer les variables une par une, dans l'ordre :

1. tirer $\bar x_1 \sim p_\theta(x_1)$
2. tirer $\bar x_2 \sim p_\theta(x_2 \mid \bar x_1)$
3. ...
4. tirer $\bar x_n \sim p_\theta(x_n \mid \bar x_{<n})$

Chaque tirage exige un forward pass complet, qui dépend des tirages précédents. Pour un LLM qui génère 1000 tokens, c'est 1000 forward passes en série — *non* parallélisables. C'est la raison principale pour laquelle l'inférence des LLM est coûteuse, et pourquoi beaucoup d'optimisations (KV-cache, speculative decoding, batching) ciblent ce point.

### D. Les deux modes : teacher forcing à l'entraînement, séquentiel en génération

Cette sous-section est **transversale à toute la famille autorégressive** — elle vaut pour FVSBN, NADE, MADE, PixelCNN, WaveNet, GPT, sans exception. C'est le point conceptuel le plus important à internaliser avant d'attaquer les architectures concrètes.

Un modèle autorégressif s'utilise dans **deux modes très différents**, et la confusion entre les deux est l'une des sources principales d'erreur quand on lit du code.

#### Mode A — Évaluer la vraisemblance d'un $x$ donné (entraînement)

On dispose d'un échantillon **complet** du dataset, $x = (x_1, \ldots, x_n)$. On veut calculer $\log p_\theta(x) = \sum_i \log p_\theta(x_i \mid x_{<i})$ pour mettre à jour $\theta$.

Pour chaque position $i$, le modèle calcule :

$$\hat x_i \;=\; f_\theta(x_{<i}) \;\in\; [0, 1]$$

qui est la **probabilité prédite** que $x_i = 1$ sachant le passé (cas binaire ; en multiclasse c'est un vecteur softmax). **C'est un nombre réel, pas un tirage.** Il représente la croyance du modèle.

On compare ensuite $\hat x_i$ à la **vraie valeur** $x_i$ via la cross-entropy :

$$-\log p_\theta(x_i \mid x_{<i}) = \begin{cases} -\log \hat x_i & \text{si } x_i = 1, \\ -\log(1 - \hat x_i) & \text{si } x_i = 0. \end{cases}$$

C'est la **binary cross-entropy** standard. On la somme sur tous les $i$ et toutes les images du batch, et on rétropropage.

> [!warning] Teacher forcing
> Point crucial : pour calculer $\hat x_i$ pendant l'entraînement, on injecte en entrée **les vrais $x_{<i}$ du dataset** — pas des prédictions précédentes du modèle. On dit qu'on fait du *teacher forcing*. 
> 
> Conséquence majeure : les $\hat x_1, \hat x_2, \ldots, \hat x_n$ ne dépendent pas les uns des autres — chacun dépend uniquement des vrais $x_{<i}$, qu'on connaît depuis le début. **Toutes les $\hat x_i$ peuvent donc être calculées en parallèle** dans un seul forward pass (à condition que l'architecture le permette, ce qui est le cas pour MADE, PixelCNN et Transformer causal).
> 
> C'est précisément ce qui rend l'entraînement des LLM faisable à grande échelle : on traite un batch de séquences en un seul passage forward + backward.

#### Mode B — Générer un nouveau $\bar x$ (sampling)

Là on n'a rien à part les poids appris $\theta$. On veut produire un échantillon **de zéro**.

1. Le modèle calcule $\hat x_1 = f_\theta(\emptyset)$, la probabilité a priori de $x_1$.
2. On **tire au sort** $\bar x_1 \sim \text{Bern}(\hat x_1)$ : avec probabilité $\hat x_1$ on obtient $1$, avec probabilité $1 - \hat x_1$ on obtient $0$.
3. On injecte $\bar x_1$ comme entrée → on calcule $\hat x_2 = f_\theta(\bar x_1)$ → on tire $\bar x_2 \sim \text{Bern}(\hat x_2)$.
4. ... et ainsi de suite jusqu'à $\bar x_n$.

> [!warning] Bernoulli = tirage aléatoire, **pas** un seuillage à 0.5
> $\text{Bern}(\hat x_i)$ n'est *pas* "si $\hat x_i > 0.5$ alors 1, sinon 0". C'est un **tirage stochastique** : avec probabilité $\hat x_i$ on obtient $1$, sinon $0$. En NumPy : `np.random.choice([0, 1], p=[1 - x_hat, x_hat])`.
> 
> C'est cette stochasticité qui rend le modèle **génératif** — chaque appel produit un échantillon différent. Un seuillage déterministe ne produirait qu'une seule image possible par modèle, ce qui n'a aucun intérêt.

#### Tableau récapitulatif

> [!warning] L'asymétrie entraînement / génération
> | Aspect | Entraînement (Mode A) | Génération (Mode B) |
> |---|---|---|
> | Entrées du modèle | Vrais $x_{<i}$ du dataset | Pixels déjà tirés $\bar x_{<i}$ |
> | Sortie $\hat x_i$ | Probabilité prédite | Probabilité prédite |
> | Que fait-on de $\hat x_i$ ? | Cross-entropy avec le vrai $x_i$ | Tirage $\bar x_i \sim \text{Bern}(\hat x_i)$ |
> | Parallélisable sur les $i$ ? | **Oui** (grâce au teacher forcing) | **Non** (séquentiel par nature) |
> | Apprentissage des poids ? | Oui (SGD + backprop) | Non, poids figés |
> 
> **Cette asymétrie est *fondamentale* à toute la famille autorégressive** et explique l'essentiel de leur économie computationnelle : on peut paralléliser l'entraînement (donc passer à l'échelle), mais l'inférence reste séquentielle (donc coûteuse).

## II. Le problème : paramétrer $p(x_i \mid x_{<i})$

Comme on l'a vu dans `[[00_Fondations#F. Le problème résiduel : les CPDs restent trop grosses]]`, écrire la chain rule ne suffit pas. Si l'on stocke chaque CPD $p(x_i \mid x_{<i})$ dans une table, on a besoin de $2^{i-1}$ entrées pour le seul facteur $i$ (cas binaire). Pour le dernier facteur, $2^{n-1}$ entrées. Total :

$$\sum_{i=1}^{n} (2^{i-1} - 1) \;=\; 2^n - 1 - n.$$

On retombe exactement sur le mur combinatoire de la jointe brute. La factorisation seule ne gagne rien — il faut **paramétrer chaque CPD par une fonction** $f_\theta$ de taille fixe.

L'enjeu de toute la suite : choisir une famille de fonctions $f_\theta(x_{<i})$ qui soit (i) suffisamment expressive, (ii) suffisamment compacte, (iii) qui permette un calcul **parallèle** de toutes les CPDs en entraînement (cf. §I.D).

## III. Premières solutions paramétrées

Les trois architectures FVSBN, NADE, MADE introduisent — par incréments — les trois idées techniques qui font tourner les modèles autorégressifs modernes :

1. **FVSBN** : la première paramétrisation, une LR par CPD ;
2. **NADE** : le **partage de paramètres** entre CPDs ;
3. **MADE** : le **masking des poids** qui permet de calculer toutes les CPDs en parallèle dans un seul réseau.

Ces trois idées se retrouvent toutes dans le Transformer causal des LLM modernes.

### A. FVSBN — Fully Visible Sigmoid Belief Network

L'idée la plus simple possible : chaque CPD est une **régression logistique** sur les variables précédentes.

![[fvsbn1 1.png|183]]
*Figure. Architecture de FVSBN : chaque sortie $\hat x_i$ est une régression logistique sur les pixels précédents $x_1, \ldots, x_{i-1}$, avec ses propres poids $w_j^{(i)}$ — aucun partage entre les CPDs.*

**Définition**

Soit une image $X$ et un ```ordre "raster scan"``` pour chaque pixel $i$, on définit la probabilité suivante : 

$$\hat x_i \;=\; f_i(x_{<i}) \;=\; \sigma\!\left(w_0^{(i)} + \sum_{j=1}^{i-1} w_j^{(i)} \, x_j\right) \;\in\; [0, 1],$$

avec $\sigma$ la sigmoïde. Et la CPD correspondante :

$$p_{\theta_i}(x_i \mid x_{<i}) \;=\; \text{Bern}(\hat x_i).$$

 $\text{Bern}(\hat x_i)$ : avec probabilité $\hat x_i$ on obtient $1$, avec probabilité $1 - \hat x_i$ on obtient $0$.

Chaque CPD a ses **propres paramètres** $\theta_i = \{w_0^{(i)}, w_1^{(i)}, \ldots, w_{i-1}^{(i)}\}$ — **pas de partage**. Le nœud $i$ possède $i$ paramètres (le biais + les $i-1$ poids).

> [!note]- Lecture des indices
> - $i$ indexe **la position** dans la séquence (le pixel à prédire).
> - $j$ indexe **les entrées** sur lesquelles se base la prédiction (les pixels précédents).
> - $w_j^{(i)}$ est donc le poids que le prédicteur du pixel $i$ donne au pixel $j$ ($j < i$).
> - Il y a $n$ prédicteurs au total, chacun avec ses propres poids — d'où l'absence de partage.

**Compte de paramètres**

Total des paramètres :

$$\sum_{i=1}^{n} i \;=\; \frac{n(n+1)}{2} \;=\; O(n^2).$$

Énorme gain par rapport au tabulaire ($O(2^n)$) — pour $n = 1024$, c'est $\sim 5 \cdot 10^5$ paramètres au lieu de $10^{308}$. Tractable.

**Entraînement**

On suit le protocole du Mode A décrit en §I.D :

1. Pour chaque image $x$ du batch et chaque position $i$, calculer $\hat x_i$ à partir des **vrais $x_{<i}$** (teacher forcing).
2. Calculer la binary cross-entropy entre $\hat x_i$ et le **vrai** $x_i$ :

$$\mathcal{L}(\theta) \;=\; -\sum_{i=1}^{n} \Big[ x_i \log \hat x_i + (1 - x_i) \log(1 - \hat x_i) \Big].$$

3. Rétropropager, mettre à jour les $w_j^{(i)}$ par SGD.

Les $n$ prédictions $\hat x_i$ sont **indépendantes les unes des autres** (chacune utilise les vrais $x_{<i}$, pas des prédictions précédentes), donc parallélisables sur GPU.

**Génération**

On suit le Mode B :

1. Calculer $\hat x_1 = \sigma(w_0^{(1)})$, tirer $\bar x_1 \sim \text{Bern}(\hat x_1)$.
2. Calculer $\hat x_2 = \sigma(w_0^{(2)} + w_1^{(2)} \bar x_1)$, tirer $\bar x_2 \sim \text{Bern}(\hat x_2)$.
3. ... séquentiellement jusqu'à $\bar x_n$.

L'image finale est $\bar x = (\bar x_1, \ldots, \bar x_n)$. **Tirage aléatoire à chaque étape**, pas un seuillage.

![[fvsbn2 1.png|336]]
*Figure. Visualisation. À gauche : données d'entraînement (Caltech 101 Silhouettes). À droite : échantillons générés par le modèle. D'après* Learning Deep Sigmoid Belief Networks with Data Augmentation*, 2015.*

**Limites**

Deux faiblesses majeures :

- **Trop linéaire.** Une régression logistique impose une frontière de décision linéaire entre $x_i$ et son histoire. Les structures non linéaires des données réelles (corrélations spatiales complexes, motifs visuels) sont mal capturées.
- **Pas de partage de paramètres.** Chaque CPD est indépendamment paramétrée. On ne peut pas réutiliser ce qu'on a appris en modélisant $x_2 \mid x_1$ pour modéliser $x_3 \mid x_1, x_2$. C'est du gaspillage statistique : on apprend $n$ modèles séparés alors qu'ils décrivent un même type d'objet (un pixel sachant son histoire).

Le second point est le plus important. Il motive directement NADE.

### B. NADE — Neural Autoregressive Density Estimator

> [!quote] L'apport de NADE (Larochelle & Murray, 2011)
> Remplacer la LR par un MLP **dont les poids sont partagés entre toutes les CPDs**. Le réseau apprend une seule fois "comment regarder le passé" et cette représentation sert pour toutes les variables.

#### Architecture

NADE introduit une **couche cachée intermédiaire** entre l'entrée et la sortie. Pour chaque position $i$, on calcule un vecteur caché $h_i \in \mathbb{R}^d$ qui résume le contexte $x_{<i}$, puis on prédit $\hat x_i$ à partir de $h_i$ :

$$h_i = \sigma\!\left(W_{\cdot,<i} \, x_{<i} + c\right), \qquad \hat x_i = \sigma\!\left(\alpha^{(i)} \cdot h_i + b_i\right), \qquad p_{\theta_i}(x_i \mid x_{<i}) = \text{Bern}(\hat x_i).$$

Ici :

- **$W \in \mathbb{R}^{d \times n}$** est une matrice de poids **partagée entre tous les $i$** — chaque colonne $W_{\cdot, j}$ correspond à une "lecture" de l'entrée $x_j$. C'est *la* différence avec FVSBN.
- **$c \in \mathbb{R}^d$** est le biais de la couche cachée, lui aussi partagé.
- **$\alpha^{(i)} \in \mathbb{R}^d$** et **$b_i \in \mathbb{R}$** sont les paramètres de sortie spécifiques à chaque position $i$ — on en a besoin parce que chaque CPD doit pouvoir prédire quelque chose de différent.
- **$W_{\cdot, <i}$** désigne les $i - 1$ premières colonnes de $W$ — c'est par cette **sélection de colonnes** que la propriété autorégressive est respectée : $h_i$ ne peut voir que $x_1, \ldots, x_{i-1}$.

![[nade1.png|242]]
Figure image du NADE


L'usage de $\hat x_i$ (entraînement vs génération) est exactement celui décrit en §I.D — teacher forcing pour évaluer la loss, tirage séquentiel pour échantillonner.

#### Forward propagation pas à pas

C'est là que la mécanique devient claire. Visualisons comment $h_1, h_2, h_3, \ldots$ se construisent successivement à partir de la **même** matrice $W$.

![[nade2.png|540]]

On note $W$ la matrice partagée :

$$W = \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{d1} & w_{d2} & \cdots & w_{dn} \end{bmatrix} \in \mathbb{R}^{d \times n}.$$

À chaque position $i$, on n'utilise que les $i - 1$ premières colonnes de $W$ — on appelle cette sous-matrice $A_i = W_{\cdot, <i}$ :

$$A_1 = \emptyset, \quad A_2 = \big[W_{\cdot, 1}\big], \quad A_3 = \big[W_{\cdot, 1}, W_{\cdot, 2}\big], \quad \ldots, \quad A_n = \big[W_{\cdot, 1}, \ldots, W_{\cdot, n-1}\big].$$

Le calcul devient :

$$h_i = \sigma\!\left(A_i \, x_{<i} + c\right), \qquad i = 1, \ldots, n.$$

> [!example]- Déroulé sur $n = 3$ pixels
> Supposons $W \in \mathbb{R}^{d \times 3}$ et un échantillon $x = (x_1, x_2, x_3)$.
> 
> **Étape 1 — prédire $x_1$.** $h_1$ n'a accès à rien, $A_1$ est vide :
> $$h_1 = \sigma(c), \qquad \hat x_1 = \sigma(\alpha^{(1)} \cdot h_1 + b_1).$$
> 
> **Étape 2 — prédire $x_2$.** $h_2$ utilise la première colonne de $W$ :
> $$h_2 = \sigma\!\left(W_{\cdot, 1} \, x_1 + c\right), \qquad \hat x_2 = \sigma(\alpha^{(2)} \cdot h_2 + b_2).$$
> 
> **Étape 3 — prédire $x_3$.** $h_3$ utilise les deux premières colonnes :
> $$h_3 = \sigma\!\left(W_{\cdot, 1} \, x_1 + W_{\cdot, 2} \, x_2 + c\right), \qquad \hat x_3 = \sigma(\alpha^{(3)} \cdot h_3 + b_3).$$
> 
> **Observation clé.** Les vecteurs $h_1, h_2, h_3$ partagent **les mêmes colonnes** de $W$. Quand on apprend $W_{\cdot, 1}$ pour calculer $h_2$, la même colonne sert aussi à calculer $h_3, h_4, \ldots, h_n$. C'est ça, le partage de paramètres : un signal d'apprentissage venant de n'importe quelle position $i \geq 2$ met à jour $W_{\cdot, 1}$, et toutes les autres positions en bénéficient.

> [!note]- Formulation équivalente : sélection ≡ masquage de l'entrée
> Dans certains exposés (notamment la slide du cours Stanford), la sélection de colonnes est présentée comme un **masquage du vecteur d'entrée** : on multiplie $x$ par un masque binaire qui annule les composantes $\geq i$ avant la multiplication matricielle. Concrètement pour $h_2$ (avec $n = 3$) :
> 
> $$h_2 = \sigma\!\left( W \, \big( x \odot m^{(2)} \big) + c \right), \qquad m^{(2)} = (1, 0, 0)^T.$$
> 
> Le résultat est strictement le même que $h_2 = \sigma(W_{\cdot, 1} \, x_1 + c)$ : les colonnes correspondant aux composantes annulées ne contribuent pas à la somme. Les deux formulations désignent la même opération.
> 
> **Attention au vocabulaire.** Ce "masquage" du vecteur d'entrée n'est **pas** le même que celui de MADE : ici on masque $x$, dans MADE on masquera $W$. C'est le déplacement du masque (de l'entrée vers les poids) qui rendra MADE parallélisable. À garder en tête pour ne pas confondre.

#### Backward propagation : où vont les gradients ?

La loss est la même que pour FVSBN — binary cross-entropy par position, sommée sur tous les $i$ :

$$\mathcal{L}(\theta) \;=\; -\sum_{i=1}^{n} \Big[ x_i \log \hat x_i + (1 - x_i) \log(1 - \hat x_i) \Big].$$

Ce qui est spécifique à NADE, c'est **où les gradients vont** quand on rétropropage cette loss. Décomposons la loss en termes locaux $\mathcal{L}_i = -\big[x_i \log \hat x_i + (1 - x_i) \log(1 - \hat x_i)\big]$ et regardons quels paramètres chacun met à jour.

> [!example] Exemple concret : la position $i = 3$
> Le terme $\mathcal{L}_3$ dépend de $\hat x_3$, qui est calculé à partir de $h_3$, lui-même calculé à partir de $W_{\cdot, 1}, W_{\cdot, 2}, c$ et de l'entrée $(x_1, x_2)$, puis combiné avec $\alpha^{(3)}, b_3$.
> 
> Donc le gradient de $\mathcal{L}_3$ met à jour :
> - $\alpha^{(3)}, b_3$ — paramètres de sortie spécifiques à $i = 3$, **uniquement modifiés par cette position** ;
> - $W_{\cdot, 1}, W_{\cdot, 2}, c$ — paramètres partagés, **aussi mis à jour par d'autres positions** ($\mathcal{L}_4, \mathcal{L}_5, \ldots$ utilisent aussi $W_{\cdot, 1}$ et $W_{\cdot, 2}$ via leurs propres $h_4, h_5, \ldots$).

> [!warning] Deux familles de paramètres, deux régimes de gradients
> - **Paramètres de sortie** $\{\alpha^{(i)}, b_i\}$ : chacun ne reçoit du gradient que de **sa position** $i$. Régime FVSBN — un signal d'apprentissage par CPD.
> - **Paramètres partagés** $\{W, c\}$ : chaque colonne $W_{\cdot, j}$ reçoit du gradient de **toutes les positions $i > j$** qui l'utilisent dans leur $h_i$. Les gradients s'**accumulent**.
> 
> Cette accumulation est *exactement* ce qui rend NADE statistiquement efficace : la colonne $W_{\cdot, 1}$ par exemple est mise à jour par toutes les positions $i \geq 2$. Le réseau apprend "comment lire $x_1$" à partir d'un signal venant de $n - 1$ positions — pas une seule comme dans FVSBN.

![[nade3.png|267]]
*Figure. Illustration de la backprop. Pour un exemple où $x_3 = 1$, la vraie distribution est $p = (0, 1)$ et la prédiction du modèle est par exemple $q = (0.7, 0.3)$ ; la loss est leur cross-entropy. Le gradient ne se propage qu'à travers les connexions actives (non-masquées). Seules les colonnes $W_{\cdot, 1}, W_{\cdot, 2}$ utilisées pour calculer $h_3$ sont mises à jour par cette position — la colonne $W_{\cdot, 3}$ ne l'est pas (elle n'est utilisée que par les positions $i \geq 4$).*

#### Abstraction et génération

**NADE n'est pas un modèle de représentation.** Une fois entraîné, on peut bien sûr récupérer une séquence de vecteurs cachés $h_1, h_2, \ldots, h_n$ pour un $x$ donné. Mais ce ne sont pas des "représentations abstraites" au sens d'un autoencodeur : $h_i$ encode juste l'information nécessaire pour prédire $x_i$ à partir de $x_{<i}$, pas $x$ tout entier. NADE n'est pas un modèle à variable latente — pour ça, voir `[[02_VAE]]`.

**Génération.** On suit le Mode B de §I.D, dévidé séquentiellement :

1. Calculer $h_1 = \sigma(c)$, puis $\hat x_1 = \sigma(\alpha^{(1)} \cdot h_1 + b_1)$. Tirer $\bar x_1 \sim \text{Bern}(\hat x_1)$.
2. Injecter $\bar x_1$ comme entrée, calculer $h_2 = \sigma(W_{\cdot, 1} \bar x_1 + c)$, puis $\hat x_2 = \sigma(\alpha^{(2)} \cdot h_2 + b_2)$. Tirer $\bar x_2 \sim \text{Bern}(\hat x_2)$.
3. ... continuer jusqu'à $\bar x_n$.

C'est lent — un pixel à la fois — mais c'est la nature de tout modèle autorégressif.

![[nade4.png|520]]

*Figure. Visualisation sur MNIST. À gauche : échantillons générés par NADE. À droite : probabilités conditionnelles $\hat x_i$ apprises pour chaque pixel. D'après* The Neural Autoregressive Distribution Estimator*, 2011.*

![[nade5.png|432]]

#### Compte de paramètres

> [!note]- Compte de paramètres NADE
> - Première couche partagée : $W \in \mathbb{R}^{d \times n}$ donne $nd$ paramètres, $c \in \mathbb{R}^d$ donne $d$.
> - Couches de sortie (non partagées) : $\alpha^{(i)} \in \mathbb{R}^d$ donne $d$, $b_i \in \mathbb{R}$ donne $1$, répété $n$ fois → $n(d+1)$.
> - Total : $nd + d + nd + n = 2nd + d + n = O(nd)$.
> 
> **Linéaire en $n$**. Pour $n = 1024$ et $d = 500$, c'est $\sim 10^6$ paramètres. Tractable, et surtout indépendant de la complexité combinatoire des CPDs.

#### Pourquoi le partage est l'idée centrale

Le partage de paramètres dans NADE est l'ancêtre direct de :

- les **embeddings** partagés des LLM (un seul tableau de vecteurs pour tout le vocabulaire) ;
- le **partage des poids dans le Transformer causal** (la même couche d'attention regarde toutes les positions) ;
- le **parameter tying** dans tous les modèles autorégressifs modernes.

L'intuition est statistique : si toutes les CPDs décrivent le même type de phénomène (qu'est-ce qu'un "pixel suivant" plausible ; qu'est-ce qu'un "token suivant" plausible), elles doivent pouvoir partager des représentations.

#### Limite : pas trivialement parallèle

Pour évaluer la log-vraisemblance complète d'une image, NADE doit calculer **$n$ vecteurs cachés différents** $h_1, h_2, \ldots, h_n$ — un par position. Comme chaque $h_i$ utilise une sous-matrice $A_i$ de taille différente, on ne peut pas faire un seul produit matriciel : il faut $n$ étapes séquentielles (ou une astuce de propagation incrémentale qui ajoute une colonne à chaque étape).

Sur GPU moderne, cela laisse beaucoup de parallélisme inutilisé. **MADE va résoudre exactement ce problème** : un seul réseau, un seul forward pass, toutes les $\hat x_i$ en parallèle.

### C. MADE — Masked Autoencoder for Distribution Estimation

> [!quote] L'apport de MADE (Germain et al., 2015)
> Un **seul** réseau (un autoencodeur), un **seul** forward pass, qui sort **toutes les CPDs** en parallèle — tout en respectant la propriété autorégressive. La magie : du **masking sur les poids** (pas sur l'entrée comme NADE).

> [!warning] Le masking de MADE n'est pas celui de NADE
> Confusion fréquente. À retenir :
> - **NADE** masque le **vecteur d'entrée** $x$ (ou de façon équivalente, sélectionne des colonnes de $W$). Un masque par position $i$. Conséquence : $n$ forward passes successifs pour évaluer toutes les CPDs.
> - **MADE** masque les **poids** $W$ eux-mêmes. Un seul masque global, fixé une fois pour toutes. Conséquence : un seul forward pass pour évaluer toutes les CPDs.
> 
> C'est le **déplacement du masque** (de l'entrée vers les poids) qui débloque la parallélisation. L'idée du Transformer causal est exactement la même appliquée à la matrice d'attention.

#### Le problème de l'autoencodeur standard

Un autoencodeur fully-connected viole la propriété autorégressive : sa sortie $\hat x_i$ dépend de **toutes** les entrées $x_1, \ldots, x_n$, y compris $x_i$ lui-même. On ne peut pas l'utiliser tel quel pour modéliser $p(x_i \mid x_{<i})$.

L'idée de MADE est de **partir d'un autoencodeur classique** et de **désactiver les connexions interdites** par un système de masques.

#### Construction du masque : numéroter et déconnecter

L'algorithme se fait en trois étapes :

1. **Numéroter les entrées.** On fixe un ordre $1, 2, \ldots, n$ sur les entrées $x_1, \ldots, x_n$.

2. **Numéroter les neurones cachés.** À chaque neurone caché, on assigne **aléatoirement** un nombre $m \in \{1, \ldots, n-1\}$. Ce nombre signifie : *"ce neurone aura le droit de regarder les $m$ premières entrées, et rien de plus"*.

3. **Définir les règles de connexion** (autorégressivité) :
    - une connexion entre une entrée $x_j$ et un neurone caché numéroté $m$ est **conservée** si $j \leq m$ ;
    - une connexion entre deux neurones cachés numérotés $m_1$ et $m_2$ (couches successives) est **conservée** si $m_1 \leq m_2$ ;
    - la sortie $\hat x_i$ ne peut être connectée qu'aux neurones cachés numérotés $\leq i - 1$ (car $\hat x_i$ ne doit dépendre que de $x_{<i}$).

![[made1.png|502]]
*Figure. À gauche : MLP fully-connected classique (toutes les connexions actives). Au centre : les trois matrices de masque $M^{W^1}, M^{W^2}, M^V$ (une par couche) qui désactivent les connexions interdites. À droite : le réseau effectif après masquage — seules les connexions respectant la règle autorégressive subsistent.*

#### Lecture du graphe : pourquoi la règle marche

Prenons deux exemples concrets pour voir comment les règles font tomber les bonnes connexions.

**Exemple 1 — un neurone caché numéroté 2 (couche 2).** Ce neurone est censé n'avoir vu que $x_1$ et $x_2$ (sa numérotation indique "2 entrées au max"). Pour respecter ça, il ne peut être connecté qu'à des neurones de la couche précédente qui eux-mêmes ne dépendent que de $\{x_1, x_2\}$ — donc des neurones numérotés $\leq 2$. C'est exactement la règle de connexion entre couches cachées.

**Exemple 2 — la sortie $\hat x_3$.** Elle prédit $p(x_3 \mid x_1, x_2)$, donc elle ne doit dépendre que de $x_1$ et $x_2$. Par le même argument, elle ne peut être connectée qu'aux neurones cachés numérotés $\leq 2$ dans la dernière couche cachée.

![[made2.png|532]]
*Figure. À gauche : un neurone caché numéroté 2 (en couche 2). Seules ses connexions vers des neurones numérotés $\leq 2$ de la couche précédente sont actives. À droite : la sortie $\hat x_3$ ne reçoit que des neurones cachés numérotés $\leq 2$. Dans les deux cas, l'autorégressivité est préservée par construction.*

#### Implémentation : matrices de masque par produit Hadamard

Concrètement, on prend les matrices de poids $W^1, W^2, V$ (entrée → couche 1, couche 1 → couche 2, couche 2 → sortie) et on applique un **masque binaire** par produit de Hadamard :

$$W^\ell_{\text{effectif}} = W^\ell \odot M^\ell.$$

Le masque $M^\ell$ est calculé une seule fois, au début de l'entraînement, à partir des numéros assignés. Les poids correspondant aux connexions interdites sont multipliés par $0$ et restent inactifs à jamais.

> [!example]- Exemple concret : un masque sur $W^2$
> Supposons une couche cachée à 5 neurones, avec des numéros (par exemple) $(2, 2, 4, 2, 5)$ assignés aux 5 neurones. Le masque $M^{W^2}$ qui implémente la règle "$m_1 \leq m_2$" est :
> 
> $$M^{W^2} \;=\; \begin{bmatrix} 1 & 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 & 0 \\ 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}$$
> 
> Lecture : la ligne $i$ correspond au neurone $i$ de la couche cible (numéro $m_2^{(i)}$), la colonne $j$ au neurone $j$ de la couche source (numéro $m_1^{(j)}$). L'entrée vaut $1$ ssi $m_1^{(j)} \leq m_2^{(i)}$.
> 
> Pendant l'entraînement, le réseau utilise $W^2 \odot M^{W^2}$ — les poids aux entrées nulles ne contribuent jamais aux activations.

![[made3.png|529]]
*Figure. Illustration concrète du produit Hadamard $W^2 \odot M^{W^2}$. Les entrées du masque sont $0$ ou $1$ selon la règle d'autorégressivité, et le masque éteint les poids correspondants. Cette opération est calculée une seule fois avant l'entraînement.*

#### Loss, backprop, ordre

**Loss.** Exactement la même que pour NADE/FVSBN : binary cross-entropy par position, sommée :

$$\mathcal{L}(\theta) \;=\; -\sum_{i=1}^{n} \Big[ x_i \log \hat x_i + (1 - x_i) \log(1 - \hat x_i) \Big].$$

**Backward.** La rétropropagation traverse uniquement les **connexions actives** (non-masquées). Les poids masqués ont un gradient nul par construction (le masque agit comme dropout fixe), donc ils ne sont jamais mis à jour. L'autorégressivité est préservée à l'entraînement comme à l'inférence.

**L'ordre est un hyperparamètre.** Le choix de l'ordre sur les entrées influence le modèle. Sur images, l'ordre raster (haut-gauche → bas-droite) est standard ; sur texte, l'ordre temporel s'impose. On peut aussi entraîner plusieurs masques avec des ordres différents et faire un **ensemble** (technique appelée *NADE ordering trick* dans le papier original).

> [!warning] Avantage décisif de MADE
> Toutes les CPDs $\hat x_1, \ldots, \hat x_n$ sont calculées dans **un seul forward pass** du réseau masqué. Pendant l'entraînement (mode A, §I.D), la log-vraisemblance complète d'un échantillon $x$ s'évalue en un seul passage forward + un seul passage backward, parallélisables sur tout le batch. 
>
> C'est *le* gain qui rend les modèles autorégressifs scalables — et c'est l'idée que le Transformer causal va reprendre via son masque d'attention triangulaire.

#### Abstraction et génération

**Abstraction.** Comme NADE, MADE n'est pas un modèle de représentation : les neurones cachés encodent l'information nécessaire pour prédire les sorties, mais pas une représentation abstraite globale de $x$. Pour ça → `[[02_VAE]]`.

**Génération.** On suit le Mode B de §I.D, dévidé séquentiellement (comme tout modèle autorégressif) :

1. Faire un forward pass avec $x = (0, 0, \ldots, 0)$ comme entrée fictive ; ne lire que $\hat x_1$. Tirer $\bar x_1 \sim \text{Bern}(\hat x_1)$.
2. Faire un nouveau forward pass avec $x = (\bar x_1, 0, \ldots, 0)$ ; ne lire que $\hat x_2$. Tirer $\bar x_2 \sim \text{Bern}(\hat x_2)$.
3. Continuer jusqu'à $\bar x_n$, en injectant à chaque étape les pixels déjà tirés.

> [!note]- Pourquoi génération séquentielle malgré la parallélisation à l'entraînement
> À l'entraînement on a accès à *tout* $x$ via teacher forcing → on peut calculer toutes les CPDs en parallèle. À la génération, $x$ est ce qu'on est en train de construire — pour calculer $\hat x_i$ il faut d'abord avoir tiré $\bar x_1, \ldots, \bar x_{i-1}$. C'est l'asymétrie fondamentale de §I.D, et elle vaut pour MADE comme pour tout AR.

![[made4.png]]
*Figure. Schéma de la génération séquentielle pour MADE : à chaque étape on injecte les pixels déjà tirés et on lit la nouvelle CPD correspondante. Forward pass complet à chaque étape, mais une seule sortie utilisée par étape — d'où le coût $O(n)$ à l'inférence.*

### D. Récapitulatif

> [!warning] Les trois idées de FVSBN → NADE → MADE
> | Modèle | Idée ajoutée | Où est le masque ? | Complexité | Limite |
> |---|---|---|---|---|
> | FVSBN | LR comme CPD | (pas de masque, pas de partage) | $O(n^2)$ | Linéaire, pas de partage |
> | NADE | Partage de poids entre CPDs | Sur l'**entrée** $x$ | $O(nd)$ | $n$ forward passes successifs |
> | MADE | Masque déplacé sur les poids | Sur les **poids** $W$ | $O(nd)$ | Choix de l'ordre, capacité d'un MLP |
> 
> Ces trois idées — paramétrisation par NN, partage de paramètres, masquage des poids — sont *toutes les trois* présentes dans le Transformer causal des LLM modernes. FVSBN/NADE/MADE ne sont pas des reliques historiques : ce sont les briques élémentaires.

## IV. RNADE — extension au continu

NADE/MADE modélisent des variables discrètes (Bernoulli ou catégorielle). Pour modéliser des variables **continues** (signaux audio, intensités de pixels en réel), il faut remplacer la Bernoulli en sortie par une distribution continue. Sauf qu'une gaussienne unique ne suffit pas — les conditionnelles réelles ont souvent plusieurs modes.

### A. Mixture Density Networks (encadré)




![[rnade1.png|529]]
Figure le mixture density networks ?


Formally:
$$
p(y \mid x) = \sum_{c=1}^C \alpha_c(x) D(y \mid \lambda_{1,c}(x), \lambda_{2,c}(x), ...)
$$
To obtain the parameters for the mixture, a DNN is modified to output multiple parameter vectors. We start off with a single layer DNN and a ReLU activation:
$$
\begin{aligned}
h_1(x) &= max(W_1^T x + b_1, 0) \\
\alpha(x) &= softmax(W_{\alpha}^T h_1(x) + b_{\alpha}) \\
\lambda_1(x) &= (W^{T}_{\lambda_1} h_1(x) + b_{\lambda_1}) \\
\lambda_2(x) &= (W^{T}_{\lambda_2} h_1(x) + b_{\lambda_2}) \\
\end{aligned}
$$
The mixing coefficient must to one $\sum \alpha(x) =1$. Therefore, we are using a softmax function to constraint the output. To enforce that the standard deviation is strictly positive we have to pick another kind of activation function, we propre the ELU activation function with an offset
$$
\begin{aligned}
\mu(x) &= \lambda_1(x) \\
\sigma(x) &= ELU(\lambda_2(x)) +1
\end{aligned}
$$
Finally the cost function is the MLE, where we plugin the parameters of the mixture of gaussians and then evaluate with our training data points
$$
\underset{\Theta}{\arg \min }~ l(\Theta)=-\frac{1}{|\mathbb{D}|} \sum_{(\mathbf{x}, y) \in \mathbb{D}} \log p(y \mid \mathbf{x})
$$



> [!note]- Mixture Density Networks (Bishop, 1994)
> Une MDN remplace la sortie d'un réseau par les paramètres d'un **mélange de gaussiennes** :
> 
> $$p(y \mid x) = \sum_{c=1}^{C} \alpha_c(x) \, \mathcal{N}\!\left(y; \mu_c(x), \sigma_c^2(x)\right),$$
> 
> avec des contraintes sur les sorties du réseau pour assurer que (i) les $\alpha_c$ somment à 1 (softmax), (ii) les $\sigma_c$ sont strictement positifs (exp ou ELU + offset).
> 
> Idée transversale : on l'utilisera aussi pour les VAE conditionnels et certains diffusion models. Voir le papier de référence : [Bishop 1994](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf).

### B. RNADE : NADE + MDN

RNADE (Uria, Murray, Larochelle, 2014) est l'idée la plus simple : on garde l'architecture NADE et on remplace la Bernoulli de sortie par un **mélange uniforme de $K$ gaussiennes** :

$$p(x_i \mid x_{<i}) = \sum_{j=1}^{K} \frac{1}{K} \, \mathcal{N}\!\left(x_i ; \mu_i^j, \sigma_i^j\right), \qquad \hat x_i = (\mu_i^1, \ldots, \mu_i^K, \sigma_i^1, \ldots, \sigma_i^K) = f(h_i).$$

Applications : modélisation de signaux audio, densités d'attributs continus en tabular learning.

![[rnade2.png|368]]


## V. Architectures pour les images

Pour les images, les architectures se spécialisent : on n'utilise pas un MLP générique mais des **convolutions** qui exploitent la structure spatiale 2D. Le défi : respecter la propriété autorégressive (ne pas voir le pixel courant ni les futurs) tout en gardant la structure conv 2D.

### A. PixelRNN

Modélisation pixel par pixel en raster scan. Chaque pixel RGB demande trois conditionnelles :

$$p(x_t \mid x_{<t}) = p(x_t^R \mid x_{<t}) \cdot p(x_t^G \mid x_{<t}, x_t^R) \cdot p(x_t^B \mid x_{<t}, x_t^R, x_t^G),$$

chacune étant une softmax sur 256 valeurs (intensité 8-bit).

Le cœur architectural est une **LSTM convolutionnelle** : les gates standard d'une LSTM sont remplacées par des convolutions causales :

$$\left[\Gamma_o^i, \Gamma_f^i, \Gamma_u^i, \tilde c_i\right] = \sigma\!\left(K^{ss} \star a_{i-1} + K^{is} \star x_i\right),$$

avec $K^{is}$ noyau "input → state" et $K^{ss}$ noyau "state → state". La sparsité des convolutions remplace les matrices fully-connected d'une LSTM vanilla.

> [!note]- Variantes Row-LSTM et Diagonal-LSTM
> Le papier original (Van den Oord et al., 2016) propose deux variantes selon la forme du voisinage convolutionnel utilisé pour le contexte d'état à état :
> - **Row-LSTM** : 1D conv $3 \times 1$ qui lit la ligne précédente. Rapide (parallèle par ligne), mais champ réceptif **triangulaire** qui ignore certains pixels en haut à droite.
> - **Diagonal-LSTM** : convolutions le long des diagonales (avec une astuce de *skew* pour rendre la diagonale verticale). Champ réceptif **global**, mais plus complexe à implémenter.
> 
> Détail surtout d'intérêt historique — les architectures modernes (Pixel Transformer, diffusion U-Net) ont largement remplacé ces variantes.

### B. PixelCNN

Plus rapide à entraîner que PixelRNN : on remplace les LSTM convolutionnelles par des **convolutions standard masquées**. Le masque sur le noyau de convolution empêche le pixel courant de voir les pixels futurs (et lui-même selon la position dans la chaîne RGB).

$$\text{Kernel mask pour } x_i = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

(masque de type "A" — la variante "B" autorise le pixel central pour les couches > 1).

**Avantage** : entraînement entièrement parallèle (mode A de §I.D appliqué aux pixels d'une image), comme MADE.

**Limite : blind spot.** Une pile de convolutions masquées laisse une zone du voisinage causalement valide **invisible** à la position courante — quelques pixels au-dessus à droite ne sont jamais propagés. C'est le *blind spot problem*. Gated PixelCNN (Van den Oord et al., 2016) résout le problème en combinant deux flux convolutionnels (horizontal et vertical).

## VI. Architectures pour le texte : la voie des LLM

C'est ici que le cours rejoint la pratique moderne. **Tous les LLM (GPT, Claude, Llama, ...) sont des modèles autorégressifs au sens strict de la §I**, avec deux choix architecturaux :

### A. RNN / LSTM language models (l'ancienne génération)

Avant 2017, les LM neuronaux étaient des **RNN** :

$$h_{t+1} = \tanh(W_{hh} h_t + W_{xh} x_{t+1}), \qquad o_{t+1} = W_{hy} h_{t+1}, \qquad p(x_{t+1} \mid x_{\leq t}) = \text{softmax}(o_{t+1}).$$

Le RNN compresse tout l'historique dans un vecteur d'état $h_t$ de taille fixe — élégant en théorie (Turing-complet), mais en pratique :
- entraînement **séquentiel** (chaque pas dépend du précédent → pas de parallélisme sur la longueur),
- difficultés des gradients (vanishing/exploding, partiellement réglées par LSTM/GRU),
- mémoire limitée du contexte lointain.

### B. Transformer causal — l'architecture des LLM

Le Transformer (Vaswani et al., 2017), en version *décodeur causal*, est l'architecture qui domine depuis 2018. Trois idées clés :

1. **Attention** : chaque position $i$ regarde **directement** toutes les positions $\leq i$ via un mécanisme d'attention pondérée, sans passer par un état caché compressé.
2. **Masque causal** : un masque triangulaire (avec $-\infty$ avant softmax) empêche la position $i$ de voir les positions $\geq i$. *C'est exactement le masking de MADE, appliqué à la matrice d'attention.*
3. **Calcul parallèle à l'entraînement** : toutes les positions sont traitées en parallèle dans un seul forward pass — exactement le mode A de §I.D, appliqué aux tokens d'une séquence.

> [!warning] GPT = MADE + attention + scaling
> Le Transformer causal hérite des trois idées de FVSBN/NADE/MADE :
> - **Paramétrisation par NN** des CPDs (FVSBN → MLP en sortie).
> - **Partage de poids** sur les positions (NADE → blocs d'attention identiques).
> - **Masking des poids** pour parallélisation (MADE → masque d'attention causal).
> 
> Le saut quantitatif (passer du jouet académique au LLM utilisable) ne tient pas à une idée conceptuelle nouvelle mais à **l'échelle** : milliards de paramètres, milliers de milliards de tokens, et un mécanisme d'attention qui exploite le parallélisme matériel des GPU/TPU bien mieux que les RNN ou les conv masquées.

Le détail architectural du Transformer (attention multi-head, positional encoding, layer norm, etc.) est traité dans la note dédiée du dossier `[[Natural Language Processing (NLP)]]`.

## VII. Audio : WaveNet (TODO)

> [!todo] WaveNet à compléter
> WaveNet (Van den Oord et al., 2016) modélise des signaux audio bruts (16kHz, 16-bit) avec une architecture autorégressive : **convolutions causales dilatées**. Les dilatations exponentielles permettent un champ réceptif énorme sans exploser le nombre de paramètres. Application centrale : *text-to-speech* de qualité quasi-naturelle (Google Assistant à l'époque).
> 
> Note à étoffer après lecture du papier — référence : [Van den Oord et al., 2016](https://arxiv.org/abs/1609.03499).

## VIII. Synthèse

### A. Forces et faiblesses

> [!warning] Bilan des modèles autorégressifs
> **Forces :**
> - **Densité exacte calculable** — pas de borne ELBO, pas d'approximation.
> - **MLE direct** par SGD — entraînement stable, théoriquement bien compris.
> - **Parallélisme à l'entraînement** (MADE, PixelCNN, Transformer) — passage à l'échelle réussi pour les LLM. Voir §I.D.
> - **Conditionnement facile** — il suffit de fixer un préfixe $x_{<k}$ pour conditionner.
> 
> **Faiblesses :**
> - **Génération séquentielle** — coût $O(n)$ forward passes pour échantillonner, non parallélisable. C'est le goulot d'étranglement de l'inférence LLM.
> - **Choix d'un ordre** — naturel pour texte/audio, arbitraire pour images, problématique pour structures non séquentielles (graphes, ensembles).
> - **Pas d'espace latent explicite** — pas de représentation compacte $z$ qu'on peut interpoler, manipuler, désentanglement. Pour ça → VAE, flows.

### B. Tableau de positionnement

| Aspect | Autorégressif | VAE | Flow | GAN | Diffusion |
|---|---|---|---|---|---|
| Structure | Chain rule, ordre total | Latent $z$ + décodeur | Bijection $z \leftrightarrow x$ | Latent $z$ + générateur | Chaîne de Markov de débruitage |
| Densité $p(x)$ | **Exacte** | Borne (ELBO) | Exacte | Inaccessible | Approchée |
| Apprentissage | MLE direct (§V bis fondations) | ELBO | MLE | Adversarial | Score matching |
| Échantillonnage | **Lent (séquentiel)** | Rapide | Rapide | Rapide | Itératif (lent) |
| Espace latent | Non | Oui | Oui | Oui | Implicite |

Les autorégressifs sont les **rois de la densité exacte et de la MLE**, au prix d'une génération lente. C'est le compromis structurant de la famille.

### C. Quel est le bon usage ?

Modèle autorégressif si :
- on veut une **densité exacte** (compression, détection d'anomalies, comparaison de modèles via log-likelihood) ;
- les données ont un **ordre naturel** (texte, audio, séries temporelles) ;
- on accepte une génération séquentielle (les LLM acceptent ce coût parce que le contexte d'usage le permet).

Autre famille si :
- on veut **interpoler dans un espace latent** (VAE, flow, GAN) ;
- on veut une génération **rapide** pour des images en haute définition (diffusion accélérée, GAN) ;
- les données n'ont **pas d'ordre naturel** (ensembles, graphes — bien que des variantes autorégressives existent).

---

## Pour aller plus loin

- **Cours.** Stefano Ermon, *CS236 Deep Generative Models*, Stanford. Leçon 2 — la présente note.
- **NADE.** H. Larochelle, I. Murray. *The Neural Autoregressive Distribution Estimator.* AISTATS 2011.
- **MADE.** M. Germain, K. Gregor, I. Murray, H. Larochelle. *MADE: Masked Autoencoder for Distribution Estimation.* ICML 2015.
- **RNADE.** B. Uria, I. Murray, H. Larochelle. *RNADE: The real-valued neural autoregressive density-estimator.* NeurIPS 2013.
- **PixelRNN/PixelCNN.** A. Van den Oord, N. Kalchbrenner, K. Kavukcuoglu. *Pixel Recurrent Neural Networks.* ICML 2016. Et *Conditional Image Generation with PixelCNN Decoders.* NeurIPS 2016.
- **WaveNet.** A. Van den Oord et al. *WaveNet: A Generative Model for Raw Audio.* arXiv 2016.
- **Transformer.** A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- **GPT.** A. Radford et al. *Language Models are Unsupervised Multitask Learners.* (GPT-2, 2019) et suivants.
