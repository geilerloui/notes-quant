---
title: Energy-based Models
---
# Energy-based Models

> Deuxième famille du chapitre génératif : les **EBM** (energy-based models). Très flexibles — on peut y mettre n'importe quel réseau, sans contrainte d'architecture (pas d'ordre comme en autorégressif, pas d'inversibilité comme en flow) — mais au prix d'une **constante de normalisation $Z_\theta$ intractable** qui rend l'entraînement par MLE direct impossible. C'est ce verrou qui motivera plus tard le passage au **score matching** (note `[[06_Score-Based & Diffusion Models|06]]`). Au-delà de la théorie, on verra trois exemples canoniques : modèle d'Ising, Product of Experts, et **Restricted Boltzmann Machines (RBM)**.

## I. Le cadre EBM

### A. Pourquoi pas un réseau de neurones directement ?

Une distribution $p(x)$ doit vérifier deux propriétés :

- **Non-négativité** : $p(x) \ge 0$
- **Sum-to-one** : $\sum_x p(x) = 1$ (ou $\int p(x)\, dx = 1$)

La non-négativité est facile à imposer. Par exemple, pour n'importe quel réseau $f_\theta$ :

- $g_\theta(x) = f_\theta(x)^2$
- $g_\theta(x) = \exp(f_\theta(x))$

Mais la sum-to-one est dure : en général $\int g_\theta(x)\, dx = Z(\theta) \ne 1$. Donc $g_\theta$ n'est pas une densité valide.

**Solution naïve : normaliser.** On définit

$$p_\theta(x) = \frac{1}{Z(\theta)}\, g_\theta(x), \qquad Z(\theta) = \int g_\theta(x)\, dx.$$

Par construction $\int p_\theta(x)\, dx = 1$. **Mais** ça ne marche que si on connaît $Z(\theta)$ **analytiquement** en fonction de $\theta$ — ce qui restreint à des familles très spécifiques :

- Gaussienne : $g(x) = e^{-(x-\mu)^2/2\sigma^2}$, $Z = \sqrt{2\pi\sigma^2}$.
- Exponentielle : $g(x) = e^{-\lambda x}$, $Z = 1/\lambda$.
- Etc.

C'est restrictif, mais ces formes simples servent de **briques** pour construire des modèles plus complexes :

- **Autoregressif** — produits d'objets normalisés : $\int_x \int_y p_\theta(x)\, p_{\theta'(x)}(y)\, dy\, dx = \int_x p_\theta(x)\, dx = 1$.
- **Variables latentes** — mélanges d'objets normalisés : $\int_x \big[\alpha\, p_\theta(x) + (1-\alpha)\, p_{\theta'}(x)\big]\, dx = \alpha + (1-\alpha) = 1$.

> [!question] Et si on relâchait cette contrainte ?
> Peut-on utiliser des modèles où $Z(\theta)$ n'est **pas** calculable analytiquement ? C'est la motivation des EBM.

### B. Définition

Un **modèle énergétique** est défini par :

$$\boxed{\;p_\theta(x) = \frac{1}{Z(\theta)}\, \exp(f_\theta(x)), \qquad Z(\theta) = \int \exp(f_\theta(x))\, dx\;}$$

Le réseau $f_\theta$ est **quelconque** (pas de contrainte d'inversibilité, d'ordre, etc.). La quantité $Z(\theta)$ est la **fonction de partition** (ou *partition function*).

**Pourquoi l'exponentielle plutôt qu'un carré ?**

- *Variations de probabilité*. Les images ont des probabilités qui varient sur de nombreux ordres de grandeur ; la log-probabilité est l'échelle naturelle. Un carré demanderait un $f_\theta$ très peu lisse.
- *Familles exponentielles*. Beaucoup de distributions classiques s'écrivent sous cette forme.
- *Physique statistique*. Ces distributions apparaissent naturellement sous le principe d'entropie maximale. $-f_\theta(x)$ s'interprète comme **l'énergie** du state $x$ : une configuration de basse énergie (donc $f_\theta(x)$ élevé) est plus probable. D'où le nom *energy-based*.

### C. Avantages et limites

**Avantages.**

- **Flexibilité extrême.** On peut utiliser n'importe quel $f_\theta$ — pas de contrainte structurelle.

**Limites (nombreuses).**

- **Sampling de $p_\theta(x)$ difficile.** Pour générer un échantillon, il faut connaître les probabilités relatives, ce qui demande $Z(\theta)$.
- **Évaluation et optimisation de la vraisemblance difficiles.** Toujours le problème de $Z(\theta)$.
- **Pas d'apprentissage de représentations direct** (on peut ajouter des variables latentes pour y remédier — voir RBM en section III).

> [!warning] La malédiction de la dimension
> Calculer $Z(\theta)$ numériquement (quand pas de forme fermée) scale **exponentiellement** avec la dimension de $x$ — infaisable au-delà de quelques dizaines de dimensions. C'est *le* verrou central des EBM.

**Bonne nouvelle :** certaines tâches **n'ont pas besoin de $Z(\theta)$**, on le voit dans la section suivante.

## II. Applications des EBM

### A. Comparaisons relatives : ratios

Pour évaluer $p_\theta(x)$ ou $p_\theta(x')$ individuellement, il faut $Z(\theta)$. Mais leur **ratio** :

$$\frac{p_\theta(x)}{p_\theta(x')} = \exp\big(f_\theta(x) - f_\theta(x')\big)$$

ne fait pas intervenir $Z(\theta)$. On peut donc savoir laquelle des deux configurations est la plus probable, ce qui suffit pour :

- **Détection d'anomalie** : un $x$ avec $f_\theta(x)$ très bas (énergie très haute) est anormal.
- **Débruitage** : choisir le $\tilde{x}$ qui maximise $p_\theta(\tilde{x})$ pour des candidats donnés.
- **Classification / labellisation de séquence** : pour la reconnaissance d'objets, on apprend une fonction d'énergie qui capture la relation entre images $X$ et labels $Y$, et on prédit $\hat{y} = \arg\max_y p_\theta(x, y)$ — pas besoin de $Z$.

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im2 (4).png]]

### B. Modèle d'Ising

Image vraie $y \in \{0, 1\}^{3 \times 3}$ et image corrompue $x \in \{0, 1\}^{3 \times 3}$ (on observe $x$, on cherche à retrouver $y$).

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im3 (5).png]]

On modélise la jointe par :

$$p(y, x) = \frac{1}{Z} \exp\!\Big(\sum_i \psi_i(x_i, y_i) + \sum_{(i,j) \in E} \psi_{ij}(y_i, y_j)\Big)$$

- $\psi_i(x_i, y_i)$ : le pixel corrompu $x_i$ dépend du pixel original $y_i$ (terme de cohérence pixel-à-pixel).
- $\psi_{ij}(y_i, y_j)$ : deux pixels voisins ont tendance à avoir la même valeur (régularisation spatiale).

**Reconstruction** : on maximise $p(y \mid x)$. En pratique, on peut généraliser en remplaçant les $\psi$ par un réseau de neurones plus expressif.

### C. Product of Experts

Façon naturelle de **combiner plusieurs modèles génératifs**. Supposons qu'on a entraîné trois modèles $q_{\theta_1}(x)$, $r_{\theta_2}(x)$, $t_{\theta_3}(x)$ sur le même dataset (peut-être avec des architectures différentes). Chacun est un *expert* qui score un $x$. Si on suppose les experts indépendants, on les ensemble par produit :

$$q_{\theta_1}(x)\, r_{\theta_2}(x)\, t_{\theta_3}(x).$$

**Problème** : ce produit n'est plus normalisé, même si chaque facteur l'est. Il faut renormaliser :

$$p_{\theta_1, \theta_2, \theta_3}(x) = \frac{1}{Z(\theta_1, \theta_2, \theta_3)}\, q_{\theta_1}(x)\, r_{\theta_2}(x)\, t_{\theta_3}(x).$$

> [!note] AND vs OR
> Le produit d'experts agit comme un **AND** : la proba est nulle dès qu'un expert assigne zéro. À l'inverse, les **mélanges** (modèles à variables latentes) agissent comme un **OR** : il suffit qu'un expert assigne une masse non-nulle.

## III. Restricted Boltzmann Machines (RBM)

### A. Motivation

On considère des images $32 \times 32$ pixels — soit $1024$ variables aléatoires $(X_1, \ldots, X_{1024})$. On veut apprendre la jointe $P(X_1, \ldots, X_{1024})$.

![[im17 1.png|209]]

Si on suppose chaque pixel ne dépend que de ses voisins, on peut factoriser la distribution sur un **réseau de Markov** :

$$P \propto \prod_i \phi(D_i)$$

où $D_i$ est l'ensemble des variables formant une clique maximale (groupes de pixels voisins).

**Utilité de la jointe.** Étant donnée $P(X_1, \ldots, X_{1024})$, on peut :

- Classifier une nouvelle image (ciel ouvert ou non ?).
- Générer de nouvelles images.
- Débruiter / compléter des images partielles.

C'est exactement ce qu'on attend d'un modèle génératif.

**Ajout de variables latentes.** Pour modéliser des dépendances plus globales, on ajoute des **variables cachées** $H = (H_1, \ldots, H_n)$ représentant l'information non observée (typiquement : *jour/nuit/nuageux*). On a donc :

- $V = (V_1, \ldots, V_{1024})$ : variables visibles (pixels)
- $H = (H_1, \ldots, H_n)$ : variables cachées

et on modélise la jointe $P(V, H)$.

![[im16 1.png|181]]

Les interactions entre pixels passent désormais par les latents.

**Deux interprétations clés** une fois $P(V, H)$ apprise :

- *Abstraction.* $P(H \mid V) = P(V, H) / \sum_H P(V, H)$ donne la **représentation latente** la plus probable pour une image — c'est l'analogue d'un encodeur (PCA, autoencoder).
- *Génération.* $P(V \mid H) = P(V, H) / \sum_V P(V, H)$ permet de générer une image à partir d'un latent fixé — c'est l'analogue d'un décodeur.

Pour la suite, on suppose $V \in \{0, 1\}^m$ et $H \in \{0, 1\}^n$ (variables binaires).

**Boltzmann vs Restricted Boltzmann.**

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im8 (1).png]]

Dans une RBM, on **restreint les connexions** : pas d'arêtes entre nœuds visibles, ni entre nœuds cachés. Seulement des arêtes visible-caché. C'est cette restriction qui rend l'inférence tractable.

### B. Structure des RBM

Sur un graphe biparti $V$-$H$ (RBM), les cliques maximales sont les paires $(v_i, h_j)$. La jointe s'écrit donc comme un produit de **potentiels de clique**, plus optionnellement des potentiels de nœud :

$$P(V, H) = \frac{1}{Z} \prod_i \prod_j \phi_{ij}(v_i, h_j)\, \prod_i \psi_i(v_i)\, \prod_j \xi_j(h_j).$$

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im9 (2).png]]

La fonction de partition $Z$ somme sur les $2^m \times 2^n$ configurations possibles — **intractable** dès que $m, n$ sont raisonnables.

**Forme paramétrique.** On choisit :

$$\phi_{ij}(v_i, h_j) = e^{w_{ij} v_i h_j}, \qquad \psi_i(v_i) = e^{b_i v_i}, \qquad \xi_j(h_j) = e^{c_j h_j}.$$

La jointe devient :

$$\boxed{\;P(V, H) = \frac{1}{Z}\, e^{-E(V, H)},\quad E(V, H) = -\sum_{i,j} w_{ij} v_i h_j - \sum_i b_i v_i - \sum_j c_j h_j\;}$$

$E(V, H)$ est la **fonction d'énergie**. La forme $e^{-E}/Z$ s'appelle la **distribution de Boltzmann** ou de Gibbs en physique statistique.

### C. RBM comme réseau neuronal stochastique

**Conditionnelles tractables.** Le calcul-clé qui rend la RBM utile :

$$
\begin{aligned}
p(h \mid v) &= \frac{p(v, h)}{p(v)} = \frac{(1/Z)\, e^{b^\top v + c^\top h + v^\top W h}}{\sum_h (1/Z)\, e^{b^\top v + c^\top h + v^\top W h}} \\
&= \frac{e^{c^\top h}\, e^{v^\top W h}}{Z'} = \frac{1}{Z'} \prod_{j=1}^n \exp\!\big(c_j h_j + v^\top W_{:j}\, h_j\big)
\end{aligned}
$$

C'est un **produit** sur $j$, donc **les $h_j$ sont indépendants conditionnellement à $v$**. Par symétrie, les $v_i$ sont indépendants conditionnellement à $h$.

Sur une coordonnée :

$$p(h_j = 1 \mid v) = \frac{\exp(c_j + v^\top W_{:j})}{\exp(c_j + v^\top W_{:j}) + \exp(0)} = \sigma(c_j + v^\top W_{:j})$$

avec $\sigma$ la fonction sigmoïde. Au total :

$$\boxed{\;p(h \mid v) = \prod_j \sigma(c_j + v^\top W_{:j}),\qquad p(v \mid h) = \prod_i \sigma(b_i + W_{i:}\, h)\;}$$

> [!important] Interprétation neuronale
> Une RBM est un **réseau de neurones stochastique** : la probabilité d'activation d'un neurone (visible ou caché) à $1$ est donnée par une sigmoïde sur la somme pondérée de ses entrées — exactement comme dans un MLP, mais l'activation est stochastique. Et comme il n'y a pas de labels, $h$ apprend une **représentation abstraite** de $v$, à la manière d'un autoencodeur.

### D. Apprentissage

**Objectif (log-vraisemblance d'un point d'entraînement).**

$$\ln \mathcal{L}(\theta) = \ln p(V \mid \theta) = \ln \sum_H e^{-E(V, H)} - \ln \sum_{V, H} e^{-E(V, H)}.$$

**Gradient générique.**

$$\frac{\partial \ln \mathcal{L}(\theta)}{\partial \theta} = -\underbrace{\sum_H p(H \mid V)\, \frac{\partial E(V, H)}{\partial \theta}}_{\text{phase positive (données)}} + \underbrace{\sum_{V, H} p(V, H)\, \frac{\partial E(V, H)}{\partial \theta}}_{\text{phase négative (modèle)}}.$$

**Gradients spécifiques.** Pour les paramètres concrets, on dérive :

$$\boxed{\begin{aligned}
\nabla_W \mathcal{L}(\theta) &= \sigma(Wv + c)\, v^\top - \mathbb{E}_v\!\big[\sigma(Wv + c)\, v^\top\big] \\
\nabla_b \mathcal{L}(\theta) &= v - \mathbb{E}_v[v] \\
\nabla_c \mathcal{L}(\theta) &= \sigma(Wv + c) - \mathbb{E}_v\!\big[\sigma(Wv + c)\big]
\end{aligned}}$$

> [!warning] Le verrou : l'espérance sous le modèle
> Les trois gradients contiennent une **espérance $\mathbb{E}_v[\cdot]$ sous la distribution du modèle**, qui est intractable (elle demande $Z$). C'est exactement le problème générique des EBM. La solution : approximer cette espérance par **échantillonnage** — d'où la suite (Gibbs / Contrastive Divergence).

### E. Sampling : Gibbs et Contrastive Divergence

**Pourquoi pas Monte Carlo uniforme ?** On ne peut pas tirer chaque configuration avec probabilité uniforme — sur l'ensemble des combinaisons noir/blanc d'une image, la quasi-totalité a une probabilité quasi-nulle. Il faut échantillonner depuis $p(V, H)$, donc une chaîne de Markov.

**Gibbs sampling.** On itère :

1. Initialisation : $x_0 \in \{0, 1\}^{n+m}$ aléatoire.
2. À chaque pas : choisir $i \sim q(i)$ (uniforme), puis échantillonner $X_i$ depuis $p(X_i \mid X_{-i})$.

Pour un visible $V_i$ donné, la conditionnelle est :

$$P(V_i = 1 \mid V_{-i}, H) = \sigma\!\Big(\sum_j w_{ij}\, v_j + c_i\Big).$$

Donc on tire $z$ avec cette proba et on met $V_i$ à $1$ avec probabilité $z$, à $0$ sinon. Idem pour $H_j$ par symétrie.

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im11 (1).png]]

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im20.png]]

![[im11 (1) 1.png]]

**Estimateur Monte Carlo.** Une fois en distribution stationnaire, on remplace les espérances par des moyennes empiriques :

$$\mathbb{E}_v\!\big[\sigma(Wv + c)\, v^\top\big] \approx \frac{1}{k} \sum_{i=1}^k \sigma\!\big(Wv^{(i)} + c\big)\, v^{(i)\top},\quad\text{etc.}$$

**Pseudo-code.**

1. Initialiser $v^{(0)}$ aléatoire, $W, b, c$ aléatoires.
2. Gibbs : $v^{(0)} \to h^{(0)}$ via $\sigma$ + tirage Bernoulli, puis $h^{(0)} \to v^{(1)}$, etc.
3. Update : appliquer les formules de gradient avec les espérances estimées.

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im12.png|372]]

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im13.png|227]]

**Contrastive Divergence (CD-$k$).** En pratique Gibbs est trop lent : à chaque pas de SGD on devrait relancer une chaîne longue. L'astuce de **Hinton 2002** :

- Initialiser la chaîne **à un point de données réel** $v^{(t)}$ (pas un point aléatoire).
- Lancer Gibbs seulement $k$ pas (typiquement $k = 1$).
- Remplacer l'espérance modèle par un **point estimate** au point obtenu $\tilde{v}$ :

$$\mathbb{E}_{p(V, H)}\!\big[v_j h_i\big] \approx \sigma(w_i\, \tilde{v} + c_i)\, \tilde{v}_j.$$

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im14.png]]

Le gradient approximé devient :

$$\frac{\partial \mathcal{L}(\theta)}{\partial w_{ij}} \approx \sigma(w_i\, v + c_i)\, v_j - \sigma(w_i\, \tilde{v} + c_i)\, \tilde{v}_j.$$

**Intuition.** Au fur et à mesure que le modèle s'améliore, $\tilde{v}$ ressemble de plus en plus aux données — et le gradient tend naturellement vers zéro. CD est biaisé mais beaucoup plus rapide que Gibbs complet, et marche très bien en pratique.

![[images/3-Apprentissage automatique/05_Generative Models/energy models/im15.png]]
