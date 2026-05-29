## 0. Le point de départ — un AE est une PCA potentiellement non-linéaire

L'**autoencodeur** (AE) est un réseau de neurones entraîné à **reconstruire son entrée** : on lui passe un vecteur $x \in \mathbb{R}^d$, il le compresse vers une représentation interne $z \in \mathbb{R}^k$ (avec typiquement $k \ll d$), puis le décompresse pour produire $\hat{x} \in \mathbb{R}^d$ aussi proche de $x$ que possible. Le **goulot d'étranglement** (*bottleneck*) en dimension $k$ force le réseau à apprendre une représentation économique des données.

Conceptuellement, l'AE est le **frère non-linéaire de la PCA** :

- **PCA** (cf. `[[PCA]]`) compresse en projetant linéairement sur le sous-espace des $k$ plus grandes valeurs propres de $\Sigma$. Optimal au sens Eckart-Young pour toute compression linéaire de rang $k$.
- **AE** compresse via un réseau de neurones — donc en général **non-linéaire**. Il peut suivre des **manifolds courbés** que la PCA ne sait pas voir (cf. l'exemple du swiss roll dans `[[PCA]]`, §6).

Le résultat-clé qu'on démontrera plus bas (§3) : **un AE linéaire entraîné optimalement converge vers la PCA**. Toute la "magie" des AE par rapport à PCA, c'est juste la non-linéarité — sans elle, on retombe exactement sur Eckart-Young.

Plan de cette note :

- **§1** : architecture vanilla — encodeur, décodeur, tied weights, loss MSE / cross-entropy.
- **§2** : capacité du bottleneck — undercomplete vs overcomplete.
- **§3** : preuve "AE linéaire = PCA" via SVD et Eckart-Young.
- **§4** : denoising autoencoders — comment forcer l'apprentissage de features utiles même en overcomplete.
- **§5** : pour aller plus loin (sparse AE, contractive AE, VAE).

---

## 1. Vanilla Autoencoder

### (i) Architecture encoder–decoder

L'AE est constitué de deux fonctions paramétriques :

- **Encoder** $g_\phi : \mathbb{R}^d \to \mathbb{R}^k$ — compresse l'entrée en un code $z$ de dimension $k$.
- **Decoder** $f_\theta : \mathbb{R}^k \to \mathbb{R}^d$ — reconstruit $\hat{x}$ depuis le code.

Pour la version la plus simple (une couche cachée), ce sont des affines + non-linéarité :

$$
\boxed{\;z = g_\phi(x) = \sigma(W x + b), \qquad \hat{x} = f_\theta(z) = \sigma(W' z + c)\;}
$$

avec $W \in \mathbb{R}^{k \times d}$, $W' \in \mathbb{R}^{d \times k}$, biais $b \in \mathbb{R}^k$, $c \in \mathbb{R}^d$, et $\sigma$ une non-linéarité (sigmoïde, ReLU, tanh…). Le code $z$ vit dans l'**espace latent** $\mathbb{R}^k$ — c'est la représentation compressée.

![[images/3-Apprentissage-automatique/Unsupervised Learning/Autoencoders/auto_1_architecture.png|500]]
*Figure 1. Architecture vanilla AE. L'entrée $x \in \mathbb{R}^d$ passe par l'encodeur (taille décroissante), atteint le bottleneck $z \in \mathbb{R}^k$ avec $k \ll d$, puis remonte par le décodeur (taille croissante) jusqu'à $\hat{x} \in \mathbb{R}^d$.*

En pratique on empile plusieurs couches (encodeur et décodeur deviennent profonds) — c'est le **stacked autoencoder**, l'architecture standard.

### (ii) Hyperparamètres

Quatre choix structurels :

1. **Taille du code** $k$ — plus petit = compression plus forte = bottleneck plus serré.
2. **Profondeur** — l'encodeur et le décodeur peuvent avoir autant de couches qu'on veut.
3. **Largeurs des couches** — en général décroissantes côté encodeur, croissantes côté décodeur (symétrie).
4. **Loss** — MSE ou cross-entropy selon le type de données.

### (iii) Loss : MSE vs cross-entropy

**Pour des entrées réelles continues** (images en niveaux de gris non normalisées, signaux…) : **MSE**.

$$
\mathcal{L}_{\text{MSE}}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^n \big\|x^{(i)} - f_\theta\big(g_\phi(x^{(i)})\big)\big\|_2^2.
$$

**Pour des entrées binaires ou dans $[0,1]$** (pixels normalisés, masques binaires) : **cross-entropy** :

$$
\mathcal{L}_{\text{CE}}(\theta, \phi) = -\frac{1}{n} \sum_{i=1}^n \sum_k\, \Big[\,x^{(i)}_k \log \hat{x}^{(i)}_k + (1 - x^{(i)}_k) \log(1 - \hat{x}^{(i)}_k)\,\Big].
$$

Dans ce cas, l'activation finale du décodeur doit être une sigmoïde (pour que $\hat{x} \in [0, 1]^d$).

> [!note] Gradient identique au signe près
> Pour les deux losses, le gradient par rapport à la **pré-activation** $\hat{a}$ de la couche de sortie vaut
> $$\frac{\partial \mathcal{L}}{\partial \hat{a}^{(i)}} = \hat{x}^{(i)} - x^{(i)}.$$
> Le reste de la backprop est identique — c'est pour ça qu'on dit que MSE et CE *"donnent le même gradient"* sur le dernier layer. Pratique : on peut switcher de loss sans toucher au reste du code.

### (iv) Tied weights — l'option historique

Une astuce historique : forcer $W' = W^\top$, c'est-à-dire utiliser la **même matrice de poids** (à transposition près) pour l'encodeur et le décodeur. On parle de **tied weights** (poids liés).

$$
\hat{x} = \sigma\big(W^\top \sigma(W x + b) + c\big).
$$

**Avantages** : moitié moins de paramètres, régularisation implicite, et — comme on va le voir — **lien direct avec la PCA**.

**Inconvénients** : moins de flexibilité, plus utilisé dans les architectures modernes (variations type VAE, transformers, etc. n'utilisent pas tied weights).

---

## 2. Capacité du bottleneck

L'idée centrale de l'AE c'est le **bottleneck** : forcer l'information à passer par un espace plus petit pour qu'elle se compresse. Mais qu'est-ce qui empêche le réseau d'apprendre l'**identité** ($\hat{x} = x$ pour tout $x$) si la capacité est suffisante ?

### (i) Undercomplete : la compression forcée

Un AE est **undercomplete** quand la dimension du code est plus petite que celle de l'entrée : $k < d$.

![[images/3-Apprentissage-automatique/Unsupervised Learning/Autoencoders/auto_2_undercomplete.png|350]]
*Figure 2. AE undercomplete : le bottleneck est plus étroit que l'entrée. Le réseau **ne peut pas** apprendre l'identité — il est forcé de compresser. Cette compression doit retenir l'information utile à la reconstruction, donc l'AE apprend des features pertinentes.*

C'est le cas qui nous intéresse pour la **réduction de dimension** : l'undercomplete force naturellement le réseau à découvrir une structure dans les données.

### (ii) Overcomplete : pas d'utilité directe

Un AE est **overcomplete** quand $k \geq d$.

![[images/3-Apprentissage-automatique/Unsupervised Learning/Autoencoders/auto_3_overcomplete.png|350]]
*Figure 3. AE overcomplete : le bottleneck est aussi large (ou plus) que l'entrée. Le réseau peut trivialement apprendre l'identité — il suffit de "copier" chaque entrée dans une coordonnée du code.*

> [!warning] L'overcomplete pur n'apprend rien d'utile
> Sans contrainte supplémentaire, un AE overcomplete peut atteindre une loss de zéro en apprenant simplement la fonction identité. Les features cachées n'ont aucune raison d'être informatives. Pour rendre un overcomplete utile, **il faut ajouter une contrainte** : sparsité, bruit (denoising, voir §4), régularisation contractive, etc.

### (iii) Le trade-off général

Plus largement, le bon AE équilibre deux pressions opposées :

- **Sensibilité à l'entrée** suffisante pour reconstruire fidèlement (sinon $\hat{x}$ ne ressemble à rien).
- **Insensibilité aux fluctuations parasites** pour ne pas mémoriser le bruit (sinon overfitting, et le code n'est qu'un index déguisé).

La forme générique de la loss en pratique :

$$
\mathcal{L}(x, \hat{x}) + \eta \cdot \mathcal{R}(g_\phi, f_\theta)
$$

où $\mathcal{R}$ est un terme de régularisation (sparsité sur le code, pénalité sur les poids, bruit injecté, etc.). C'est ce qui sépare les variantes (sparse AE, contractive AE, denoising AE…).

---

## 3. Le résultat-clé : AE linéaire = PCA

### (i) Setup

On considère un AE **purement linéaire** (aucune non-linéarité) avec :

- **Encodeur** linéaire : $z = W_e x$, avec $W_e \in \mathbb{R}^{k \times d}$.
- **Décodeur** linéaire : $\hat{x} = W_d z$, avec $W_d \in \mathbb{R}^{d \times k}$.
- **Loss MSE** sur les données **centrées** : $\mathcal{L} = \frac{1}{n} \sum_i \|x^{(i)} - \hat{x}^{(i)}\|_2^2$.

En notation matricielle (avec $X \in \mathbb{R}^{n \times d}$ centrée, une ligne par observation) :

$$
\hat{X} = X W_e^\top W_d^\top, \qquad \mathcal{L} = \frac{1}{n}\, \|X - X W_e^\top W_d^\top\|_F^2.
$$

> [!warning] Convention de notation
> Je suis ici la convention SVD standard $X = U S V^\top$ de `[[01_Algèbre Linéaire]]` et `[[PCA]]`. Avec $X$ ligne-individus, l'encodeur en notation matricielle s'écrit $Z = X W_e^\top$ (chaque ligne de $Z$ est le code d'un individu). Les conventions de transposition varient selon les sources — l'important est que **le sous-espace appris** est ce qui compte, et il sera identique à celui de la PCA quelle que soit la convention choisie.

### (ii) Le problème devient une approximation de rang $k$

Posons $A = W_e^\top W_d^\top \in \mathbb{R}^{d \times d}$. Comme produit de deux matrices passant par une dimension intermédiaire $k$, $A$ a **rang au plus $k$**. Le problème devient :

$$
\min_{W_e, W_d}\, \|X - X A\|_F^2 = \min_{A : \text{rang}(A) \leq k}\, \|X - X A\|_F^2.
$$

Et $X A$ est une matrice de rang $\leq k$ également. Donc on cherche **la meilleure approximation de rang $\leq k$ de $X$** :

$$
\min_{\hat{X} : \text{rang}(\hat{X}) \leq k}\, \|X - \hat{X}\|_F^2.
$$

### (iii) Eckart-Young donne la solution

C'est exactement le théorème d'Eckart-Young (cf. `[[01_Algèbre Linéaire]]`, fin de section SVD). La solution optimale est la **troncature SVD** :

$$
\boxed{\;\hat{X}^\star = X_k = U_k S_k V_k^\top \quad \text{où } X = U S V^\top \text{ est la SVD de } X\;}
$$

avec $U_k$, $S_k$, $V_k$ les $k$ premières composantes singulières.

### (iv) Identification : $W_e^\top = V_k$

Reste à matcher $A^\star = W_e^\top W_d^\top$ avec la solution Eckart-Young. On a $X A^\star = X_k = U_k S_k V_k^\top$, et on sait que $X V_k = U_k S_k$ (depuis $X = U S V^\top \Rightarrow X V = U S$). Donc

$$
X A^\star = (X V_k)\, V_k^\top = X (V_k V_k^\top).
$$

Une solution explicite saute aux yeux :

$$
\boxed{\;W_e^\top = V_k, \qquad W_d^\top = V_k^\top\;}
$$

C'est-à-dire $W_e = V_k^\top$ et $W_d = V_k$. Conséquences :

- L'encodeur **projette** sur les $k$ premiers vecteurs propres de $\Sigma = X^\top X / n$ — exactement la matrice de projection de la PCA.
- Le décodeur **remonte** dans l'espace original via la transposée — c'est l'opération inverse de la projection.
- Les codes appris sont $Z = X V_k = U_k S_k$ — exactement les **scores** de la PCA (cf. `[[PCA]]`, §4).
- **Tied weights** $W_d = W_e^\top$ : automatiquement satisfait à l'optimum.

> [!important] L'AE linéaire entraîné optimalement converge vers la PCA
> Pas d'approximation, pas d'à-peu-près : **les poids optimaux d'un AE linéaire avec loss MSE sont exactement la projection PCA**. C'est une équivalence formelle, démontrée par Eckart-Young.

### (v) Unicité — la solution est en fait à rotation près

Petite subtilité : la solution n'est pas littéralement unique. Pour toute matrice orthogonale $Q \in \mathbb{R}^{k \times k}$,

$$
W_e^\top = V_k Q, \qquad W_d^\top = Q^\top V_k^\top
$$

donne la même reconstruction $\hat{X} = X V_k Q Q^\top V_k^\top = X V_k V_k^\top$. Donc l'AE linéaire peut converger vers **n'importe quelle base orthonormale du sous-espace PCA** — pas nécessairement la base "canonique" $V_k$ classée par variance décroissante.

**Conséquence pratique** : si tu compares les poids d'un AE linéaire entraîné avec ceux de `sklearn.PCA`, ils peuvent ne pas matcher composante par composante, mais ils engendrent le **même sous-espace**. La distinction n'est pas anodine pour l'interprétation (les composantes individuelles d'un AE linéaire n'ont pas nécessairement de variance maximale au sens PCA).

### (vi) Conséquence : ce que la non-linéarité ajoute

Si AE linéaire = PCA, alors **toute la valeur ajoutée des AE vient de la non-linéarité** $\sigma$. Avec des activations non-linéaires, l'encodeur peut apprendre une **projection sur une variété courbe** (manifold) plutôt qu'un sous-espace plat. C'est ce qui permet aux AE de traiter le swiss roll, les images naturelles, et plus largement tout ce qui ne vit pas sur un sous-espace linéaire.

> [!tip] Pourquoi on garde des AE même quand PCA existe
> Trois raisons :
> 1. **Non-linéarité** : on peut suivre des manifolds courbes.
> 2. **Profondeur** : empiler des couches permet d'apprendre des représentations hiérarchiques (low-level → high-level features).
> 3. **Flexibilité** : on peut conditionner l'AE (variables auxiliaires), le rendre génératif (VAE), le rendre robuste (DAE, CAE), etc. Aucune de ces extensions n'est aussi naturelle en PCA pure.

---

## 4. Denoising Autoencoders (DAE)

### (i) Motivation : rendre l'overcomplete utile

Le problème de l'AE overcomplete (cf. §2.ii) : il peut apprendre l'identité, donc ne rien apprendre. La **régularisation par le bruit** (Vincent et al., 2008–2010) est l'astuce qui résout ce problème.

L'idée : on **corrompt l'entrée** avec un bruit aléatoire, et on demande au réseau de reconstruire l'**entrée propre originale**. Le réseau ne peut plus se contenter de copier — il doit comprendre la structure des données pour "deviner" ce qui a été corrompu.

### (ii) Architecture

On note $\tilde{x}$ la version bruitée de $x$, obtenue par un processus stochastique $\tilde{x} \sim q(\tilde{x} \mid x)$. Le DAE minimise :

$$
\boxed{\;\mathcal{L}_{\text{DAE}}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^n \big\|x^{(i)} - f_\theta\big(g_\phi(\tilde{x}^{(i)})\big)\big\|_2^2\;}
$$

**Note importante** : la loss compare $\hat{x} = f_\theta(g_\phi(\tilde{x}))$ avec $x$ (l'original propre), **pas avec $\tilde{x}$** (l'entrée bruitée). C'est ce qui force le débruitage.

![[images/3-Apprentissage-automatique/Unsupervised Learning/Autoencoders/auto_4_denoising_schema.png|500]]
*Figure 4. Architecture DAE : l'entrée $x$ est d'abord corrompue en $\tilde{x}$ via un processus de bruit $q(\tilde{x} \mid x)$, puis passe par l'AE qui doit reconstruire le $x$ original. Le bottleneck peut être overcomplete sans que le réseau s'effondre sur l'identité — le bruit empêche cette stratégie triviale.*

### (iii) Types de bruit

Deux choix standards :

**Masking noise** (analogue à du dropout sur l'entrée) :

$$
\tilde{x}_j = \begin{cases} 0 & \text{avec probabilité } q \\ x_j & \text{avec probabilité } 1 - q \end{cases}
$$

Une fraction $q$ des coordonnées est mise à zéro aléatoirement. Le réseau doit apprendre à **reconstruire les valeurs manquantes** à partir des autres — donc à modéliser les dépendances entre coordonnées.

**Gaussian noise** :

$$
\tilde{x} = x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I).
$$

Plus naturel pour les données continues. $\sigma$ est un hyperparamètre — plus $\sigma$ est grand, plus la contrainte de débruitage est forte.

### (iv) Interprétation : projeter sur la manifold des données

L'interprétation géométrique est élégante. Supposons que les données vivent sur une **variété de basse dimension** (manifold) dans $\mathbb{R}^d$. Bruiter $x$ revient à le **pousser hors de la manifold**.

![[images/3-Apprentissage-automatique/Unsupervised Learning/Autoencoders/auto_5_manifold.png|450]]
*Figure 5. Interprétation manifold du DAE. Les données $x$ vivent sur une variété courbe (la ligne grise). Le bruit $\tilde{x}$ pousse les points hors de la variété (flèches grises). Le DAE apprend à **projeter $\tilde{x}$ vers $x$**, c'est-à-dire à retrouver le point original sur la variété (flèches bleues). En faisant cela à grande échelle sur tout le dataset, le réseau apprend la **forme de la variété**.*

C'est cette interprétation qui donne aux DAE leur pouvoir : **apprendre à débruiter = apprendre la géométrie locale des données**. Le DAE est en quelque sorte un proto-modèle génératif — il a appris à projeter sur le support des données. Cette idée mène directement aux **score matching** et aux **modèles de diffusion** (cf. `[[06_Score-Based & Diffusion Models|Score-Based & Diffusion]]`, où le résultat de Vincent 2011 sur le denoising score matching est exactement cette équivalence formalisée).

### (v) Exemples visuels

Sur les images, l'effet du DAE est spectaculaire — les filtres appris ressemblent à des **détecteurs d'arêtes** et de **traits de pinceau**, exactement le genre de features bas-niveau qui apparaissent dans les couches profondes des CNN supervisés. Plus le bruit est fort, plus les filtres deviennent globaux (car le réseau doit raisonner à plus grande échelle pour deviner les zones manquantes).

![[images/3-Apprentissage-automatique/Unsupervised Learning/Autoencoders/auto_6_filtres.png|550]]
*Figure 6. Filtres appris par un DAE sur des chiffres MNIST avec différents niveaux de bruit. À gauche ($q = 0$, AE classique overcomplete) : aucune structure visible, les filtres sont aléatoires. Au milieu ($q = 0.25$) : émergence de détecteurs d'arêtes et de traits courts. À droite ($q = 0.5$) : filtres plus larges qui captent des structures de plus grande échelle (parties de chiffres entiers).*

### (vi) Stacked DAE — un peu d'histoire

Le papier original ([Vincent, Larochelle, Bengio, Manzagol, JMLR 2010](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)) introduit le **stacked denoising autoencoder** : on empile plusieurs DAE entraînés couche par couche, chacun débruitant la sortie de l'encodeur précédent. C'était l'un des premiers schémas de **pré-entraînement non supervisé** utiles pour les réseaux profonds, dans une époque où la vanishing gradient empêchait l'entraînement direct.

Depuis ~2014 (l'arrivée de l'initialisation Xavier, des skip connections et des bonnes architectures), le stacked DAE n'est plus la norme — on entraîne directement les réseaux profonds en supervisé. Mais l'idée du bruit comme régularisation a essaimé :

- **Dropout** est essentiellement du masking noise appliqué aux **activations** plutôt qu'aux entrées.
- **Score matching** et **diffusion models** sont les héritiers conceptuels du DAE (cf. `[[06_Score-Based & Diffusion Models|06]]`).
- **BERT** et les transformers masqués reprennent l'idée *"corrompre une partie de l'entrée, reconstruire le reste"*.

---

## 5. Pour aller plus loin

Quelques variantes notables :

- **Sparse Autoencoder** — au lieu de réduire la taille du code, on garde un code overcomplete mais on **pénalise sa sparsité** (norme $L_1$ sur $z$, ou contrainte KL sur l'activation moyenne). Le code apprend à n'utiliser que quelques unités à la fois, donnant des features interprétables.
- **Contractive Autoencoder (CAE)** — on pénalise la **norme de Frobenius du Jacobien** de l'encodeur, $\|\partial g_\phi / \partial x\|_F^2$. Cela force l'encodeur à être **localement invariant** aux petites perturbations de $x$, sauf dans les directions qui suivent la manifold.
- **Variational Autoencoder (VAE)** — l'AE devient un **modèle génératif probabiliste**. L'encodeur produit une distribution $q_\phi(z \mid x)$ (typiquement gaussienne), l'objectif devient un ELBO. Voir `[[03_VAE]]` dans `Generative Models`.
- **Convolutional Autoencoder** — encodeur et décodeur sont des CNN. Standard pour les images, base architecturale de Stable Diffusion (le VAE de l'espace latent).
- **Masked Autoencoder (MAE)** — version moderne du DAE pour vision (He et al. 2021). On masque 75% des patches d'une image, le réseau (transformer) reconstruit les patches manquants. Préentraînement massivement scalable.

Toutes ces variantes partagent la même structure encodeur–décodeur de base. Le passage de l'AE déterministe au VAE marque la frontière vers les **modèles génératifs** : à partir de là, on quitte le domaine "réduction de dimension" pour rentrer dans `Generative Models`.
