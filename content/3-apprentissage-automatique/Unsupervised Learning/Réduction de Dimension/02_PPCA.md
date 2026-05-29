---
title: PPCA (Probabilistic PCA)
---
# PPCA — Probabilistic PCA

> Cette note prolonge `[[PCA]]` en lui donnant un cadre **probabiliste**. PCA classique cherche la projection linéaire de variance maximale (problème purement algébrique : SVD de la matrice de covariance). PPCA reformule la même question comme **inférence dans un modèle génératif à variable latente continue** : $z \sim \mathcal{N}(0, I)$, $x = Wz + \mu + \epsilon$. Le résultat-clé est que les directions optimales **redonnent les composantes principales** (à une rotation près) — mais on gagne au passage une distribution sur $z$, une vraie likelihood $p(x)$, et tout le bagage probabiliste : valeurs manquantes, model selection, génération de nouveaux $x$.
>
> C'est aussi le **pont conceptuel direct vers le VAE** (`[[02_VAE]]`) : remplacer la transformation linéaire $f(z) = Wz + \mu$ par un réseau de neurones non-linéaire $f_\theta(z)$, et on obtient un VAE. Tout le reste est identique structurellement.

## Conventions de notation

> [!warning] Notation tenue partout dans cette note
> | Symbole | Sens |
> |---|---|
> | $x \in \mathbb{R}^D$ | Variable observée (dimension haute) |
> | $z \in \mathbb{R}^M$ | Variable latente (dimension réduite, $M < D$) |
> | $W \in \mathbb{R}^{D \times M}$ | Matrice de transformation latent → observé (paramètre) |
> | $\mu \in \mathbb{R}^D$ | Moyenne globale (paramètre) |
> | $\sigma^2$ | Variance du bruit isotropique (paramètre) |
> | $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ | Bruit gaussien isotropique |
> | $C = WW^\top + \sigma^2 I$ | Covariance de la marginale $p(x)$ |
> | $M_\text{matrix} = W^\top W + \sigma^2 I$ | Matrice $M \times M$ utile pour la posterior (attention, notation Bishop classique mais collision avec la dimension $M$) |
> 
> **Paramètres à apprendre** : $\theta = (W, \mu, \sigma^2)$.

## I. Motivation : pourquoi rendre PCA probabiliste ?

PCA classique fait très bien son travail — extraire les directions de variance maximale via SVD. Mais en l'état, c'est un objet **purement géométrique**. Pas de likelihood, pas de génération, pas de bayésien possible. Rendre PCA probabiliste débloque plusieurs choses qui sont impossibles en PCA standard :

1. **Une vraie likelihood $p(x)$.** Ça permet la *model selection* (comparer plusieurs $M$ par BIC/AIC/cross-validation), et de détecter les outliers (points à très faible $p(x)$).
2. **Gérer les valeurs manquantes.** En PCA classique, si une coordonnée de $x_n$ manque, on est coincé. En PPCA, on traite la coordonnée manquante comme une variable latente additionnelle, et on l'intègre.
3. **Génération de nouveaux $x$.** Tirer $z \sim p(z)$ puis $x \sim p(x \mid z)$ donne un point synthétique cohérent avec le modèle.
4. **Pont conceptuel vers VAE.** PPCA = "VAE linéaire". Comprendre PPCA, c'est comprendre 80% de la mécanique d'un VAE.
5. **Cadre bayésien.** On peut mettre des priors sur $W, \mu, \sigma^2$ et faire de l'inférence bayésienne (Bayesian PCA, qui détermine automatiquement le bon $M$).

> [!note] Le résultat-clé spoiler
> À $\sigma^2$ fixé, la solution MLE de PPCA pour $W$ vaut $W_\text{ML} = U_M (\Lambda_M - \sigma^2 I)^{1/2} R$ où $U_M$ est la matrice des $M$ premiers vecteurs propres de la covariance empirique $S$, $\Lambda_M$ est la matrice diagonale de leurs valeurs propres, et $R$ est une rotation arbitraire. **On retrouve les composantes principales de PCA classique** (à une rotation près, et avec un facteur d'échelle ajusté pour le bruit).
> 
> Tout le travail mathématique de §II–§V consiste à dériver ce résultat et à comprendre ce qu'il signifie.

## II. Le modèle génératif

### A. Les briques

**Variable latente** : $z \in \mathbb{R}^M$ vit dans un espace de dimension réduite. On lui met un prior **gaussien standard isotropique** :

$$\boxed{\;p(z) = \mathcal{N}(z \mid 0, I)\;}$$

**Transformation latent → observé** : on applique une transformation **linéaire** $W \in \mathbb{R}^{D \times M}$, on translate par $\mu \in \mathbb{R}^D$, et on ajoute du bruit isotropique gaussien $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ :

$$x = Wz + \mu + \epsilon.$$

Ce qui donne la **conditionnelle** :

$$\boxed{\;p(x \mid z) = \mathcal{N}(x \mid Wz + \mu, \sigma^2 I)\;}$$

Et le bruit lui-même :

$$\boxed{\;p(\epsilon) = \mathcal{N}(\epsilon \mid 0, \sigma^2 I)\;}$$

### B. Intuition visuelle

Le modèle se lit en deux étapes :

1. **Tirer $z$** dans le petit espace $\mathbb{R}^M$ selon le prior $\mathcal{N}(0, I)$ — une simple gaussienne sphérique.
2. **Mapper $z$ vers $\mathbb{R}^D$** : la transformation $Wz + \mu$ envoie $z$ sur un point d'un **hyperplan affine** (sous-espace de dimension $M$ translaté par $\mu$) plongé dans $\mathbb{R}^D$. Puis on ajoute un nuage gaussien isotropique de variance $\sigma^2 I$ autour de ce point.

![[ppca_generative_model.png]]
*Figure. Le modèle génératif de PPCA en image (cas $D = 2$, $M = 1$). **Gauche** : le prior $p(z) = \mathcal{N}(0, 1)$ sur l'espace latent 1D. On tire $\hat{z}$. **Droite** : l'espace observé $\mathbb{R}^2$. La transformation $z \mapsto Wz + \mu$ trace une **droite** (l'hyperplan affine de dim $M = 1$). Pour $\hat{z}$ donné, on obtient un point $W\hat{z} + \mu$ sur cette droite, puis le bruit isotropique $\sigma^2 I$ crée le nuage circulaire autour. **La marginale $p(x)$** est une gaussienne 2D allongée le long de la droite, de covariance $WW^\top + \sigma^2 I$.*

### C. Qu'est-ce qu'on cherche ?

Étant donné un dataset $X = \{x_1, \ldots, x_N\}$, on veut **trouver les paramètres** $\theta = (W, \mu, \sigma^2)$ qui maximisent la log-vraisemblance :

$$\theta^* = \arg\max_\theta \sum_{n=1}^{N} \log p(x_n \mid \theta).$$

C'est du **MLE classique** sur un modèle à variable latente — exactement le setup d'EM (§II.A de `[[05_Expectation_Maximization]]`). On verra en §V que pour PPCA, on a la chance d'avoir une **forme fermée** (pas besoin d'EM), mais EM marche aussi et est utile en pratique (§VI).

## III. La marginale $p(x)$ — calcul en forme fermée

### A. Pourquoi c'est calculable

La conjugaison gaussienne sauve la mise : prior gaussien $p(z)$ + likelihood gaussienne $p(x \mid z)$ linéaire en $z$ → marginale $p(x) = \int p(x \mid z) p(z) \, dz$ **encore gaussienne**. Pas d'intégrale en haute dimension à craindre.

Au lieu de faire l'intégrale tedious, on calcule directement la moyenne et la covariance.

### B. Moyenne de $p(x)$

$$\mathbb{E}[x] = \mathbb{E}[Wz + \mu + \epsilon] = W \underbrace{\mathbb{E}[z]}_{=0} + \mu + \underbrace{\mathbb{E}[\epsilon]}_{=0} = \mu.$$

Logique : la moyenne globale du modèle, c'est $\mu$.

### C. Covariance de $p(x)$

$$
\begin{aligned}
\text{Cov}[x] &= \mathbb{E}\big[(x - \mu)(x - \mu)^\top\big] \\
&= \mathbb{E}\big[(Wz + \epsilon)(Wz + \epsilon)^\top\big] \quad \text{(car } x - \mu = Wz + \epsilon\text{)} \\
&= \mathbb{E}\big[Wzz^\top W^\top + 2 W z \epsilon^\top + \epsilon \epsilon^\top\big] \\
&= W \underbrace{\mathbb{E}[zz^\top]}_{(\star)} W^\top + 2 W \underbrace{\mathbb{E}[z \epsilon^\top]}_{(\star\star)} + \underbrace{\mathbb{E}[\epsilon \epsilon^\top]}_{(\star\star\star)} \\
&= W \, I \, W^\top + 0 + \sigma^2 I \\
&= W W^\top + \sigma^2 I.
\end{aligned}
$$

Avec les trois propriétés utilisées :

- **$(\star)$** : $z \sim \mathcal{N}(0, I)$, donc $\text{Cov}(z) = \mathbb{E}[zz^\top] - \mathbb{E}[z]\mathbb{E}[z]^\top = I - 0 = I$, donc $\mathbb{E}[zz^\top] = I$.
- **$(\star\star)$** : $z \perp \epsilon$ par hypothèse, donc $\mathbb{E}[z\epsilon^\top] = \mathbb{E}[z]\mathbb{E}[\epsilon]^\top = 0$.
- **$(\star\star\star)$** : $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, donc par le même argument que $(\star)$, $\mathbb{E}[\epsilon \epsilon^\top] = \sigma^2 I$.

### D. Résultat

$$\boxed{\;p(x) = \mathcal{N}(x \mid \mu, C), \quad \text{avec } C = WW^\top + \sigma^2 I.\;}$$

> [!warning] Lecture de la covariance $C = WW^\top + \sigma^2 I$
> Deux contributions séparées et identifiables :
> - **$WW^\top$** : covariance dans les directions **structurelles** (les colonnes de $W$). Rang $M$ — capture la variance "vraie" du signal dans le sous-espace latent.
> - **$\sigma^2 I$** : covariance **isotropique** dans toutes les directions, y compris orthogonales au sous-espace. Capture le bruit.
> 
> Quand $\sigma^2 \to 0$, $C$ devient singulière de rang $M$ — toute la masse de $p(x)$ s'effondre sur l'hyperplan affine $\mu + \text{vect}(W)$. C'est la **limite PCA classique** (§VII).

## IV. La posterior $p(z \mid x)$ — combler le `???`

### A. Forme fermée (toujours grâce à la conjugaison gaussienne)

Par les règles standard de conditionnement gaussien (cf. Bishop §2.3.3), la posterior se calcule analytiquement :

$$\boxed{\;p(z \mid x) = \mathcal{N}\!\left(z \;\Big|\; \tilde{M}^{-1} W^\top (x - \mu),\; \sigma^2 \tilde{M}^{-1}\right)\;}$$

où

$$\tilde{M} := W^\top W + \sigma^2 I \in \mathbb{R}^{M \times M}.$$

(C'est la matrice qu'on notait $M$ dans ton cours et chez Bishop, mais ici je note $\tilde{M}$ pour ne pas confondre avec la dimension latente $M$.)

### B. Que signifie cette posterior ?

C'est la question que ton cours a laissée ouverte. Décortiquons.

**La moyenne** $\mathbb{E}[z \mid x] = \tilde{M}^{-1} W^\top (x - \mu)$ donne **la projection** de $x$ (centré par $\mu$) sur l'espace latent. C'est l'équivalent probabiliste du "score" en PCA : la coordonnée de $x$ dans la base latente.

- Quand $\sigma^2 \to 0$ : $\tilde{M} \to W^\top W$, donc $\mathbb{E}[z \mid x] \to (W^\top W)^{-1} W^\top (x - \mu)$. C'est **exactement la pseudo-inverse de Moore-Penrose** appliquée à $(x - \mu)$, c'est-à-dire la projection orthogonale sur le sous-espace de $W$. **PCA classique**.
- Quand $\sigma^2$ est grand : $\tilde{M} \to \sigma^2 I$, donc $\mathbb{E}[z \mid x] \to \frac{1}{\sigma^2} W^\top (x - \mu)$. La projection est "shrinkée" vers 0 — l'incertitude domine, et le modèle préfère prédire un $z$ petit (proche du prior).

**La covariance** $\text{Cov}[z \mid x] = \sigma^2 \tilde{M}^{-1}$ donne **l'incertitude** sur la position dans l'espace latent :

- Quand $\sigma^2 \to 0$ : variance $\to 0$, on connaît $z$ parfaitement (en PCA classique, il n'y a aucune incertitude — chaque $x$ a une projection unique).
- Quand $\sigma^2$ est grand : variance grande, $z$ est très incertain pour un $x$ donné — beaucoup de configurations latentes auraient pu produire ce $x$.

![[ppca_posterior_z_given_x.png]]
*Figure. La posterior $p(z \mid x)$ pour un point $x$ donné (cas $D = 2$, $M = 1$). **Gauche** : la droite latente (hyperplan affine de dim 1 dans l'espace observé $\mathbb{R}^2$), le point $x$ observé, et la projection orthogonale $W \mathbb{E}[z \mid x] + \mu$ sur la droite. **Droite** : la distribution $p(z \mid x)$ sur l'espace latent 1D — gaussienne centrée sur $\mathbb{E}[z \mid x]$, avec une variance $\sigma^2 / (W^\top W + \sigma^2)$ qui mesure à quel point la projection est incertaine.*

> [!note] Lien avec EM / VI
> Tu reconnais ici la **même structure conceptuelle** qu'en `[[05_Expectation_Maximization]]` : la posterior $p(z \mid x)$ est exactement la quantité qu'on calcule au E-step. En PPCA, on a la chance qu'elle soit **calculable en forme fermée** (parce que le modèle est gaussien conjugué). Donc PPCA = cas particulier où EM est trivialement applicable — pas besoin de VI. Dès qu'on remplace la transformation linéaire $Wz + \mu$ par un réseau neuronal $f_\theta(z)$ (→ VAE), on perd la conjugaison et il faut passer à VI (`[[06_Variational_Inference]]`).

## V. MLE en forme fermée

### A. La log-vraisemblance

$$
\begin{aligned}
\log p(X \mid \mu, W, \sigma^2) &= \sum_{n=1}^{N} \log p(x_n \mid \mu, W, \sigma^2) = \sum_{n=1}^{N} \log \mathcal{N}(x_n \mid \mu, C) \\
&= -\frac{ND}{2} \log(2\pi) - \frac{N}{2} \log |C| - \frac{1}{2} \sum_{n=1}^{N} (x_n - \mu)^\top C^{-1} (x_n - \mu).
\end{aligned}
$$

On va dériver par rapport à chacun des trois paramètres $\mu$, $W$, $\sigma^2$ et résoudre.

### B. Estimateur de $\mu$

$$\sum_{n=1}^{N} C^{-1} (x_n - \mu) = 0 \quad\Longrightarrow\quad \mu_\text{ML} = \frac{1}{N} \sum_{n=1}^{N} x_n.$$

Sans surprise : **moyenne empirique du dataset**. C'est valable indépendamment de $W$ et $\sigma^2$.

### C. Estimateur de $W$ — le résultat de Tipping & Bishop (1999)

Soit $S = \frac{1}{N} \sum_n (x_n - \mu_\text{ML})(x_n - \mu_\text{ML})^\top$ la **matrice de covariance empirique**. On diagonalise $S = U \Lambda U^\top$ avec $\Lambda = \text{diag}(\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_D)$.

Le maximum de vraisemblance pour $W$ s'écrit alors :

$$\boxed{\;W_\text{ML} = U_M \big(\Lambda_M - \sigma^2 I\big)^{1/2} R\;}$$

où :
- $U_M \in \mathbb{R}^{D \times M}$ est la matrice des $M$ premiers vecteurs propres de $S$ (en colonnes).
- $\Lambda_M = \text{diag}(\lambda_1, \ldots, \lambda_M)$ est la diagonale des $M$ plus grandes valeurs propres.
- $R \in \mathbb{R}^{M \times M}$ est une **matrice orthogonale arbitraire** ($R R^\top = I$).

**Et l'estimateur de $\sigma^2$** :

$$\boxed{\;\sigma^2_\text{ML} = \frac{1}{D - M} \sum_{j=M+1}^{D} \lambda_j\;}$$

C'est la **moyenne des valeurs propres laissées de côté** (les $D - M$ plus petites). Interprétation : on identifie ce qui ne rentre pas dans les $M$ directions principales comme du bruit, et on l'estime par sa variance moyenne.

> [!warning] La rotation arbitraire $R$
> $W_\text{ML}$ n'est défini qu'**à une rotation près** : $W$ et $W R$ donnent la même log-vraisemblance, parce que la marginale $p(x) = \mathcal{N}(\mu, WW^\top + \sigma^2 I)$ ne dépend de $W$ que via $WW^\top$, et $(WR)(WR)^\top = WR R^\top W^\top = WW^\top$.
> 
> **Conséquence** : il n'y a pas une seule solution mais une famille équivalente. En général on prend $R = I$ pour simplifier, ce qui aligne les colonnes de $W$ avec les vecteurs propres de $S$. Mais le sous-espace engendré par $W$ — celui qui compte vraiment pour la modélisation — est unique.

### D. Interprétation : décomposition spectrale signal + bruit

C'est l'intuition la plus parlante du résultat MLE. On veut faire matcher la covariance du modèle $C = WW^\top + \sigma^2 I$ avec la covariance empirique $S = U \Lambda U^\top$. Décomposons $S$ en deux parties :

$$
\underbrace{S}_{\text{empirique}} = \underbrace{U_M \Lambda_M U_M^\top}_{\textcolor{green}{\text{signal}}} + \underbrace{U_- \Lambda_- U_-^\top}_{\textcolor{orange}{\text{bruit (queues)}}}
$$

où $U_M, \Lambda_M$ sont les top-$M$ et $U_-, \Lambda_-$ sont les $D - M$ restantes.

**Le modèle PPCA** essaie d'attraper le signal via $WW^\top$ et le bruit via $\sigma^2 I$. Idéalement :

$$
\textcolor{green}{C_\text{signal}} = WW^\top + \sigma^2 I_M = U_M (\Lambda_M - \sigma^2 I + \sigma^2 I) U_M^\top = U_M \Lambda_M U_M^\top \;\checkmark
$$

(où $WW^\top = U_M (\Lambda_M - \sigma^2 I) U_M^\top$ vient directement de la formule MLE en élevant $W_\text{ML}$ au carré). Le $\sigma^2 I$ "rattrape" ce qu'il faut pour faire l'égalité exacte.

Pour les directions hors signal, le modèle prédit $\sigma^2 I$ (bruit isotropique), tandis que l'empirique a $U_- \Lambda_- U_-^\top$ — la moyenne des $\lambda_j$ ($j > M$) est précisément la meilleure approximation isotropique des queues, d'où la formule $\sigma^2 = \frac{1}{D-M} \sum_{j=M+1}^D \lambda_j$.

![[ppca_spectrum_decomposition.png]]
*Figure. Spectre des valeurs propres de $S$. **En vert** : les $M$ plus grandes valeurs propres — le signal capturé par $WW^\top$. **En orange** : les $D - M$ plus petites — modélisées par $\sigma^2 I$, avec $\sigma^2$ égal à leur moyenne. Plus on réduit la dimension $M$, plus la barre verte rétrécit, plus la barre orange s'étend et $\sigma^2$ grossit. La hauteur jaune correspond à la moyenne des valeurs propres restantes.*

## VI. EM pour PPCA — quand la forme fermée n'est plus pratique

On a une forme fermée en §V, donc pourquoi parler d'EM ? Deux raisons concrètes :

1. **Grande dimension $D$** : calculer la SVD de $S \in \mathbb{R}^{D \times D}$ coûte $\mathcal{O}(D^3)$. Pour $D$ très grand (images, texte vectorisé), c'est rédhibitoire. EM, lui, ne manipule que des matrices de taille $D \times M$ avec $M \ll D$.
2. **Valeurs manquantes** : si certains $x_n$ ont des coordonnées non observées, la forme fermée ne marche plus directement. EM les traite naturellement (on les intègre comme des latents additionnels).

> [!note]- Algorithme EM pour PPCA
> **E-step** : pour chaque $x_n$, calculer la posterior $p(z_n \mid x_n)$ — gaussienne, forme fermée (§IV) :
> 
> $$\mathbb{E}[z_n \mid x_n] = \tilde{M}^{-1} W^\top (x_n - \mu), \qquad \mathbb{E}[z_n z_n^\top \mid x_n] = \sigma^2 \tilde{M}^{-1} + \mathbb{E}[z_n \mid x_n] \mathbb{E}[z_n \mid x_n]^\top.$$
> 
> **M-step** : maximiser l'espérance de la log-jointe par rapport à $W$ et $\sigma^2$ (rappel : $\mu$ est fixé à $\bar{x}$ une fois pour toutes). Les updates :
> 
> $$W^\text{new} = \left[\sum_n (x_n - \mu) \mathbb{E}[z_n \mid x_n]^\top\right] \left[\sum_n \mathbb{E}[z_n z_n^\top \mid x_n]\right]^{-1},$$
> 
> $$\sigma^{2, \text{new}} = \frac{1}{ND} \sum_n \left\{\|x_n - \mu\|^2 - 2 \mathbb{E}[z_n \mid x_n]^\top W^{\text{new}\,\top} (x_n - \mu) + \text{tr}\!\left(\mathbb{E}[z_n z_n^\top \mid x_n] W^{\text{new}\,\top} W^\text{new}\right)\right\}.$$
> 
> **Itérer** jusqu'à convergence de l'ELBO. Comme PPCA est un modèle conjugué gaussien, l'ELBO atteint $\log p(x)$ à convergence (KL = 0), donc EM redonne **exactement** la même solution que la forme fermée — juste par un autre chemin.
> 
> Coût par itération : $\mathcal{O}(NDM)$ au lieu de $\mathcal{O}(D^3)$. Avantage massif quand $D$ est grand et $M$ petit.

## VII. Limites et place dans la généalogie

### A. Limite $\sigma^2 \to 0$ : on retombe sur PCA

Quand on prend $\sigma^2 \to 0$ dans la solution MLE :

- $W_\text{ML} \to U_M \Lambda_M^{1/2} R$ — les colonnes de $W$ sont proportionnelles aux vecteurs propres de $S$, mises à l'échelle par la racine des valeurs propres.
- La posterior $p(z \mid x)$ se réduit à un Dirac centré sur la projection orthogonale de $x$ sur le sous-espace.
- La marginale $p(x)$ devient singulière (covariance de rang $M$) — toute la masse sur l'hyperplan.

**On retrouve PCA classique** : projection déterministe sur le sous-espace de variance maximale. PPCA est donc une **généralisation stricte** de PCA — qui ajoute une notion de bruit explicite et qui donne un cadre probabiliste.

![[ppca_sigma_effect.png]]
*Figure. Effet de $\sigma^2$ sur la marginale $p(x)$ en 2D, avec $M = 1$ (sous-espace 1D = une droite). **Gauche** : $\sigma^2$ petit — la gaussienne 2D est très allongée, presque collée sur la droite. **Milieu** : $\sigma^2$ modéré — la gaussienne s'épaissit perpendiculairement. **Droite** : $\sigma^2$ grand — la gaussienne devient presque isotropique, le signal latent est noyé. La limite $\sigma^2 \to 0$ donne PCA classique (ellipse aplatie sur la droite).*

### B. Pourquoi PPCA est le pont vers VAE

Récapitulons la structure :

$$x = f(z) + \epsilon, \quad z \sim \mathcal{N}(0, I), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I).$$

En PPCA : $f(z) = Wz + \mu$, **linéaire**. Tout est tractable : marginale gaussienne, posterior gaussienne, MLE en forme fermée.

En VAE : $f(z) = f_\theta(z)$ est un **réseau de neurones** non-linéaire. Ce qui pète :
- La marginale $p(x) = \int p(x \mid z) p(z) \, dz$ n'a plus de forme fermée.
- La posterior $p(z \mid x)$ non plus.
- → On doit passer par **VI** : approximer $p(z \mid x)$ par un $q_\phi(z \mid x)$ paramétrique (un autre réseau de neurones — l'encodeur).

> [!warning] Place dans la généalogie
> $$
> \underbrace{\text{PCA}}_{\substack{\text{déterministe} \\ \text{SVD}}} \;\xrightarrow{\text{ajoute du bruit gaussien}}\; \underbrace{\text{PPCA}}_{\substack{f(z) = Wz + \mu \\ \text{tout en forme fermée}}} \;\xrightarrow{f \text{ devient un NN}}\; \underbrace{\text{VAE}}_{\substack{f_\theta(z) \text{ non-linéaire} \\ p(z\mid x) \text{ approximée par VI}}}
> $$
> 
> PPCA est l'**étape intermédiaire conceptuelle** qui clarifie tout. Si tu comprends PPCA, tu comprends pourquoi VAE *doit* utiliser VI (la non-linéarité tue la conjugaison gaussienne), et pourquoi l'encodeur d'un VAE imite exactement la posterior $p(z \mid x)$ qu'on calculait analytiquement en PPCA.

### C. Quand utiliser PPCA en pratique ?

- **Compression / dénoising** : $M$ petit, on garde le signal et on jette le bruit.
- **Imputation de valeurs manquantes** : grâce à EM (§VI).
- **Modélisation probabiliste de signaux gaussiens** où la covariance a une structure low-rank.
- **Sanity check avant de partir sur du VAE** : si PPCA marche bien sur tes données, un VAE n'apportera probablement pas grand-chose. Si PPCA rate complètement, c'est que le problème est non-linéaire — VAE devient pertinent.

---

## Pour aller plus loin

- **Bishop, *Pattern Recognition and Machine Learning*, §12.2.** La référence canonique pour PPCA. La dérivation MLE complète est faite, ainsi qu'EM.
- **Tipping & Bishop (1999).** *Probabilistic Principal Component Analysis.* L'article fondateur où la solution forme fermée est obtenue pour la première fois.
- **Roweis (1998).** *EM Algorithms for PCA and SPCA.* Le premier à proposer EM pour PCA, avec des arguments computationnels.
- **Murphy, *Probabilistic Machine Learning: Advanced Topics*, §28.3.** Vue moderne, lien avec Factor Analysis (la généralisation où $\sigma^2 I$ devient $\Psi$ diagonale non-isotropique).
- **Kingma & Welling (2013).** *Auto-Encoding Variational Bayes.* Voir comment PPCA se généralise en VAE via VI.
