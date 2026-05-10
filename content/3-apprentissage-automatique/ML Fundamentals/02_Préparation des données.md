# (ii) Préparation des données

> Tout ce qu'on fait sur les données **avant** que le modèle ne les voie. C'est l'étape qui n'apparaît jamais dans les cours d'algos mais qui occupe 80 % du temps en pratique.

---

## 1. Preprocessing

### 1.1 Scaling

Beaucoup de modèles sont sensibles à l'échelle des features : régression régularisée (ridge, lasso), KNN, SVM, NN. Sans scaling, une feature en milliers (revenu annuel) domine une feature en unités (âge), même si la seconde est plus informative.

- **`StandardScaler`** : centre-réduit, $x' = (x - \mu) / \sigma$. Le plus courant.
- **`MinMaxScaler`** : ramène dans $[0, 1]$, $x' = (x - x_{min}) / (x_{max} - x_{min})$.
- **`RobustScaler`** : utilise médiane et IQR plutôt que moyenne et $\sigma$. Robuste aux outliers.

> **Modèles insensibles au scaling.** Arbres, random forests, gradient boosting — chaque split se fait sur une seule feature, l'échelle n'a pas d'importance.

### 1.2 Encoding

Les features catégorielles doivent être transformées en numérique.

- **One-hot encoding** : une colonne 0/1 par catégorie. Standard. Inconvénient : explosion du nombre de colonnes si cardinalité élevée.
- **Ordinal encoding** : entier par catégorie quand l'ordre a un sens (small/medium/large → 0/1/2).
- **Label encoding** : entier arbitraire. À éviter sauf pour les arbres (les autres modèles vont interpréter les entiers comme un ordre).
- **Target encoding** (mean encoding) : remplacer chaque catégorie par la moyenne de $Y$ pour cette catégorie. Puissant pour les hautes cardinalités, mais **grosse source de data leakage** — doit être fait dans la CV, pas avant.
- **Embeddings** : représentation apprise (cardinalité très élevée, NLP, recommandation).

> **Lien avec data leakage.** Tout transformer (scaler, target encoder, etc.) doit être **fitté sur le train uniquement** et appliqué au test. En sklearn : utiliser `Pipeline` rend cette discipline automatique.


One-hot encoding : 


| Initial Table                             | One-hot encoded "Favorite Color" Feature  |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260510171259.png\|197]] | ![[Pasted image 20260510171355.png\|262]] |

Label encoding : Une limite de ce modèle c'est que les nombres qu'on a choisi son arbitraire et ta des algorithmes de machine learning will treat the order of the numbers as if they might mean something, and that can cause problems.


| Initial Table                             | Label encoding "Favorite Color" Feature   |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260510171259.png\|197]] | ![[Pasted image 20260510171523.png\|196]] |


Target encoding : On va assigner à chaque label de "Favorite color" une valeur qui est proportionnelle à la cible. Pour "Blue" on a deux "No" et un seul "Yes" donc on va encoder 1/3=0.33; pour "Red" uniquement un seul "No" donc on assigne 0 etc.

Because we used the Target, the thing we want to predict, to determinite what values to replace the discrete options, this method is called Target Encoding.

| Initial Table                             | Target encoding "Favorite Color" Feature  |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260510171259.png\|197]] | ![[Pasted image 20260510171843.png\|200]] |

Un des problèmes de cette méthode est que less data supports the value we replaced Red with, we have less confidence that we replaced Red with the best value then we have for Blue and Green. So in order to deal with this Target Encoding is usually done using a Weighted Mean. Le mec te dit ça s'appelle aussi Bayesian Mean Encoding

Sinon on peut utiliser le weighted Mean

$$
\text { Weighted Mean }=\frac{n \times \text { Option Mean }+m \times \text { Overall Mean }}{n+m}
$$
$n=$ Weight for Option Mean (usually the number of rows)
$m=$ Weight for Overall Mean (user defined)


Remarque: ON peut noter qu'on a utilisé la cible pour construire notre feature data leakage results in models taht work great with training data, but not so well with testing data. In other words, data leakage results in models that are overfit. La good news est qu'il y'a plusieurs mtéhode qui permettent de réduire ce data leakage so that you can use Target Encoding without overfitting your model.

![[Pasted image 20260510172537.png|173]]





---

## 2. Valeurs manquantes

### 2.1 Modèles qui gèrent nativement les NaN

- **XGBoost, LightGBM, CatBoost** : choisissent la meilleure direction (gauche/droite) pour les NaN à chaque split. Pas besoin d'imputer.
- **Random forests classiques (sklearn)** : non, il faut imputer (ou utiliser des variantes spécifiques).
- **Régression linéaire/logistique, NN, SVM, KNN** : non, il faut imputer.

### 2.2 Stratégies d'imputation

*À compléter si besoin.*

- **Suppression** : drop des lignes ou colonnes. OK si peu de NaN.
- **Imputation simple** : moyenne, médiane, mode. Rapide mais perd l'info.
- **Imputation par modèle** : KNNImputer, IterativeImputer (sklearn). Plus précis mais coûteux.
- **Indicator variable** : ajouter une colonne `is_missing` en plus de l'imputation, pour ne pas perdre l'info "était manquante".

> **Pourquoi le NaN est informatif.** Le fait qu'une valeur soit manquante n'est presque jamais aléatoire (MCAR rare). Souvent c'est un signal métier (client qui n'a pas rempli un champ = profil différent). D'où l'intérêt de l'indicator variable.

---

## 3. Feature engineering, extraction, selection

Trois problèmes distincts qu'on confond souvent. Le mot **feature extraction** a en plus deux acceptions différentes selon le contexte (génération automatique vs compression), qu'on détaille en 3.2.

| | Feature engineering | Feature extraction | Feature selection |
| :--- | :--- | :--- | :--- |
| Question | "Quelles features je **crée** à la main à partir des données brutes ?" | "Comment je **transforme automatiquement** les données brutes en features ?" | "Parmi mes $p$ candidats, lesquelles je **garde** ?" |
| Nature | **Artisanat** : dépend du domaine, du problème, de l'intuition métier | **Algorithmique** : transformation systématique sans intervention métier | Relativement **systématique** : Lasso, importance des arbres, méthodes statistiques |
| Effet sur $p$ | Augmente $p$ (en général de quelques unités) | Soit explose $p$ (génération automatique type tsfresh), soit réduit $p$ (compression type PCA) | Réduit $p$ |
| Quand ça compte | Toujours, surtout pour les modèles linéaires | Données à structure (séries temporelles, image, texte, signal) | $p \gg n$ ou besoin d'interprétabilité |

### 3.1 Feature engineering

*À développer.*

Quelques idées génériques :
- **Combinaisons** : produits, ratios, différences entre features (ex : prix au m² = prix / surface)
- **Transformations** : log pour les données très skewées, polynômes pour capter du non-linéaire
- **Agrégations temporelles** : moyennes glissantes, écarts-types, pentes (cf. expérience Qube ENS)
- **Encodage cyclique** : pour heure/jour/mois → $(\sin(2\pi t/T), \cos(2\pi t/T))$

C'est de l'artisanat. Il n'y a pas de méthode systématique — l'essentiel vient de la connaissance métier.

### 3.2 Feature extraction

Le terme recouvre **deux opérations très différentes** dans la littérature, ce qui crée beaucoup de confusion. Les deux ont en commun de transformer automatiquement les données, mais l'une augmente la dimension et l'autre la réduit.

#### 3.2.1 Génération automatique de features (sens "tsfresh")

L'idée : à partir de données brutes structurées (typiquement une série temporelle, mais aussi des images, du texte, des graphes), **calculer automatiquement un grand catalogue de features descriptives** — sans aucune intuition métier, en appliquant tout ce qu'on sait calculer.

**Exemple typique : tsfresh pour les séries temporelles.** À partir d'un signal $(x_1, \dots, x_T)$, la librairie calcule ~750 features par défaut :
- Statistiques de base : moyenne, variance, skewness, kurtosis, min, max, médiane
- Mesures de complexité : entropie (Shannon, sample, approximate), nombre de changements de signe
- Autocorrélations et corrélations partielles à différents lags
- Coefficients de Fourier (FFT) et d'ondelettes
- Tests statistiques (Augmented Dickey-Fuller pour la stationnarité, etc.)
- Comportements spécifiques : nombre de pics, longueur de la plus longue séquence croissante, etc.

> **Pourquoi cette approche.** Quand on n'a pas d'expertise métier ou qu'on travaille sur beaucoup de signaux hétérogènes, on ne sait pas a priori quelles statistiques sont pertinentes. Plutôt que deviner, on calcule tout, puis on filtre.

**Pipeline standard.** Génération massive → feature selection statistique (cf. 3.3) → modèle. Le papier tsfresh (Christ et al. 2018) propose justement un test FRESH (statistique) pour filtrer les features non pertinentes, en contrôlant le FDR via Benjamini-Yekutieli.

**Autres exemples du même esprit.**
- **`featuretools`** : Deep Feature Synthesis sur des bases relationnelles (calcule des agrégations à plusieurs niveaux : `MEAN(orders.AMOUNT)`, `STD(orders.products.PRICE)`, etc.).
- **HOG, SIFT, SURF** : descripteurs d'images calculés systématiquement avant l'ère du deep learning.
- **MFCC** : coefficients cepstraux pour l'audio, pipeline standard avant les CNN/transformers acoustiques.

> **Le piège évident.** Cette approche explose $p$ — tsfresh peut transformer 100 séries temporelles en un dataset $100 \times 750$. Sans feature selection rigoureuse derrière, c'est la garantie d'overfit et de bruit. Le couple **génération massive + selection sévère** est non négociable.

> **Lien avec le deep learning.** Les CNN, transformers, etc. apprennent eux-mêmes ces représentations à partir des données brutes (images, séquences, texte). Historiquement, le deep learning a remplacé ces pipelines extraction+modèle par un seul modèle bout-en-bout. Mais sur les petits datasets de séries temporelles tabulaires, tsfresh + gradient boosting reste très compétitif et bien plus rapide qu'un réseau récurrent.

#### 3.2.2 Compression / dimension reduction (sens "PCA")

L'idée inverse : passer de $X \in \mathbb{R}^p$ à $Z = f(X) \in \mathbb{R}^k$ avec $k \ll p$, où $Z$ concentre l'information et écarte le bruit. Contrairement à la *selection*, on ne garde pas un sous-ensemble des features originales — on en construit de **nouvelles** par combinaison.

**Méthodes linéaires non supervisées.**
- **PCA** : projette sur les directions de variance maximale. Hypothèse implicite : variance = information. Marche bien quand les features sont corrélées.
- **ICA** : cherche des composantes statistiquement indépendantes (pas seulement décorrélées). Utile pour la séparation de sources (EEG, audio).
- **NMF** : décomposition $X \approx WH$ avec $W, H \geq 0$. Donne des composantes additives, plus interprétables que PCA (ex : topics dans du texte).

**Méthodes linéaires supervisées.**
- **LDA (Fisher)** : cherche les directions qui maximisent la séparation entre classes.
- **PLS** : variante supervisée de PCA qui utilise $Y$ pour orienter les composantes. Standard en chimiométrie.

**Méthodes non linéaires.**
- **Kernel PCA** : PCA dans un espace transformé via un noyau. Capte des structures non linéaires.
- **t-SNE, UMAP** : préservent la structure locale (voisinages). Surtout utiles pour la **visualisation** en 2D/3D, pas comme features pour un modèle aval (instables, pas de transformation déterministe sur de nouvelles données pour t-SNE).
- **Autoencoders** : NN qui apprend une représentation compressée $Z$ minimisant $\|X - g(f(X))\|^2$. Généralisation non linéaire de PCA.

> **Piège classique.** PCA est non supervisée : rien ne garantit que les directions de variance maximale soient celles qui discriminent $Y$. Si on a un label, LDA ou PLS sont souvent préférables.

#### 3.2.3 Vue d'ensemble

| | Génération (3.2.1) | Compression (3.2.2) |
| :--- | :--- | :--- |
| Effet sur $p$ | $p \uparrow \uparrow$ (de 1 à 750) | $p \downarrow$ (de 1000 à 50) |
| Exemples | tsfresh, featuretools, HOG, MFCC | PCA, autoencoders, UMAP |
| Suite logique | **selection** (3.3) pour filtrer | modèle directement |
| Quand l'utiliser | Données brutes structurées sans expertise métier | Données déjà tabulaires avec features très corrélées |

> **Les deux peuvent se chaîner.** Pipeline réaliste sur des séries temporelles : signal brut → tsfresh (3.2.1) → 750 features → selection statistique (3.3) → 50 features → PCA optionnelle (3.2.2) → modèle.

### 3.3 Feature selection

Trois familles classiques :

- **Filter** : on classe les features par un critère statistique (corrélation, mutual information, chi²) **indépendamment du modèle**. Rapide, mais ignore les interactions entre features.
- **Wrapper** : on entraîne le modèle sur différents sous-ensembles de features et on garde le meilleur (forward selection, backward elimination, RFE). Coûteux mais précis.
- **Embedded** : la sélection est intégrée à l'entraînement du modèle. C'est aujourd'hui la méthode dominante.
   - **Lasso / Elastic Net** : la pénalité $L_1$ met des coefficients à zéro
   - **Importance des arbres** (RF, gradient boosting) : on garde les features qui réduisent le plus l'impureté

> **Quand est-ce que la feature selection est encore utile ?** Avant 2010, beaucoup de modèles (régression linéaire/logistique sans régularisation) plantaient quand $p > n$. Aujourd'hui c'est moins critique : la régularisation gère, RF/gradient boosting ignorent les features inutiles, deep learning n'en a pas besoin. **Cas où ça reste utile** :
> - $p \gg n$ extrême (génomique, $p \sim 10^4$, $n \sim 10^2$)
> - Contrainte d'**interprétabilité** (medical, credit scoring : on veut peu de features pour pouvoir expliquer)
> - Coût de collecte/calcul élevé en production
> - **Couplage avec génération automatique** (tsfresh, featuretools) : c'est l'étape qui rend l'approche viable
