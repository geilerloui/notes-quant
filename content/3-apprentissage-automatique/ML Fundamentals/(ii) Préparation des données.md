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

## 3. Feature engineering vs feature selection

Deux problèmes inverses qu'on confond souvent.

| | Feature engineering | Feature selection |
| :--- | :--- | :--- |
| Question | "Quelles features je **crée** à partir des données brutes ?" | "Parmi mes 1000 candidats, lesquelles je **garde** ?" |
| Nature | **Artisanat** : dépend du domaine, du problème, de l'intuition métier | Relativement **systématique** : Lasso, importance des arbres, méthodes statistiques |
| Quand ça compte | Toujours, surtout pour les modèles linéaires | Quand $p \gg n$ ou pour l'interprétabilité |

### 3.1 Feature engineering

*À développer.*

Quelques idées génériques :
- **Combinaisons** : produits, ratios, différences entre features (ex : prix au m² = prix / surface)
- **Transformations** : log pour les données très skewées, polynômes pour capter du non-linéaire
- **Agrégations temporelles** : moyennes glissantes, écarts-types, pentes (cf. expérience Qube ENS)
- **Encodage cyclique** : pour heure/jour/mois → $(\sin(2\pi t/T), \cos(2\pi t/T))$

C'est de l'artisanat. Il n'y a pas de méthode systématique — l'essentiel vient de la connaissance métier.

### 3.2 Feature selection

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
