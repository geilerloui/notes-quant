# (vi) Interprétabilité

> Comprendre **pourquoi** un modèle fait telle prédiction, et plus généralement quelles features influencent la sortie. Deux niveaux : **global** (le modèle dans son ensemble) et **local** (une prédiction donnée). Cette note couvre la philosophie (intrinsèque vs post hoc), les méthodes globales (PDP, ICE, ALE, permutation importance), les méthodes locales (LIME, SHAP avec ses variantes Kernel/Tree/Deep), et les méthodes spécifiques aux réseaux de neurones (LRP, DeepLIFT).

---

## I. Pourquoi se soucier de l'interprétabilité

> [!warning] Définition (Interprétabilité)
> L'**interprétabilité** est le degré auquel un humain peut comprendre la cause d'une décision.
>
> On distingue parfois **interprétabilité / explicabilité** (au niveau du modèle global) et **explanation** (pour des prédictions individuelles).

Quatre raisons principales de s'y intéresser :

- **Réglementaire** : credit scoring, médical, justice prédictive. Le client/patient/justiciable a le droit de savoir pourquoi il a été refusé/diagnostiqué/condamné.
- **Debugging** : un modèle peut avoir une excellente performance sur un mauvais signal (target leakage, biais dans les données). L'interprétabilité aide à le détecter.
- **Confiance utilisateur** : un trader ne fera pas confiance à un signal qu'il ne comprend pas, même très performant.
- **Découverte scientifique** : comprendre quelles features sont prédictives peut révéler des patterns métier inconnus.

---

## II. Taxonomie des approches

### A. Intrinsèque vs post hoc

| Approche | Quand | Exemples |
| :--- | :--- | :--- |
| **Intrinsèque (interpretable by design)** | Le modèle est interprétable **par construction** | Régression linéaire (coefficients lisibles), arbre de décision peu profond, GAM |
| **Post hoc** | Modèle entraîné d'abord (souvent boîte noire), on l'explique **ensuite** | SHAP, LIME, permutation importance, partial dependence plots, Grad-CAM |

> 💡 **Étymologie.** *Post hoc* vient du latin *post hoc, ergo propter hoc* ("après cela, donc à cause de cela"). Au sens large = **après l'événement**, ou **a posteriori**. En interprétabilité ML, ça désigne les méthodes qui s'appliquent **après l'entraînement** sur un modèle déjà fixé. Tu entraînes un XGBoost ou un réseau de neurones, puis tu appliques une méthode qui explique ses prédictions sans modifier le modèle.

> [!warning] Cas hybride — Intrinsic puis post hoc
> Après avoir entraîné un decision tree (modèle intrinsèquement interprétable), on peut quand même appliquer de la permutation feature importance par-dessus.

### B. Model-specific vs model-agnostic

| Type | Description |
| :--- | :--- |
| **Model-specific** | Limité à un type de modèle. Ex : poids d'une régression linéaire, outil ne marchant que sur les NN. |
| **Model-agnostic** | Applicable à n'importe quel modèle ML, post hoc, sans accès aux poids du modèle. |

### C. Le piège conceptuel des méthodes post hoc

> [!warning] La critique de Cynthia Rudin
> Dans *Stop Explaining Black Box ML Models for High Stakes Decisions* (Nature Machine Intelligence, 2019), Rudin formalise une critique récurrente : les explications post hoc sont par définition des **approximations** du vrai comportement du modèle. Elles peuvent diverger entre elles (SHAP et LIME donnent parfois des explications contradictoires sur la même prédiction), et donnent une **fausse impression de transparence**.
>
> Pour les enjeux critiques (médical, justice, crédit), Rudin recommande d'utiliser un modèle **intrinsèquement interprétable** plutôt qu'un modèle complexe + une couche d'explication post hoc. Si une régression logistique régularisée ou un GAM donne une performance proche de XGBoost (souvent le cas sur des données tabulaires bien préparées), l'utiliser évite le risque d'expliquer faussement une boîte noire.

### D. Évaluation de l'interprétabilité

D'après Doshi-Velez et Kim (2017), trois niveaux principaux :

- **Application level evaluation (real task)** — mettre l'explication en production et la faire tester par les utilisateurs finaux.
- **Human level evaluation (simple task)** — expériences avec des non-experts. Par exemple, montrer plusieurs explications et demander à l'utilisateur de choisir la meilleure.
- **Function level evaluation (proxy task)** — pas besoin d'humains, métriques automatiques.

### E. Propriétés des méthodes d'explication

> [!note]- Quatre propriétés clés
> - **Expressive Power** — la "langue" ou structure des explications. IF-THEN rules, arbres, somme pondérée, langage naturel...
> - **Translucency** — combien la méthode dépend de l'inspection des paramètres internes du modèle. Méthodes intrinsèques (régression linéaire) = haute translucidité. Méthodes purement input/output = zéro translucidité. Trade-off : haute translucidité = plus d'info mais moins portable.
> - **Portability** — gamme de modèles applicables. Faible translucidité ↔ haute portabilité.
> - **Algorithmic Complexity** — coût de calcul.

### F. Propriétés des explications individuelles

> [!note]- Cinq propriétés clés
> - **Accuracy** — comment l'explication prédit-elle sur des données non vues ?
> - **Fidelity** — comment l'explication approxime-t-elle la prédiction du black box model ? Crucial. Une explication à faible fidélité est inutile. Distinction entre fidélité **globale** et **locale** (LIME, Shapley sont locales).
> - **Consistency** — l'explication varie-t-elle entre modèles entraînés sur la même tâche avec prédictions similaires ? Délicat à cause du **Rashomon effect** : deux modèles peuvent utiliser des features différentes pour la même prédiction.
> - **Stability** — explications similaires pour instances similaires (à modèle fixé). Faible stabilité = méthode à haute variance ou non-déterministe.
> - **Comprehensibility** — humanly understood. L'éléphant dans la pièce, difficile à mesurer mais crucial.

---

## III. Méthodes intrinsèques

### A. Régression linéaire / logistique

Les coefficients $\beta_j$ se lisent directement (effet d'une augmentation d'une unité de $x_j$ sur $y$).

> 💡 **Attention.** Interprétation valable **uniquement** si features standardisées et indépendantes. La multicolinéarité fait n'importe quoi avec les signes.

### B. Arbre de décision

Règles si-alors lisibles à condition de rester peu profonds.

#### B.1 Importance individuelle

Pour expliquer la prédiction d'une observation, on suit son chemin dans l'arbre et on mesure la décroissance d'impureté (Gini ou MSE) à chaque nœud rencontré.

![[DT-1.png]]
**Figure 1.** Pour une observation, on suit le chemin et on mesure la décroissance de Gini à chaque split.

#### B.2 Feature importance

On parcourt **tous les splits** du modèle qui utilisent cette feature et on somme la réduction d'impureté qu'elle apporte. La somme totale est normalisée à 100, donc chaque importance s'interprète comme **part de l'importance globale du modèle**.

> 💡 **Piège classique.** Biais en faveur des features à haute cardinalité (un float continu peut faire baisser Gini "par chance" plus facilement qu'une variable binaire).

### C. GAM (Generalized Additive Models)

$$y = f_1(x_1) + f_2(x_2) + \ldots + f_p(x_p),$$

où chaque $f_j$ peut être une fonction non linéaire mais marginale. On peut **tracer chaque $f_j$ séparément** — interprétation propre.

---

## IV. Permutation Feature Importance (post hoc, model-agnostic)

> [!warning] Idée centrale
> Si on **shuffle aléatoirement** une feature dans le validation set en gardant le reste intact, **de combien la performance se dégrade-t-elle** ? Plus la dégradation est grande, plus la feature est importante.

### A. Algorithme

> [!note]- Pseudo-code
> 1. Entraîner le modèle sur $X_{\text{train}}, y_{\text{train}}$.
> 2. Faire des prédictions sur le training set, calculer le score initial (plus haut = mieux).
> 3. Pour chaque feature $i$ :
>    - Permuter la colonne $i$ dans le training set ($X_{\text{train\_permuted}}$).
>    - Refaire des prédictions, calculer `score_permuted`.
>    - **Importance** = `score_permuted - score`. Plus c'est négatif, plus la feature est importante.
> 4. Répéter le tout plusieurs fois pour réduire l'effet du tirage aléatoire et moyenner.

> [!example] Football Man-of-the-Game
> Modèle qui prédit si l'équipe aura le joueur "Man of the Game" basé sur les stats. Après permutation feature importance :
>
> ![[permutation-4.png]]
>
> Les valeurs en haut = features les plus importantes. Le premier nombre par ligne = dégradation moyenne d'accuracy après shuffle. Il y a un peu d'aléa à cause du shuffle, on répète plusieurs fois.

### B. Exemples

> [!example] Cervical cancer (catégoriel)
> Random forest pour prédire le risque de cancer cervical. Mesure d'erreur : $1 - \text{AUC}$. Les features avec un facteur $\approx 1$ ne sont pas importantes.
>
> ![[permutation-2.png]]

> [!example] Bike rentals (continu)
> SVM pour prédire le nombre de vélos loués selon météo et calendrier. Mesure : MAE.
>
> ![[permutation-3.png]]

### C. Avantages et limites

> [!warning] Avantages
> - **Interprétation propre** — l'importance est la dégradation d'erreur quand on détruit l'info de la feature.
> - **Métrique au choix** (AUC, MAE...) — comparable d'un problème à l'autre.
> - **Tient compte des interactions** — en permutant une feature, on détruit aussi ses interactions avec les autres.
> - **Pas besoin de réentraîner**.

> [!warning] Limites
> - Besoin du **vrai outcome** (donc utilisable seulement sur un set labellisé).
> - **Coûteux** si on veut un IC : il faut shuffle plusieurs fois par feature.
> - Sensible aux **features corrélées** : on évalue le modèle sur des combinaisons jamais vues si on shuffle aveuglément.

> [!note]- Implémentation Python (extrait)
> ```python
> def calculate_permutation_importance(model, X, y, scoring_function, n_repeats=3, seed=42):
>     model.fit(X, y)
>     y_hat = model.predict(X)
>     score = scoring_function(y, y_hat)
>     
>     importances = {col: 0 for col in X.columns}
>     for n in range(n_repeats):
>         for col in X.columns:
>             X_temp = X.copy()
>             X_temp[col] = X[col].sample(frac=1, random_state=seed+n).values
>             y_hat_perm = model.predict(X_temp)
>             score_perm = scoring_function(y, y_hat_perm)
>             importances[col] += (score_perm - score) / n_repeats
>     return importances
> ```

> 💡 **Critique de la méthode** ([Stop Permuting Features](https://towardsdatascience.com/stop-permuting-features-c1412e31b63f)) : permuter brise les corrélations naturelles et crée des points hors distribution. Des variantes "conditionnelles" existent.

---

## V. Méthodes globales — Dependence Plots

### A. Partial Dependence Plot (PDP)

> [!warning] Définition
> Le **PDP** (ou PD plot) montre l'effet marginal d'**une ou deux features** sur la sortie prédite.
>
> $$\hat f_{x_S, \text{PDP}}(x_S) = \mathbb{E}_{x_C}\big[ \hat f(x_S, x_C) \big] = \int_{x_C} \hat f(x_S, x_C) \, p(x_C) \, dx_C$$
>
> où $x_S$ sont les features qu'on plot (1 ou 2), $x_C$ les autres features.

**Estimation par méthode Monte Carlo** sur le training set :

$$\hat f_{x_S}(x_S) = \frac{1}{n} \sum_{i=1}^n \hat f(x_S, x_C^{(i)}).$$

Pour la classification, on plotte la probabilité de chaque classe contre les valeurs de $x_S$ — une ligne par classe.

> 💡 **Méthode globale.** Le PDP considère **toutes les instances** et donne une vue globale de la relation feature ↔ prédiction.

#### A.1 Exemple — feature catégorielle

On veut examiner l'effet marginal des valeurs de `street` sur le prix d'une maison. On prend les valeurs uniques `A, B, C` et on remplace `street` par chaque valeur pour **tous** les samples du dataset, puis on moyenne les prédictions.

![[pdp-1.png]]
![[pdp-2.png]]
**Figure 2.** Calcul du PDP : remplacement systématique + moyenne.

![[pdp-3.png]]

La hauteur de chaque catégorie correspond à :

$$\hat f_{\text{street}}(A) = \frac{1}{8} \sum_{i=1}^8 \hat f(A, x_C^{(i)}).$$

#### A.2 Exemple — feature continue (bike rentals)

Prédire le nombre de vélos loués selon la météo. Pour `temperature`, le modèle prédit en moyenne un grand nombre de locations pour une météo chaude mais pas trop. Quand `humidity` dépasse 60%, les loueurs sont inhibés. Pour `wind speed`, on a un drop > 25 km/h, mais probablement peu de données (regarder le rug en bas).

![[pdp-4.png]]

#### A.3 Exemple — interaction 2D

PDP de la probabilité de cancer cervical en fonction de l'**interaction** entre âge et nombre de grossesses. Augmentation du risque à 45 ans. Pour < 25 ans, les femmes ayant 1-2 grossesses ont un risque prédit plus faible que celles ayant 0 ou plus de 2 grossesses.

![[pdp-5.png]]

> 💡 **Attention causalité.** Il s'agit potentiellement d'une **corrélation**, pas d'un effet causal !

#### A.4 Avantages et limites

> [!warning] Avantages
> - **Intuitif** — facile à expliquer à un non-expert.
> - **Interprétation claire** si la feature de PDP n'est pas corrélée avec les autres.

> [!warning] Limites
> - **Maximum 2 features** (au-delà, illisible).
> - Ne montre pas la **distribution de la feature** par défaut (pas de rug → on ne sait pas où il y a peu de données).
> - **Hypothèse d'indépendance** : si $x_S$ est corrélée à $x_C$, on évalue le modèle sur des combinaisons impossibles.
> - **Effets hétérogènes cachés** : la PDP est une moyenne — peut masquer des effets opposés sur deux sous-groupes (cf. ICE).
>
> ![[pdp-8.png]]
>
> Sur cet exemple, la moitié des points sont au-dessus de la moyenne, l'autre en dessous — la PDP cache complètement l'hétérogénéité.

#### A.5 Implémentation Python (essence)

```python
def pdp(feature, df, x_labels, y_label, model):
    df_copy = df.copy()
    unique_vals = np.unique(df_copy[feature].values)
    y = []
    for val in unique_vals:
        df_copy[feature] = val
        X = df_copy[x_labels]
        y.append(np.average(model.predict(X)))
    return unique_vals, y
```

### B. Individual Conditional Expectation (ICE)

> [!warning] Idée
> ICE plot = **une ligne par instance** au lieu de la moyenne PDP. Une ligne montre comment la prédiction d'une instance change quand la feature varie. Le **PDP est la moyenne des ICE**.

> 💡 **Pourquoi c'est utile.** Le PDP cache les effets hétérogènes créés par des **interactions**. Si la PDP est plate alors qu'il y a deux sous-groupes avec des effets opposés, l'ICE le révèle.

#### B.1 Algorithme

Reprenons l'exemple de prédiction de prix avec 4 observations. $\hat y$ est la prédiction pour chaque observation, et `mean` est la moyenne (= PDP).

![[ICE-1.png]]

À la différence du PDP, ICE **garde toutes les sorties** pour chaque observation :

![[ICE-2.png]]
![[ICE-3.png]]
**Figure 3.** À gauche le tableau de sortie PDP. À droite le tableau ICE.

#### B.2 Exemples

> [!example] Cancer cervical (catégoriel)
> Sur le PDP on voyait que la probabilité augmente vers 50 ans. Mais est-ce vrai pour **toutes** les femmes ? L'ICE révèle que pour la plupart, oui. Mais pour quelques-unes ayant déjà une probabilité élevée jeune, l'effet de l'âge est faible.
>
> ![[ICE-4.png]]

> [!example] Bike rentals (continu)
> Toutes les courbes suivent à peu près le même pattern → pas d'interactions visibles → PDP suffit.
>
> ![[ICE-5.png]]

#### B.3 Centered ICE (c-ICE)

Parfois difficile de comparer les courbes ICE quand elles partent de prédictions différentes. Solution : les **centrer** au lower bound de la feature :

$$\hat f_{\text{cent}}^{(i)} = \hat f^{(i)} - \mathbf{1} \cdot \hat f(x^a, x_C^{(i)}),$$

où $x^a$ est le point d'ancrage (typiquement le min).

> [!example] Centered ICE — cancer cervical et bike rentals
> ![[ICE-6.png]]
> ![[ICE-7.png]]

#### B.4 Avantages et limites

> [!warning] Avantages
> - **Plus intuitif** que PDP.
> - **Révèle les relations hétérogènes**.

> [!warning] Limites
> - **Une seule feature** par plot (sinon surfaces qui se chevauchent).
> - **Mêmes problèmes que PDP** avec features corrélées.
> - **Surcharge visuelle** si trop de courbes (solution : transparence ou échantillonnage).

### C. Accumulated Local Effects (ALE)

#### C.1 M-plots — la solution intermédiaire qui ne marche pas

> [!warning] Le problème du PDP
> Le PDP peut donner des résultats irréalistes. Si dataset = $\{$living area, rooms$\}$ et target = valeur de la maison, le PDP remplace `living area` par 30 m² **pour toutes les observations**, y compris des maisons à 10 chambres — combinaison absurde.

**Idée alternative : moyenner sur la distribution conditionnelle :**

$$\hat f_{x_S, M}(x_S) = \mathbb{E}_{x_C \mid x_S}\big[\hat f(x_S, x_C) \mid X_S = x_S\big] = \int \hat f(x_S, x_C) \, p(x_C \mid x_S) \, dx_C.$$

Approximation :

$$\hat f_{x_S, M}(x_S) = \frac{1}{n(x_S)} \sum_{i \in N(x_S)} \hat f(x_S, x_C^{(i)}).$$

> 💡 **Pourquoi ça ne résout pas tout.** Si on moyenne les prédictions des maisons d'environ 30 m², on capture l'effet **combiné** de la surface ET du nombre de chambres (à cause de leur corrélation). Si en réalité seul le nombre de chambres compte, le M-plot va quand même montrer un effet de la surface.

![[ALE-1.png|463]]
![[ALE-2.png]]
**Figure 4.** À gauche PDP, à droite M-Plot.

#### C.2 ALE — l'idée

ALE résout le problème en regardant les **différences locales** plutôt que la moyenne. Pour un intervalle étroit, on regarde **l'effet local** de la feature à valeurs proches, ce qui élimine la confusion due à la corrélation. Le détail mathématique mérite une note dédiée — voir références.

> 💡 **À retenir.** ALE > PDP quand les features sont corrélées. C'est le standard moderne pour features fortement corrélées.

**Bibliographie ALE :**
- [Maths détaillées](https://ema.drwhy.ai/accumulatedLocalProfiles.html)
- [Christoph Molnar's book](https://christophm.github.io/interpretable-ml-book/ale.html)

---

## VI. Méthodes locales — LIME

> 💡 **Référence.** Ribeiro et al. 2016. LIME = Local Interpretable Model-agnostic Explanations. Idée : pour expliquer **une prédiction** spécifique, on entraîne un **modèle simple** (linéaire ou arbre) **localement**, autour du point d'intérêt.

### A. Framework général

On définit une explication comme un modèle $g \in G$ (classe de modèles interprétables : linéaires, arbres). $\Omega(g)$ mesure la complexité de $g$ (profondeur d'arbre, nombre de poids non-nuls...). $f$ est le black box. $\pi_x(z)$ mesure la proximité de $z$ à $x$. $\mathcal L(f, g, \pi_x)$ est la distance entre $g$ et $f$ dans le voisinage de $x$.

L'explication produite par LIME :

$$\boxed{\xi(x) = \arg\min_{g \in G} \mathcal L(f, g, \pi_x) + \Omega(g).}$$

### B. Cas des modèles linéaires

$g(z') = w_g \cdot z'$ et **locally weighted square loss** :

$$\mathcal L(f, g, \pi_x) = \sum_{z, z' \in \mathcal Z} \pi_x(z) \big[f(h_x(z')) - g(z')\big]^2,$$

avec $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$ — un **kernel exponentiel** sur une distance $D$ (cosinus pour le texte, $L_2$ pour les images).

### C. LIME pour données tabulaires

L'algorithme en 6-7 étapes :

> [!note]- Pseudo-code
> 1. **Entraîner** $f$ (le black box) sur le dataset.
> 2. **Choisir** une observation $x_{\text{interest}}$ à expliquer.
> 3. **Générer des perturbations** :
>    - Définir un vecteur binaire interprétable $x' = (1, 1, 0, 1, 0, 0, \ldots)$ (les features auxquelles on s'intéresse).
>    - Échantillonner des $z_i' = (1, 0, 0, 1, \ldots)$ autour de $x'$.
>    - Pour le tabulaire, ajouter un bruit gaussien sur les continues, échantillonner depuis la distribution pour les catégorielles.
> 4. **Similarity** : appliquer $\pi_x(z) = \exp(-D(x, z)^2 / \sigma^2)$ pour chaque paire ($x_{\text{interest}}, z_i$). Plus de poids autour de $x_{\text{interest}}$.
> 5. **Forward pass** : passer chaque $z_i$ dans $f$ pour obtenir $f(z_i)$.
> 6. **Fit local** : entraîner $g$ sur les $(z_i', f(z_i))$ avec poids $\pi_x(z_i)$. Feature selection au choix (forward, top-K, lasso path).
> 7. **Lecture** : les coefficients de $g$ donnent l'explication.

> [!example] Boston Housing
> $x_{\text{interest}}$ = une observation. Black box = random forest. On veut expliquer pourquoi $\hat{\text{medv}} = 25.38$ (vraie valeur 24).
>
> Modèle local linéaire à 3 features max :
>
> ![[lime-2.png]]
>
> ![[lime-0.png]]
>
> Plus on est près de $x_{\text{interest}}$, plus le poids est grand (kernel exponentiel).
>
> ![[lime-3.png]]
>
> Conclusion : `pratio` a un impact négatif, `rm` a un impact positif. Modèle local :
>
> $$s_x = w_0 + w_1 \cdot rm + w_2 \cdot \text{pratio} + w_3 \cdot \text{lstat}.$$

### D. LIME pour le texte

Adaptation simple : pour le texte, **pas de perturbation gaussienne**. On supprime simplement des mots (mettre à 0). $x'$ est la liste binaire de présence/absence des mots.

> [!example] Détection de spam YouTube (1 = spam, 0 = normal)
> ![[lime-8.png]]
>
> Pour chaque variation $z_i'$, on calcule la **proximité** comme $1 - \text{(fraction de mots supprimés)}$. Si on supprime 1 mot sur 7, proximité = 0.86.
>
> ![[lime-9.png]]
>
> Résultat : seul "channel!" a un poids positif → c'est le mot qui pousse vers la classe spam.
>
> ![[lime-10.png]]

### E. LIME pour les images

Les images sont segmentées en **super-pixels** (algorithme `Quickshift`). $x'$ devient le vecteur binaire de présence/absence des super-pixels.

> [!example] InceptionV3 sur photo d'un labrador
> ![[lime-4.png]]
> ![[lime-5.png]]
>
> Pour générer les perturbations, on **éteint** ou **allume** des super-pixels :
>
> ![[lime-6.png]]
>
> Distance $\pi_x$ = cosinus entre image originale et perturbée. Forward pass = InceptionV3. Fit local linéaire. Résultat : zone de l'image qui a la plus forte association avec la prédiction "Labrador".
>
> ![[lime-7.png]]

> [!warning] Limites de LIME
> - **Sensible au choix du voisinage** ($\sigma$, taille des perturbations).
> - **Instable** : deux runs sur la même observation peuvent donner des explications différentes.
> - **Pas de fidelity garantie globalement** — n'approche $f$ que **localement**.
> - **Casse les propriétés** d'additivité et de cohérence (cf. SHAP qui les respecte).

---

## VII. SHapley Additive exPlanations (SHAP)

> 💡 **Référence.** Lundberg & Lee 2017. Approche basée sur les **valeurs de Shapley** de la théorie des jeux coopératifs (Shapley 1953, prix Nobel 2012).

### A. Motivation et définition

> [!example] Motivation
> On gère une location de motos. On veut comprendre l'influence de la pub sur les locations journalières. Trois prédicteurs binaires : pub, jour de semaine, pluie.
>
> | Jour | Add | Weekday | Rain | Nb_rentals |
> | :---: | :---: | :---: | :---: | :---: |
> | 1 | 1 | 1 | 1 | 10 |
> | 2 | 0 | 1 | 0 | 8 |
> | 3 | 1 | 0 | 0 | 20 |
> | 4 | 0 | 1 | 0 | 12 |
>
> Black box entraîné. **Comment savoir si la pub a vraiment marché ce jour-là ?**

> [!warning] Définition (Additive feature attribution)
> Pour le modèle interprétable $g$, l'attribution distribue la prédiction entre les variables :
>
> $$g(z') = \phi_0 + \sum_{i=1}^M \phi_i z_i'$$
>
> avec $z_i \in \{0, 1\}$.

### B. Trois propriétés désirables

> [!warning] Local accuracy
> $g$ doit donner la **même prédiction** que $f$ sur l'input non transformé. Implique que les $\phi_i$ somment à $f(x) - \phi_0$ :
>
> $$\sum_i \phi_i + \phi_0 = f(x).$$

> [!warning] Missingness
> Si une feature est absente, son poids est 0. Pertinent pour les datasets avec features constantes.

> [!warning] Consistency
> Si l'effet d'ajouter $x_i$ est plus grand pour un modèle $f'$ que pour un modèle $f$, le poids $\phi_i$ doit être plus grand pour $f'$ que pour $f$.

> 💡 **LIME casse les propriétés 1 et 3.** LIME approche $f$ localement par fit, sans garantie qu'il prédit la même chose en $x$ (casse local accuracy) ni de cohérence entre modèles. **C'est précisément ce qui motive SHAP.**

> [!warning] Théorème principal
> Le **seul** ensemble de poids satisfaisant les trois propriétés est donné par les **valeurs de Shapley**.

### C. Calcul des Shapley values

Pour calculer la valeur de Shapley d'une feature, on liste **toutes les coalitions** la contenant, et pour chacune on mesure la **contribution marginale** de cette feature à la coalition.

> [!example] Calcul pour la feature "Add"
> Coalitions impliquant "Add" :
> - $\{x_a\}$
> - $\{x_a, x_w\}$
> - $\{x_a, x_r\}$
> - $\{x_a, x_w, x_r\}$
>
> Pour chaque coalition $S$, contribution :
>
> $$\delta_i^{(S)} = f_S(S) - f_{S \setminus \{i\}}(S \setminus \{i\}).$$
>
> Par exemple :
>
> $$\delta_{x_a}^{\{x_a, x_r\}} = f_{\{x_a, x_r\}}(\{x_a, x_r\}) - f_{\{x_r\}}(\{x_r\}).$$
>
> Valeur de Shapley finale :
>
> $$\phi_a = \frac{1}{p} \sum_S \binom{p-1}{|S|-1}^{-1} \delta_a^{(S)},$$
>
> où $p$ est le nombre total de features. C'est la **moyenne pondérée** des contributions marginales.

> [!warning] Problème pratique
> Calcul **exponentiel** en nombre de features ($2^p$ coalitions). D'où les approximations Kernel SHAP, Tree SHAP, Deep SHAP.

### D. Architecture du package `shap`

![[shap-1.png]]
**Figure 5.** Le package `shap` propose plusieurs explainers selon le type de modèle (Tree, Kernel, Deep, Linear).

---

## VIII. Kernel SHAP

> 💡 **Idée centrale.** Au lieu de **réentraîner** des modèles avec des subsets de features (intractable), on utilise le **modèle complet** $f$ déjà entraîné, et on remplace les features absentes par leur **espérance marginalisée** sur les données.

### A. Marginalisation

Pour 3 features, le modèle partiel sans $x_3$ est estimé par :

$$f_{\{x_1, x_2\}}(x_1, x_2) \xrightarrow[\text{Kernel SHAP}]{} \int f(x_1, x_2, x_3) \, p(x_3) \, dx_3.$$

### B. Le kernel magique

Au lieu du kernel LIME standard, Kernel SHAP utilise :

$$\pi_x^{\text{SHAP}}(z') = \frac{p - 1}{\binom{p}{|z'|} \, |z'| \, (p - |z'|)},$$

qui met plus de poids sur les coalitions avec **peu** ou **presque toutes** les features. C'est précisément ce kernel qui garantit les **trois propriétés Shapley**.

### C. Loss

$$\mathcal L(f, g, \pi_x) = \sum_{z' \in Z} \pi_x(z') \big[f(h_x(z')) - g(z')\big]^2,$$

avec $g(z') = \phi_0 + \sum_{j=1}^M \phi_j z_j'$.

### D. Exemple complet en 2D

> [!example] Cas pédagogique 2 features
> Dataset $X$ + classifieur $f$. Objectif : expliquer la prédiction $f(1.3, 0) = 0.233$.
>
> ![[kshap-1.png]]

#### D.1 Hypothèse d'indépendance

Hypothèse forte de SHAP :

$$f(h_x(z')) = \mathbb{E}_{z_{\bar S} \mid z_S}[f(z)] \approx \mathbb{E}_{z_{\bar S}}[f(z)].$$

Avec 2 features, $Z = \{(1, 0), (0, 1)\}$. On calcule :

$$\begin{aligned}
\mathbb{E}_{z_2} f(x_1, z_2) &\equiv f(1.3, \cdot) = \frac{1}{n} \sum_i f(1.3, z_2^{(i)}) = 0.140 \\
\mathbb{E}_{z_1} f(z_1, x_2) &\equiv f(\cdot, 0) = \frac{1}{n} \sum_i f(z_1^{(i)}, 0) = 0.764
\end{aligned}$$

![[kshap-2.png]]

#### D.2 Calcul des Shapley values

> [!note]- Détail de la dérivation
> $\phi_0$ = moyenne du modèle :
>
> $$\phi_0 = \mathbb{E}[f] = \frac{1}{n} \sum_i f(x^{(i)}) = 0.511.$$
>
> Loss :
>
> $$\mathcal L = \frac{1}{2}\big[\mathbb{E}_{z_2} f(x_1, z_2) - (\phi_0 + \phi_1)\big]^2 + \frac{1}{2}\big[\mathbb{E}_{z_1} f(z_1, x_2) - (\phi_0 + \phi_2)\big]^2.$$
>
> Avec les contraintes :
> - **Missingness** : $\mathbb{E}[f] = \phi_0$.
> - **Local accuracy** : $\phi_2 = f(x) - \phi_0 - \phi_1$.
>
> Réécriture :
>
> $$\mathcal L = \frac{1}{2}\big[(\mathbb{E}_{z_2} f - \mathbb{E}[f]) - \phi_1\big]^2 + \frac{1}{2}\big[(\mathbb{E}_{z_1} f - f(x)) + \phi_1\big]^2.$$
>
> C'est un problème de régression linéaire 1D à 2 samples :
>
> $$X = \begin{pmatrix} 1 \\ -1 \end{pmatrix}, \quad y = \begin{pmatrix} \mathbb{E}_{z_2} f - \mathbb{E}[f] \\ \mathbb{E}_{z_1} f - f(x) \end{pmatrix}.$$
>
> Résultat : $\phi_0 = 0.511, \phi_1 = -0.452, \phi_2 = 0.173$.

#### D.3 Interprétation

- **Moyenne** du modèle sur le training set : 0.511.
- **$x_1$ a un poids -0.452** — pour des exemples similaires avec $x_1$ différent, la prédiction est beaucoup plus grande (push down, bleu).
- **$x_2$ a un poids 0.173** — corrélé positivement (push up, rouge).

![[kshap-3.png]]
**Figure 6.** Visualisation graphique des Shapley values.

---

## IX. Tree SHAP

> 💡 **Idée.** Pour les modèles à arbres (Random Forest, GBM, XGBoost, LightGBM), on peut calculer les Shapley values **exactement** (pas d'approximation) en temps polynomial. Implémentation : `shap.TreeExplainer`. **Standard de fait** en pratique.

### A. Exemple à la main

![[tree_shap-1.png]]
**Figure 7.** Decision tree avec un sample par feuille.

Soit $x_{\text{interest}} = \{F = \text{Yes}, C = \text{Yes}\}$. Coalitions possibles avec 2 features :

$$S = \{\emptyset, \{F\}, \{C\}, \{F, C\}\}.$$

> [!note]- Calcul de $\phi_F$
> $$\phi_F = \frac{0! \cdot 1!}{2!}\big[f_x(\{F\}) - f_x(\emptyset)\big] + \frac{1! \cdot 0!}{2!}\big[f_x(\{F, C\}) - f_x(\{C\})\big]$$
>
> $$\phi_F = \frac{1}{2}[40 - 20] + \frac{1}{2}[80 - 40] = 30.$$
>
> **Calcul de $f_x(\{F\})$** : on suit le chemin avec uniquement la feature $F$. Si on tombe à mi-arbre, on calcule la moyenne des feuilles enfants → $40$.
>
> **Calcul de $f_x(\{C\})$** :
>
> $$f_x(\{C\}) = \frac{n_{\text{left}}}{L} \cdot \text{pred\_left} + \frac{n_{\text{right}}}{L} \cdot \text{pred\_right}$$
>
> Avec $L = 4, n_{\text{left}} = 2, n_{\text{right}} = 2$, et $C = \text{Yes}$ donne 0 à gauche, 80 à droite → $40$.

> [!note]- Calcul de $\phi_C$
> Symétrique :
>
> $$\phi_C = \frac{1}{2}[40 - 20] + \frac{1}{2}[80 - 40] = 30.$$

Total :

$$\phi = \phi_0 + \phi_F + \phi_C = 20 + 30 + 30 = 80.$$

Pour le **Modèle B** (autre arbre) :

$$\phi_F = 30, \quad \phi_C = 35, \quad \phi = 25 + 30 + 35 = 90.$$

> 💡 **À retenir.** Pour les modèles à arbres, SHAP est **exact, rapide, et la méthode de référence**. Si tu fais du XGBoost en production avec besoin d'explication, `shap.TreeExplainer` est le bon choix.

---

## X. Deep SHAP

> 💡 **Idée.** Deep SHAP combine **DeepLIFT** (méthode de propagation de relevance pour réseaux profonds) avec les **Shapley values**. Pour les NN, c'est l'analogue de Tree SHAP pour les arbres.

> [!warning] Référence
> [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/pdf/1705.07874.pdf) (Lundberg & Lee, NIPS 2017).

---

## XI. Méthodes spécifiques aux réseaux de neurones

### A. Forward (perturbation-based) vs Backward (backpropagation-based)

> [!warning] Deux familles
> - **Backward / Backpropagation-based** — partent de la sortie et propagent vers l'entrée pour identifier l'impact des inputs (LRP, DeepLIFT, Grad-CAM).
> - **Forward / Perturbation-based** — partent des pixels et perturbent pour observer comment la sortie change.

### B. Layer-Wise Relevance Propagation (LRP)

#### B.1 Principe

Chaque neurone **redistribue** vers la couche inférieure autant de relevance qu'il en a reçu de la couche supérieure. On peut ainsi remonter jusqu'aux pixels d'entrée et savoir lesquels ont influencé la prédiction.

![[lrp-7.png]]

Pour les réseaux profonds avec ReLU :

$$a_k = \max\!\left(0, \sum_{0, j} a_j w_{jk}\right).$$

#### B.2 Trois règles de propagation

> [!warning] (i) Basic Rule (LRP-0)
> $$R_j = \sum_k \frac{a_j w_{jk}}{\sum_{0, j} a_j w_{jk}} R_k$$
>
> Équivalent à $\text{Gradient} \times \text{Input}$. Mais le gradient des deep nets est typiquement bruité → on a besoin de règles plus robustes.

> [!warning] (ii) Epsilon Rule (LRP-$\epsilon$)
> $$R_j = \sum_k \frac{a_j w_{jk}}{\epsilon + \sum_{0, j} a_j w_{jk}} R_k$$
>
> Le rôle de $\epsilon$ : absorber la relevance quand les contributions au neurone $k$ sont faibles ou contradictoires.

> [!warning] (iii) Gamma Rule (LRP-$\gamma$)
> Favorise les contributions positives :
>
> $$R_j = \sum_k \frac{a_j \cdot (w_{jk} + \gamma w_{jk}^+)}{\sum_{0, j} a_j \cdot (w_{jk} + \gamma w_{jk}^+)} R_k$$

**Forme générale :**

$$R_j = \sum_k \frac{a_j \cdot \rho(w_{jk})}{\epsilon + \sum_{0, j} a_j \cdot \rho(w_{jk})} R_k.$$

#### B.3 Quel LRP pour quelle couche ?

> 💡 **LRP composite.** En pratique, on combine LRP-0 / $\epsilon$ / $\gamma$ selon la couche :
> - LRP-0 uniforme → trop d'artifacts locaux.
> - LRP-$\epsilon$ uniforme → enlève le bruit, garde un nombre limité de features.
> - LRP-$\gamma$ uniforme → plus dense, plus lisible humain, mais peut prendre des concepts non liés.
> - **Composite LRP** combine les trois pour avoir le meilleur de chaque.

![[lrp-6.png]]

#### B.4 Exemple — MNIST

On part d'une matrice $X \in \mathbb{R}^{12 \times 784}$ (12 images de 784 pixels). On forward dans le NN, qui prédit pour chaque image un vecteur de 10 probabilités.

```
# digit "2" : [0]0.0 [1]3.6 [2]49.1 [3]8.9 ... 
# digit "1" : [0]0.0 [1]27.0 [2]0.0 [3]0.0 ...
# digit "0" : [0]49.1 [1]0.0 [2]10.6 [3]0.0 ...
```

Initialisation de la matrice de relevance par la **prédiction la plus haute**, puis backpropagation par LRP composite.

![[lrp-4.png]]
![[lrp-5.png]]
**Figure 8.** Pixels en rouge = contribution positive à la prédiction. En bleu = négative.

#### B.5 Application — Sentiment Analysis

LRP heatmaps sur des phrases tests, avec couleur intensité normalisée par phrase.

![[lrp-2.png]]

#### B.6 Detecting Clever Hans avec analyse spectrale

> 💡 **L'effet Clever Hans en ML.** Un modèle peut avoir une accuracy excellente mais pour les **mauvaises raisons** (apprend un artefact du dataset, pas le vrai signal).

L'analyse spectrale de relevance permet d'identifier différentes **stratégies de prédiction** sur le dataset, en clusterisant les heatmaps LRP. Quatre stratégies différentes pour classifier un cheval :
- (b) Détecter un cheval (et un cavalier).
- (c) Détecter un **source tag** dans les images portrait.
- (d) Détecter des haies en bois et autres éléments d'équitation.
- (e) Détecter un **source tag** dans les images paysage.

![[lrp-1.png]]
**Figure 9.** Le SVHN apprend partiellement un tag de source — pas la présence du cheval lui-même !

> 💡 **Lien avec l'effet Clever Hans en finance.** Cas typique : un modèle de credit scoring qui apprend à prédire le **code postal** plutôt que le risque réel — Clever Hans en pratique. Les méthodes d'interprétabilité comme LRP permettent de détecter ces problèmes avant déploiement.

**Bibliographie LRP :**
- [Initial paper Bach 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
- [Tutorial LRP](http://iphome.hhi.de/samek/pdf/MonXAI19.pdf)
- [LRP for sentiment analysis](https://arxiv.org/pdf/1706.07206.pdf)
- [Clever Hans paper](https://www.nature.com/articles/s41467-019-08987-4.pdf)

### C. DeepLIFT

> 💡 **Idée.** DeepLIFT (Deep Learning Important FeaTures) compare l'activation de chaque neurone à une **activation de référence** (typiquement zéro ou une image moyenne) et propage les différences. Plus stable que les gradients bruts.
>
> Référence : [DeepLIFT paper](https://arxiv.org/pdf/1704.02685.pdf).

### D. Self-Attention pour la Feature Importance

> 💡 **Référence.** [Feature Importance Estimation with Self-Attention Networks](https://arxiv.org/pdf/2002.04464.pdf) (Skrjl et al. 2020) — proposent SAN (Self-Attention Network).

Trois tâches distinctes en feature importance :
- **Feature selection** — assigner 0 (irrelevant) ou 1 (relevant) à chaque feature.
- **Feature ranking** — score arbitraire pour trier.
- **Thresholding** — partitionner relevant / irrelevant à partir du ranking.

---

## XII. Interprétabilité ≠ causalité

> [!warning] Le piège fondamental
> Toutes ces méthodes donnent des **corrélations conditionnelles**, **pas** des effets causaux.
>
> Si une feature a une grande importance SHAP, ça ne veut **pas** dire qu'agir dessus changera la sortie en production. Pour ça, il faut un cadre causal (DAG, do-calculus, expérimentation) — voir [[SCM Pearl]].

> 💡 **En finance.** Si un modèle de scoring crédit met haut SHAP value sur "âge", ça ne veut pas dire que rajeunir un client baisserait son risque — ça veut dire que les jeunes du dataset ont eu plus de défauts (peut-être à cause de revenus plus faibles, etc.). L'intervention causale demande un autre framework.

---

## XIII. Bibliographie complète

**Livres et tutoriels** :
- [Interpretable ML book — Christoph Molnar](https://christophm.github.io/interpretable-ml-book/) — référence
- [Tutorial en français — Avisia](https://www.avisia.fr/news/tribune-expert/interpreter-modeles-machine-learning/)
- [Kaggle ML Explainability course](https://www.kaggle.com/learn/machine-learning-explainability)

**Papers fondamentaux** :
- Ribeiro et al. 2016 — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME)
- Lundberg & Lee 2017 — *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Rudin 2019 — *Stop Explaining Black Box ML Models for High Stakes Decisions* (Nature Machine Intelligence)
- Bach et al. 2015 — *On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation* (LRP)
- Shrikumar et al. 2017 — *Learning Important Features Through Propagating Activation Differences* (DeepLIFT)

**Articles techniques** :
- [Tutorial Shapley](https://data4thought.com/shapley.html) — intuition mathématique
- [Tutorial Kernel SHAP](https://data4thought.com/kernel_shap.html)
- [Push the limits of explainability](https://medium.com/swlh/push-the-limits-of-explainability-an-ultimate-guide-to-shap-library-a110af566a02)
- [Stop Permuting Features](https://towardsdatascience.com/stop-permuting-features-c1412e31b63f)
- [Interventional Tree Explainer](https://hughchen.github.io/its_blog/index.html#background_distribution)

---

## Annexe — récapitulatif des méthodes

| Méthode | Type | Scope | Modèle |
| :--- | :--- | :--- | :--- |
| **Coefficients linéaires** | Intrinsèque | Global | Linéaire |
| **Decision tree rules** | Intrinsèque | Global+local | Arbres |
| **GAM** | Intrinsèque | Global | Additif |
| **Permutation Importance** | Post hoc | Global | Agnostic |
| **PDP** | Post hoc | Global | Agnostic |
| **ICE** | Post hoc | Global+local | Agnostic |
| **ALE** | Post hoc | Global | Agnostic |
| **LIME** | Post hoc | Local | Agnostic |
| **Kernel SHAP** | Post hoc | Local | Agnostic |
| **Tree SHAP** | Post hoc | Local | Tree-based |
| **Deep SHAP** | Post hoc | Local | NN |
| **LRP** | Post hoc | Local | NN (ReLU) |
| **DeepLIFT** | Post hoc | Local | NN |

> 💡 **Le résumé en une phrase.** Pour les modèles à arbres en production : **Tree SHAP**. Pour les NN : **LRP composite** ou **Deep SHAP**. Pour le tabulaire générique en exploration : **Permutation Importance + PDP/ICE**. Pour expliquer une prédiction individuelle quand on a le temps de calcul : **Kernel SHAP**. Pour la critique méthodologique : **Cynthia Rudin 2019**.
