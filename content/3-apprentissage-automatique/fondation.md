---
title: Fondations
---

# Fondations du Machine Learning

> Prédiction, biais-variance et régularisation.

---

## 1. Inférer vs prédire

En statistique classique on fait de l'**inférence** : à partir de données, que peut-on dire sur les mécanismes sous-jacents ? On estime des paramètres, on teste des hypothèses. La question centrale est *"qu'est-ce que mes données me disent sur le monde ?"*

En machine learning, l'objectif est différent. On s'intéresse à la **prédiction** : mon modèle entraîné sur des données passées va-t-il bien se comporter sur de nouvelles données qu'il n'a jamais vues ? La question devient *"est-ce que ma fonction généralise ?"*

Ce changement d'objectif change tout le cadre d'évaluation. On ne cherche plus à comprendre — on cherche à généraliser.

---

## 2. L'erreur de prédiction (EPE)

### (i) Le modèle de bruit

On suppose qu'il existe une vraie relation entre les features $X$ et la cible $Y$, perturbée par du bruit :

$$Y = f(x) + \varepsilon \qquad \mathbb{E}[\varepsilon] = 0, \quad \text{Var}(\varepsilon) = \sigma^2$$

Deux points importants. D'abord, le $\varepsilon$ n'est pas une hypothèse sur le modèle qu'on choisit — c'est une hypothèse sur le **monde**. Il représente les variables non observées et la variabilité intrinsèque du phénomène. Que l'on utilise une régression linéaire ou un random forest, ce bruit est là dans les données.

Ensuite, un modèle non-paramétrique comme un random forest ne supprime pas ce bruit. Ce qu'il fait c'est réduire le biais structurel — asymptotiquement il peut approximer n'importe quelle $f$. Mais le $\sigma^2$ reste irréductible.

### (ii) Décomposition de l'EPE

L'**Expected Prediction Error** mesure l'erreur de prédiction moyenne de notre estimateur $\hat{f}$ :

$$\text{EPE} = \mathbb{E}\left[(Y - \hat{f}(x))^2\right]$$

On substitue $Y = f(x) + \varepsilon$ et on développe $(f(x) - \hat{f}(x) + \varepsilon)^2$ :

$$= \mathbb{E}\left[(f(x) - \hat{f}(x))^2\right] + 2\underbrace{\mathbb{E}[\varepsilon]\cdot\mathbb{E}[f(x)-\hat{f}(x)]}_{=0} + \mathbb{E}[\varepsilon^2]$$

Le terme croisé est nul car $\varepsilon$ est indépendant de $\hat{f}$ et $\mathbb{E}[\varepsilon] = 0$. On obtient :

$$\boxed{\text{EPE} = \underbrace{\mathbb{E}\left[(f(x) - \hat{f}(x))^2\right]}_{\text{erreur réductible}} + \underbrace{\sigma^2}_{\text{irréductible}}}$$

---

## 3. La décomposition biais-variance

### (i) Dérivation

On décompose l'erreur réductible en ajoutant et soustrayant $\mathbb{E}[\hat{f}(x)]$ — ajouter zéro intelligemment :

$$\mathbb{E}\left[\Big(\underbrace{f(x) - \mathbb{E}[\hat{f}(x)]}_{B\ \text{(constante)}} + \underbrace{\mathbb{E}[\hat{f}(x)] - \hat{f}(x)}_{C\ \text{(v.a., moyenne 0)}}\Big)^2\right]$$

On développe $(B + C)^2 = B^2 + 2BC + C^2$. Le terme croisé $\mathbb{E}[BC] = B \cdot \mathbb{E}[C] = 0$. Il reste :

$$\boxed{\text{EPE} = \underbrace{\left(f(x) - \mathbb{E}[\hat{f}(x)]\right)^2}_{\text{biais}^2} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{variance}} + \sigma^2}$$

### (ii) Interprétation

Le **biais** mesure l'erreur systématique : en moyenne sur tous les datasets d'entraînement possibles, est-ce que $\hat{f}$ vise juste ? Un modèle linéaire ajusté sur une vraie relation quadratique aura un biais structurel non nul — peu importe la quantité de données, la droite ne peut pas capturer la courbe.

La **variance** mesure la sensibilité aux données : si je ré-entraîne sur un autre dataset tiré de la même distribution, est-ce que $\hat{f}(x)$ change beaucoup ? Un modèle trop flexible va mémoriser le bruit de chaque dataset.

> **Attention :** ce n'est pas le même biais/variance qu'en inférence. En inférence, $\text{biais}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$ portait sur un paramètre scalaire. Ici on parle de la fonction entière $\hat{f}$ — sa capacité à représenter $f$ et sa stabilité face aux données.

### (iii) Exemple numérique

Supposons $f(x) = x^2$. On compare régression linéaire (2 paramètres) vs polynôme degré 10 (11 paramètres), entraînés sur des centaines de datasets de 20 points. On regarde les prédictions en $x = 2$ (vraie valeur : $f(2) = 4$).

- **Linéaire** → prédictions stables ($\text{Var}$ faible), mais moyenne $\approx 2.5$ ($\text{biais}^2$ élevé) : une droite ne peut structurellement pas capturer $x^2$.
- **Degré 10** → prédictions qui sautent partout ($\text{Var}$ élevée), mais moyenne $\approx 4.0$ ($\text{biais}^2$ faible) : assez de liberté pour approximer $x^2$, mais aussi pour mémoriser le bruit.

### (iv) Le trade-off

Biais et variance évoluent en sens inverse avec la complexité du modèle. L'erreur totale dessine une courbe en U :

<div style="text-align: center;">

![Figure 1. Trade-off biais-variance : l'erreur totale (violet) est la somme du biais² (orange) et de la variance (bleu). Le minimum définit la complexité optimale du modèle.](images/fondation/im1.png)

</div>

Figure 1. Trade-off biais-variance : l'erreur totale (violet) est la somme du biais² (orange) et de la variance (bleu). Le minimum définit la complexité optimale du modèle.

Trop simple, le modèle sous-fit (*underfitting*, fort biais). Trop complexe, il sur-fit (*overfitting*, forte variance). L'objectif est de trouver le point minimum de l'erreur totale.

---

## 4. Au-delà du trade-off classique : le double descent

La courbe en U est le cadre classique. Belkin et al. (2019) montrent qu'elle est incomplète. Si on continue à augmenter la complexité au-delà du **seuil d'interpolation** — le point où le modèle fit parfaitement les données d'entraînement (erreur train $= 0$) — l'erreur de test explose puis **redescend** :

<div style="text-align: center;">

![Figure 3. Double descent. Dans le régime classique (gauche), on retrouve la courbe en U. Au seuil d'interpolation, l'erreur de test explose. Dans le régime overparamétrisé (droite), elle redescend en dessous du minimum classique.](images/fondation/im3.png)

</div>

Figure 3. Double descent. Dans le régime classique (gauche), on retrouve la courbe en U. Au seuil d'interpolation, l'erreur de test explose. Dans le régime overparamétrisé (droite), elle redescend en dessous du minimum classique.

L'intuition : parmi tous les modèles qui interpolent parfaitement les données, certains sont plus lisses que d'autres. Avec plus de paramètres que d'équations, le système est sous-déterminé. La descente de gradient converge naturellement vers la solution de **norme minimale** (*minimum norm solution*), qui est souvent la plus régulière.

C'est une **régularisation implicite** — par opposition au $\lambda$ explicite de Ridge/Lasso. L'architecture et l'algorithme d'optimisation jouent le rôle de régularisateur. C'est ce qui explique pourquoi les grands réseaux de neurones, bien qu'interpolant parfaitement le train set, généralisent néanmoins bien.

---

## 5. La limite du cadre : interpolation vs extrapolation

Tout ce qu'on a construit repose sur une hypothèse implicite fondamentale : les données de test sont tirées de la **même distribution** que les données d'entraînement — l'hypothèse i.i.d. (*independent and identically distributed*).

Cette hypothèse définit une limite de validité précise :
- **Interpolation** : $x_{\text{new}}$ appartient au domaine couvert par les données d'entraînement. Les garanties théoriques tiennent.
- **Extrapolation** : $x_{\text{new}}$ est en dehors. L'EPE ne dit plus rien.

En régression 1D c'est visible. En haute dimension, la limite est sournoise : la zone couverte par les données est une région très creuse dans un espace à $p$ dimensions. Un point peut sembler raisonnable sur chaque feature individuellement, mais correspondre à une combinaison jamais vue.

<details>
<summary>Application en finance — market impact et distribution shift</summary>

En finance ce problème est amplifié par deux mécanismes.

D'abord, les marchés sont **non-stationnaires** : un modèle entraîné sur un régime de faible volatilité extrapole dès qu'il entre dans un régime différent, même si les valeurs individuelles des features semblent normales.

Ensuite, le modèle lui-même peut modifier la distribution : un modèle de market impact suffisamment utilisé change le marché qu'il prédit. Il sort mécaniquement de sa distribution d'entraînement.

Le terme générique pour ce phénomène est le **distribution shift**. C'est un champ de recherche actif en ML (domain adaptation, out-of-distribution generalization).

</details>

---

## 6. Évaluer la généralisation : le sampling

### (i) Train / Validation / Test

On ne peut pas mesurer la généralisation sur les données d'entraînement — on mesurerait la mémorisation. On découpe en trois ensembles aux rôles distincts :

- **Train set** : estime $\hat{\theta}$. Le modèle voit ces données.
- **Validation set** : choisit les hyperparamètres ($\lambda$, architecture...). Indirectement contaminé par nos décisions de modélisation.
- **Test set** : estime l'erreur finale. Ne doit **jamais** être consulté avant la toute fin — sinon on optimise implicitement pour lui.

### (ii) K-fold cross-validation

Quand le dataset est petit, sacrifier 20% en validation est coûteux. La **k-fold cross-validation** résout ce problème :

<div style="text-align: center;">

![Figure 4. K-fold cross-validation (k=5). À chaque fold, un bloc différent sert de validation (orange) et les quatre autres servent d'entraînement (vert). Chaque observation passe exactement une fois en validation. Le score CV est la moyenne des k erreurs.](images/fondation/im4.png)

</div>

Figure 4. K-fold cross-validation (k=5). À chaque fold, un bloc différent sert de validation (orange) et les quatre autres servent d'entraînement (vert). Chaque observation passe exactement une fois en validation. Le score CV est la moyenne des k erreurs.

En pratique pour choisir $\lambda$ : on fait tourner la CV pour chaque valeur candidate, on garde celle qui minimise l'erreur CV, puis on ré-entraîne le modèle final sur **tout** le train set avec ce $\lambda$. Le test set n'a toujours pas été touché.

### (iii) Walk-forward validation pour les séries temporelles

Le k-fold suppose que les observations sont échangeables — qu'on peut les mélanger librement. En séries temporelles c'est faux : utiliser le futur pour prédire le passé constitue une fuite d'information (*data leakage*). On utilise le **walk-forward validation** :

<div style="text-align: center;">

![Figure 5. Walk-forward validation. Le train set grandit à chaque split, la validation est toujours dans le futur par rapport au train. Le test set est bloqué à la fin de la série — la période la plus récente, celle qui ressemble le plus à la production.](images/fondation/im5.png)

</div>

Figure 5. Walk-forward validation. Le train set grandit à chaque split, la validation est toujours dans le futur par rapport au train. Le test set est bloqué à la fin de la série — la période la plus récente, celle qui ressemble le plus à la production.

### (iv) Data leakage

Le problème plus général est le **data leakage** : toute information du futur ou du test set qui se retrouve dans le train, souvent de façon invisible. Exemples courants :

- Normaliser les features sur tout le dataset **avant** de splitter.
- Imputer des valeurs manquantes avec la moyenne globale avant de splitter.
- Calculer des features sur des fenêtres glissantes incluant des points futurs.

Dans tous ces cas le modèle semble excellent en CV et s'effondre en production.

> **Règle** : le split est toujours la première opération, avant tout preprocessing. Tout ce qui est appris sur les données (moyenne, variance, encodages) doit être appris sur le train set uniquement, puis appliqué au test set.

---

## Synthèse

| Concept | Rôle |
|---|---|
| EPE = biais² + variance + $\sigma^2$ | Décompose l'erreur de prédiction en trois termes |
| Biais | Erreur structurelle du modèle — indépendante de la taille des données |
| Variance | Sensibilité aux fluctuations des données d'entraînement |
| Double descent | Au-delà du seuil d'interpolation, régularisation implicite par le minimum norm |
| Hypothèse i.i.d. | Condition de validité du cadre — extrapolation et distribution shift l'invalident |
| Train / Val / Test + CV | Infrastructure pour estimer honnêtement la généralisation |
