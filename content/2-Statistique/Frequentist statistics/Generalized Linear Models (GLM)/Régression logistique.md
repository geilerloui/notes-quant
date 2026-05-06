# Régression logistique

## Introduction conceptuelle

### Pourquoi pas la régression linéaire ?

**Contexte :** On veut prédire des classes binaires (0/1) mais la régression linéaire classique pose problème :

- **Régression linéaire :** $\hat{y} = w_0 + w_1x$ → peut donner -5, 2.3, 127... 
- **Classes binaires :** $y \in \{0, 1\}$ seulement

**Problèmes :**
1. Prédictions hors $[0,1]$ → impossible à interpréter comme probabilités
2. Hétéroscédasticité → variance non-constante 
3. Linéarité inadaptée → transition abrupte entre classes

### Plan d'attaque

**Idée clé :** Au lieu de modéliser directement y, on modélise quelque chose de "linéaire" qu'on peut transformer.

**Les 3 espaces interconnectés :**
1. **Espace des classes :** $y \in \{0, 1\}$
2. **Espace log-odds :** $\log(p/(1-p)) \in \mathbb{R}$ (linéaire !)  
3. **Espace probabilité :** $p \in [0, 1]$

Navigation : Classe ↔ Log-odds (linéaire) ↔ Probabilité

**Stratégie :** 
1. **Modéliser linéairement** les log-odds : $\log(p/(1-p)) = w_0 + w_1x_1 + \ldots$
2. **Transformer** via sigmoïde : $p = \sigma(\text{log-odds}) = \frac{1}{1+e^{-\text{log-odds}}}$
3. **Classifier** : $\hat{y} = 1$ si $p > 0.5$, sinon $\hat{y} = 0$

**Résultat :** Frontière de décision linéaire dans l'espace des features, mais prédictions probabilistes !

### Comprendre les odds et log-odds

**Transformation magique :**
- **Odds :** rapport $p/(1-p) \in [0, +\infty]$
- **Log-odds :** $\log(p/(1-p)) \in \mathbb{R}$ → espace linéaire !

**Avantages :**
- Symétrique : $\log\text{-odds}(p) = -\log\text{-odds}(1-p)$
- Bornes infinies : $p \to 0 \Rightarrow \log\text{-odds} \to -\infty$, $p \to 1 \Rightarrow \log\text{-odds} \to +\infty$
- Linéarité : $\log\text{-odds} = w_0 + w_1x_1 + w_2x_2 + \ldots$

#### Exemples visuels des odds

| Définition des odds                            | Probabilité vs Odds                            | Probabilité de perdre                              |
| ---------------------------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| ![[images/2-Statistiques/Frequentist/regression-logistique/im1.png\|242]] | ![[images/2-Statistiques/Frequentist/regression-logistique/im2.png\|299]] | ![[images/2-Statistiques/Frequentist/regression-logistique/im3 (1).png\|328]] |

**Définition :** Les odds sont le rapport entre "quelque chose qui arrive" et "quelque chose qui n'arrive pas".

$$
Odds = \frac{\text{something happening (ie my team \textcolor{oceanblue}{wins})}}{\text{something not happening (my team \textcolor{cornellred}{not winning}}}
$$

**Exemple :** Si les chances de gagner sont de 5 contre 3, alors les odds sont $5/3 = 1.67$.

**Connexion avec les probabilités :**

$$
\frac{p}{(1-p)} = \frac{\text{probability of \textcolor{oceanblue}{wining}}}{1-\text{probability of \textcolor{oceanblue}{wining}}} = \frac{5/8}{3/8} = \frac{5}{3} = Odds
$$

#### Motivation pour les log-odds

| Asymétrie des odds | Symétrie des log-odds |
|-------------------|-------------------|
| ![[images/2-Statistiques/Frequentist/regression-logistique/im4.png\|433]] | ![[images/2-Statistiques/Frequentist/regression-logistique/im5.png\|450]] |

**Problème d'asymétrie :** Les odds de perdre vs gagner créent une asymétrie (odds $< 1$ vs odds $> 1$).

**Solution - Les log-odds :** Pour éviter ce problème, on utilise les log-odds qui symétrient la situation :
- Odds contre 1 à 6 → $\log(1/6) = -1.79$
- Odds en faveur 6 à 1 → $\log(6) = +1.79$


EN FAIT LES LOG ODDS NE SONT DEFINI QUE POUR UNE POPULATION GENERALE comme en stat si j'ai un sample de la population eg X=50 ans le mien est y=1 survécu mais en réalité dans la population on a 1000 personnes de 50 ans avec 80% de survécu et 20% de non log(odds(X=50 ans)) = log80%/20%=0.60

### La fonction sigmoïde

**Définition :**
$$
\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1 + e^x}
$$

**Dérivée utile :**
$$
\frac{d \sigma}{d x}=\sigma(x)(1-\sigma(x))
$$

**Application en régression logistique :**
$$
\sigma(x) = p(y=1 \mid x) =  \frac{e^{w_0 + w_1 x}}{1+ e^{w_0 + w_1 x}} = \frac{e^{log(odds)}}{1+e^{log(odds)}}
$$

**Exemple de transformation :** Pour mapper un point des log-odds vers l'espace probabilité :

$$
p(y=1 \mid \textcolor{cornellred}{x_1}) = \frac{e^{-2.1}}{1+e^{-2.1}} = 0.10 \quad 
p(y=1 \mid \textcolor{cornellred}{x_6}) = \frac{e^{2.4}}{1+e^{2.4}} = 0.90
$$

---

## Cas 2D continu

### Dataset d'exemple

On va initialiser $w_0=-5$ et $w_1=0.15$ pour le calcul des $\text{log-odds}_i=w_0+w_1 x_i^{(1)}$ et $\sigma(x)=\frac{e^{\text{log-odds}}}{1+e^{\text{log-odds}}}$

$$
\begin{array}{|c|c|c|c|c|c|}
\hline
\text{Souris} & \text{Poids (g)} & \text{Obèse } y & \text{log-odds} & \sigma(x) & \hat{y} \\
\hline
1 & 20 & 0 & -2.0 & 0.12 & 0 \\
2 & 22 & 0 & -1.7 & 0.15 & 0 \\
3 & 25 & 0 & -1.25 & 0.22 & 0 \\
4 & 28 & 1 & -0.8 & 0.31 & 0 \\
5 & 30 & 0 & -0.5 & 0.38 & 0 \\
6 & 32 & 1 & -0.2 & 0.45 & 0 \\
7 & 34 & 0 & 0.1 & 0.52 & 1 \\
8 & 38 & 1 & 0.7 & 0.67 & 1 \\
9 & 42 & 1 & 1.3 & 0.79 & 1 \\
\hline
\end{array}
$$

### Étape 1 — Données brutes

La logique de la régression logistique est de naviguer entre trois espaces : 
Obésité $(y)$ → log-odds → probabilité $p(y=1|x)$

![[Pasted image 20260418145222.png|377]]

### Étape 2 — Modélisation linéaire des log-odds

On va initialiser $w_0=-5$ et $w_1=0.15$ pour le calcul des $\text{log-odds}_i=w_0+w_1 x_i^{(1)}$

**Note importante :** Les log-odds peuvent atteindre ±∞, résolvant le problème des bornes de la régression linéaire classique.

![[Pasted image 20260418151041.png|361]]

### Étape 3 — Optimisation (MLE + Gradient Descent)

Pour trouver la meilleure droite (étape 2) qui donne la meilleure courbe (étape 4), on multiplie la probabilité de chaque point. On "bouge" ensuite notre droite jusqu'à ce que ce produit soit maximum.

**Fonction de vraisemblance :**
$$L(\beta)=\prod_{y_i=1} P\left(x_i\right) \times \prod_{y_i=0}\left(1-P\left(x_i\right)\right)$$

où $p(x_i) = \sigma(w_0 + w_1 x_i)$

**Gradient Descent :** On répète les étapes suivantes :

$$
\mathbf{w}:=\mathbf{w}-\alpha \sum_{n=1}^{N}\left(p_{n}-y_{n}\right) \phi_{n}
$$

### Étape 4 — Transformation sigmoïde et prédiction

$\sigma(x)=\frac{e^{\text{log-odds}}}{1+e^{\text{log-odds}}}$

Une fois qu'on a les paramètres $w_0$ et $w_1$, on peut calculer les probabilités pour chaque échantillon : $p(y=1|x_1) = 0.12$, $p(y=1|x_2) = 0.15$, etc. On fixe un seuil $\eta = 0.5$ : chaque échantillon au-dessus de ce seuil est classifié comme obèse.

![[Pasted image 20260418151313.png|363]]

---

## Extension 3D continue

### Données

![[Pasted image 20260418152205.png|283]]

### Visualisation 3D des données et plan des log-odds

| Visualisation 3D des données | Plan des log-odds |
|------------------------------|-------------------|
| ![[Pasted image 20260418152730.png]] | ![[Pasted image 20260418154035.png]] |

**Plan des log-odds :** Ce graphique montre UN choix de paramètres : $w_0=-6$, $w_1=0.08$, $w_2=0.15$

### Surface sigmoïde : Impact des coefficients

Il peut être surprenant que la sigmoïde ressemble à un plan courbe, mais ceci est dû aux coefficients $w_i$ qui sont petits. S'ils étaient beaucoup plus grands, ce serait beaucoup plus abrupte.

#### Coefficients faibles

Ce graphique montre UN choix de paramètres : $w_0=-6$, $w_1=0.08$, $w_2=0.15$

| Surface 3D                           | Projection 2D                        |
| ------------------------------------ | ------------------------------------ |
| ![[Pasted image 20260418155040.png]] | ![[Pasted image 20260418155550.png]] |

#### Coefficients forts

Ce graphique montre UN choix de paramètres : $w_0=-15$, $w_1=0.3$, $w_2=0.4$

**Équations :**

$\text{log-odds} = -15 + 0.3 \times \text{poids} + 0.4 \times \text{âge}$

**Frontière de décision :**

$\text{âge} = \frac{15 - 0.3 \times \text{poids}}{0.4}$

**Probabilité :**

$P(\text{obèse}) = p(y=1)= \sigma(-15 + 0.3 \times \text{poids} + 0.4 \times \text{âge})$

| Surface 3D | Projection 2D |
|------------|---------------|
| ![[Pasted image 20260418155807.png\|305]] | ![[Pasted image 20260418160136.png\|305]] |

---

## Cas discret

### Dataset avec variables catégorielles

![[Pasted image 20260418175327.png|401]]

**Caractéristique :** Avec des données discrètes, la régression logistique produit une grille de probabilités plutôt qu'une surface continue. Chaque combinaison de features catégorielles correspond à une probabilité spécifique.

---

## Extensions avancées

### Interprétation des coefficients et R²

**À développer :** Interprétation des coefficients, calcul du pseudo-R², mesures de performance.

% Table des coefficients de régression logistique
$$
\begin{array}{|l|c|c|c|c|}
\hline
\textbf{Variable} & \textbf{Estimate} & \textbf{Std. Error} & \textbf{z value} & \textbf{Pr(>|z|)} \\
\hline
\text{(Intercept)} & 1.239 & 10.774 & 0.115 & 0.9084 \\
\text{Poids} & -0.483 & 0.772 & -0.625 & 0.5321 \\
\text{Âge} & 1.052 & 1.084 & 0.970 & 0.3321 \\
\hline
\end{array}
$$

% Métriques du modèle
**Métriques du modèle :**
- **Accuracy :** 88.9% (8/9 prédictions correctes)
- **Pseudo R² :** 0.463
- **AIC :** 12.637
- **Log-Likelihood :** -3.319

% Équation du modèle estimé
**Équation du modèle estimé :**
$$\text{log-odds} = 1.239 - 0.483 \times \text{Poids} + 1.052 \times \text{Âge}$$

**Probabilité prédite :**
$$P(\text{Obèse} = 1) = \sigma(1.239 - 0.483 \times \text{Poids} + 1.052 \times \text{Âge})$$
Interprétation: 
- **Coefficient du poids : -0.483**
- **Coefficient de l'âge : 1.052**

Ces coefficients représentent **l'impact sur les log-odds** d'une augmentation d'1 unité de la variable. 
On transforme en odds ratio:
$OR_{poids} = e^{(-0.483)} = 0.617$

Pour l'âge
$OR_{âge} = e^(1.052) = 2.863$

**✅ Interprétation correcte :** "Chaque gramme supplémentaire **multiplie les odds par 0.617**, soit les **réduit de 38.3%**"

**✅ Interprétation correcte :** "Chaque semaine supplémentaire **multiplie les odds par 2.863**, soit les **augmente de 186.3%**"

![[Pasted image 20260418185333.png]]


### Régularisation

**Effet sur les surfaces :** La régularisation contrôle la "raideur" de la surface sigmoïde :
- **Forte régularisation** → surface douce, coefficients réduits
- **Faible régularisation** → surface abrupte, risque de sur-apprentissage

**À intégrer :** Comparaisons visuelles montrant l'impact de différents niveaux de régularisation.