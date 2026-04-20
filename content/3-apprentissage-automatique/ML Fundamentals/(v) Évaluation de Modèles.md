# I - Fondements Théoriques

Le vocabulaire (positifs, négatifs) est issu de problématiques de **détection** (signal, dépistage médical) et peut se ramener aux concepts d'**erreurs de première et seconde espèces**. On appelle **faux positif** une observation classée positive alors qu'elle appartient au groupe des sains, etc. 

Si l'on désigne par **s** le seuil au delà duquel on classe comme positif, on définit :
- **G₁** : groupe des **malades** (positifs)
- **G₂** : groupe des **sains** (négatifs)

$$
\text{sensibilité (\% de vrai positif)} = 1-\beta(s) = p(S > s | G_1)
$$
$$
\text{spécificité (\% de vrai négatif)} = 1 - \alpha(s) = p(S < s | G_2)
$$

| Colonne 1                                 | Colonne 2                                 |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260420151117.png\|294]] | ![[Pasted image 20260420150336.png\|304]] |


**Trade-off sensibilité/spécificité :**
- Si $s = -\infty$ → toute observation classée en $G_1$ → **sensibilité = 100%** mais **spécificité = 0%**
- En **augmentant $s$** → ↘️ sensibilité mais ↗️ spécificité
- En **diminuant $s$** → ↗️ sensibilité mais ↘️ spécificité

> **💡 Intuition clé :** La courbe ROC visualise ce trade-off en montrant l'évolution des vrais positifs $(1-\beta)$ en fonction des faux positifs $(\alpha)$ quand on fait varier le seuil.

## Matrice de confusion

En pratique, on résume ces concepts dans une matrice de confusion qui compte les 4 cas possibles : vrais positifs (VP), vrais négatifs (VN), faux positifs (FP) et faux négatifs (FN).

![[Pasted image 20260420152113.png|386]]
Table. Matrice de confusion

### 📋 **Exemple concret**
Sur 100 patients (50 malades, 50 sains), avec un modèle ayant une **sensibilité de 80%** et une **spécificité de 90%** :

- **VP = 40** (détecte 40/50 malades)
- **FN = 10** (rate 10/50 malades) 
- **VN = 45** (détecte 45/50 sains)
- **FP = 5** (se trompe sur 5/50 sains)

**Vérification :** Sensibilité = 40/50 = 80%, Spécificité = 45/50 = 90%

# II - Méthodes de validations

## A. Problématique : Train/Test vs Réalité

- Pourquoi on ne peut pas juste tester sur les données d'entraînement
- Overfitting et généralisation
- Le problème de data leakage

## B. Les Méthodes de validations

Le classique 60/20/20

Validation croisée etc
# III - Métriques d'Évaluation
## A. Cas Normal/Équilibré

### Métriques à seuil fixe

Rappel de la **matrice de confusion** pour les définitions :

![[Pasted image 20260420152113.png|386]]
Table. Matrice de confusion

**(a) L'Accuracy :**
$$
\text{Accuracy} = \frac{VP + VN}{VP + VN + FP + FN}
$$
- **En gros :** Proportion de prédictions correctes (VP + VN) sur le total
- **Le piège :** Si 99% de gens sains et 1% de malades, un modèle qui dit "tout le monde est sain" aura 99% d'accuracy mais sera inutile

**(b) Précision et Rappel :**
- **Précision** (ne pas se tromper quand on prédit "Positif") : $\frac{VP}{VP + FP}$
- **Rappel/Sensibilité** (ne rater aucun "vrai" positif) : $\frac{VP}{VP + FN}$

**(c) Le F1-score :**
Moyenne harmonique entre **Précision** et **Rappel** :
$$
F1 = 2 \times \frac{\text{Précision × Rappel}}{\text{Précision + Rappel}}
$$

### Exemple concret : German Credit Dataset

Prenons un exemple concret pour illustrer ces métriques. Le **German Credit Dataset** (UCI) prédit le risque de crédit. Voici la distribution des scores prédits $q_{\theta}(y=1|x)$ par notre modèle :

![[Pasted image 20260420154559.png|550]]
Figure. Distribution de probabilité empiriques des prédictions $q_{\theta}(y=1|x)$

Avec un **seuil s = 0.5**, on obtient la matrice de confusion suivante :

![[Pasted image 20260420153124.png|377]]
Table. Matrice de confusion avec un seuil $s = 0.50$

**Calculs des métriques :**
- Accuracy = (VP + VN) / Total = ...
- Précision = VP / (VP + FP) = ...
- Rappel = VP / (VP + FN) = ...
- F1-score = ...

### Métriques à seuil variable : ROC, AUC

**Problématique :** Plutôt que de fixer arbitrairement un seuil (s = 0.5), analysons comment le modèle se comporte pour **tous les seuils possibles**.

#### Théorie : La courbe ROC

**Contexte historique :** La courbe ROC ("Receiver Operating Characteristic") était utilisée par les opérateurs radar pendant la 2ème guerre mondiale pour distinguer signal utile vs bruit.

| Distribution des classes | Trade-off sensibilité/spécificité |
| --- | --- |
| ![[Pasted image 20260420150336.png\|304]] | **En variant le seuil s :**<br/>• Si $s = -\infty$ → sensibilité = 100%, spécificité = 0%<br/>• Si $s = +\infty$ → sensibilité = 0%, spécificité = 100%<br/>• **Trade-off :** ↗️ sensibilité ⟺ ↘️ spécificité |

> **💡 Intuition clé :** La courbe ROC trace l'évolution de la **sensibilité** (axe Y) vs **1-spécificité** (axe X) quand on fait varier le seuil de $-\infty$ à $+\infty$.

#### L'AUC (Area Under Curve)

L'**AUC** quantifie la performance globale du modèle :

$$
AUC = \int_{s=-\infty}^{s=+\infty} (1 - \beta(s)) d\alpha(s)
$$

**Interprétation intuitive :** AUC = probabilité qu'un échantillon positif ait un score plus élevé qu'un échantillon négatif choisis au hasard.

- **AUC = 0.5** : modèle aléatoire (ligne diagonale)
- **AUC = 1.0** : modèle parfait
- **AUC > 0.8** : généralement considéré comme bon

#### Exemple sur German Credit Dataset

![[Pasted image 20260420153229.png|417]]
Figure. Courbe ROC pour le German Credit Dataset

**Analyse :**
- AUC ≈ 0.XX (à calculer depuis la courbe)
- **Performance** : [à interpréter selon la valeur]



## B. Cas Déséquilibré
### Métriques à seuil fixe

Motivation : Échec de la métrique Accuracy


Papier: Bad practices in evaluation methodology relevant to class imbalacned pb, Babac, Machlica 2018

### Métriques à seuil variable : Precision-Recall Curve, AUC-PR

Tutoriel sur sur le Precision Recall Curve
https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/

On utilise Precision recall Curve 


et $AUC_{PR}$ 

## C. Comment choisir une métrique ?

Après avoir vu toutes ces métriques (accuracy, F1, ROC-AUC, PR-AUC...), la question cruciale devient : **laquelle utiliser ?** La réponse dépend de trois facteurs clés :

### 1. Le contexte business et les coûts d'erreur

Tous les modèles font des erreurs, mais **toutes les erreurs ne se valent pas**. L'histoire de la **matrice de coûts** (cost matrix) remonte aux années 1960 en recherche opérationnelle, où l'on a formalisé l'idée que les conséquences des faux positifs et faux négatifs peuvent être très différentes.

**Matrice de coûts générale :**
```
                   Prédiction
Réalité      │  Négatif    │  Positif
─────────────┼─────────────┼─────────────
Négatif      │      0      │   Coût FP
Positif      │   Coût FN   │      0
```

**Exemples typiques :**
- **Médical** : Coût FN >> Coût FP (rater cancer >> fausse alerte)
- **Spam** : Coût FP >> Coût FN (bloquer vrai email >> laisser spam)
- **Crédit** : Coût FN >> Coût FP (prêter à risqué >> refuser bon client)

### 2. L'équilibre des classes

- **Classes équilibrées** (50/50) → Accuracy, F1-score fiables
- **Classes déséquilibrées** (99/1) → Accuracy = piège ! Privilégier PR-curve

### 3. L'objectif de l'application

- **Seuil fixe imposé** → Métriques à seuil fixe (Accuracy, F1)
- **Seuil à optimiser** → Courbes (ROC, PR) pour choisir le point optimal
- **Comparaison de modèles** → AUC (ROC ou PR selon équilibre)

### Exemple : German Credit Dataset (continued)

**Contexte :** Prédire le défaut de paiement d'un client

**Analyse de la matrice de coûts :**
```
                   Prédiction
Réalité      │  Bon client │  Défaut
─────────────┼─────────────┼─────────────
Bon client   │      0      │ Opportunité perdue (~€100)
Défaut       │ Perte prêt  │      0
             │ (~€5000)    │
```

**Ratio de coûts : ~50:1** → Le coût d'accepter un mauvais client est **50 fois plus élevé** que refuser un bon client.

**Conséquences pour le choix de métrique :**
- ❌ **Accuracy** : optimise performance globale, ignore les coûts asymétriques
- ❌ **Precision** : focus sur éviter les FP (pas notre priorité principale)
- ✅ **Recall** : focus sur éviter les FN (notre priorité !)
- ✅ **Courbe ROC** : pour choisir un seuil qui minimise le coût total
- ✅ **Courbes business** : pour quantifier directement l'impact financier 


MAIS YA PAS LHISTOIRE DU COST SENSTIVEI LOGISTIC REGRESSION SINON ??????????????????????????????????????????????????????

# IV - Courbe business

Introduction en fonction des use cases: 
* marketing, fraud: 

## A. Courbe de discrimination : credit score 

Exemple (Credit Fraud UCI suite).
On peut visualiser aussi les fonctions de répartitions des desux groupres "courbe de discrimination cumulative". On va multiplier le score de probabilité par 100 .

La figure suivante : si l'on décide qu'un client ayant un score inférieur à 550 est "mauvais" on détecte environ 58.3% de cette catégorie, tout en ne déclarant "mauvais" que 14.3% des "bons". Inversement si le seuil pour être classé "bon" est 750, on reconnaît environ la moitié de cette catégorie, et seuls 17.3% des "mauvais" sont considérés à tort comme des "bons".

![[Pasted image 20260420153925.png]] 
![[Pasted image 20260420154513.png|193]]


## B. Lift curve 


https://analyticsgyanblog.wordpress.com/category/lift-curve/
https://www.youtube.com/watch?v=IwCUZQllVVI
https://www.youtube.com/watch?v=fZ7xgnAfKM4


## C. Gain chart







