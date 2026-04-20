

JE SAIS PAS SI METRICS C VRAIMENT LE MEILLEUR TITRES LA


# I - Méthodes de validations

Validation croisé , par groupe, stratified k-fold ... 

Resampling methods méthodes de rééchantillonage,

cross validation = validation croisée 

validation croisée à k-blocs 



# II - Scores 
# Cadre général

Cas théorique: Le vocabulaire (positifs, négatifs) est issu de problématiques de détection (signal, dépistage médical) et peut se ramener aux concepts d'erreurs de première et seconde espèces du chapitre 14. On appelle faux positif une observation classée en G2 alors qu'elle appartient à G1 etc. Si l'on désigne par s le seuil au delà duquel on classe en G1 on définit la: 

$$
\text{sensibilité (\% de vrai positif)} = 1-\beta(s) = p(S > s | G_1)$$
$$
\text{spécificité (\% de vrai négatif)} = 1 - \alpha(s) = p(S < s | G_2)
$$

| Colonne 1                                 | Colonne 2                                 |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260420151117.png\|294]] | ![[Pasted image 20260420150336.png\|304]] |


Si $s=-\infty$ toute observation est classée en $G_1$ donc $1-\beta=1$ mais $\alpha=1$. En augmentant $s$ on diminue la sensibilité mais on augmente la spécificité. La courbe ROC donne alors l'évolution de la proportion de vrais positifs $1-\beta$ en fonction de la proportion de faux positifs $\alpha$.

Matrice de confusion

![[Pasted image 20260420152113.png|530]]
Table. Matrice de confusion


--- 

Métriques basé sur un seuil fixe
(a) L'Accuracy :
$$
\text { Accuracy }=\frac{V P+V N}{V P+V N+F P+F N}
$$
- **En gros :** C'est le nombre de fois où on a eu juste (les deux zones avec des petits points sur ton graphique) divisé par tout le monde.
- **Le piège :** Si tu as 99% de gens sains et 1% de malades, un modèle qui dit "tout le monde est sain" aura 99% d'accuracy mais sera inutile car il ne détectera aucun malade.

(b) Le F1-score : C'est la moyenne harmonique entre la **Précision** et le **Rappel** (Recall).
$$
F 1=2 \times \frac{\text { Précision × Rappel }}{\text { Précision }+ \text { Rappel }}
$$
- Précision (Ne pas se tromper quand on prédit "Positif") : $\frac{V P}{V P+F P}$
- Rappel / Sensibilité (Ne rater aucun "vrai" positif) : $\frac{V P}{V P+F N}$

Métrique de balayge (on fait varier tous les seuils $s \in \mathbb{R}$) 

--- 

Cas Optimal:


Cas réel: German credit dataset de UCI la 


Distribution de probabilités des scores $q_{\theta}(y=1 |x)$ je sais pas où on mets ça mais on a un seuil s=0.5 ici. 

![[Pasted image 20260420154559.png|550]]
Figure. Distribution de probabilité empiriques des prédiction $q_{\theta}(y=1 |x)$


![[Pasted image 20260420153124.png|377]]
Table. Matrice de confusion avec un seuil $s=0.50$


PEUT ETRE METTRE LE SCORE AUC QUON OBTIENT ICI + cas optimal limite du score etc ? + peut etre que c'est utile pour comparer des modèles
![[Pasted image 20260420153229.png|417]]


## l'AUC

On fait varier le seuil s car c'état utilisé par les radars

On divise la population en deux: 
* vrai malade "vrai positif" et "faux négatif"
* vrai sains en "vrais négatif" et "faux positif" $\alpha=5\%$ et $1-\alpha=95\%$ 


courbe ROC (Received Operator courbe)

on a aussi

$$
AUC = \int_{s=-\infty}^{s=+\infty} (1 - \beta(s)) d\alpha(s)
$$

## F1 score, accuracy




## Ce que je pas comprends c comment on choisit une métrique ?





## Le cas déséquilibré

On utilise Precision recall Curve 


et $AUC_{PR}$ 

Papier: Bad practices in evaluation methodology relevant to class imbalacned pb, Babac, Machlica 2018


Tutoriel sur sur le Precision Recall Curve
https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/



## Courbe business

Introduction en fonction des use cases: 
* marketing, fraud: 

### Courbe de discrimination : credit score 

On peut visualiser aussi les fonctions de répartitions des desux groupres "courbe de discrimination cumulative". On va multiplier le score de probabilité par 100 .

La figure suivante : si l'on décide qu'un client ayant un score inférieur à 550 est "mauvais" on détecte environ 58.3% de cette catégorie, tout en ne déclarant "mauvais" que 14.3% des "bons". Inversement si le seuil pour être classé "bon" est 750, on reconnaît environ la moitié de cette catégorie, et seuls 17.3% des "mauvais" sont considérés à tort comme des "bons".

![[Pasted image 20260420153925.png]] 
![[Pasted image 20260420154513.png|193]]


## Lift curve 


https://analyticsgyanblog.wordpress.com/category/lift-curve/
https://www.youtube.com/watch?v=IwCUZQllVVI
https://www.youtube.com/watch?v=fZ7xgnAfKM4


## Gain chart







