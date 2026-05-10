# I - Évaluation en classification binaire

Le choix des métriques dépend fortement de la proportion de positifs $\pi = n_1 / N$ dans l'échantillon.

| $\pi$ (taux de positifs) | Qualification | Exemples | Métriques adaptées |
| :------------------------ | :------------- | :------- | :----------------- |
| 40 – 50 %                 | Équilibré     | Classification binaire "propre", benchmarks académiques | Accuracy, F1, ROC-AUC |
| 20 – 40 %                 | Légèrement déséquilibré | Credit scoring (ex. German Credit ~30%) | ROC-AUC, PR-AUC, F1 |
| 5 – 20 %                  | Déséquilibré | Churn, réponse marketing | PR-AUC, rappel à précision fixée |
| 1 – 5 %                   | Fortement déséquilibré | Maladies rares, défauts industriels rares | PR-AUC, rappel à précision fixée |
| < 1 %                     | Extrême       | Fraude carte bancaire, anomalies réseau | PR-AUC, rappel à précision fixée, cost-sensitive |

> **Règle pratique** : au-dessus de ~5% de positifs, ROC et PR donnent des diagnostics cohérents. En dessous, l'**AUC-ROC devient trompeuse** (le FPR est écrasé par le grand nombre de TN), et la **PR curve** est plus honnête.

---
## A. Cas normal / équilibré

### 1. Métriques à seuil fixe

#### Cadre

On prend un cas concret qu'on va dérouler de bout en bout. Une plateforme e-commerce entraîne un modèle pour détecter, à chaque inscription, les **faux comptes** (positif = fake, créé par un bot ou pour abuser des promos). Les comptes prédits "fake" sont automatiquement bloqués en attente de vérification d'identité.

Sur **15 000 nouvelles inscriptions** observées :
- **900 sont réellement des faux comptes**, donc $n_1 = 900$ et $n_0 = 14\,100$
- Le modèle en détecte correctement **720** comme faux → $TP = 720$
- Le modèle classe **420 comptes légitimes comme faux** à tort → $FP = 420$

On en déduit :
- $FN = n_1 - TP = 900 - 720 = 180$ (faux comptes ratés)
- $TN = n_0 - FP = 14\,100 - 420 = 13\,680$ (légitimes bien classés)

**Matrice de confusion**

| $y \downarrow / \hat{y} \rightarrow$ | $\hat{y}=0$ | $\hat{y}=1$ |      Total      |
| :----------------------------------- | :---------: | :---------: | :-------------: |
| **$y=0$**                            | TN = 13 680 |  FP = 420   | $n_0 = 14\,100$ |
| **$y=1$**                            |  FN = 180   |  TP = 720   |   $n_1 = 900$   |
| **Total**                            |   13 860    |    1 140    |  $N = 15\,000$  |

**Ratios à seuil fixe**

| Ratio           | Calcul                          | Valeur |
| :-------------- | :------------------------------ | :----: |
| **Spécificité** | $TN / n_0 = 13\,680 / 14\,100$  | 97.0 % |
| **Sensibilité** | $TP / n_1 = 720 / 900$          | 80.0 % |
| **Précision**   | $TP / (TP + FP) = 720 / 1\,140$ | 63.2 % |

> [!note]- 📐 Définitions formelles des trois ratios
> Chaque ratio s'écrit de deux façons équivalentes : en **probabilité conditionnelle** (vision statistique, en termes des erreurs $\alpha$ et $\beta$) ou en **ratio de comptages** (vision ML).
>
> **Sensibilité** (= rappel = TPR) — *"parmi les vrais positifs, combien j'en détecte ?"*
> $$\text{Sensibilité} = P(\hat{y}=1 \mid G_1) = 1 - \beta = \frac{TP}{TP + FN} = \frac{TP}{n_1}$$
>
> **Spécificité** (= TNR) — *"parmi les vrais négatifs, combien je classe correctement ?"*
> $$\text{Spécificité} = P(\hat{y}=0 \mid G_0) = 1 - \alpha = \frac{TN}{TN + FP} = \frac{TN}{n_0}$$
>
> **Précision** (= PPV) — *"quand j'alerte, à quel point j'ai raison ?"*
> $$\text{Précision} = P(G_1 \mid \hat{y}=1) = \frac{TP}{TP + FP}$$
>
> Avec $\alpha$ = erreur de 1ère espèce (faux positif), $\beta$ = erreur de 2nde espèce (faux négatif).

#### Storytelling business

Pour communiquer ces résultats, on combine **chiffres bruts** (volumétrie absolue) et **pourcentages contextualisés** entre parenthèses. Deux formulations équivalentes selon l'audience :

> **Version "chiffres bruts (puis %)"**
> Sur les 15 000 nouvelles inscriptions, le modèle en détecte 1 140 comme faux comptes, dont 420 (37 % des alertes) sont en réalité des comptes légitimes qu'on va bloquer à tort. À l'inverse, 180 faux comptes (20 % des vrais faux) passent à travers le filet.

> **Version "% (puis chiffres bruts)"**
> Le modèle attrape 80 % des faux comptes (720 sur 900) mais ses alertes ne sont justifiées que dans 63 % des cas (720 alertes correctes sur 1 140 émises) — soit 420 utilisateurs légitimes bloqués injustement chaque période.

L'idée est de toujours **éviter le pourcentage seul** : 37 % d'alertes injustifiées ne dit rien tant qu'on n'a pas l'ordre de grandeur (37 % de 10 alertes ≠ 37 % de 1 140 alertes côté charge opérationnelle).

### 2. Métriques agrégées

#### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{N} = \frac{720 + 13\,680}{15\,000} = 96.0\,\%$$

> **Interprétation business** — le modèle prédit correctement 96 cas sur 100. **Mais attention au déséquilibre des classes** : ici les faux comptes ne représentent que 6 % des inscriptions, donc un modèle qui prédirait "tout est légitime" obtiendrait déjà $14\,100 / 15\,000 = 94.0\,\%$ d'accuracy sans rien faire. Les 96 % cachent en fait un modèle moyennement performant sur la classe qui nous intéresse (sensibilité de seulement 80 %, précision de 63 %). **Ne jamais citer l'accuracy seule en présentation business sur classes déséquilibrées.**

#### F1-score

Moyenne **harmonique** de précision et rappel :

$$F_1 = 2 \cdot \frac{\text{Précision} \cdot \text{Rappel}}{\text{Précision} + \text{Rappel}} = 2 \cdot \frac{0.632 \cdot 0.800}{0.632 + 0.800} \approx 70.6\,\%$$

> **Pas d'interprétation business directe.** Contrairement à l'accuracy, le F1-score n'a pas de lecture en "X cas sur 100". C'est une métrique de **comparaison de modèles** : un modèle doit être bon à la fois en précision *et* en rappel pour avoir un bon F1 (si l'un des deux est nul, $F_1 = 0$). On l'utilise pour ranker des modèles entre eux, pas pour communiquer un résultat à un manager.

### 3. Métriques à seuil variable (ROC-AUC)

Les ratios précédents (sensibilité, spécificité) dépendent tous du seuil $s$ :

$\text{sensibilité} = 1-\beta(s) = P(S > s \mid G_1) \qquad \text{spécificité} = 1-\alpha(s) = P(S \leq s \mid G_0)$

Faire varier $s \in ]-\infty; +\infty[$ déforme ces deux quantités en sens opposés. On le voit directement sur les distributions des scores conditionnelles aux deux groupes : le seuil coupe l'axe des scores et délimite les 4 zones de la matrice de confusion.

| Colonne 1                                 | Colonne 2                                 |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260420151117.png\|294]] | ![[Pasted image 20260420150336.png\|304]] |
*Figure 1. Distribution des scores conditionnelle à chaque groupe ; le seuil $s$ découpe les 4 zones TP/FP/TN/FN.*

**Définition (La courbe ROC).** La courbe ROC (*Receiver Operating Characteristic*, héritage des travaux radar de la Seconde Guerre mondiale) trace $1-\beta(s)$ (sensibilité) en fonction de $\alpha(s)$ (1 − spécificité = FPR), pour $s$ parcourant $\mathbb{R}$. Il y a un **arbitrage fondamental** :
- Si $s \to -\infty$ : tout est classé positif → sensibilité = 100%, spécificité = 0%
- Si $s \to +\infty$ : tout est classé négatif → sensibilité = 0%, spécificité = 100%
- En **augmentant** $s$ : ↘️ sensibilité, ↗️ spécificité
- En **diminuant** $s$ : ↗️ sensibilité, ↘️ spécificité

IMAGE
*Figure 2. Courbe ROC générique.*

**Définition (Area Under the Curve, AUC).** L'AUC quantifie la performance globale du modèle, indépendamment du seuil :

$$AUC = \int_{s=-\infty}^{s=+\infty} (1 - \beta(s)) \, d\alpha(s)$$
> [!note]- 📐 Interprétation probabiliste de l'AUC
>**Interprétation probabiliste** (équivalente, plus parlante) : si on tire indépendamment un positif $x^+ \in G_1$ et un négatif $x^- \in G_0$, alors
>
$AUC = P\big(S(x^+) > S(x^-)\big)$
>
>C'est la probabilité que le modèle attribue un score plus élevé à un positif tiré au hasard qu'à un négatif tiré au hasard.

**Repères** (à relativiser selon le domaine, il n'y a pas de seuil universel) :
- $AUC = 0.5$ : modèle aléatoire (diagonale)
- $AUC = 1.0$ : séparation parfaite des deux groupes
- En credit scoring, $0.70$–$0.80$ est typique ; en imagerie médicale, on vise plutôt $0.90+$

**Exemple (German Credit Dataset).** On visualise d'abord la distribution des scores prédits dans chaque groupe :

![[Pasted image 20260426203729.png|471]]
*Figure 3. Distribution des scores prédits — bons payeurs vs. mauvais payeurs.*

Puis la courbe ROC correspondante qui nous donne : $\boxed{AUC=0.78}$ 

![[Pasted image 20260426203748.png|309]]
*Figure 4. Courbe ROC sur German Credit.*

Pour le **choix opérationnel du seuil**, on tabule les métriques pour différentes valeurs de $s$ :

| Seuil PD | Taux acceptation | Sensibilité | Précision | Bons refusés (FP) |
| :------: | :--------------: | :---------: | :-------: | :---------------: |
|   0.30   |       40.3 %     |    86.0 %   |   43.2 %  |        339        |
|   0.50   |       59.6 %     |    70.7 %   |   52.5 %  |        192        |
|   0.70   |       77.1 %     |    46.3 %   |   60.7 %  |         90        |

>**Récit** :  
>- **0.30 = banque prudente** — on accepte 40 % des dossiers, on attrape 86 % des défauts, mais 339 bons clients sont refusés à tort. _"Mieux vaut perdre un bon client que prêter à un défaut."_
>- **0.50 = neutre** — le seuil "naïf" par défaut, équilibre médiocre des deux côtés.
>- **0.70 = banque commerciale agressive** — on accepte 77 % des dossiers, seulement 90 bons refusés, mais on rate plus d'un défaut sur deux. _"On vise la croissance, on absorbera les pertes."_

**Choix business du seuil via les CDF.** Une fois le seuil candidat identifié, on visualise les **fonctions de répartition conditionnelles** des deux groupes ("courbes de discrimination cumulatives"), avec les scores multipliés par 100 pour la lecture business :

![[Pasted image 20260420153925.png|564]] 
![[Pasted image 20260420154513.png|193]]
*Figure 5. CDF des scores pour chaque groupe.*

> Lecture : si l'on décide qu'un client ayant un score inférieur à 550 est "mauvais", on détecte environ **58.3 %** des mauvais (sensibilité) tout en ne flaggant à tort que **14.3 %** des bons (1 − spécificité). Inversement, si le seuil pour être classé "bon" est 750, on reconnaît environ la moitié des bons mais **17.3 %** des mauvais sont considérés à tort comme bons.

---

## B. Cas déséquilibré

### 1. Le piège de l'accuracy

Quand les classes sont très déséquilibrées, l'accuracy devient inexploitable. Sur **Credit Card Fraud** (284 807 transactions, 492 fraudes, $\pi = 0.17\,\%$), un modèle naïf qui prédit "tout légitime" pour tout le monde donne :

| $y \downarrow / \hat{y} \rightarrow$ | $\hat{y}=0$ (légitime) | $\hat{y}=1$ (fraude) |   Total   |
| :----------------------------------- | :--------------------: | :------------------: | :-------: |
| **$y=0$** (légitime)                 |    TN = 284 315        |        FP = 0        |  284 315  |
| **$y=1$** (fraude)                   |     FN = 492           |        TP = 0        |    492    |
| **Total**                            |    284 807             |          0           |  284 807  |

**Accuracy = 99.83 %**, mais **sensibilité = 0 %**. La colonne $\hat{y}=1$ est entièrement vide : le modèle ne détecte **aucune** fraude. La matrice rend le mensonge visible — pas besoin de calculer des métriques pour voir le problème.

Une régression logistique entraînée sur ces mêmes données atteint **99.92 % d'accuracy**. Soit **0.09 point d'écart** pour passer de "rate 100 % des fraudes" à "rate 38 %". L'accuracy est insensible au gain réel du modèle — il faut d'autres métriques.

### 2. PR-AUC : la métrique honnête en très déséquilibré

**Pourquoi ROC-AUC ment.** La courbe ROC trace la sensibilité contre le FPR $= FP/n_0$. Quand $n_0$ est énorme (99.83 % des données ici), même beaucoup de FP en valeur absolue donnent un FPR ridiculement bas. Sur Credit Card Fraud :

$\text{AUC-ROC} = 0.974 \quad \text{vs} \quad \text{PR-AUC} = 0.757$

Le ROC paraît excellent ; la PR-AUC montre que le modèle est en réalité correct mais loin d'être parfait. Pour comparaison, un modèle aléatoire aurait PR-AUC $= \pi = 0.0017$ — le modèle apprend bien quelque chose, mais le ROC surévalue largement.

![[Pasted image 20260426210818.png]]
*Figure 6. ROC vs PR curve sur Credit Card Fraud.*

**Pourquoi PR-AUC est honnête.** La précision $TP/(TP+FP)$ a au dénominateur les **alertes émises**, pas les négatifs réels. Trop de FP fait s'effondrer la précision immédiatement, sans dilution. Comme ROC-AUC, c'est un nombre **agrégé** sur tous les seuils : sert au diagnostic et à la comparaison de modèles, pas au choix de seuil.

**Choix de seuil par contrainte métier.** Sans matrice de coût explicite, on traduit le besoin métier en contrainte sur précision ou rappel, et on lit le seuil correspondant sur la courbe PR. Sur Credit Card Fraud :

| Précision min |  Seuil  | Précision | Rappel |  TP  |  FP  |  FN  |
| :-----------: | :-----: | :-------: | :----: | :--: | :--: | :--: |
|     50 %      |  0.014  |   50.0 %  | 85.4 % | 420  | 420  |  72  |
|     80 %      |  0.129  |   80.0 %  | 75.0 % | 369  |  92  | 123  |
|     90 %      | 0.9996  |   90.4 %  | 25.0 % | 123  |  13  | 369  |
|     95 %      | 1.0000  |   95.8 %  | 18.7 % |  92  |   4  | 400  |

> **Lecture du trade-off** : si la fraude bloque automatiquement la carte, on veut une précision haute (80 %+) pour ne pas fâcher les bons clients — au prix de rater 25 % des fraudes. Si on alerte juste un humain pour vérification, on peut descendre à 50 % de précision et attraper 85 % des fraudes.

**Remarque sur `class_weight='balanced'` (le faux ami).** Le réflexe classique pour traiter le déséquilibre est de pondérer les classes pendant l'entraînement (`class_weight` en logreg/SVM/RF, `scale_pos_weight` en XGBoost/LightGBM — c'est le même mécanisme général). Sur Credit Card Fraud :

|                   | Baseline | Balanced |
| :---------------- | :------: | :------: |
| AUC-ROC           |  0.974   |  0.979   |
| **PR-AUC**        | **0.757**| **0.730**|
| Sensibilité @ 0.5 |   62 %   |   91 %   |
| **Précision @ 0.5** | **87 %** | **6.5 %**|
| FP @ 0.5          |    46    |  6 466   |

La sensibilité monte mais la précision s'effondre (6 466 FP au lieu de 46 !), et la **PR-AUC baisse**. La repondération **décalibre les PD** sans améliorer le classement réel des observations. Le bon levier reste de **baisser le seuil sur le modèle non pondéré** — c'est précisément ce qu'on a fait dans la table de contraintes ci-dessus.

---

## C. Matrice de coût

### 1. Définition

Tous les modèles font des erreurs, mais **toutes les erreurs ne se valent pas**. La **matrice de coûts** (cost matrix) formalise cette asymétrie en associant un coût à chaque type d'erreur :

| Réalité ↓ / Prédit → | Prédit 0 (négatif) | Prédit 1 (positif) |
| :------------------- | :----------------: | :----------------: |
| **Réel 0**           |         0          |     $C_{FP}$       |
| **Réel 1**           |     $C_{FN}$       |         0          |

Exemples typiques d'asymétrie :
- **Médical** : $C_{FN} \gg C_{FP}$ (rater un cancer ≫ fausse alerte)
- **Spam** : $C_{FP} \gg C_{FN}$ (bloquer un vrai email ≫ laisser passer un spam)
- **Crédit** : $C_{FN} \gg C_{FP}$ (prêter à un défaut ≫ refuser un bon client)

Pour un client de PD prédite $\hat{p}$, le coût espéré dépend de la décision :

$$\text{Coût(accepter)} = C_{FN} \cdot \hat{p}, \qquad \text{Coût(refuser)} = C_{FP} \cdot (1-\hat{p})$$

On refuse si $\text{Coût(refuser)} < \text{Coût(accepter)}$, ce qui donne le **seuil optimal** :

$$s^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$

> [!note]- 📐 Dérivation
> On cherche le seuil $s$ qui rend les deux coûts égaux (point d'indifférence) :
> $$C_{FN} \cdot s = C_{FP} \cdot (1-s)$$
> $$s \cdot (C_{FN} + C_{FP}) = C_{FP} \implies s^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$
> En-dessous de $s^*$, accepter coûte moins ; au-dessus, refuser coûte moins.

### 2. Application au German Credit

La documentation UCI du dataset fournit la matrice de coûts officielle : $C_{FP} = 1$ (refuser un bon client) et $C_{FN} = 5$ (accepter un défaut). Donc :

$$s^* = \frac{1}{1+5} \approx 0.167$$

On compare 4 politiques sur les 1000 clients du dataset :

| Politique          |  Seuil   |   FP    |   FN   | Coût total |
| :----------------- | :------: | :-----: | :----: | :--------: |
| Tout accepter      |   0.00   |    0    |  300   |   1 500    |
| Naïf               |   0.50   |   192   |   88   |    632     |
| **Optimal métier** | **0.17** | **462** | **17** |  **547**   |
| Tout refuser       |   1.00   |   700   |   0    |    700     |

Au seuil optimal, le modèle refuse 462 bons clients à tort (66 % des bons !) mais ne laisse passer que 17 défauts sur 300 (sensibilité de 94 %). C'est volontairement très restrictif : rater un défaut coûte 5× plus que refuser un bon, donc on **achète** des FP pour minimiser les FN.

> **A retenir.** Le bon seuil n'est pas un problème mathématique mais un problème **métier**. Le modèle estime des PD ; les coûts dictent la coupure. Sans la matrice de coût, n'importe quel data scientist dirait "ce modèle refuse 66 % des bons clients, il est nul". Avec la matrice de coût, c'est le **bon** modèle.
