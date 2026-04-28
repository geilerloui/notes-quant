# (iv) Méthodes ensemble

> Combiner plusieurs modèles pour faire mieux qu'un seul. Trois familles : **bagging** (variance ↓), **boosting** (biais ↓), et l'**architecture multi-modèles** (cascade, stacking, two-stage).

---

## 1. Bagging et Boosting (rappel)

Ces deux familles sont vues en détail dans [[(i) Modèles d'Arbres]] (random forests, gradient boosting). Résumé :

| | Bagging | Boosting |
| :--- | :--- | :--- |
| Idée | Entraîner $B$ modèles **indépendamment** sur des bootstraps, moyenner | Entraîner $B$ modèles **séquentiellement**, chacun corrige les erreurs du précédent |
| Effet principal | Réduit la **variance** | Réduit le **biais** |
| Exemple | Random Forest | XGBoost, LightGBM, CatBoost |

---

## 2. Stacking

*À développer.*

Idée : entraîner $K$ modèles différents (ex : logreg, RF, XGBoost), puis entraîner un **méta-modèle** qui prend les prédictions des $K$ comme features d'entrée pour produire la prédiction finale.

```
X → [logreg]    → p_1 \
X → [RF]        → p_2  → [méta-modèle] → ŷ
X → [XGBoost]   → p_3 /
```

- Pour éviter le data leakage, les prédictions des modèles de base doivent être **out-of-fold** (générées par CV).
- Le méta-modèle est typiquement simple (logreg) pour ne pas réintroduire de variance.

---

## 3. Cascade model (Andrew Ng)

*À développer.*

Idée : enchaîner plusieurs modèles **en série**, chacun filtrant ce que le précédent laisse passer. Typique en détection avec très fort déséquilibre.

```
X → [modèle rapide, recall haut] → si négatif : STOP
                                  → si positif : passer au modèle suivant
                                  → [modèle plus précis, plus coûteux] → décision finale
```

Exemple : détection d'objets en vision par ordinateur. Le premier modèle élimine 99 % des fenêtres clairement vides. Le second, plus coûteux, traite seulement les 1 % restants.

Avantage : **coût computationnel** drastiquement réduit en production.

---

## 4. Two-stage pipeline (le truc QRT)

*À développer.*

Idée : séparer le problème en deux étapes avec **deux modèles distincts**, où la sortie du premier devient input du second.

Exemple credit scoring :
1. **Stage 1** : modèle de classification PD (probabilité de défaut)
2. **Stage 2** : modèle de régression LGD (loss given default) sachant que défaut

Au lieu d'un seul modèle qui prédit la perte espérée directement, on décompose en `E[perte] = PD × LGD`, chaque facteur étant modélisé séparément. Permet de mieux capter la structure du problème et facilite l'interprétation.

> **À noter sur le contexte QRT** : utilisé en alpha research pour décomposer un signal en deux composantes (ex : direction + magnitude ; probabilité d'event + sizing).
