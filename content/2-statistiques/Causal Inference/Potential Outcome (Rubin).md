## 🧭 Big picture

**Le but de l'inférence causale.** Savoir si X *cause* Y, pas juste si X est corrélé à Y. Autrement dit : est-ce que **changer X**, toutes choses égales par ailleurs, **change Y** ?

**Le problème fondamental.** Pour répondre proprement il faudrait observer la même unité avec et sans traitement *en même temps* :
$$
\text{Effet individuel} = Y_i(1) - Y_i(0)
$$
Mais on n'observe qu'une seule des deux moitiés. L'autre est le **contrefactuel**, à jamais inobservable.

**La parade.** On abandonne l'effet individuel et on vise une moyenne :
$$
ATE = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]
$$
Si on a deux groupes vraiment comparables (un traité, un pas), alors $ATE \approx \bar Y_{\text{traités}} - \bar Y_{\text{contrôles}}$. Sinon on mesure un mélange "effet du traitement + effet des différences pré-existantes" (les confondeurs).

> 💡 **Toute l'inférence causale = construire deux groupes comparables.**

**Les 4 régimes selon comment on construit la comparabilité :**

| Régime | Comment on rend les groupes comparables | Crédibilité |
|---|---|---|
| Contrôlée | À la main, en cherchant des jumeaux | Parfait en théorie, irréaliste |
| Randomisée (RCT / A/B test) | Tirage au sort → la randomisation équilibre tout, *même les confondeurs cachés* | Gold standard |
| Naturelle | Le hasard du monde a fait le tirage à notre place (loi, frontière, lotterie) | Très bonne si bien identifiée |
| Observationnelle | Personne n'a randomisé. On *bricole* statistiquement avec les méthodes du III. | Plus fragile, dépend des hypothèses |

**Arbre de décision quand on a une question causale :**

```
1. Puis-je randomiser ?
   OUI → RCT / A/B test → différence de moyennes
   NON → 2

2. Y a-t-il un événement extérieur qui randomise pour moi ?
   OUI → expérience naturelle → IV ou RDD
   NON → 3

3. Dans mes données, est-ce que je crois capter tous les confondeurs ?
   OUI → G-formula ou Matching (sous unconfoundedness)
   NON → 4

4. Ai-je un instrument valide / un seuil net / un avant-après comparable ?
   Instrument           → IV
   Seuil                → RDD
   Avant/après 2 groupes → Diff-in-Diff
   Rien                 → faire de la prédiction, pas de la causalité.
```

**Méthode → hypothèse identifiante → estimand.** Chaque méthode tient debout grâce à *une hypothèse différente*, et n'estime *pas forcément le même objet* :

| Méthode | Hypothèse identifiante | Estimand |
|---|---|---|
| RCT (différence de moyennes) | Randomisation correcte | ATE |
| G-formula / Matching | Unconfoundedness + Positivity | ATE ou ATT |
| Instrumental Variables | Pertinence + Exclusion restriction | LATE (compliers) |
| Sharp RDD | Continuité au cutoff | Effet local au cutoff |
| Fuzzy RDD | Continuité + saut de proba | LATE au cutoff |
| Diff-in-Diff | Parallel trends | ATT |

> 💡 Deux méthodes peuvent donner deux nombres différents qui sont **tous les deux corrects**, parce qu'elles mesurent des estimands différents.

**Le fil rouge de tout le cours.**
* Section II : les 4 *contextes de données* possibles.
* Section III : les méthodes pour reconstruire un quasi-RCT à partir de données observationnelles.
* Section IV : les estimands (ATE, ATT, CATE, LATE) qu'on essaie d'estimer dans tous les cas.

Tout ce qu'on fait dans III, c'est **simuler à la main ce qu'un RCT fait gratuitement** : rendre les groupes comparables.

---

## I - Introduction (vocabulaire, ATE, contrefactuel, confounders)

Vocabulaire : 
* **Unité d'analyse (Unit of analysis - UA).** En causalité on ne parle pas d'observation mais d'unité d'analyse eg un individu, un bloc de bâtiment...
* **Variable de sortie (outcome variable).** C'est le Y
* Les features : 
	* **Variable d'intérêt (Policy/Treatment variable - VI).** C'est la seule feature dont on veut tester l'intérêt.
	* **Variable de contrôle (Control variables - VC).** Les autres features qui sont "fixes" pour isoler l'effet de la politique/du traitement.

**Définition (Causalité).** On définit la Causalité comme l'étude de comment une intervention, le fait de changer volontairement la variable d'intérêt, affecte aka la variable de sortie Y.

**Propriétés (Causalité).**
* Si il y'a zéro corrélations entre les variables après avoir nettoyé les données alors il n'y a pas de causalité.
* Deux variables corrélés n'implique pas qu'il y'a causalité.

**Exemple.**
>La Unité d'analyse (UA) est le quartier, la Variable d'intérêt (VI) est le nombre d'arbres, la Variable de contrôle (VC) est le nombre de crime. Si il y'a beaucoup d'arbres => il y'a peu de crime, cependant, il manque la variable "richesse du quartier".

**Propriété (Effet causal individuel - Unit-level causal effect).**
Chaque Unité d'analyse (UA) a son propre effet causal eg effet du médicament mais on ne peut pas savoir l'effet sur le patient à la fois si il l'avait pris et si ils l'avaient pas pris.
|
| => Solution (ATE - Average Treatment Effect). 
$$
\text{ATE = moyenne avec la politique - moyenne sans la politique}
$$
Il y'a trois interprétations : 
$$
\begin{cases}
 \text{ATE > 0 : Bonne politique }\\
 \text{ATE < 0 : Mauvaise politique }\\
 \text{ATE = 0 : Pas d'effet }
\end{cases}
$$

**Définition (Hétérogénéité).** On dit qu'il y'a hétérogénéité si l'effet causal individuel $Y_i(1)-Y_i(0)$ varie selon les individus $i$. Autrement dit, le traitement n'a pas le même effet pour tout le monde.
> eg. Medicaid baisse le cholestérol de certaines personnes mais pas d'autres.

⚠️ En pratique on n'observe jamais $Y_i(1)-Y_i(0)$ directement (problème fondamental). On détecte donc l'hétérogénéité en calculant des **CATE** sur des sous-groupes (hommes vs femmes, jeunes vs vieux) et en regardant s'ils diffèrent.

**Définition (Problème fondamental de l'inférence causal).** On ne pourra jamais faire 
$$
\text{(Ton cholestérol avec medicaid) - (Ton cholestérol sans medicaid)}
$$
Ou mathématiquement : 
$$
\boxed{~~Y_i(1) - Y_i(0)~~}
$$
Avec :
* $i$ c'est toi aka la Unit of Analysis
* $1$ : tu as le traitement medicaid
* $0$ : tu n'as pas medicaid
* $Y_i$ ton niveau de cholestérol (outcome)

⚠️ Dans la vraie vie pour la personne $1$ on verra eg le 40. On dit que le $30$ c'est le contrefactuel (counterfactual).

**Exemple (Le cas idéal).** C'est le cas "idéal" où on connaitrait à la fois avec et sans traitement pour chaque patient. La Variable d'intérêt (VI) est le taux de cholestérol. Une valeur de 40 se lit : le niveau de cholestérol est de 40 sans medicaid et de 30 avec. Lorsqu'on dit qu'on a un "unit causal effect" de +10 ça signifie que le traitement a amélioré le score de +10 points (score à définir).
$$
\begin{array}{l|llll}
\text{Person} & \text{Sex} & \text{Outcome} & \text{Outcome} & \text{Unit causal effect} \\
(i) &  & \text{with treatment} & \text{without treatment} & \\
& & Y_i(1) & Y_i(0) & Y_i(1) - Y_i(0)\\
\hline
1 & \text{male} & 40 & 30 & 10 \\
2 & \text{male} & 20 & 20 & 0 \\
3 & \text{female} & 10 & 15 & -5 \\
4 & \text{female} & 30 & 30 & 0
\end{array}
$$
L'hétérogénéité ce lit dans la dernière colonne, on voit que le unit causal effect, est différent pour chaque patient.
On peut calculer l'effet global :
$$
ATE = \frac{10+0+(-5)+0}{4} = \frac{5}{4}
$$
L'ATE est >0 on en déduit que le traitement est efficace. Cependant si on regarde par "Sexe" : 
$$
CATE(men) = \frac{10+0}{2} = 5; ~~~~ CATE(female) = \frac{-5+0}{2} = -\frac{5}{2}
$$
En reprenant notre échelle ci-dessus on en conclut que le traitement est bon pour les hommes mais mauvais pour les femmes.

**Définition (Facteurs de confusion - Confounders).** C'est une variable qui pollue la comparaison car elle change d'un groupe à l'autre.

**Exemple.**
> UA = quartier, Y = crime, VI = nombre d'arbre, le confounder = le revenu élevé. On a un mécanisme de confusion : 
> * Les gens riches sélectionnent les quartiers avec arbre (revenu est lié au traitement)
> * Les gens riches commettent statistiquement moins de crimes (le revenu est lié à l'outcome).

**Définition (Actual Outcome vs Counterfactual outcome).** Une fois que l'action est faite (eg tu as pris medicaid), les deux résultats potentiel (potential outcomes) changent de statut.
* Actual Outcome : ton résultat
* Counterfactual Outcome (le counterfactual). C'est la donnée manquante, ce qui aurait pu se passer si tu avais fais un autre choix.

**Définition (Inférence causale).**
$$
\boxed{\text{Inférence causale = Utiliser des hypothèses pour estimer l'inobservable}}
$$
Plus spécifiquement : 
* Inférence statistique : On apprends à partir des données qu'on voit
* Inférence causale : on essaie d'apprendre sur des données qu'on ne verra jamais (le contrefactuel), c'est pour ça que c'est beaucoup plus difficile que l'inférence statistique.

**Définition (SUTVA - Stable Unit Treatment Value Assumption).** Si les hypothèses du SUTVA sont violés alors ton ATE, CATE sont faux car il y'a des fuites entre les groupes.

**Exemple (Effet de débordement - Spillover Effect).** On a le groupe contrôle qui ne fait rien et le groupe traité qui reçoit une formation de trading. Le problème c'est qu'un des mecs du groupe traité est ami avec des gens du groupe contrôle et il leur apprend le trading. Ce qui va biaiser les résultats.

## II - Les types d'expériences
### A. Contrôlée

**Objectif (Expérience contrôlé).** C'est de recréer artificiellement $Y_i(1)-Y_i(0)$ en utilisant deux Unités d'Analyse (Unit of Analysis) différentes mais identiques.

**Exemple.**
> On trouve deux personnes identiques (même features). Sauf que l'un reçoit le traitement $(D=1)$ l'autre non ($D=0$). Ce qui nous permettra de soustraire l'outcome de l'un par l'autre.

⚠️ En réalité c'est très difficile de trouver deux individus identiques surtout quand il y'a trop de features. => La solution c'est la $\boxed{\text{~~Randomized Experiments ~~}}$

### B. Randomisée (RCT) -> ITT, LATE, SUTVA, proxies

**Objectif (Expérience Randomisée).** Au lieu de chercher deux individus quasi identiques (Expérience contrôlé) on va essayer de créer deux groupes équivalents. La moyenne du groupe $A$ devient statistiquement identique à la moyenne du groupe $B$.

**Remarque (A/B testing).** En tech / produit, le RCT s'appelle **A/B testing**. C'est exactement la même chose, juste un vocabulaire différent : l'unité d'analyse est l'utilisateur, le traitement c'est le variant B (nouvelle version), le contrôle c'est A (ancienne), et la randomisation se fait via un hash du cookie / user ID. Les mêmes problèmes se posent : non-compliers (refresh, cookies bloqués), violation du SUTVA (spillover dans les marketplaces ou réseaux sociaux), proxies (clic à 1 jour comme proxy de la rétention à 6 mois).

**Remarque (Spurious correlation).** Une corrélation forte n'implique pas une causalité. Pire : sur des séries non-stationnaires (typiquement deux prix d'actions modélisés comme des marches aléatoires indépendantes), le coefficient de corrélation est faussement très grand alors qu'il n'y a aucun lien causal. C'est le résultat de Granger & Newbold (1974) sur les *spurious regressions*. En finance quant, on évite ce piège en travaillant sur les **rendements** plutôt que sur les **prix**, ou en testant la **cointégration** quand on veut une relation de long terme entre actifs.

**Conditions (Expérience Randomisée).** Il faut que $D \perp X$ où $D$ est le traitement.
C'est à dire que si on a cent features $(X_1, X_2, .., X_{100})$ on doit vérifier que :
* la moyenne de $X_1$ est la même dans le groupe $D=0$ et $D=1$
* la moyenne de $X_2$ est la même dans le groupe $D=0$ et $D=1$
* etc

**Exemple.**
$$
\begin{array}{l|c|c|c}
\text{Feature} & \text{Contrôle (D=0)} & \text{Traité (D=1)} & \text{p-value} \\ \hline
\text{Age} & 34.5 & 34.7 & 0.85 \\
\text{Revenu} & 2{,}500\,\$ & 2{,}495\,\$ & 0.92 \\
\text{Sexe (\%F)} & 52\% & 51\% & 0.72 \\
\end{array}
$$
On doit en premier vérifier que la propriété $D \perp X$ est bien vérifié pour se faire on utilise un t-test de Welch. Si les $p-value >>0$ l'indépendance est vérifié et la randomisation est correcte.

Ensuite on peut calculer l'ATE : 
$$
ATE = \mathbb{E}[Y \mid D=1] - \mathbb{E}[Y \mid D=0]
$$

**Définition (Les non compliers).** Si on a 10,000 personnes, 5,000 ne prennent rien et 5,000 sont censés le prendre mais finalement 30% des 5,000 ne le prennent pas. Ainsi si je compare les "gagnants" aux "perdants" mon effet est dilué c'est le 

$$
\boxed{\text{~~ Intent-to-treat (ITT) ~~}}
$$
**Exemple.** Groupe contrôle (50 personnes) n'ont rien ils perdent 2kg en un mois
* Groupe lotterie (50 personnes) on leur offre un coaching 
	* Mais que 25 vont faire du sport
	* A la fin la moyenne du groupe est de 4kg

$$
\begin{array}{l|l|l}
\text{Mean control} & \text{Moyenne groupe témoin} & \text{Résultat (2kg)} \\ \hline
\text{Effet lottery (ITT)} & \text{Moyenne lotterie - moyenne contrôle} & 4-2=2\text{kg} \\ \hline
\text{ATE Ajusté = LATE} & \text{ITT / \% de gens qui ont participé} & 2/0.50=4\text{kg}
\end{array}
$$
Commentaire ? 
####  Les quatre méthodes pour les Non Compliers

**Méthodes (non compliers).**
1. Intention to treat (ITT)
2. Instrumental Variables (IV)
3. Assume random compliance
4. Bound Analysis

(1) Intention to treat (ITT). Pour le (1) on l'a calculé dans l'exemple ci-dessus; 

(2) Instrumental Variables (IV). Pour le (2) également c'est "l'effet IV" aka le LATE (Local Average Treatment Effect) :
$$
LATE = \frac{\text{résultat(moyenne traité - moyenne contrôle)}}{\text{participation (proportion de compliers)}}
$$
On dit que c'est "Local" car ça ne nous dit rien sur les gens qui auraient pris le médicament de toute façon (les Always Takers) ou ceux qui refusent quoi qu'il arrive (never taker).

Exemple. Impact d'une réduction de frais de courtage sur le volume de trading 

$$
\begin{array}{l|l|l|l}
\text{Trader} & \text{Si on donne la réduc} & \text{Si on ne donne rien} & \text{Catégorie} \\ \hline
\text{A} & \text{Trade Bcp} & \text{Ne trade pas} & \text{Complier = Celui qu'on veut cibler} \\ \hline
\text{B} & \text{Trade Bcp} & \text{Trade Bcp} & \text{Always taker = perte d'argent} \\ \hline
\text{C} & \text{Ne trade pas} & \text{Ne trade pas} & \text{Never taker}
\end{array}
$$

(3) Assume random compliance. On supprime du dataset ceux qui n'ont pas participé à l'expérience. On compare 25 personnes qui ont fait du sport aux 50 qui n'en ont pas fait.
⚠️ Problème : si les 25 qui ont participé sont tous des "hyper motivés" mon résultat sera biaisé.

(4) Bound Analysis. On accepte de l'incertitude "Quel serait l'effet si les gens qui n'ont pas répondu étaient tous des génies ou s'ils étaient tous des cas désespérés".

Exemple. On fait un intervalle $[-23\%, +37\%]$

####  Proxies et effets long terme

Définition (Les proxies). C'est lorsque l'outcome réel est impossible à mesurer directement.

Exemples : 
* Startup : Tu ne peux pas attendre 10 ans pour savoir si une stratégie évite la faillite. Tu regardes le profit annuel (proxy). Si le profit baisse c'est un signe qu'il y'a un risque de fermeture.

⚠️Un bon proxy doit rester corrélé à l'outcome même après le traitement.




### C. Naturelle

Motivation. Étude sur les impôts : on ne peut pas faire payer +10% d'impôt à un groupe et +50% à un autre juste pour voir qui travaille le plus. 

Sinon j'ai écris y'en a deux types : 
#### True natural experiment

**Pourquoi _true_ natural experiment :** le marché s'attendait à "Remain" (sondages ~52%, bookmakers 75%). Le résultat "Leave" a été un **choc exogène imprévu** appliqué de manière hétérogène selon l'exposition UK des firmes.

**Setup d'étude :**

- **Unité d'analyse (UA)** : entreprises cotées au London Stock Exchange
- **Variable d'intérêt (X)** : exposition au marché UK (% revenus UK)
- **Outcome (Y)** : rendement de l'action après le 24 juin
- **Groupe Traité** : firmes très exposées UK (FTSE 250, domestique)
- **Groupe Contrôle** : firmes peu exposées UK (FTSE 100, multinationales)

**Résultat :** le 24 juin, FTSE 250 ≈ -7% intraday vs FTSE 100 ≈ -3%. Cette différence ≈ effet causal du choc Brexit sur les firmes domestiques.

#### As-if natural experiment

**Pourquoi _as-if_ natural experiment :** le comité S&P a une part discrétionnaire dans le choix des nouvelles inclusions. Mais entre deux actions également éligibles (taille, secteur, profitabilité), choisir l'une plutôt que l'autre est _quasi-aléatoire_ du point de vue du chercheur.

**Setup d'étude :**

- **Unité d'analyse (UA)** : actions américaines de grande capitalisation
- **Variable d'intérêt (X)** : être ajoutée au S&P 500 (oui/non)
- **Outcome (Y)** : rendement de l'action sur quelques jours autour de l'annonce
- **Groupe Traité** : actions ajoutées (ex : Tesla en décembre 2020)
- **Groupe Contrôle** : actions de taille/secteur similaires mais non ajoutées

**Résultat :** "S&P 500 inclusion effect" ≈ +5% à +9% à l'annonce, dû à la demande mécanique des fonds indiciels qui doivent acheter pour tracker l'indice. Effet diminué depuis l'arrivée de l'arbitrage anticipé.

### D. Observationnelle

**Définition.** Aucune intervention, aucun tirage au sort. On observe le monde tel qu'il est : les unités d'analyse choisissent elles-mêmes (ou subissent) leur traitement.

**Exemples.**
* Effet du tabac sur le cancer (on ne randomise pas qui fume)
* Effet du MBA sur le salaire (les gens choisissent de s'inscrire)
* Effet d'aller à l'armée sur le salaire (engagement volontaire)

⚠️ C'est le cas le plus courant en pratique mais aussi le plus difficile : ceux qui choisissent le traitement sont systématiquement différents de ceux qui ne le choisissent pas (confondeurs partout). Une simple différence de moyennes est biaisée.

=> Toute la section III est dédiée aux méthodes pour traiter ce cas.

## III - Méthodes d'estimation

Ces méthodes ont été développées surtout pour le cas **observationnel** (le plus difficile), mais elles servent aussi pour les **expériences naturelles** (qui empruntent IV/RDD/DiD selon la structure du choc) et parfois même pour les **RCT** (eg IV pour gérer les non-compliers, vu en II.B).

| Méthode | RCT | Naturelle | Observationnelle |
|---|---|---|---|
| Diff de moyennes | ✅ par défaut | rare | ❌ biaisé |
| Régression linéaire | ✅ pour ajuster | ✅ | ✅ |
| G-formula / Matching | ✅ pour ajuster covariates | possible | ✅ usage principal |
| **IV** | ✅ pour les non-compliers | ✅ usage principal | ✅ si confondeur caché |
| **RDD** | ❌ | ✅ usage principal | ✅ si seuil exploitable |
| **Diff-in-Diff** | rare | ✅ usage principal | ✅ |

> 💡 Le fil rouge : toutes ces méthodes essaient de **simuler à la main ce qu'un RCT fait gratuitement** — rendre les groupes comparables.

### A. Régression linéaire (et ses pièges : médiation, modération)

Exemple. Y=GDP, X=property right (PR) ie que 
* score de 10 si l'état te protège si tu as un bien
* score 0-2 : instable, le dictateur peut tout te prendre
Or si $\rho(X,Y) >>0$ il peut y'avoir des sources de biais.

Propriétés (Source de biais).
1. Variables de confusion :  Une variable Z eg le climat joue sur les deux variables
2. Causalité Inverse : Une fois qu'un pays est riche les citoyens demandent des droits on a donc une relation inverse PR <- Y
3. La simultanéité : X cause Y pendant que Y cause X

#### La Médiation

On introduit une variable $M$ le "Médiateur"

Graphe a mettre

Exemple. 
* X : le montant du coupon de réduction eg de 0$ à 10$
* Y : score de fidélité 6 mois plus tard (de 0 à 100)
* M : Le nombre de commandes passées durant le 1er mois.

i) Effet total : ($X \rightarrow Y$)
$$
fidelite = \beta_0 + \beta_1 \cdot coupon + \varepsilon
$$
On obtient $\beta_1=0.5$ ie que pour chaque $ de réduction donné, la fidélité augmente de +0.5 points.

ii) Lien vers le médiateur : ($X \rightarrow M$)
$$
commande = \beta_0 + \beta_1 \cdot coupon + \varepsilon
$$
On obtient $\beta_1=1.2$ ie que pour chaque $ de réduction donné ça entraine +1.2 commandes supplémentaires.

iii) ($X + M \rightarrow Y$)
$$
fidelite = \beta_0 + \beta_1 \cdot coupon+ \beta_2 \cdot commandes + \varepsilon
$$
Scénario A : "Médiation totale" - Le coupon n'est que un prétexte
$$
\beta_2=0.4 ~(\text{p-val}\approx 0), ~~\beta_1\approx 0 ~(\text{p-val}>> 0)
$$
On en conclut que le coupon n'apporte aucune fidélité en soi. C'est le faite de commander qui rend fidèle.

Scénario B: "Médiation partielle"
$$
\beta_2=0.3, ~~ \beta_1=0.2
$$
Pour rappel on avait obtenu "0.5" pour l'effet total ici on a "0.2" : 
* Donc "commander" crée un attachement à la marque
* Mais aussi, indépendamment, donner des coupons augmentent aussi la fidélité

iv) Calcul de l'ACME (Average Causal Mediation Effect).
ACME = Effet de X sur M (1.2) x effet de M sur Y (0.4) = 0.48

Ce qui veut dire que sur tes 0.5 points de fidélité initiaux, 0.48 passent par le fait de commander.

v) Storytelling : Si le Boss dit "On arrête les coupons c'est trop cher" on peut lui répondre : 
"Si on arrête les coupons, on perd 1-2 commandes par client et comme chaque commande génère +0.4 de fidélité, notre fidélité globale va chuter de 0.48 points" (? pas sûr de la fin)

Remarques : 
* Apparemment il faut aussi regarder l'écart-type de la fidélité dans les données : si tout le monde a un score entre 49 et 51 alors 0.48 c'est une énorme perte.

#### La Modération

**Définition (Modération).** Une variable $Z$ est un *modérateur* de l'effet de $X$ sur $Y$ si **l'effet de $X$ sur $Y$ dépend du niveau de $Z$**.

**L'enchaînement mental hétérogénéité → modération → interaction.**
* Tu observes que ton ATE (5/4) cache des CATE très différents (5 pour les hommes, -2.5 pour les femmes) → **hétérogénéité** (constat empirique).
* Tu te demandes pourquoi → la variable "sexe" *modère* l'effet du traitement → **modération** (explication).
* Pour quantifier ça dans une régression, tu mets un **terme d'interaction** $X \cdot Z_{\text{sexe}}$ → outil mathématique.

Graphe à rajouter

**Exemple (Multi-modération).** 
* X : Télétravail (Treatment)
* $Z_1$ : Homme vs femme - Sexe (Modérateur 1)
* $Z_2$ : Junior vs senior - Ancienneté (Modérateur 2)
* $Y$ : Productivité

On va résoudre :

$$
Y = \beta_0 + \beta_1  X \cdot Z_1 + \beta_2 X \cdot Z_2 + \varepsilon
$$
En gros on veut savoir si l'effet du traitement (Télétravail) a un impact sur la productivité et est modérée par l'ancienneté et le sexe.




### B. La G-Formula

On va voir une méthode pour calculer l'ATE en utilisant une méthode qu'on appelle la standardisation (G-formula) c'est une version automatisé des calculs à la main 

⚠️ Pour que la G-formula marche il faut respecter deux conditions : 
1. Unconfoundedness : J'ai inclus dans mon GLM toutes les variables qui influencent à la fois le treatment et les résultats
2. Positivity (Overlap). Il faut qu'il y'ait au moins quelques personnes traitées et non traitées dans chaque catégorie

⚠️ Limites de la G-formula : 
* Le problème de l'Extrapolation : A cause du modèle de clonage eg si dans tes données tu n'as aucun senior qui a fait la formation, ton modèle va inventer un résultat
|
| => Solution : Le matching : au lieu de créer un clone on va chercher dans la BDD un jumeau.


Exemple.

|**Individu**|**Âge (Z)**|**Formation (X)**|**Salaire RÉEL (Y)**|
|---|---|---|---|
|**A**|20|1|**3000**|
|**B**|20|0|**2500**|
|**C**|50|1|**6000**|
|**D**|50|0|**5000**|
Matrice 1 : Scénario "Tout le monde est formé" : On garde l'âge réel, mais on **force** $X$ à 1 pour tout le monde. On utilise notre "recette" pour prédire le salaire.

|**Individu**|**Âge (Z)**|**Formation (X)**|**Salaire PRÉDIT (Y^1​)**|
|---|---|---|---|
|**A**|20|1|**3000** (déjà connu)|
|**B**|20|1|**3500** (le clone de B formé)|
|**C**|50|1|**6000** (déjà connu)|
|**D**|50|1|**6500** (le clone de D formé)|
Matrice 2 : Scénario "Personne n'est formé" On garde l'âge réel, mais on **force** $X$ à 0 pour tout le monde.

|**Individu**|**Âge (Z)**|**Formation (X)**|**Salaire PRÉDIT (Y^0​)**|
|---|---|---|---|
|**A**|20|0|**2000** (le clone de A non-formé)|
|**B**|20|0|**2500** (déjà connu)|
|**C**|50|0|**5000** (le clone de C non-formé)|
|**D**|50|0|**5000** (déjà connu)|
L'ATE (Average Treatment Effect), c'est simplement la moyenne de la colonne $\hat{Y}_1$ moins la moyenne de la colonne $\hat{Y}_0$.
- Moyenne $\hat{Y}_1=(3000+3500+6000+6500) / 4=4750$
- Moyenne $\hat{Y}_0=(2000+2500+5000+5000) / 4=3625$
- ATE $=4750-3625=\mathbf{1 1 2 5}$

Pourquoi c'est puissant ? Regarde bien : pour l'individu B, on a **estimé** qu'il aurait gagné 3500 s'il avait été formé. Pour l'individu A, on a **estimé** qu'il n'aurait gagné que 2000 sans formation.

### C. Le Matching

i) Imagine tu as un soldat (vétéran) 25 ans blanc
* On ne sait pas combien il aurait gagné s'il n'étaient pas allé à l'armée
* Solution "Matching" dans la bdd tu cherches un civil (non vétéran) qui a 25 ans qui est blanc => tu regardes la différence de salaire entre les deux jumeaux
* On répète ça pour tout le monde on obtient une liste de différence : moyenne de la liste = ATE

ii) Le propensity score : C'est si tu as 50 variables le jumeau parfait est difficile à trouver à la place on calcule un score $[0,1]$ on matche les gens qui ont la même note

### D. Instrumental Variables

**Motivation :** G-Formula et Matching reposent sur l'hypothèse d'unconfoundedness, or en réalité il y'a souvent des variables invisibles ou impossibles à mesurer.

> Eg. $Z$ est caché, on ne l'a pas dans les données. Mais si $Z$ influence à la fois $X$ et $Y$, le matching sera toujours biaisé.

**Solution.** IV est une méthode pour trouver la causalité même quand il y'a des variables cachées. On cherche un instrument $(Z)$ qui : 
1. Influence le traitement $(X)$
2. N'a aucun rapport avec $Y$ sauf via $X$

**Remarque :** 
* Médiation $X \rightarrow M \rightarrow Y$
* Instrument $Z \rightarrow X \rightarrow Y$

**Définitions :** 
* **Variable endogène** : variable du modèle qui est *corrélée au terme d'erreur* (donc à des facteurs cachés). Typiquement, le treatment $X$ est endogène quand il y a un confondeur non-mesuré.
* **Variable exogène** : variable qui n'est *pas* corrélée au terme d'erreur. C'est ce qu'on demande de l'instrument $Z$.

**Conditions formelles pour un instrument $Z$ valide :**
1. **Pertinence** : $\text{Cov}(X, Z) \neq 0$ (l'instrument bouge réellement le treatment)
2. **Exclusion restriction** : $\text{Cov}(\varepsilon_Y, Z) = 0$ (l'instrument n'agit sur $Y$ que via $X$, pas par d'autres canaux)

**Estimation par 2SLS (Two-Stage Least Squares).** La méthode standard d'estimation d'IV en pratique :
* **Étape 1** : régresser $X$ sur $Z$ → on récupère $\hat X = \hat\gamma Z$ (la part de $X$ "expliquée par l'instrument", donc *exogène*)
* **Étape 2** : régresser $Y$ sur $\hat X$ → le coefficient obtenu est $\hat\beta$, l'effet causal de $X$ sur $Y$

> 💡 **L'idée.** $X$ contient deux choses mélangées : une partie causale + une partie polluée par le confondeur. L'étape 1 isole la partie de $X$ qui *vient uniquement de $Z$* (donc propre). L'étape 2 utilise cette partie propre pour estimer l'effet sur $Y$.

**Formule fermée (cas scalaire).** Quand $X$, $Y$, $Z$ sont uni-dimensionnels, 2SLS se réduit à :
$$
\hat\beta = \frac{\text{Cov}(Y, Z)}{\text{Cov}(X, Z)}
$$
À retenir : c'est *un ratio de covariances*. Le numérateur capte combien $Z$ bouge $Y$ (effet réduit), le dénominateur combien $Z$ bouge $X$ (force de l'instrument). Le ratio donne l'effet causal de $X$ sur $Y$.

**Exemple (Card 1995 — Effet des études sur le salaire).**

Question : *combien une année d'études supplémentaire augmente-t-elle le score à un test cognitif (proxy de capital humain) ?*

* **UA** : lycéens américains (cohorte 1980, dataset `CollegeDistance`, $n = 4739$)
* **$Y$** : score à un test composite
* **$X$** : nombre d'années d'études
* **Confondeurs cachés** : capacité innée, motivation, milieu familial → impossibles à mesurer proprement, et ils influencent à la fois $X$ (combien tu fais d'études) et $Y$ (ton score). Une simple OLS de $Y$ sur $X$ est donc biaisée.
* **Instrument $Z$** : *distance au plus proche college 4 ans*

Pourquoi c'est un bon instrument ?
* **Pertinence** ✅ : habiter loin d'un college augmente le coût (déménagement, transport) → réduit la probabilité de poursuivre les études. $\text{Cov}(X, Z) \neq 0$.
* **Exclusion** (plausible) : la distance géographique au college n'a *pas* d'effet direct sur ton score cognitif, sauf via le fait qu'elle change ton niveau d'études.

**Calcul.** En appliquant la formule fermée sur les données : $\hat\beta = \text{Cov}(Y, Z) / \text{Cov}(X, Z) \approx 3.55$

→ Une année d'études supplémentaire augmente le score d'environ **3.5 points**. C'est l'estimateur **LATE** (effet local sur les *compliers* : ceux dont la décision d'aller à l'université dépend effectivement de la distance au college). Pour les *always takers* (qui y vont quoi qu'il arrive) ou les *never takers* (qui n'y vont jamais), IV ne dit rien.

> ⚠️ **Limites pratiques.** L'exclusion restriction est *invérifiable* à partir des données seules — c'est une hypothèse qu'on défend par l'argumentation économique. Si la distance au college était corrélée à autre chose (ex : qualité des écoles secondaires locales), l'instrument serait invalide et $\hat\beta$ serait biaisé. Trouver un *bon* instrument est l'art principal en IV — souvent plus difficile que l'estimation elle-même.


### E. Régression sur Discontinuité

Régression sur Discontinuité aka Regression Discontinuity Design.
Eg Bourse d'études:
* Running variable (X) : ta note à un examen (0 à 20)
* Cutoff : "15" si tu as 15.0 tu as la bourse si tu as 14.9 tu n'as rien
* Outcome (Y) : ton futur salaire

* On a une discontinuité pour ceux qui ont la bourse et ceux qui ne l'ont pas.
* Le saut de 10k est magique pour les chercheurs car si tu compares un élève qui a eu entre 10 et 19, est-ce que l'écart de salaire vient de la bourse ou d'une variable cachée (confounder), comme le fait qu'il soit plus fort / travailleur ?
* Et entre 14.9 et 15 ils ont le même niveau, une différence de 0.1 c'est un coup de chance.
* Danger des confounders : plus on s'éloigne du 15 (vers 10 ou vers 20) plus les autres facteurs (revenus des parents, intelligence innée) recommencent à influencer le salaire.

1. Approche non-paramétrique : 
On choisit une bandwidth (largeur de bande) eg le cutoff est à 15/20 on ne regarde que les élèves entre 14.5 et 15.5
* On calcule la moyenne des salaires pour ceux entre 14.5 et 14.99 (groupe contrôle)
* On calcule moyenne des salaires pour ceux entre 15.0 et 15.5 (groupe de traitement)
=> Effet causal = moyenne des traités - moyenne des contrôles

⚠️ Problème : le choix de la bandwidth : 
* Si trop grande, on a des confounders
* Si trop petite, pas assez de données donc estimation biaisée

2. Approche paramétrique :
On trace deux droites ou 2 courbes 
i) Une courbe qui fit les points à gauche du cutoff
ii) Une courbe qui fit les points à droite du cutoff
iii) On regarde l'écart vertical entre ces deux lignes au point précis du cutoff

3. Fuzzy RDD (Discontinuité "floue")
Avant on a vu le sharp RDD (Discontinuité nette) ie si tu as 15.0 tu as la bourse, le saut de probabilité de recevoir le traitement passe de 0% à 100% au niveau du cutoff.

Fuzzy RDD c'est : à 15.0 "tu es éligible" à la bourse, la plupart l'ont mais certains oublient de renvoyer le dossier. À 14.9 tu n'es pas éligible mais qq-uns arrivent par piston ou dérogation. Le saut de probabilité passe eg de 10% à 70%.

=> Le problème c'est que maintenant le saut est "dilué" par les gens qui n'ont pas suivi la règle : 
$$
\text{Effet causal} = \frac{\text{saut de outcome (salaire)}}{\text{saut de proba de traitement}}
$$
* Saut de outcome (salaire). De combien le salaire a-t-il sauté au cutoff ?
* Saut de proba de traitement : De combien la probabilité d'avoir la bourse a sauté au cut-off ?

Eg si le salaire saute de +5k$ au seuil de 15/20 mais que passer le seuil n'augmente tes chances d'avoir la bourse que de +50% alors l'effet réel de la bourse pour ceux qui la prennent est de 5,000/0.5 = +10k$

Après j'ai qq notes : 
* sharp RDD = ATE
* Fuzzy RDD = LATE


### F. Diff-in-Diff

**Idée.** On a deux groupes (un traité, un contrôle) et on a des données **avant** ET **après** le traitement. On compare l'évolution des deux groupes plutôt que leurs niveaux.

**Pourquoi c'est plus malin qu'une simple comparaison de moyennes :**
* Comparer juste les niveaux après traitement → biaisé si les groupes sont différents au départ
* Comparer juste avant/après dans le groupe traité → biaisé si autre chose a changé entre temps (tendance générale)
* DiD = on **soustrait les deux différences**, donc on enlève à la fois les différences de niveau initiales ET les tendances communes

**La formule :**
$$
\text{DiD} = (\bar Y_{\text{traité, après}} - \bar Y_{\text{traité, avant}}) - (\bar Y_{\text{contrôle, après}} - \bar Y_{\text{contrôle, avant}})
$$

**Exemple (Brexit, en reprenant II.C) :**

|                          | Avant 24 juin | Après 24 juin | Différence |
| ------------------------ | ------------- | ------------- | ---------- |
| FTSE 250 (UK exposé)     | 100           | 93            | -7%        |
| FTSE 100 (multinational) | 100           | 97            | -3%        |
|                          |               |               |            |

$$
\text{DiD} = (-7\%) - (-3\%) = -4\%
$$

→ Le Brexit a causé une baisse de **-4% supplémentaire** sur les firmes UK-exposées vs les multinationales. Les -3% du FTSE 100 sont attribués au "choc général" (peur, livre qui chute, etc.) ; le delta de -4% est attribué à l'effet *spécifiquement domestique* du Brexit.

**Hypothèse identifiante : Parallel trends.**

⚠️ DiD ne marche que si **en l'absence de traitement, les deux groupes auraient évolué parallèlement**. C'est-à-dire : avant le choc, FTSE 250 et FTSE 100 doivent avoir des trajectoires qui montent et descendent *en même temps*. Si l'un avait déjà tendance à sur-performer l'autre, DiD attribue cette tendance à l'effet causal et c'est faux.

**Comment vérifier en pratique :**
* Tracer les deux séries sur la période *avant traitement* et regarder visuellement si elles sont parallèles
* Faire un *placebo test* : appliquer DiD sur des dates antérieures où il n'y a pas eu de traitement → on doit trouver un effet ≈ 0

**Estimand : ATT** (Average Treatment effect on the Treated). On mesure l'effet du traitement *sur ceux qui ont été traités* — pas sur la population entière.

**Régression équivalente (façon économiste) :**
$$
Y_{it} = \alpha + \beta_1 \cdot \text{Traité}_i + \beta_2 \cdot \text{Après}_t + \beta_3 \cdot (\text{Traité}_i \cdot \text{Après}_t) + \varepsilon_{it}
$$
Le coefficient **$\beta_3$** sur le terme d'interaction = **l'estimateur DiD**. C'est ça qu'on regarde et c'est sur ça qu'on fait le test de significativité.

**Cas d'usage typique :**
* Brexit, S&P 500 inclusion, changement de loi dans un État vs un autre
* A/B test où on a des données *pré-expérience* (cf. CUPED qui exploite cette idée)
* Évaluation de politiques publiques (Obamacare dans certains États seulement, etc.)

**Limites :**
* Si les tendances ne sont pas parallèles → biais
* Si le traitement est anticipé → les acteurs s'ajustent avant et "polluent" la pré-période
* Si SUTVA est violé (spillover entre groupes) → DiD est faux
* N'estime que l'**ATT**, pas l'ATE général

## IV - Estimands : ATE, ATT, CATE, LATE


$$
\begin{array}{l|l|l}
\text{ATE} & \text{Moyenne de l'effet pour tout le monde (homme et femme)} & \text{Si je force tout le pays à se vacciner, que se passe-t-il ?} \\ \hline
\text{ATT} & \text{Moyenne de l'effet pour ceux qui ont choisi d'être traités} & \text{Mes clients qui ont utilisé le coupon, ont-ils vraiment plus acheté ?} \\ \hline
\text{CATE} & \text{Moyenne pour un sous-groupe} & \text{Est-ce que ma pub marche mieux sur les jeunes ?}
\end{array}
$$



Puis on a une définition des quatre groupes : Never takers, Compliers, Always takers et Defiers

![[im12 1.png|468]]