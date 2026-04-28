## I - Introduction

Vocabulaire : 
* Unité d'analyse (Unit of analysis - UA). En causalité on ne parle pas d'observation mais d'unité d'analyse eg un individu, un bloc de bâtiment...
* Variable de sortie (outcome variable). C'est le Y
* Les features : 
	* Variable d'intérêt (Policy/Treatment variable - VI). C'est la seule feature dont on veut tester l'intérêt.
	* Variable de contrôle (Control variables - VC). Les autres features qui sont "fixes" pour isoler l'effet de la politique/du traitement.

Définition (Causalité). On définit la Causalité comme l'étude de comment une intervention, le faite de changer volontairement la variable d'intérêt, affecte aka la variable de sortie Y.

Propriétés (Causalité).
* Si il y'a zéro corrélations entre les variables après avoir netttoyé les données alors il n'y a pas de causalité.
* Deux variables corrélés n'implique pas qu'il y'a causalité.

Exemple.
La Unité d'analyse (UA) est le quartier, la Variable d'intérêt (VI) est le nombre d'arbres, la Variable de contrôle (VC) est le nombre de crime. Si il y'a beaucoup d'arbres => il y'a peu de crime, cependant, il manque la variable "richesse du quartier".

Propriété (Effet causal individuel - Unit-level causal effect).
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

Définition (Le terme d'interaction TI). C'est que une des Variable de contrôle (VC) est lié à la Variable d'intérêt (VI) 
> eg on image deux résultats potentiels (potential outcomes) : $Y(medicaid)$ et $Y(\text{pas medicaid})$ mes VS je n'en ai que 1 sur 2 (??). Ma Variable d'intérêt (VI) c'est l'obésité on dit qu'elle a un terme d'interaction (TI) qui est l'activité physique ou repas trop gras. 

Définition (Hétérogénéité). L'effet d'une variable dépend d'une autre variable.
> eg. Medicaid a un effet sur ton cholestérol SSI tu manges mal

Définition (Problème fondamental de l'inférence causal). On ne pourra jamais faire 
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
En reprenant notre échelle ci-dessus on en conclut que le traitement est bon pour les femmes mais mauvais pour les femmes.

Définition (Facteurs de confusion - Confounders). C'est une variable qui pollue la comparaison car elle change d'un groupe à l'autre.

Exemple.
> UA = quartier, Y = crime, VI = nombre d'arbre, le confounder = le revenu élevé. On a un mécanisme de confusion : 
> * Les gens riches sélectionnent les quartiers avec arbre (revenu est lié au traitement)
> * Les gens riches commettent statistiquement moins de crimes (le revenu est lié à l'outcome).

Définition (Actual Outcome vs Counterfactual outcome). Une fois que l'action est faite (eg tu as pris medicaid), les deux résultats potentiel (potential outcomes) changent de statut.
* Actual Outcome : ton résultat
* Counterfactual Outcome (le counterfactual). C'est la donnée manquante, ce qui aurait pu se passer si tu avais fais un autre choix.

Définition (Inférence causale).
$$
\boxed{\text{Inférence causale = Utiliser des hypothèses pour estimer l'inobservable}}
$$
Plus spécifiquement : 
* Inférence statistique : On apprends à partir des données qu'on voit
* Inférence causale : on essaie d'apprendre sur des donneés qu'on ne verra jamaisa (le contrefactuel), c'est pour ça que c'est beaucoup plus difficile que l'inférence statistique.

## II - Expérience contrôlé

Objectif (Expérience contrôlé). C'est de recréer artificiellement $Y_i(1)-Y_i(0)$ en utilisant deux Unité d'Analyse (Unit of Analysis) différent mais identique.

Exemple.
> On trouve deux personnes identiques (même features). Sauf que l'un reçoit le traitement $(D=1)$ l'autre non ($D=0$). Ce qui nous permettra de soustraite l'outcome de l'un par l'autre.

⚠️ En réalité c'est très difficile de trouver deux individu identique surtout quand il y'a trop de features. => La solution c'est la $\boxed{\text{~~Randomized Experiments ~~}}$

### A. Expérience Randomisée (Randomized Experiments)

Objectif (Expérience Randomisée). Au lieu de chercher deux invididus quasi identique (Expérience contrôlé) on va essayer de créer deux groupes équivalents. La moyenne du groupe $A$ devient statistiquement identique à la moyenne du groupe $B$.

Remarque (Spurious correlation). TO do

Conditions (Expérience Randomisée). Il faut que $D \perp X$ où $D$ est le traitement.
C'est à dire que si on a cent features $(X_1, X_2, .., X_{100})$ on doit vérifier que :
* la moyenne de $X_1$ est la même dans le groupe $D=0$ et $D=1$
* la moyenne de $X_2$ est la même dans le groupe $D=0$ et $D=1$
* etc

Exemple.
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

Définition (Les non compliers). Si on a 10,000 personnes, 5,000 ne prennent rien et 5,000 sont censé le prendre mais finalement 30% des 5,000 ne le prennent pas. Ainsi si je compare les "gagnants" aux "perdants" mon effet est diluée c'est le 

$$
\boxed{\text{~~ Intent-to-treat (ITT) ~~}}
$$
Exemple. Groupe contrôle (50 personnes) n'ont rien ils perdent 2kg en un mois
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

Méthodes (non compliers).
1. Intention to treat (ITT)
2. Instrumental Variables (IV)
3. Assume random compliance
4. Bound Analysis

(1) Intention to treat (ITT). Pour le (1) on l'a calculé dans l'exemple ci-dessus; 

(2) Instrumental Variables (IV). pour le (2) également c'est "l'effet IV" aka le LATE (Le Local Average Treatment effect) :
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
⚠️ Problème si les 25 qui ont participé sont tous des "hyper motivés" mon résultat sera biaisé.

(4) Bound Analysis. On accepte de l'incertitude "Quel serait l'effet si les gens qui n'ont pas répondu étaient tous des génies ou s'ils étaient tous des cas désespérés".

Exemple. On fait un intervalle $[-23\%, +37\%]$

####  Long-term Average Treatment Effects

Définition (Les proxies). C'est lorsque l'outcome réel est impossible à mesurer directement.

Exemples : 
* Startup : Tu ne peux pas attendre 10 ans pour savoir si une stratégie évite la faillite. Tu regardes le profit annuel (proxy). Si le profit baisse c'est un signe qu'il y'a un risque de fermeture.

⚠️Un bon proxy doit rester corrélé à l'outcome même après le traitement.

Définition (SUTVA - Stable Unit Treatment Value Assumption). Si les hypothèses du SUTVA sont violés alors ton ATE, CATE sont faux car il y'a des fuites entre les groupes.

Exemple (Effet de débordement - Spillover Effect). On a le groupe contrôle qui ne fait rien et le groupe traité qui reçoit une formation de trading. Le problème c'est que un des mecs du groupe traité est amis avec des gens du groupe contrôle et il leur apprends le trading. Ce qui va biaiser les résultats.

####  Long-term Average Treatment Effects

### B. Expérience naturel (Natural Experiments)

Motivation. Étude sur les impôts on ne peut pas faire payer +10% d'impôt à un groupe et +50% à un autre juste pour voir qui travaille le plus. 

Sinon j'ai écris y'en a deux types : 

True natural experiment  = exemple ?

As-if natural experiment = La France (Groupe traité) applique une taxe sur les transactions financières mais pas l'Allemagne (Groupe contrôle) si le volume baisse en France est ce à cause de la taxe ou la situation morose. 

### C. Expérience observationnel

## III - Régression Linéaire

Exemple. Y=GDP, X=property right (PR) ie que 
* score de 10 si l'état te protège si tu as un bien
* score 0-2 instable le dictateur peut tout te prendre
Or si $\rho(X,Y) >>0$ il peut y'avoir des sources de biais.

Prorpriétés (Source de biais).
1. Variables de confusion :  Une variable Z eg le climat joue sur les deux variables
2. Causalité Inverse : Une fois qu'un pays est riche les citoyens demandent des droits on a donc une relation inverse PR <- Y
3. La simultanéité : X cause Y pendant que Y cause X

### A. La Médiation

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

iv) Calcule de l'ACME (Average Causal Mediation Effect).
ACME = Effet de X sur M (1.2) x effet de M sur Y (0.4) = 0.48

CE qui veut dire que sur tes 0.5 points de fidélités initiaux, 0.48 passent par le faite de commander.

v) Storytelling : Si le Boss dit "On arrête les coupons c'est trop cher" on peut lui répondre : 
"Si on arrête les coupons, on perd 1-2 commandes par client et comme chaque commande génère +0.4 de fidélité, note fidélité globale va chuter à 0.48 points" (? pas sur de la fin)

Remarques : 
* Apparament faut aussi regarder l'écart type de la fidélité dans les données , si tout le monde a un score entre 49 et 51 alors 0.48 c'est une énorme perte.

### B. La Modération

Graphe à rajouter

Exemple (Multi-modération). 
* X : Télétravail (Treatment)
* $Z_1$ : Homme vs femme - Sexe (Modérateur 1)
* $Z_2$ : Junior vs senior - Ancienneté (Modérateur 2)
* $Y$ : Productivité

On va résoudre :

$$
Y = \beta_0 + \beta_1  X \cdot Z_1 + \beta_2 X \cdot Z_2 + \varepsilon
$$
En gros on veut savoir si l'effet du traitement (Télétravail) a un impact sur la productivité et est modérée par l'ancienneté et le sexe.

### C. Je sais pas trop

$$
\begin{array}{l|l|l}
\text{ATE} & \text{Moyenne de l'effet pour tout le monde (homme et femme)} & \text{Si je force tout le pays à se vacciner, que se passe-t-il ?} \\ \hline
\text{ATT} & \text{Moyenne de l'effet pour ceux qui ont choisi d'être traités} & \text{Mes clients qui ont utilisé le coupon, ont-ils vraiment plus acheté ?} \\ \hline
\text{CATE} & \text{Moyenne pour un sous-groupe} & \text{Est-ce que ma pub marche mieux sur les jeunes ?}
\end{array}
$$




### D. La G-Formula

On va voir une méthode pour calculer l'ATE en utilisant une méthode qu'on appelle la standardisation (G-formula) c'est une version automatisé des calculs à la main 

⚠️ Pour que la G-formula marche il faut respecter deux conditions : 
1. Unconfoundness : J'ai inclus dans mon GLM toutes les variables qui influencent à la fois le treatment et les résultats
2. Positivity (Overlap). Il faut qu'il y'ait au moins quelques personnes traités et non traités dans chaque catégorie

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

### E. Le Matching

i) Imagine tu as un soldat (vétéran) 25 ans blanc
* On ne sait pas combien il aurait gagné s'il n'étaient pas allé à l'armée
* Solution "Matching" dans la bdd tu cherches un civil (non vétéran) qui a 25 ans qui est blanc => tu regardes la différence de salaire entre les deux jumeaux
* On répète ça pour tout le monde on obtient une liste de différence : moyenne de la liste = ATE

ii) Le propensity score : C'est si tu as 50 variables le jumeau parfait est difficile à trouver à la place on calcule un score $[0,1]$ on matche les gens qui ont la même note

## IV - Instrumental Variables

Motivation : G-Formula et Matching sur l'hypohtèse d'unconfoundness or en réalité il y'a souvent des variables invisibles ou impossible à mesurer.

Eg. $Z$ est câché on ne l'a pas dans les données. Mais si $Z$ influence à la fois $X$ et $Y$ le matching sera toujours biaisé.

Solution. IV est une méthode pour trouver la causalité même quand il y'a des variables câchés. On cherche un instrument $(Z)$ qui : 
1. Influence le traitement $(X)$
2. N'a aucun rapport avec $Y$ sauf via $X$

Remarque : 
* Médiation $X \rightarrow M \rightarrow Y$
* Instrument $Z \rightarrow X \rightarrow Y$

Définitions : 
* Variable endogène
* Variable exogène

Exemple (Effet des études sur le salaire).


## V - Régression sur Discontinuité

Régression sur Discontinuité aka Regression Discontinuity Design.
Eg Bourse d'études:
* Running variable (X) : ta note à un examen (0 à 20)
* Cutoff : "15" si tu as 15.0 tu as la bourse si tu as 14.9 tu n'as rien
* Outcome (Y) : ton futur salaire

* On a une discontinuité pour ceux qui ont la bourse et ceux qui ne l'ont pas.
* Le saut de 10k est magique pour les chercheurs car si tu compares un élèves qui a eu entre 10 et 19. Est cec que l'écart de salaire vient de la bourse ou d'une variable câché (confounders) qui est il est plus fort / travailleur.
* Et entre 14.9 et 15 ils ont le même niveau différence de 0.1 c'est un coup de chance
* Danger des confounders : plus on s'éloigne du 15 (vers 10 ou vers 20) plus les autres facteurs (revenus des parents, intelligence inné) recomment à influencer le salaire.

1. Approche non-paramétrique : 
On choisit une bandwidth (largeur de bande) eg le cutoff est à 15/20 ou regarde que les élèves entre 14.5 et 15.5
* On calcule la moyenne des salaires pour ceux entre 14.5 et 14.99 (groupe contrôle)
* On calcule moyenne des salaires pour ceux entre 15.0 et 15.5 (groupe de traitement)
=> Effet causal = moyenne des traités - moyenne des contrôles

⚠️ Problème le choix de la bandwidth : 
* Si trop grand on a des confounders
* Si trop petit pas assez de données donc estimation biaisé

2. Approche paramétrique :
On trace deux droites ou 2 courbes 
i) Une courbe qui fit les points à gauche du cutoff
ii) Une courbe qui fit les points à droite du cutoff
iii) On regarde l'écart vertical entre ces deux lignes au point précis du cutoff

3. Fuzzy RDD (Discontinuité "floue")
Avant on a vu le sharp RDD (Discontinuité net) ie si tu as 15.0 tu as la bourse, le saut de probabilité de recevoir le traitement passe de à 0% à 100% au niveau du cutoff

Fuzzy RDD c'est a 15.0 "tu es éligible" à la bourse la plupart l'ont mais certains oublient de renvoyer le dossier. A 14.9 tu n'es pas éligible mais qq arrivent par piston ou dérogation. Le saut de probabilité passe eg de 10% à 70%.

=> Le problème c'est que maintenant le saut est "dilué" par les gens qui n'ont pas suivi la règle : 
$$
\text{Effet causal} = \frac{\text{saut de outcome (salaire)}}{\text{saut de proba de traitement}}
$$
* Saut de outcome (salaire). De combien le salaire a-t-iol sauté au cutoff ?
* Saut de proba de traitement : De combien la probabilité d'avoir la bourse a sauté au cut-off

Eg si le salaire saute de +5l$ au seuil de 15/20 mais que passer le seuil n'augmente tes chances d'avoir la bourse que de +50% alors l'effet réel de la bourse pour ceux qui la prennent est de 5,000/0.5=+10k$

Après j'ai qq notes : 
* sharp RDD = ATE
* Fuzzy RDD = LATE

Puis on a une définition des quatre groupes : Never takers, compliers, always takers et defiers

![[im12 1.png|468]]