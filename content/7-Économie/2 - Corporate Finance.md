# Comptabilité

**Bilan ou Balance sheeet :**

- Coucou

![image.png](images/6-Économie/02_Corporate_Finance/01_image.png)

Compte de résultat ou ? :

- Le Income Statement américain est plus proche d’un “compte de résultat fonctionnel” en français
- Note le bilan comptable et le compte de résultat matche pas (mais y’a bien une colonne en commun le “Résultat de l’exercice”

![image.png](images/6-Économie/02_Corporate_Finance/02_image.png)

# X

# X. Comprendre l’effet de levier

Rappels de comptabilité : Un bilan se découpe en deux grandes parties :

**Actif (ce que l’entreprise possède) = Passif (ce qu’elle doit + ce qui lui appartient)**

👉 Passif = **Capitaux propres (Equity)** + **Dettes (Liabilities)**

### Exemple :

- **Actifs** : 200 € (immeubles, machines, brevets, cash, etc.)
- **Passifs** :
    - Dette (long terme + court terme) = 100 €
    - Capitaux propres (Equity) = 100 €

Donc : Actifs (200) = Dette (100) + Equity (100). ✅

![image.png](images/6-Économie/02_Corporate_Finance/03_image.png)

**Effet de levier avec les formules comptables/financières :**

a) Le **ROA (Return on Assets)** mesure la rentabilité de l’entreprise par rapport à ses actifs totaux.

$ROA = \frac{\text{Résultat net}}{\text{Actifs totaux}}$

- **Résultat net** = bénéfice après paiement des intérêts et impôts.
- **Actifs totaux** = tout ce que possède l’entreprise (immeubles, machines, cash, etc.).

👉 Le ROA dit : « combien rapporte chaque euro investi dans les actifs ».

b) le **ROE (Return on Equity)** mesure la rentabilité pour les actionnaires uniquement.

$R O E=\frac{\text { Résultat net }}{\text { Capitaux propres }}$

- **Résultat net** = ce qui reste après avoir payé les intérêts aux créanciers.
- **Capitaux propres (Equity)** = argent des actionnaires (capital social + bénéfices accumulés).

👉 Le ROE dit : « combien rapporte chaque euro mis par les actionnaires ».

c) Lien entre ROA, ROE et effet de levier

En présence de dette, le **ROE** peut être plus grand que le **ROA** grâce à l’effet de levier :

$ROE=ROA+(ROA−i)\times \frac{D}{E}$

où :

- $ROA$ = rendement des actifs
- $i$ = taux d’intérêt de la dette
- $D$ = montant de la dette
- $E$ = capitaux propres

👉 Interprétation :

- Si $ROA>i$ l’endettement **augmente** la rentabilité des actionnaires (effet de levier positif).
- Si $ROA<i$, l’endettement **réduit** la rentabilité des actionnaires (effet de levier négatif).

**Preuve :** On rajoute le faite que la relation bilan est de : $A = D + E$

$\begin{aligned}  
&R O E=\frac{\text { Résultat net }}{E}=\frac{R O A \times A-i \times D}{E}\\  
&\text { Remplaçons } A \text { par } D+E \text { : }\\  
&\begin{gathered}  
R O E=\frac{R O A \times(D+E)-i D}{E} \\  
R O E=\frac{R O A \times E}{E}+\frac{R O A \times D}{E}-\frac{i D}{E} \\  
R O E=R O A+(R O A-i) \frac{D}{E}  
\end{gathered}  
\end{aligned}$

### Cas A : sans dette

- Actifs = 200
- Dette = 0
- Equity = 200
- Résultat net = 20

$ROA = \frac{20}{200} = 10\%$

$ROE = \frac{20}{200} = 10\%$

👉 Pas de différence : tout est financé par les actionnaires.

![image.png](images/6-Économie/02_Corporate_Finance/04_image.png)

### Cas B : avec dette

- Actifs = 200
- Dette = 100 (5% d’intérêt = 5)
- Equity = 100
- Résultat brut (avant intérêts) = 20
- Résultat net = 20 - 5 = 15

$ROA = \frac{15}{200} = 7,5\%$

$ROE = \frac{15}{100} = 15\%$

👉 Ici, le ROA baisse (car on retire les intérêts du numérateur), mais le **ROE explose** grâce à l’effet de levier.

![image.png](images/6-Économie/02_Corporate_Finance/05_image.png)

### Cas C: Risque de l’effet de levier

Si la rentabilité des actifs chute, par exemple à 3% :

- Résultat brut = 200 × 3% = 6
- Intérêts = 5
- Résultat net = 1

$ROA = \frac{1}{200} = 0,5\%$

$ROE = \frac{1}{100} = 1\%$

👉 Le ROE s’effondre, et si le ROA passe sous 2,5%, l’entreprise ne peut même plus payer ses intérêts = faillite de liquidité.

Cas favorable un $ROA=10\%$

$ROE = \frac{A \times ROA - i \times D}{E} = \frac{200 \times 10\% - 198 \times 5\%}{200} = 505\%$

👉Cas défavorable (ROA chute à 3%).

$ROE = \frac{A \times ROA - i \times D}{E} = \frac{200 \times 3\% - 198 \times 5\%}{200} = -195\%$  
Avec un levier énorme, **un petit choc de rentabilité efface tout** et envoie les actionnaires dans le rouge.

![image.png](images/6-Économie/02_Corporate_Finance/06_image.png)

---

📘 Comprendre l’effet de levier (leverage)

1. Définition simple

L’**effet de levier** correspond à l’utilisation de la dette pour augmenter la rentabilité des capitaux propres (ce que les actionnaires ont investi dans l’entreprise).

➡️ En d’autres termes, au lieu de financer un projet uniquement avec leur propre argent, les actionnaires utilisent aussi de l’argent emprunté. Si l’investissement rapporte plus que le coût de la dette, leur rendement est amplifié.

2. Exemple de base

Situation 1 : entreprise sans dette

- Actifs (immeubles, machines, bureaux, etc.) : **200 €**
- Financement : 200 € apportés par les actionnaires.
- Rendement des actifs : **10% par mois** → donc l’entreprise génère **20 € de bénéfice**.
- Rendement pour les actionnaires = 20 / 200 = **10%**.

Situation 2 : entreprise avec dette (effet de levier)

- Actifs : **200 €**
- Financement :
    - 100 € apportés par les actionnaires.
    - 100 € empruntés auprès de créanciers.
- Rendement des actifs : toujours 10% → **20 € de bénéfice brut**.
- Coût de la dette : 5% sur 100 € → **5 € d’intérêts à payer**.
- Bénéfice net pour les actionnaires = 20 - 5 = **15 €**.
- Rendement pour les actionnaires = 15 / 100 = **15%**.

👉 Résultat : grâce à la dette, les actionnaires passent de 10% à 15% de rendement.

C’est **l’effet de levier positif**.

3. Pourquoi ne pas emprunter "à l’infini" ?

Imaginons une entreprise ultra-endettée :

- Actifs : **200 €**
- Financement :
    - 2 € apportés par les actionnaires.
    - 198 € empruntés.
- Rendement des actifs (10%) : 20 €.
- Intérêts (5% de 198) ≈ **9,9 €**.
- Résultat net pour les actionnaires = 20 - 9,9 ≈ **10,1 €**.
- Rendement = 10,1 / 2 ≈ **505%** ! 🚀

Ça paraît génial, mais en réalité, c’est **extrêmement risqué** :

- Si les actifs chutent un peu (par ex. baisse de la valeur des immeubles), l’entreprise peut devenir **insolvable** : ses dettes > ses actifs → **faillite comptable**.
- Si les revenus chutent temporairement (par ex. crise économique), elle peut ne pas avoir assez de cash pour payer les intérêts → **faillite de liquidité**.
- Les créanciers veulent être payés **à date fixe** (les intérêts sont obligatoires), alors que les actionnaires, eux, peuvent patienter.

👉 Plus il y a de dette, plus l’effet de levier amplifie les gains **en période favorable**, mais aussi les pertes **en période de crise**.

4. Pourquoi les entreprises (et les États) refinancent leur dette au lieu de la rembourser ?

- **Économie de liquidités** : garder du cash pour investir, développer l’activité ou faire face à des imprévus, plutôt que de tout utiliser pour rembourser.
- **Coût de la dette bas** : si les taux d’intérêt sont faibles, il est souvent plus rentable de conserver la dette et d’investir l’argent ailleurs.
- **Refinancement** : quand une dette arrive à échéance, au lieu de la rembourser avec les bénéfices, on émet une nouvelle dette pour remplacer l’ancienne. C’est courant car :
    - les investisseurs (obligataires) acceptent de prêter à nouveau ;
    - ça évite d’assécher la trésorerie.
- **Fiscalité** : les intérêts de la dette sont souvent déductibles des impôts, ce qui rend la dette encore plus attractive pour les entreprises.

5. Cas particulier : banques et hedge funds

- Ils utilisent énormément de levier, parfois **20x ou 30x** leur capital.
- Exemple : avec 1 € de fonds propres, ils empruntent 29 € pour avoir 30 € à investir.
- Tant que le rendement > coût de la dette, c’est ultra rentable.
- Mais en cas de crise, une perte de seulement 3-4% des actifs peut effacer tout le capital → faillite rapide (cf. Lehman Brothers 2008).

# 📘 Lien entre ROE et prix de l’action

## 1. Comment on valorise une action en théorie

La valeur d’une action = **valeur actualisée des cash-flows futurs aux actionnaires**.

En pratique :

- soit via les **dividendes attendus** (modèle de Gordon–Shapiro),
- soit via les **bénéfices futurs** (PER, Price/Earnings Ratio).

👉 Ce que regardent les investisseurs, c’est :

1. combien l’entreprise **peut générer de bénéfices** avec son capital,
2. et si elle peut **les faire croître dans le temps**.

---

## 2. Rôle du ROE

Le ROE = capacité à générer du résultat net **par euro investi par les actionnaires**.

- Si une entreprise a un **ROE élevé et durable**, cela veut dire qu’elle arrive à transformer efficacement le capital des actionnaires en bénéfices.
- Les marchés récompensent ça en lui donnant un **multiple plus élevé (PER)** → donc le prix de l’action StS_tSt augmente.

⚠️ Mais attention : le marché regarde **le couple ROE / risque**.

- Si ton ROE est élevé mais obtenu avec un levier énorme et donc un risque de faillite → le marché te donne un discount.
- Si ton ROE est élevé grâce à un business solide et peu risqué (ex. Apple, LVMH) → le marché paie cher ton action.

---

## 3. Exemple simple

Deux entreprises avec le même bénéfice (10 M€) :

- **Entreprise A** : Equity = 200 M€ → ROE = 10/200 = 5%
- **Entreprise B** : Equity = 50 M€ → ROE = 10/50 = 20%

➡️ Si les investisseurs pensent que B peut maintenir 20% de ROE de façon durable, ils valoriseront B beaucoup plus cher qu’A.

Mais si ce 20% vient d’un levier risqué (banques avant 2008), alors le prix peut chuter brutalement en cas de crise.

---

## 4. Cas des banques

Tu dis « sinon les banques auraient toutes un prix énorme ».

👉 Justement, beaucoup de banques ont historiquement eu des ROE artificiellement gonflés grâce à l’effet de levier.

Mais :

- le marché **intègre le risque de faillite** → donc elles ne sont pas valorisées comme des "machines à cash".
- après 2008, les régulateurs ont imposé plus de capital (Equity ↑) → ROE ↓ → valorisations plus modérées

---

1. Rappel du PER

Le PER (Price/Earnings Ratio) est :  
$P E R=\frac{S_t}{E P S}$

où :

- $S_t=$ prix de l'action,
- $E P S=$ Earnings Per Share $=$ bénéfice net $/$ nombre d'actions.

👉C'est combien d'années de bénéfices les investisseurs sont prêts à "payer" pour acheter une action.

- Un **PER élevé** = marché croit que les bénéfices vont croître durablement.
- Un **PER faible** = marché pense que les bénéfices sont stagnants ou risqués.

# 📘 2. Lien ROE ↔ croissance des bénéfices

Le ROE joue sur la **croissance future des bénéfices**.

Pourquoi ?

- Si une entreprise a un ROE élevé et **réinvestit ses bénéfices** (au lieu de les distribuer), alors ses capitaux propres grossissent vite → et donc ses bénéfices futurs aussi.
- La formule classique est :

$g= ROE \times \text{taux de rétention}$

où $g =$ croissance des bénéfices,

et taux de rétention = part des bénéfices gardés dans l’entreprise (non distribués en dividendes).

---

# 📘 3. Modèle de valorisation (Gordon–Shapiro simplifié)

En finance, une action se valorise souvent ainsi :

$S_t= \frac{D_1}{k - g}$

où :

- $D_1$ = dividende attendu l’an prochain,
- $k$ = coût du capital (rendement exigé par les investisseurs),
- $g$ = taux de croissance des dividendes/bénéfices.

Et comme $g= ROE \times \text{taux de rétention}$ :

👉 Si le ROE est durablement élevé → g↑ → le dénominateur (k−g) ↓ → donc St↑.

# 📘 3. Relier tout ça

1. Dividende attendu :
    
    $D1=EPS \times (1 - b)$
    
    car si une entreprise garde $b$ (rétention) alors elle distribue $1−b$
    
2. On remplace dans Gordon-Shapiro :
    

$S_t = \frac{EPS \times (1 - b)}{k - (ROE \times b)}$

---

⇒ Aussi pour le délire j’ai regardé un peu Gordon-Shapiro et comment on calcule “k” avec medaf ..

# 📘 4. Conséquences

- Si **ROE ↑** (et que $ROE>k$), alors $g↑ → S_t↑$.
- Si **taux de rétention b ↑**, l’entreprise distribue moins aujourd’hui mais fait croître ses bénéfices plus vite $→ S_t↑$ (tant que les actionnaires croient à la croissance).
- Le **PER** devient :

$PER= \frac{S_t}{EPS} = \frac{(1 - b)}{k - (ROE \times b)}$

👉 On voit bien que **PER dépend directement du ROE** et du taux de rétention.

✅On en déduit que :

- $ROE_t$ est bien une série qui capture l’efficacité de l’entreprise.
- Si le marché croit qu’il est **haut et durable**, il anticipe une forte croissance g.
- Ce $g$ entre dans la formule → augmente le PER → et donc le prix $S_t$

Ainsi :

$R O E ↑ \rightarrow g \rightarrow \text{PER attendu ↑ } \rightarrow \text{flux d'achats ↑ } \rightarrow S_t ↑$

## 5. Réponse à la question de départ

Les entreprises **ne remboursent pas leur dette avec leurs bénéfices** parce que :

1. La dette permet d’amplifier le **ROE** via l’effet de levier.
2. Un ROE élevé rend l’entreprise plus attractive → soutient le **PER** et le prix de l’action $S_t$
3. Rembourser = réduire la dette = réduire le levier = réduire le ROE → moins sexy pour les investisseurs.
4. En plus :
    - les intérêts de la dette sont déductibles fiscalement,
    - le refinancement garde du cash dispo pour investir ailleurs,
    - ça augmente la flexibilité financière.

✅ Donc la logique complète est :

**Refinancement → maintien du levier → ROE plus élevé → signal positif → PER plus élevé → prix de l’action plus haut → plus facile d’attirer des investisseurs.**

# Elon Musk

En analysant Tesla, on remarque que **63 % de ses actifs appartiennent aux créanciers** et seulement le reste aux actionnaires. Si Elon Musk détient environ **20 %** de Tesla, il possède en valeur comptable près de **3,5 milliards de dollars**, bien loin des 185 milliards souvent cités.

Ce que l’on observe ici, c’est la **valeur comptable** : en d’autres termes, si quelqu’un voulait créer une entreprise identique à Tesla, avec les mêmes usines, machines, brevets, il faudrait environ **45 milliards de dollars**. Pour financer ces actifs, Tesla a demandé **22,6 milliards aux actionnaires** (dont 5,7 milliards de pertes cumulées) et **28,1 milliards aux créanciers**.

Cette approche revient à regarder le passé : comment Tesla a financé ses actifs. C’est utile pour comprendre la structure, mais cela ne dit rien sur **combien Tesla vaut si on veut l’acheter aujourd’hui**. La vraie valeur de marché dépend du **potentiel de bénéfices futurs** que l’entreprise pourra générer. Or, jusqu’ici, Tesla a **perdu plus d’argent qu’elle n’en a gagné**.

En résumé, les actionnaires ont investi **22,6 milliards**, qui valent aujourd’hui **16,9 milliards** en valeur comptable, tandis que le reste des actifs appartient aux créanciers. C’est donc le rôle des marchés financiers d’anticiper les bénéfices futurs, qui iront à 100 % aux actionnaires. Mais comme l’a montré Robert Shiller, les marchés ne font pas toujours ce travail correctement, car la **spéculation** brouille souvent cette anticipation.

![image.png](images/6-Économie/02_Corporate_Finance/07_image.png)

Tesla a environ **947,9 millions d’actions** en circulation. Avec un cours de **8 940,44 $**, sa **capitalisation boursière** atteint environ **834 milliards de dollars**. Si Elon Musk détient **20 %**, cela représente **166,8 milliards**.

Pourtant, en valeur comptable, Tesla ne possède que **45 milliards d’actifs**, dont seulement **16,9 milliards appartiennent réellement aux actionnaires** (le reste étant financé par les créanciers). En théorie, si l’entreprise devait être liquidée, les créanciers seraient remboursés en premier, et **les actionnaires ne récupéreraient que ces 16,9 milliards** – et encore, souvent une liquidation se fait à prix bradé, donc moins.

Les marchés financiers, eux, anticipent beaucoup plus : ils valorisent les actions de Tesla **18 fois la valeur des actifs actuels** et environ **49 fois les capitaux propres comptables**. Autrement dit, la valeur de marché repose bien plus sur les **bénéfices futurs espérés** que sur la valeur des actifs présents.

👉 On peut comparer ce multiple avec d’autres entreprises pour juger si Tesla est survalorisée ou non par rapport à son secteur.

|Paramètre|Tesla|General Motors (GM)|Toyota|Volkswagen|
|---|---|---|---|---|
|**Véhicules vendus / livrés (2019)**|$0.367 millions|$7,7 millions|$10.7 millions|$11 millions|
|**Chiffre d’affaires (2019)**|$24 Mds USD|$137 Mds|$263 Mds|$308 Mds|
|**Bénéfice net (2019)**|$-70 M (perte)|$8.5 Mds|$25 Mds|$22.4 Mds|
|**Valeur comptable (2019)**|$45 Mds|$240 Mds|$498 Mds|$595 Mds|
|**Capitalisation boursière (janv. 2021)**|$834 Mds|$63 Mds|$246 Mds|$99 Mds|

A partir de ça on se demande si la Capitlaisation boursière de Tesla n’est pas un peu exagéré. Mais revennons à la richesse de Ellon Musk si il arrive à vendre ses $168.8Mds

---

# X. Corporate Actions

## X.1 Les dividendes

**Les dividendes n’enrichissent pas réellement les actionnaires.**

On a vu que Tesla peut financer ses actifs de trois façons : grâce à l’argent prêté par les créanciers, aux apports des actionnaires, ou aux bénéfices accumulés. Quand l’entreprise décide de payer un dividende, elle fait simplement un **transfert** d’argent de la société vers le compte bancaire des actionnaires.

Ce transfert implique une diminution de ce que “l’entreprise doit à ses actionnaires”, c’est-à-dire des bénéfices accumulés inscrits en capitaux propres. Résultat : après le versement, les actionnaires possèdent **autant qu’avant**, mais réparti différemment :

- avant le dividende : valeur comptable (par ex. 1 000 000 €),
- après le dividende : cash sur leur compte (900 000 €) + valeur comptable réduite (100 000 €),
- total = toujours 1 000 000 €.

👉 Autrement dit, le dividende ne crée pas de richesse supplémentaire : il change seulement la **forme** sous laquelle les actionnaires détiennent leur valeur.

![Ce que possède les actionnaires est le même.
Mais la valeur comptable baisse de la valeur des dividendes.](images/6-Économie/02_Corporate_Finance/08_image.png)

Ce que possède les actionnaires est le même.  
Mais la valeur comptable baisse de la valeur des dividendes.

![image.png](images/6-Économie/02_Corporate_Finance/09_image.png)

---

## X.2 Rachats d’actions (buyback)

Quand une entreprise rachète ses propres actions, c’est une **tactique comptable** : en réduisant le nombre de titres en circulation, elle fait monter mécaniquement le cours. C’est aussi une façon d’**éviter la fiscalité des dividendes**. Apple en a fait sa spécialité : sans ces rachats massifs, sa valorisation serait bien plus faible. Ce mécanisme ne profite pas aux vendeurs, mais surtout **à ceux qui conservent leurs actions**, car leur part de l’entreprise vaut davantage.

Quand une entreprise fait des bénéfices, la direction doit décider quoi en faire. En théorie, elle pourrait augmenter les salaires. Mais selon les règles comptables, **tout ce que possède une entreprise appartient soit aux créanciers, soit aux actionnaires**. Les bénéfices ne peuvent donc revenir qu’aux actionnaires. Une augmentation de salaire est perçue comme **prendre aux actionnaires pour donner aux salariés**, et n’est tolérée qu’à la marge, sous forme de primes. La norme managériale est claire : l’objectif est de **maximiser le bénéfice pour les actionnaires**, tandis que le salarié est vu comme une ressource remplaçable.

Dès lors, il reste deux usages possibles des bénéfices :

1. **Réinvestir** (brevets, recherche, placements) → signe que la direction sait comment créer plus de valeur.
2. **Rendre l’argent aux actionnaires** (dividendes ou rachats d’actions). Le dividende est taxé, tandis que le rachat fait monter le cours et l’EPS, ce qui favorise les actionnaires restants.

👉 Réinvestir = _“on sait quoi faire de l’argent”_.

👉 Distribuer (dividende/rachat) = _“on ne sait pas quoi en faire, donc on vous le rend”_.

Différence dividende et rachat d’actions ?

- rachat: plus flexible si elle rémunère avec du cash ceux qui vendent leurs actions ça fait monter le prix des actions.

![image.png](images/6-Économie/02_Corporate_Finance/10_image.png)

Le **dividende** est un versement direct aux actionnaires. C’est fixe, visible, mais rigide : une fois instauré, il est difficile de le réduire sans envoyer un signal négatif.

Le **rachat d’actions** est plus flexible : l’entreprise utilise son cash pour racheter des actions à ceux qui veulent vendre. Résultat : le **nombre d’actions en circulation diminue**, ce qui fait mécaniquement grimper le **bénéfice par action (EPS)** et donc le cours de bourse.

$EPS = \frac{\text{Bénéfice net}}{\text{Nombre d’actions}}$

Si le **EPS** augmente ⇒ le **cours de l’action** ($S_t$) tend à augmenter, car chaque action représente une part plus importante des bénéfices futurs.

$EPS↑    ⇒    St↑$

Contrairement à l’idée reçue, ce n’est pas seulement une question de **supply/demand** : le prix de l’action reflète surtout un **ratio financier lié aux bénéfices futurs**. En tant que propriétaire, je détiens un morceau des bénéfices de l’entreprise. Comme je ne connais pas le futur, j’utilise un calcul simple :

**Exemple Apple**

- **2006–2013** : Apple émettait environ **+1,6 % d’actions par an**, principalement pour rémunérer via stock-options. Le nombre d’actions est passé à **26,5 milliards**.
- **À partir de 2013** : Apple change de stratégie et rachète ses actions à un rythme de **–5,2 % par an**, jusqu’à descendre à environ **16,8 milliards d’actions**.
- Sur les graphiques, on voit que lorsque les rachats s’accélèrent, le **cours de l’action grimpe fortement**, même si la croissance des bénéfices devient plus linéaire (avec un boost ponctuel lors du COVID).

👉 En clair : **avant 2013**, la croissance du bénéfice tirait le cours d’Apple. **Après 2013**, ce sont surtout les **rachats massifs d’actions** qui ont maintenu la hausse de son cours.

![image.png](images/6-Économie/02_Corporate_Finance/11_image.png)

![image.png](images/6-Économie/02_Corporate_Finance/12_image.png)

**📌 Dividende**

1. **Cash immédiat** : chaque actionnaire reçoit une part des bénéfices, ce qui augmente son pouvoir d’achat. C’est perçu comme moins risqué car c’est un revenu réel. Inconvénient : il faut payer des **impôts** sur ces dividendes.
2. **Effet mécanique sur le cours** : après le détachement du dividende, le prix de l’action (StS_tSt) baisse d’autant. Théoriquement, il devrait remonter ensuite, mais cela dépend de la confiance du marché.
3. **Signal au marché** : instaurer un dividende récurrent signifie : _“chaque année, nous serons capables de payer au moins autant que l’an passé.”_ C’est une promesse implicite de stabilité.

**📌 Rachat d’actions**

1. **Volontariat** : seuls les actionnaires qui veulent vendre participent. Ceux qui gardent leurs titres voient leur part relative de l’entreprise augmenter.
2. **Hausse de l’EPS** : comme le nombre d’actions diminue, le **bénéfice par action** grimpe (EPS=Beˊneˊfice netNb d’actionsEPS = \frac{\text{Bénéfice net}}{\text{Nb d’actions}}EPS=Nb d’actionsBeˊneˊfice net), ce qui fait monter le cours de bourse (StS_tSt).
3. **Intérêt des dirigeants** : cette méthode est particulièrement avantageuse pour les dirigeants qui détiennent des **stock-options**, car elle gonfle mécaniquement la valeur de leurs actions sans passer par une distribution de cash.

![image.png](images/6-Économie/02_Corporate_Finance/13_image.png)

On dit souvent qu’Elon Musk a un “salaire énorme” de PDG, mais en réalité, il ne touche pas de salaire fixe. Sa rémunération, comme celle d’autres dirigeants, se fait presque exclusivement en **stock-options**.

Imaginons que je sois PDG et que je reçoive des stock-options me permettant, dans cinq ans, d’acheter 10 millions d’actions à 12 €, alors que le cours est aujourd’hui de 10 €. Si dans cinq ans l’action monte à 20 €, je pourrais les acheter à 12 € et les revendre aussitôt à 20 €, soit un gain de 80 millions d’euros.

Évidemment, je n’ai pas 120 millions à avancer. En pratique, la **banque finance l’achat** en prenant mes stock-options comme garantie. Elle se rémunère via une commission ou des intérêts. Je peux généralement emprunter environ 30 % de leur valeur, sans rembourser le capital, uniquement les intérêts. Si le cours monte, je peux emprunter davantage ; s’il baisse, la banque durcit ses conditions.

En tant que PDG, je rembourse donc seulement les intérêts, très faibles, que je peux couvrir avec mes autres revenus financiers (loyers, placements). Au moment de vendre, je rembourse le prêt et conserve la plus-value. Je peux aussi décider de garder mes actions, devenir pleinement actionnaire, et continuer à les utiliser comme garantie pour financer mon train de vie.

Toute cette mécanique est gérée par des équipes spécialisées de financiers et de juristes, ce qui rend l’opération fluide et relativement simple pour moi, malgré la complexité apparente.

![image.png](images/6-Économie/02_Corporate_Finance/14_image.png)

![image.png](images/6-Économie/02_Corporate_Finance/15_image.png)

Les grands fonds comme BlackRock ou les fonds de pension prennent une commission sur ce qu’ils font gagner à leurs clients.

- Avec les dividendes, il y a une incertitude : ils dépendent des résultats, peuvent baisser ou disparaître.
- Alors qu’avec les rachats d’actions, le cours monte, ce qui fait mécaniquement grimper la valeur des portefeuilles. Même si aucune action n’a été vendue, c’est déjà considéré comme un gain, et les fonds peuvent toucher leur commission dessus.

Mais l’énorme risque, c’est qu’en cas de crise, toute cette valeur peut s’évaporer d’un coup. Les cours chutent, les portefeuilles se dégonflent, et toute la richesse “virtuelle” accumulée par les rachats s’écroule en même temps. Ceux qui vivent à crédit sur la valeur de leurs actions ou de leurs stock-options se retrouvent alors en grande difficulté, car la garantie ne vaut plus rien et la banque réclame des comptes.

![image.png](images/6-Économie/02_Corporate_Finance/16_image.png)

![image.png](images/6-Économie/02_Corporate_Finance/17_image.png)

Les entreprises sont aussi influencées par leurs concurrents. Si, dans un secteur, certaines distribuent des dividendes, des fonds spécialisés viendront chercher ces rendements, et certains actionnaires particuliers apprécient les dividendes parce qu’ils offrent une certitude immédiate.

Le rachat d’actions, lui, a une limite : le nombre d’actions disponibles. Entre 2013 et 2020, Apple a racheté près de 2 milliards d’actions sur un total de 16,8 milliards. Elle ne peut pas continuer cette politique éternellement, car au bout d’une dizaine d’années, il n’y aurait plus rien à racheter. L’émission de nouvelles actions est possible, mais ce serait pour lever des fonds, pas dans une logique de redistribution. Dans le cas d’Apple, ces rachats traduisent surtout un excès de bénéfices qu’elle ne sait pas réallouer autrement.

Cela crée aussi un jeu sur le bénéfice par action (EPS). Quand une entreprise émet de nouvelles actions, le nombre total grimpe, donc l’EPS baisse mécaniquement. À l’inverse, quand elle rachète ses propres actions, le nombre diminue et l’EPS remonte. Au final, elle peut donner l’impression de stabilité ou de croissance sans que le bénéfice total ait vraiment changé.

$EPS = \frac{\text{Bénéfice nets}}{Nombre d’actions}$

---

Une entreprise peut aussi décider de s’endetter pour racheter ses actions, c’est ce qu’on appelle l’effet de levier. Imaginons que son actif total vaille 100. Les créanciers détiennent une dette de 60 et les actionnaires représentent 40. En réalité, ces 40 ne correspondent pas forcément à ce qu’ils ont injecté au départ : selon les pertes ou les gains passés, leur apport réel peut avoir été plus élevé ou plus faible entre 0 et l’infini.

![image.png](images/6-Économie/02_Corporate_Finance/18_image.png)

**Solution 1 :**  
Imaginons qu’à l’année **Y2**, l’entreprise réalise un **bénéfice de +10** qu’elle conserve (non distribué). Avant ce bénéfice, les **actionnaires détenaient 40 de capitaux propres**. Avec ce gain, leur part passe à **50**. Le rendement pour eux est donc :  
$ROE = \frac{\text{Résultat net}}{\text{Capitaux propres}} = \frac{10}{40} = 25\%$  
Si l’entreprise **réinvestit entièrement ce bénéfice** dans son actif, on suppose qu’il rapporte **1 € l’année suivante** (soit **10 % de rentabilité sur l’actif**). Le rendement global reste donc de **10 % au niveau de l’entreprise**, mais pour les actionnaires, la rentabilité **baisse mécaniquement** car leur base de capitaux propres a augmenté (de 40 à 50).  
👉 Pour maintenir un **ROE élevé**, l’entreprise doit alors chercher à **réduire ses coûts**, **trouver des fournisseurs moins chers** ou encore **licencier du personnel**.

![image.png](images/6-Économie/02_Corporate_Finance/19_image.png)

**Solution 2 : le rachat d’actions par la dette.**

L’entreprise **s’endette de 10 supplémentaires**. L’actif (**A**) augmente de +10, mais on remplace une partie des **actionnaires par des créanciers**. Cela crée un **effet de levier** qui peut doper le **ROE**, mais avec un **risque de faillite comptable** en cas de difficultés. Il faut donc que les investisseurs aient **confiance** dans la solidité de l’entreprise.

👉 On choisit cette stratégie quand les **actionnaires exigent un rendement trop élevé** et que l’entreprise ne peut plus suivre uniquement avec ses bénéfices. Dans ce cas, elle **remplace les actionnaires par des créanciers**, qui **coûtent moins cher** (taux d’intérêt fixe). Résultat : le **cours de l’action (Sₜ)** grimpe mécaniquement, ce qui est particulièrement intéressant pour les **dirigeants détenant des stock-options**.

![image.png](images/6-Économie/02_Corporate_Finance/20_image.png)

![image.png](images/6-Économie/02_Corporate_Finance/21_image.png)

## X.3 Splits

![image.png](images/6-Économie/02_Corporate_Finance/22_image.png)

XX

![image.png](images/6-Économie/02_Corporate_Finance/23_image.png)

## X.4 Fusions/acquisitions

## X.5 Spin-offs

# Gouvernance d’entreprise et valorisation

Livre Roger martin - Fixing the game

[https://heu7reka.github.io/Episode_27.html](https://heu7reka.github.io/Episode_27.html)

notions:

- marché FI sont pas efficient
- entreprises sont éternels

Marchés FI : qu’est ce que le prix d’une action ? valeur de l’entreprise ou market cap = quantité * prix; pour définir le prix de l’entreprise on regarde la market cap de tesla, elle est au même niveau que General motors sauf que son chiffre d’faareie est 26x moins élevé que GM et quand Tesla a perdu 770million $ GM en a rapporté 9.2Mds.

![image.png](images/6-Économie/02_Corporate_Finance/24_image.png)

CE qui détermine le prix d’une action c’est la S/D ce qui justifie les déséquilibre cest les anticipations futurs des investisseurs, donc tesla vaut aussi cher que GM pas parcequ’elle est aussi rentable c’est surtout que les investisseurs que Tesla sera plus rentable dans le futur. ⇒ on doit donc prédire les bénéfices futurs. Les investisseures utilisent la méthode du doigt mouillé pour deviner ça (via compte de trésorire, résultat, financier)

Selon la théorie des marchés efficients de Fama les investisseurs forment leurs anticipations sur les bénéfices futurs en utilisant des informations pertinentes, cette théorie des marché financier ne dit pas que les marchés FI prédisent ‘lavenir ils disent qu’il faut regarder les marchés FI digérer et filtrer les informations et laisser l’offre et la demande de filtrer l’information. Cette théorie est trop simple ou complétement fausse. Eg nasa dit que y’a une éruption solaire et dit que ça va provoquer l’effondrement du CAC40 ⇒ il dit je filtre cette information car ça n’a pas de sens, mais des gens vont vraiment y croire ils vont vendre leurs actions, et d’autres investisseurs avisés vont se dire qu’ils veulent battre ces abrutis ils vont donc les vendre avant eux et donc le CAC 40 s’effondre car les investisseurs sont tous en compétitions (mécanisme auto réalisateur est à la base des bulles spéculatives. Entreprise ne produise pas moins de richesse à cause de l’éruption solaire pourtant les prix ont bien baissé. La théorie des marché FI n’expliquent pas la spéculation pour eux les prix sont toujours connectés à des vrais anticipations des richesses potentiels futurs. mais comme on l’a vu c as le cas, et bcp de mécanisme financier ont des impact sur le prix alors que c pas pertinent. Eg les BC injecte nouvelle monnaie via QE les investisseurs ont trop de cash kls vont acheter des produits financiers donc les prix monte (ça n’a aucun rapport avec de potentiel monté de bénéfice de l’entreprise). Eg2 certain fond d’invest doit investir dans certains sstocks qu’ils ne veulent pas mais c’est due à des contrainte stratégique et technique, même si le gérant du fond en utilisant le plus d’info pertinnent il sait qu’il faut vendre les entreprise pétrolière il pourra pas le faire voir en acheter.

![image.png](images/6-Économie/02_Corporate_Finance/25_image.png)

Pour les théorie des marché FI efficient on n’a pas d’autre modèle donc c’est mieux d’avoir cette approximation. Si le signal prix qu’il procure est faussé alors tous les marchésé (blé, lait, énergie…) ne colle pas.

Idée qu’il y’a des amrchés financiers efficients est un mythe en réalité c’est manipulable (marchés réels)

![image.png](images/6-Économie/02_Corporate_Finance/26_image.png)

Si on revient à l’exemple de Tesla la valorisation est très élevé par rapport à la réalité eg anticipation, planche a billet, règle planqué des fonds d’investissment.. En réalité, qq soit la valorisation d’une entreprise y’aura tjours qqn qui te dira que l’anticipation est rationnelle car l’anticipation on ne sait pas ce que ça va être.

2e partie du raiusonement, quel est la raison d’être d’une entreprise ? maximiser le rendement pour les actionnaires: soit j’augmente les bénéfices reversés sous forme de dividende SOIT lj’augmente le prix de l’action (prise de valur de l’entreprise) or comme on a vu le prix peut représenter n’importe quoi. SI une entreprise est immortel elle devait maximiser la valeur pr ces actionaires préent et futur ? son but de pdg est que l ‘entreprise survive le plus longtemps possible mais dans droit anglo saxon l’entreprise existe que pour l’actionnaire majoritaire aujourd’hui; et le pb si tu as un actionnaire qui a un objectif de 1 an qui peut être majoritaire et d’autres de 8 ans… quelle valeur voir ? le pris de l’action sensible à des bulles spéculative et qui est pas précis et inefficients ?

et face à tte ces contradiction les pdg ne sachant pas qui satiasfaire une entreprise est mortelle elle doit avoir une vision long terme

en 1976 michal jensen et meckling proc de finance publie un article Theory of the firm, ils évoquent le n on alignement entre les dirigeants de l’entreprise qui ne sont que des employés vs les actionnaires de l’entreprise, selon la théorie dominante les agents cherchent à maximiser leurs solution à moindre cout eg le plus de salaire en faisant le moins d’effort, donc les dirigeant vont chercher à détourner les profits de l’entreprises pour eux eg jet privé, facturation abusive, augmentation de salaire, restaurant luxueux .. mais dirigeant on un atout car il gère l’entreprie il maitrise l’information il peuvent cacher bcp de chose aux actionnaire voire leur dire que c’est pareil partout dans les autres boites. Dans le papier ils disent que la solution est que les pdg soient eux memes actionnaires

oo

![image.png](images/6-Économie/02_Corporate_Finance/27_image.png)

ooo

![image.png](images/6-Économie/02_Corporate_Finance/28_image.png)

ooo

![image.png](images/6-Économie/02_Corporate_Finance/29_image.png)

ooo

![image.png](images/6-Économie/02_Corporate_Finance/30_image.png)

ooo

![image.png](images/6-Économie/02_Corporate_Finance/31_image.png)

oo