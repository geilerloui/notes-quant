---
title: Macroéconomie
draft: true
---
# Macroéconomie

> Cette note couvre la macroéconomie au sens de l'analyse des **grandeurs agrégées** d'une économie nationale — production, revenu, emploi, prix, monnaie — et des politiques qui cherchent à agir dessus. L'introduction pose le vocabulaire (agents, flux, circuit) et surtout le clivage théorique qui structure toute la suite : l'activité est-elle déterminée par l'offre ou par la demande ? Ce clivage n'est pas une querelle d'école abstraite — il commande directement les réponses données au chômage, à l'inflation et à la dette publique dans les chapitres suivants.

## I. Introduction

### A. Objet de la macroéconomie

> [!warning] Définition
> La **macroéconomie** (macroeconomics) est l'étude du comportement d'ensemble d'une économie, décrite par un petit nombre de grandeurs agrégées (**agrégats**, aggregates) — production totale, niveau général des prix, taux de chômage, masse monétaire — et des relations qui les lient.
>
> Elle s'oppose à la **microéconomie** (microeconomics), qui étudie les décisions d'agents individuels (un consommateur, une entreprise) et le fonctionnement d'un marché particulier.

La distinction n'est pas seulement une affaire de taille. Passer du micro au macro suppose une opération d'**agrégation** (aggregation) qui n'est pas neutre : la somme des comportements individuels ne se comporte pas nécessairement comme un comportement individuel agrandi.

> [!important] Le sophisme de composition
> Le **sophisme de composition** (fallacy of composition) consiste à conclure qu'une propriété vraie pour un individu reste vraie pour l'ensemble.
>
> L'exemple canonique en macroéconomie est le **paradoxe de l'épargne** (paradox of thrift) : si *un* ménage épargne davantage, il s'enrichit. Si *tous* les ménages épargnent davantage simultanément, la consommation totale chute, donc les débouchés des entreprises, donc la production, donc les revenus — et l'épargne totale peut finir par **baisser**.
>
> C'est la raison d'être de la macroéconomie comme discipline séparée : certains mécanismes n'existent qu'au niveau agrégé, et le raisonnement « ce qui est bon pour un ménage est bon pour le pays » y est structurellement piégeux.

Deux conséquences pratiques, qui reviendront tout au long de la note :

- **Les identités comptables contraignent le raisonnement.** La dépense de l'un est le revenu de l'autre ; l'épargne des uns est nécessairement l'endettement des autres. On ne peut pas raisonner sur un agent sans tenir la contrainte de bouclage du système entier.
- **Les effets de rétroaction dominent.** Une variable macro agit rarement une fois : elle revient sur elle-même par le circuit (une baisse de salaire réduit un coût *et* un débouché). C'est exactement le point sur lequel les écoles vont diverger en I.D.

### B. Les agents : les secteurs institutionnels

La comptabilité nationale ne raisonne pas sur des individus mais sur des **secteurs institutionnels** (institutional sectors) : des regroupements d'unités qui ont le même comportement économique. Deux critères définissent un secteur — sa **fonction principale** (que produit-il ou que fait-il ?) et son **type de ressources** (d'où vient son argent ?).

| Secteur | Fonction principale | Ressource principale |
|---|---|---|
| **Ménages** (households) | Consommer ; fournir du travail | Salaires, revenus du patrimoine, prestations |
| **Sociétés non financières** (non-financial corporations) | Produire des biens et services marchands | Ventes |
| **Sociétés financières** (financial corporations) | Intermédiation financière, assurance, gestion d'actifs | Marge d'intérêt, primes, commissions |
| **Administrations publiques** (general government) | Produire des services non marchands ; redistribuer | Prélèvements obligatoires |
| **ISBLSM** | Services non marchands aux ménages (associations, syndicats) | Contributions volontaires |
| **Reste du monde** (rest of the world, RdM) | — | — |

> [!note]- Ce que veut dire « marchand »
> Un producteur est dit **marchand** (market producer) si le prix de vente couvre plus de la moitié de ses coûts de production ; **non marchand** (non-market producer) sinon. C'est un critère comptable, pas juridique : un hôpital public est non marchand, une clinique privée est marchande, une entreprise publique qui vend au prix du marché (la SNCF pour ses billets) est marchande.
>
> Attention à une confusion fréquente : une **collectivité publique qui produit un service gratuit n'est pas une « entreprise non marchande »**, elle appartient au secteur des administrations publiques. La distinction marchand/non marchand ne subdivise pas les entreprises, elle sert justement à trancher dans quel secteur classer une unité.

> [!note]- Le « ménage » n'est pas la famille
> Au sens de la comptabilité nationale, un ménage est l'ensemble des occupants d'un même logement, quels que soient leurs liens. Une personne seule est un ménage ; une colocation aussi. Les **ménages collectifs** (prison, caserne, maison de retraite, communauté religieuse) en font également partie. Un ménage peut par ailleurs produire : l'entrepreneur individuel est comptabilisé dans le secteur des ménages, pas dans celui des sociétés.

> [!note]- Le reste du monde n'est pas un secteur
> Le RdM ne regroupe pas des unités qui se ressemblent, mais **toutes les unités non résidentes**, quelle que soit leur nature, dès lors qu'elles réalisent une opération avec un résident. C'est un artefact comptable qui sert à fermer le circuit : sans lui, les importations et les exportations n'auraient pas de contrepartie.
>
> Noter aussi que le critère est la **résidence**, pas la nationalité : l'usine Toyota de Valenciennes est une unité résidente française ; la filiale de Renault au Brésil ne l'est pas. C'est cette distinction qui séparera plus loin le PIB (critère de territoire) du PNB (critère de nationalité).

### C. Le circuit économique

Les agents sont reliés par des **flux** (flows), qui vont toujours par paires : un flux réel (des biens, des services, du travail) et un flux monétaire de sens opposé qui le rémunère.

![[images/6-Économie/01_Macroéconomie/01_représentation_économie/01_image.png|441]]

**Figure 1.** Le circuit économique. Les rectangles bleus sont les agents, les ellipses vertes les marchés sur lesquels ils se rencontrent (travail, biens et services, financier). Les flèches rouges représentent les flux monétaires : les ménages vendent leur travail contre des salaires, dépensent sur le marché des biens et services, épargnent auprès du secteur financier qui prête à l'État et aux entreprises ; l'État prélève des impôts sur les ménages et les entreprises et injecte de la dépense publique. La flèche noire vers le RdM porte les exportations et les importations.

Ce circuit se lit en trois moments, qui s'enchaînent en boucle.

![[images/6-Économie/01_Macroéconomie/01_représentation_économie/03_image.png|354]]

**Figure 2.** Les trois moments du circuit. La production dégage une valeur ajoutée — c'est le PIB — qui est intégralement répartie en revenus ; ces revenus sont ensuite soit dépensés, soit épargnés et intermédiés par la sphère financière, et cette dépense forme la demande qui déclenche la production de la période suivante.

#### 1. La production

Les unités productives combinent du travail et du capital pour produire des biens et services. On ne retient pas la production brute mais la **valeur ajoutée** (value added) : ce que l'unité a réellement créé, une fois retranché ce qu'elle a consommé pour produire.

$$VA = P - CI$$

où $P$ est la production et $CI$ la **consommation intermédiaire** (intermediate consumption) — les biens et services détruits ou transformés dans le processus de production. Le PIB est la somme des valeurs ajoutées de toutes les unités résidentes. Le mécanisme, et la raison pour laquelle on ne peut pas simplement additionner les chiffres d'affaires, sont détaillés au chapitre suivant.

#### 2. La répartition

La valeur ajoutée est **intégralement** distribuée sous forme de revenus : salaires et cotisations pour le travail, excédent brut d'exploitation pour le capital, impôts pour les administrations. Ce n'est pas un résultat empirique mais une identité — la valeur ajoutée *est* la somme des revenus qu'elle finance.

On distingue :

- la **répartition primaire** (primary distribution) : le partage direct de la valeur ajoutée entre facteurs de production — c'est le fameux « partage de la valeur ajoutée » entre travail et capital ;
- la **redistribution** (secondary distribution) : la reprise par les administrations sous forme de prélèvements, et le reversement sous forme de prestations et de services non marchands.

#### 3. La dépense

Les revenus sont soit consommés, soit épargnés. L'épargne ne sort pas du circuit : elle est intermédiée par le secteur financier vers ceux qui dépensent plus que leur revenu — entreprises qui investissent, État en déficit. La somme des dépenses forme la demande adressée à l'appareil productif, ce qui referme la boucle.

![[images/6-Économie/01_Macroéconomie/01_représentation_économie/02_image.png|257]]

**Figure 3.** La boucle production → revenus → dépenses. Les trois sommets mesurent la même grandeur sous trois angles différents ; c'est cette coïncidence qui fonde les trois méthodes de calcul du PIB.

#### 4. L'équilibre ressources–emplois

Le bouclage du circuit s'écrit comme une identité comptable. Tout ce qui est disponible sur le territoire pendant la période (les **ressources**) est nécessairement utilisé (les **emplois**) :

$$\underbrace{P + IM}_{\text{Ressources (offre)}} = \underbrace{CI + C + I + G + EX + \Delta S}_{\text{Emplois (demande)}}$$

avec la notation suivante, qui sera utilisée dans toute la note :

| Symbole | Grandeur |
|---|---|
| $P$ | Production |
| $CI$ | Consommation intermédiaire |
| $C$ | Consommation finale des ménages |
| $I$ | Investissement (formation brute de capital fixe) |
| $G$ | Dépenses publiques |
| $EX,\ IM$ | Exportations, importations |
| $\Delta S$ | Variation des stocks |

En faisant passer $CI$ à gauche et en reconnaissant $P - CI = PIB$, on obtient la forme sous laquelle on utilisera l'identité par la suite :

$$\boxed{PIB = C + I + G + (EX - IM) + \Delta S}$$

> [!important] Une identité n'est pas une théorie
> Cette égalité est vraie **par construction**, ex post, quelles que soient les hypothèses de comportement. Elle ne dit strictement rien sur la causalité : elle n'affirme ni que la demande détermine la production, ni l'inverse.
>
> C'est précisément ce vide que les écoles de pensée viennent remplir. L'école libérale lit l'identité de gauche à droite (la production disponible se répartit entre emplois), l'école keynésienne la lit de droite à gauche (les dépenses anticipées commandent la production). **Le même tableau comptable est compatible avec les deux lectures** — le désaccord porte sur les équations de comportement qu'on ajoute autour, pas sur la comptabilité.

> [!note]- Ce que le circuit ne montre pas
> La représentation en circuit a trois angles morts qu'il faut garder en tête, et qui expliquent une bonne partie des débats des chapitres suivants :
>
> - **Elle est en flux, pas en stocks.** Le patrimoine, la dette accumulée, le capital installé n'y figurent pas. Or c'est souvent le stock qui bloque le flux : un ménage très endetté n'emprunte pas, même à taux nul (cf. chapitre 4, sur l'échec du QE).
> - **Elle suppose la monnaie transparente.** Dans le schéma, le secteur financier ne fait que transporter l'épargne des uns vers l'investissement des autres. Le chapitre 2 montrera que cette lecture est contestée : si les banques créent la monnaie en accordant du crédit, alors le financement ne suppose plus d'épargne préalable, et une flèche du circuit part de nulle part.
> - **Elle est agrégée.** Deux économies ayant le même PIB et la même répartition primaire peuvent avoir des dynamiques totalement différentes selon la distribution entre ménages, parce que la propension à consommer n'est pas la même en haut et en bas de la distribution.

### D. Les deux grandes écoles

Toute la macroéconomie appliquée se laisse organiser autour d'une seule question : **qu'est-ce qui détermine le niveau d'activité ?** Deux réponses s'affrontent, et elles impliquent des politiques économiques opposées.

#### 1. L'école libérale

Dominante depuis le tournant des années 1980, elle est l'héritière des classiques (Smith, Ricardo, Say) et des néoclassiques (Walras, Marshall), prolongée par le monétarisme et la nouvelle économie classique.

> [!warning] Hypothèses
> **H1 — Individualisme méthodologique.** Les phénomènes agrégés se déduisent des décisions individuelles. La société est la somme d'individus se rencontrant sur des marchés.
>
> **H2 — Rationalité des agents.** L'agent est égoïste, matérialiste (l'argent est le critère de décision), optimisateur sous contrainte, doté du libre arbitre, et traite au mieux l'information disponible.
>
> **H3 — Flexibilité des prix.** Les prix — y compris le salaire et le taux d'intérêt — s'ajustent librement jusqu'à l'équilibre.
>
> **H4 — Loi de Say.** « L'offre crée sa propre demande » : toute production distribue des revenus qui, dépensés, achètent cette production. Une insuffisance *durable* de demande globale est donc impossible.
>
> **H5 — Neutralité de la monnaie.** À long terme, la monnaie n'affecte que le niveau des prix, pas les grandeurs réelles. C'est un « voile » posé sur l'économie réelle.

Deux mécanismes gouvernent le système :

1. **La loi de l'offre et de la demande.** Si $S > D$ le prix baisse, si $S < D$ le prix monte, et l'équilibre est atteint en $S = D$. Le prix est le signal qui coordonne des agents qui ne se connaissent pas.
2. **La main invisible** (invisible hand, Smith, 1776). Chacun poursuivant son intérêt propre, le résultat collectif est efficient sans qu'aucune intention collective ne soit nécessaire.

**Application au marché du travail.** Le travail est un bien comme un autre : le salaire réel est son prix, et il s'ajuste jusqu'à égaliser offre et demande de travail. Il en découle une lecture particulière du chômage :

- **Sans intervention** : le salaire s'ajuste, le marché se vide, et le chômage résiduel est **volontaire** ou **frictionnel** (le temps de chercher un emploi). C'est le chômage « optimum ».
- **Avec intervention** : un salaire minimum place un plancher au-dessus du salaire d'équilibre, les indemnités de licenciement renchérissent le coût du travail et les indemnités de chômage réduisent l'incitation à reprendre un emploi. Le chômage devient **involontaire** — mais il est causé par les rigidités, pas par le marché.

Le remède est donc du côté de l'**offre** : baisser le coût du travail, flexibiliser, alléger la fiscalité sur le capital, discipliner le budget public — ce dernier point étant renforcé par l'argument de l'**éviction** (crowding out) : l'État qui s'endette capte une épargne rare qui aurait financé l'investissement privé.

> [!example] Illustration — la désinflation compétitive en France
> À partir du tournant de la rigueur de 1983, et pendant une bonne quinzaine d'années, la France mène une politique dite de **désinflation compétitive**, adossée à la stratégie du « franc fort » et à l'ancrage sur le mark dans le Système monétaire européen.
>
> La logique, purement du côté de l'offre :
>
> 1. On renonce à combattre directement le chômage, ce qui **stabilise les salaires** (un chômage élevé pèse sur les revendications salariales).
> 2. Salaires contenus ⇒ **hausse des prix plus faible** que chez les partenaires commerciaux.
> 3. Prix relatifs plus faibles ⇒ **gain de compétitivité-prix** ⇒ hausse des exportations ⇒ hausse de la production.
> 4. Hausse de la production ⇒ le chômage devait finir par baisser.
>
> Le résultat empirique est resté très en deçà : l'inflation a bien été jugulée et la balance commerciale s'est redressée, mais **le chômage de masse ne s'est pas résorbé** sur la période. C'est l'un des faits qui alimentent la critique keynésienne : le salaire n'a pas été traité comme ce qu'il est aussi, un débouché.

#### 2. L'école interventionniste

Dominante des années 1950 aux années 1970, elle procède de John Maynard Keynes (1883–1946) et de la *Théorie générale* (1936), écrite dans le contexte de la Grande Dépression — c'est-à-dire précisément face à une situation que la loi de Say déclarait impossible.

> [!warning] Hypothèses
> **H1 — La demande commande.** C'est la **demande effective** (effective demand) — la demande que les entreprises *anticipent* — qui détermine le niveau de production et donc d'emploi.
>
> **H2 — Rigidité des prix et des salaires à court terme.** Salaires nominaux et prix ne s'ajustent pas instantanément ; l'ajustement se fait alors par les **quantités** (on licencie, on déstocke) plutôt que par les prix.
>
> **H3 — Incertitude radicale.** L'avenir n'est pas un risque probabilisable. Les décisions d'investissement dépendent d'états de confiance (« animal spirits ») qui peuvent s'effondrer collectivement.
>
> **H4 — La monnaie n'est pas neutre.** Elle peut être désirée pour elle-même (préférence pour la liquidité), et cette thésaurisation est exactement le trou par lequel le circuit peut fuir.
>
> **H5 — Rejet de la loi de Say.** Un équilibre de **sous-emploi** durable est parfaitement possible : l'économie peut se stabiliser à un niveau d'activité où une partie de la main-d'œuvre reste involontairement inoccupée.

**Application au marché du travail.** Le renversement tient en une phrase : le salaire n'est pas seulement un coût pour l'entreprise qui le verse, c'est un revenu pour le ménage qui le reçoit — donc un débouché pour l'ensemble des entreprises.

$$w \nearrow \;\Rightarrow\; C \nearrow \;\Rightarrow\; \text{production} \nearrow \;\Rightarrow\; \text{demande de travail } L \nearrow$$

L'effet est amplifié par le **multiplicateur** (multiplier) : une dépense initiale devient le revenu de quelqu'un, qui en dépense une fraction $c$ (la propension marginale à consommer), qui devient un nouveau revenu, etc. La somme géométrique donne

$$\Delta Y = \frac{1}{1 - c}\,\Delta D$$

Avec $c = 0{,}8$, un euro de dépense publique initiale engendre cinq euros de revenu total. Le remède au chômage est donc du côté de la **demande** : relance budgétaire, soutien du pouvoir d'achat, investissement public.

#### 3. Le désaccord, point par point

| | **École libérale** | **École interventionniste** |
|---|:---:|:---:|
| Détermine l'activité | L'offre | La demande |
| Loi de Say | Acceptée | Rejetée |
| Ajustement des prix | Flexible | Rigide à court terme |
| Variable d'ajustement | Les prix | Les quantités |
| Nature du salaire | Un **coût** à contenir | Un **revenu** à soutenir |
| Nature du chômage | Volontaire / dû aux rigidités | Involontaire / dû à la demande |
| Monnaie | Neutre (à long terme) | Non neutre |
| Épargne | Finance l'investissement | Peut fuir le circuit |
| Rôle de l'État | Minimal (effet d'éviction) | Actif (effet multiplicateur) |
| Remède au chômage | Baisser le coût du travail | Soutenir la demande |
| Période de domination | 1980 → aujourd'hui | 1945 → 1975 |

> [!note]- Le clivage est un point de départ, pas une carte complète
> L'opposition en deux camps est commode pour entrer dans la matière, mais elle écrase des positions qui ne s'y rangent pas :
>
> - **Le monétarisme** (Friedman) est libéral sur le fond mais accorde à la monnaie un rôle central à court terme : « l'inflation est toujours et partout un phénomène monétaire ». Ses hypothèses seront testées et discutées au chapitre 2.
> - **La nouvelle économie classique** (Lucas) radicalise l'individualisme méthodologique avec les anticipations rationnelles, jusqu'à conclure à l'inefficacité complète des politiques anticipées.
> - **Les nouveaux keynésiens** font le chemin inverse : ils acceptent les micro-fondations et la rationalité, mais montrent que des rigidités réalistes suffisent à produire des équilibres de sous-emploi. C'est la base des modèles utilisés aujourd'hui par les banques centrales — dont le chapitre 4 discutera les limites.
> - **Les post-keynésiens** insistent sur la monnaie endogène : ce sont les crédits qui font les dépôts. C'est la théorie T3 du chapitre 2.
> - **L'ordolibéralisme** allemand, libéral mais non laissez-faire, tient que le marché ne fonctionne que dans un cadre de règles fortes garanties par l'État. Sa trace est partout dans les traités européens et dans la gestion de la crise de l'euro.
>
> Un même auteur peut par ailleurs relever de plusieurs cases selon la question traitée, et les positions se déplacent au fil des crises : 2008 a nettement réhabilité l'interventionnisme, y compris chez ceux qui le contestaient.

### E. Les thèmes de la macroéconomie

Les questions qui structurent la suite de la note, et pour chacune le clivage qui la traverse :

| Thème | La question | Où |
|---|---|---|
| Mesure de l'activité | Que mesure exactement le PIB, et que rate-t-il ? | Chapitre 1 |
| Croissance et cycles | La croissance est-elle tendancielle ou cyclique ? | Chapitre 1 |
| Monnaie et banques | Qui crée la monnaie, et sous quelle contrainte ? | Chapitre 2 |
| Inflation | Phénomène monétaire, de coûts, ou rapport de force ? | Chapitre 3 |
| Politique monétaire | Une banque centrale peut-elle piloter l'inflation ? | Chapitre 4 |
| Politique budgétaire | Relance ou éviction ? | Chapitre 4 |
| Chômage | Baisser les salaires ou les augmenter ? | Note dédiée |
| Dette publique, retraites, fiscalité | Soutenabilité et répartition | À venir |
