---
title: Évaluer les qualités mécaniques d'un athlète
---
# Évaluer les qualités mécaniques d'un athlète

> Le chapitre 4 a donné les lois du muscle, le chapitre 5 la méthode pour remonter aux forces à partir du mouvement. Celui-ci les applique à quelqu'un de réel — mais en gardant seulement ce qui change quelque chose : **les faits que les mesures ont révélés**, pas les protocoles de laboratoire qui les produisent.

![[Pasted image 20260728161317.png|420]]
*Trois propriétés suffisent à caractériser mécaniquement un système musculo-tendineux : la **force**, la **raideur** et la **viscosité**.*

> Source : MOOC *Le mouvement humain*, Université de Nantes — semaine 2, sections D et E.
> Précédents : *[[04_Mécanique musculaire (v2)|Mécanique musculaire]]*, *[[05_Biomécanique du mouvement (v2)|Biomécanique du mouvement]]*.

---

## 1. À l'échelle de l'athlète, la relation devient linéaire

Au niveau du muscle isolé, la relation force-vitesse est une **hyperbole** — c'est le résultat du chapitre 4.

![[Pasted image 20260728160251.png|400]]
*Rappel : hyperbole de Hill, bornée par **F max** à vitesse nulle et **V max** à force nulle.*

Mais dès qu'on mesure un **mouvement entier**, la courbe change de nature.

![[Pasted image 20260728160425.png|560]]
*Saut vertical. La relation force-vitesse (axe de gauche) est une **droite**. Elle définit **F₀**, force maximale théorique extrapolée à vitesse nulle, et **V₀**, vitesse maximale théorique extrapolée à force nulle. Remarquer que les points mesurés ne couvrent que les vitesses basses : le reste de la droite est extrapolé.*

> [!warning] La linéarité n'est pas une propriété du muscle
> C'est une propriété **du système**. Elle émerge de l'assemblage : plusieurs articulations en chaîne, des agonistes et des antagonistes qui se contrarient, des bras de levier qui changent au cours du mouvement, des masses segmentaires à déplacer. Et la « vitesse » en abscisse n'est plus celle d'une fibre, mais celle du centre de masse.
>
> Le muscle reste hyperbolique. C'est l'athlète qui est linéaire.

![[Pasted image 20260728160449.png|660]]
*Quatre mouvements, quatre équipes, quatre publications : **squat chargé**, **presse à cuisses**, **sprint en course à pied**, **pédalage**. Dans tous les cas, une droite.*

> 💡 **La puissance est maximale à charge intermédiaire.** Puisque $P = F \times V$ et que la force décroît linéairement, la puissance décrit une parabole : nulle à vitesse nulle (de la force, mais rien ne bouge), nulle à V₀ (ça bouge, mais plus de force), maximale entre les deux. Cette vitesse s'appelle **V_opt**.
>
> C'est pour ça que le travail de puissance se fait à **charges modérées**, et non avec la barre la plus lourde possible.
>
> *Réserve : l'optimum tombe exactement à V₀/2 uniquement parce qu'on a modélisé la relation par une droite. Avec la vraie hyperbole de Hill, il se situe plutôt vers 30 % de V_max. La symétrie est un artefact du modèle.*

> [!warning] F₀ et V₀ ne sont pas mesurés
> Ce sont des **extrapolations**, obtenues en prolongeant une droite tracée à partir de quelques points tous regroupés dans les vitesses basses. Utilisable pour comparer ou suivre une progression ; beaucoup moins pour affirmer « ta vitesse maximale théorique est de 3,0 m/s ».

---

## 2. Mesurer des distances pour en déduire des forces

On ne mesure aucune force. On mesure **une masse et deux longueurs**, et tout le reste se calcule.

![[Pasted image 20260728160529.png|620]]
*Les trois hauteurs du saut. **h_S** = hauteur du centre de masse au départ. **h_PO** = distance de poussée, l'amplitude sur laquelle la force est appliquée. **h** = hauteur de saut, l'élévation après décollage.*

| Symbole | Ce que c'est | Comment on l'obtient |
|---|---|---|
| $m$ | Masse totale — corps **+ charge** éventuelle | Balance |
| $h_{PO}$ | Distance de poussée | Position de départ et longueur des membres inférieurs |
| $h$ | Hauteur de saut | Élévation du centre de masse après décollage |

### La vitesse

Après le décollage, plus rien ne pousse : seule la gravité agit. Toute l'énergie cinétique verticale se convertit donc en hauteur, ce qui donne la vitesse au décollage :

$$\tfrac{1}{2} m v_{TO}^2 = mgh \quad \Longrightarrow \quad v_{TO} = \sqrt{2gh}$$

En supposant l'accélération à peu près constante pendant la poussée, la vitesse moyenne vaut la moitié de celle du décollage :

$$\bar{v} = \frac{v_{TO}}{2} = \frac{\sqrt{2gh}}{2}$$

### La force

Pendant la poussée, la force du sol te propulse et le poids te retient. Le travail de la résultante sur la distance $h_{PO}$ est égal à l'énergie cinétique acquise :

$$(\bar{F} - mg)\,h_{PO} = \tfrac{1}{2} m v_{TO}^2 = mgh$$

D'où, en isolant :

$$\boxed{\;\bar{F} = mg\left(1 + \frac{h}{h_{PO}}\right)\;}$$

> 💡 **Ce que dit cette formule.** La force moyenne ne dépend que d'un **rapport de longueurs** : ta hauteur de saut divisée par ta distance de poussée. Sauter aussi haut que la distance sur laquelle tu pousses ($h = h_{PO}$) revient exactement à produire **deux fois ton poids de corps**.

### La puissance

$$\bar{P} = \bar{F} \times \bar{v}$$

### Exemple chiffré

Pour $m = 80$ kg, $h_{PO} = 0{,}40$ m et $h = 0{,}40$ m :

$$\bar{F} = 80 \times 9{,}81 \times (1 + 1) = 1570\ \text{N}$$
$$v_{TO} = \sqrt{2 \times 9{,}81 \times 0{,}40} = 2{,}80\ \text{m/s} \quad \Longrightarrow \quad \bar{v} = 1{,}40\ \text{m/s}$$
$$\bar{P} = 1570 \times 1{,}40 \approx 2200\ \text{W}$$

Debout immobile, tu appuierais sur le sol avec 785 N. La poussée du saut en produit le double.

> [!warning] De quoi cette force est-elle la force ?
> C'est la **force verticale totale appliquée contre le sol** — ce que lirait une plateforme de force. Pas la force d'un muscle : la **résultante de toute la chaîne**, extenseurs de hanche, de genou et de cheville, diminuée de ce que freinent les antagonistes, transmise jusqu'au sol.
>
> Et c'est une **moyenne sur toute la poussée**, pas un pic. La force instantanée varie beaucoup pendant le mouvement.

### La version téléphone

On peut même se passer de mesurer $h$. Il suffit de filmer le saut au ralenti et de relever le **temps de vol** $t$ entre le décollage et la réception. La montée et la descente étant symétriques, la montée dure $t/2$ :

$$h = \tfrac{1}{2}\,g\left(\frac{t}{2}\right)^2 = \frac{g\,t^2}{8}$$

Un demi-seconde de vol donne 31 cm. Il ne reste plus qu'à injecter $h$ dans les formules précédentes.

### La logique d'ensemble

> [!warning] Un saut = un point, pas une courbe
> Un seul saut donne **un couple** $(\bar{F}, \bar{v})$ — un point. Il ne permet de séparer ni la force ni la vitesse.
>
> La courbe s'obtient en **variant la charge**, exactement comme dans l'expérience sur muscle isolé du chapitre 4 : on impose une charge, on lit la vitesse qui en résulte.
>
> - Saut à vide → vitesse élevée, force faible → point à droite
> - Saut chargé → vitesse moindre, force plus élevée → point à gauche
>
> Six charges, six points, et la droite passe par eux. **Le « profil » d'un athlète n'est rien d'autre que la pente de cette droite** : pentue = il tient la force mais perd vite de la vitesse quand on le charge ; plate = l'inverse.

### Les hypothèses derrière tout ça

- **Accélération constante** pendant la poussée — c'est ce qui autorise $\bar{v} = v_{TO}/2$. Approximation raisonnable, pas une vérité.
- **$h_{PO}$ doit être mesurée correctement**, et elle entre directement dans la force. Une erreur de 5 cm sur la distance de poussée déplace le résultat de plus de 10 %.
- **Aucune vitesse horizontale**, aucun mouvement de bras parasite : le modèle suppose un saut purement vertical.
- **$F_0$ et $V_0$ sont extrapolés** aux deux bouts d'un segment tracé à partir de points tous regroupés dans les vitesses basses.

## 3. Raideur et viscosité

Les deux dernières propriétés concernent l'élastique, pas le moteur.

> [!warning] Deux définitions à ne pas confondre
> **Raideur** — capacité de la structure à **stocker** de l'énergie élastique lorsqu'elle est étirée, puis à la **restituer** pour produire du mouvement.
>
> **Viscosité** — **dissipation** d'énergie lors de la restitution, perdue en chaleur. Muscle et tendon sont incapables de restituer 100 % de l'énergie stockée : ce qui manque a été mangé par la viscosité.

![[Pasted image 20260728161447.png|560]]
*Le mécanisme en trois temps. **Au repos**, rien n'est tendu. **À la stimulation**, le générateur de force se raccourcit (trait bleu) et **étire les structures élastiques** (trait rouge) : l'énergie est stockée. **À la restitution**, elle est rendue et l'ensemble raccourcit — c'est elle qui produit le mouvement.*

> 💡 **Le point contre-intuitif.** Pendant la phase de stockage, le générateur de force se raccourcit alors que **l'ensemble muscle-tendon ne bouge presque pas** : tout le raccourcissement des ponts part dans l'étirement du tendon. La force au bout du segment ne vient donc pas seulement des ponts actine-myosine — elle vient de l'élastique qu'ils ont tendu.

---

## Ce que ce chapitre garde

1. **À l'échelle de l'athlète, la relation force-vitesse est une droite** — et ses deux extrémités sont extrapolées, pas mesurées.
2. **On est le plus puissant à mi-chemin**, ni en force max ni en vitesse max.
3. **Mesurer des distances suffit à déduire des forces** — y compris avec un téléphone.

*Ont été retirés de ce chapitre : la fiche de test INSEP et sa lecture normative, les protocoles isométriques et isocinétiques sur ergomètre, le taux de montée en force (RFD) et l'élastographie. Méthodes de laboratoire ou notions purement descriptives — rien qui change une décision d'entraînement.*
