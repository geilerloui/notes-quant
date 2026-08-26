---
title: Couplage excitation-contraction
---
# Couplage excitation-contraction

> Un muscle reçoit un signal **électrique** et produit une force **mécanique**. Ce sont deux mondes physiques différents : des millivolts d'un côté, des newtons de l'autre. Rien ne dit *a priori* comment on passe de l'un à l'autre — et c'est précisément le sujet de ce chapitre.

> Source : MOOC *Le mouvement humain*, Université de Nantes — semaine 1.
> Suite : *Théorie de l'unité motrice*, *Électromyographie*, puis *Mécanique musculaire* (semaine 2).

---

## Ce que veut dire le titre

Le nom est opaque tant qu'on ne le découpe pas.

> [!warning] « Couplage excitation-contraction », mot à mot
> - **Excitation** — le signal électrique. Le potentiel d'action.
> - **Contraction** — le raccourcissement mécanique du muscle.
> - **Couplage** — le mécanisme qui convertit le premier en le second.
>
> Le titre ne désigne donc pas un objet, mais **une traduction** : comment un courant devient une force.

> 💡 **La question de départ, sous forme expérimentale.** On colle deux électrodes sur un mollet, on envoie du courant, et le muscle se contracte — sans aucune volonté, sans passer par le cerveau. Le muscle obéit donc à de l'électricité. **Par quel enchaînement de mécanismes ?**

![[Pasted image 20260728142000.png|280]]
*L'électrostimulation prouve que le déclencheur ultime de la contraction est électrique. Tout le chapitre consiste à ouvrir cette boîte noire.*

> 💡 **La réponse tient en cinq relais.** Chacun résout un problème que le précédent a créé. C'est le fil du chapitre.

| # | Problème à résoudre | Solution biologique |
|---|---|---|
| 1 | Le nerf ne touche pas le muscle | Transmission chimique (acétylcholine) |
| 2 | Le signal reste en surface de la fibre | Les tubules T le conduisent en profondeur |
| 3 | Le signal est électrique, pas chimique | Le réticulum sarcoplasmique libère du Ca²⁺ |
| 4 | Les sites de fixation sont masqués | Le Ca²⁺ déplace la troponine-tropomyosine |
| 5 | Il faut convertir le chimique en mouvement | Le cycle des ponts, alimenté par l'ATP |

> 💡 **La chaîne en une phrase.** Le SNC envoie un potentiel d'action → l'acétylcholine franchit la fente et **régénère** un potentiel d'action sur le muscle **si le seuil est atteint** → le tubule T le porte **en profondeur** → le réticulum sarcoplasmique **libère son calcium** → le calcium **démasque** les sites de l'actine → la myosine **tire**, avec l'ATP → les filaments **glissent** et le muscle produit une force.

---

## Le potentiel d'action, en deux mots

Le terme revient à chaque ligne du chapitre : autant le définir avant.

La membrane d'une cellule excitable — neurone ou fibre musculaire — maintient en permanence une **différence de voltage** entre son intérieur et son extérieur : environ **−70 mV** au repos. Elle est chargée comme une petite pile, parce que les ions ne sont pas répartis également des deux côtés.

Un **potentiel d'action** est un **renversement bref et brutal de ce voltage**. Si un stimulus dépolarise la membrane jusqu'à un **seuil** d'environ −55 mV, des canaux s'ouvrent en cascade, le voltage bascule jusqu'à +40 mV, puis revient au repos. L'ensemble dure **2 à 3 millisecondes**.

![[fig-potentiel-action.svg|680]]
*Deux stimulations. La première ne fait pas monter le voltage jusqu'au seuil : rien ne part. La seconde l'atteint, et le potentiel d'action se déclenche intégralement.*

> [!warning] Ce n'est pas une différence entre deux cellules
> L'erreur naturelle est de croire que le voltage se mesure **entre** le nerf et le muscle. Non : il se mesure **entre l'intérieur et l'extérieur d'une seule et même cellule**.
>
> Concrètement : une micro-électrode **dans** la fibre, une autre **dans le liquide juste à côté**. On lit −70 mV. Le motoneurone a le sien, la fibre musculaire a le sien — deux mesures indépendantes, chacune sur sa propre membrane. Et dans la fente synaptique, il n'y a **aucune électricité** : ce qui la traverse est une molécule.
>
> Autre distinction : le −70 mV permanent est le **potentiel de repos**, l'état de base. Le **potentiel d'action** est l'**événement** — un basculement bref et local de cette différence, qui se propage ensuite le long de la membrane.

> [!warning] Trois propriétés à retenir
> - **Tout ou rien.** Sous le seuil, rien. Au seuil, le signal part — et toujours avec la même amplitude. Il n'existe pas de potentiel d'action « plus fort ».
> - **Il se régénère de proche en proche.** Ce n'est pas du courant qui circule dans un fil, c'est une rangée de dominos : chaque portion de membrane déclenche la suivante. D'où le fait qu'il **ne s'atténue jamais avec la distance**.
> - **Il est suivi d'une période réfractaire**, pendant laquelle la membrane ne peut pas repartir. C'est ce qui plafonne la fréquence maximale de décharge.

> 💡 **Le même événement, deux effets.** Ce que déclenche un potentiel d'action dépend uniquement de l'endroit où il se produit :

| Où il bascule | Ce qu'il déclenche |
|---|---|
| Au bout du motoneurone | Ouverture de canaux calciques → libération d'acétylcholine dans la fente |
| Sur la membrane du muscle, puis dans les tubules T | Ouverture du réticulum sarcoplasmique → libération du calcium sur les protéines contractiles |

---

## 1. Franchir le vide entre le nerf et le muscle

La commande volontaire part du cerveau, transite par la moelle épinière, et emprunte le **motoneurone** jusqu'au muscle. Le contact final s'appelle la **jonction neuromusculaire**.

![[Pasted image 20260728142201.png|560]]
*Le trajet complet, et le zoom qui révèle le problème : entre le bouton terminal du motoneurone et la membrane du muscle, il y a un espace. Les deux tracés encadrés, ① et ②, ne sont **pas** les deux bornes d'une même mesure — ce sont **deux mesures indépendantes** : le voltage de la membrane du neurone d'un côté, celui de la membrane du muscle de l'autre.*

> [!warning] Le point contre-intuitif
> **Il n'y a aucune continuité physique entre le nerf et le muscle.** Un espace les sépare : la **fente synaptique**. C'est une synapse chimique.
>
> Conséquence : le signal électrique **ne traverse pas**. Il est converti en signal chimique, puis **régénéré** de l'autre côté. Les deux potentiels d'action ① et ② du schéma ne sont pas le même signal qui continue son chemin — c'est un signal qui meurt et un autre qui naît.

**Ce qui se passe dans la fente.** Le motoneurone libère de l'**acétylcholine**, qui traverse et dépolarise la membrane du muscle. **Si cette dépolarisation atteint le seuil (≈ −55 mV), un nouveau potentiel d'action naît dans le muscle.** Sinon, rien ne part.

![[Pasted image 20260728142702.png|620]]
*À gauche le mécanisme moléculaire, à droite sa traduction électrique. Voir la ligne « Seuil » : en dessous, rien ne se passe ; au-dessus, le potentiel d'action part et il est intégral. C'est un interrupteur, pas un variateur.*

> [!warning] Erreur fréquente sur le seuil
> Le seuil n'est **pas** une condition pour « traverser la fente ». L'acétylcholine traverse toujours.
>
> Le seuil est la condition pour qu'un **nouveau potentiel d'action naisse du côté musculaire**. La question n'est pas « le message passe-t-il ? » mais « le message est-il assez fort pour être réécrit de l'autre côté ? »

> 💡 **Loi du tout ou rien.** Un potentiel d'action ne se produit pas « à moitié ». Sous le seuil : rien. Au seuil : le PA part, toujours de la même amplitude. L'intensité de la contraction ne se code donc **pas** dans la taille du signal — elle se code dans le **nombre** d'unités motrices recrutées et dans la **fréquence** des signaux. → voir *Théorie de l'unité motrice*.

> [!note]- Le détail moléculaire, si un jour tu le veux
> 1. Le PA arrive au bouton terminal → des canaux calciques s'ouvrent → entrée de Ca²⁺.
> 2. Ce Ca²⁺ déclenche l'exocytose des vésicules d'acétylcholine.
> 3. L'acétylcholine se fixe sur les récepteurs cholinergiques du sarcolemme.
> 4. Cette fixation ouvre des canaux → entrée de Na⁺ → dépolarisation locale, le **PPSE**.
> 5. Si le PPSE atteint le seuil, le potentiel d'action musculaire part.

---

## 2. Descendre au cœur de la fibre

Le potentiel d'action se propage désormais le long du **sarcolemme**, la membrane de la fibre. Mais la machinerie contractile, elle, occupe tout le **volume** de la fibre.

![[Pasted image 20260728142941.png|420]]
*L'intérieur d'une fibre musculaire : sarcolemme en surface, myofibrilles en profondeur, mitochondries entre les deux. Le signal est en surface, le moteur est au centre — c'est là qu'il faut le porter.*

> 💡 **Pourquoi ça compte.** Sans solution à ce problème, seule la périphérie de la fibre se contracterait, le centre resterait inerte, et la contraction serait à la fois faible et désynchronisée.

**La solution est géométrique.** Le sarcolemme **s'invagine** : il plonge dans la fibre en formant des tunnels perpendiculaires, les **tubules T** (transverses).

![[Pasted image 20260728143012.png|620]]
*Le tubule T, en jaune, descend depuis la surface et vient au contact des **citernes terminales** du **réticulum sarcoplasmique** (en bleu-gris), qui enveloppe les filaments d'actine et de myosine.*

> [!warning] Le tubule T ne fabrique rien
> Il ne produit ni calcium, ni force. **Il ne transporte qu'un signal électrique.** C'est un câble, pas un générateur.
>
> Son seul rôle : amener la dépolarisation au contact du réticulum sarcoplasmique, jusqu'au cœur de la fibre.

---

## 3. Convertir l'électrique en chimique

Le **réticulum sarcoplasmique** (RS) est un réseau de tubules qui entoure les protéines contractiles. Il est **rempli de Ca²⁺**, stocké là en permanence. Quand la dépolarisation atteint le tubule T, les canaux du RS s'ouvrent et le calcium se déverse directement sur les protéines contractiles.

![[Pasted image 20260728144002.png|620]]
*Même coupe que précédemment, un instant plus tard : le RS relâche son calcium dans le cytoplasme. C'est le moment exact où le signal change de nature — d'électrique il devient chimique.*

> [!warning] Le stock est fini, et il est recyclé
> Le RS ne **crée** pas de calcium, il le **libère** — puis il le **repompe** (via la pompe SERCA) pour arrêter la contraction.
>
> Cette nuance n'est pas cosmétique : c'est elle qui explique tout le mécanisme des **fréquences de stimulation**. Si le signal suivant arrive avant que le calcium soit rentré, la concentration cytosolique ne redescend pas, elle s'accumule, et la force monte. → voir [[Sci sport - Biomécanique du fitness]] et la sommation / le tétanos dans *Théorie de l'unité motrice*.

> 💡 **Proportionnalité.** Plus la stimulation est forte, plus la quantité de Ca²⁺ libérée est élevée. Le calcium est donc bien le **messager d'intensité** à l'intérieur de la fibre.

---

## 4. Déverrouiller les sites de fixation

![[Pasted image 20260728144218.png|440]]
*Le **sarcomère** : la plus petite unité contractile du muscle. Deux **disques Z** l'encadrent, les filaments fins d'**actine** y sont ancrés, les filaments épais de **myosine** occupent le centre (ligne M, zone H).*

**Le verrou.** Au repos, la myosine ne peut pas se fixer sur l'actine, même si les deux sont côte à côte : les sites de fixation sont **physiquement masqués** par le complexe **troponine-tropomyosine**. Le calcium se fixe sur la troponine → le complexe pivote → **les sites sont démasqués** → la tête de myosine peut enfin s'accrocher.

![[Pasted image 20260728144158.png|620]]
*Avant / après. À gauche : la tropomyosine bloque, la tête de myosine reste à distance malgré son ADP+Pi. À droite : le calcium est arrivé, le complexe a pivoté, la fixation devient possible.*

> [!warning] Le calcium n'est pas le moteur
> C'est l'erreur la plus répandue du chapitre. Le calcium **ne fait pas glisser les filaments**. Il **enlève le verrou**.
>
> - **Calcium = l'autorisation.** Il ouvre la serrure.
> - **ATP = le carburant.** C'est lui qui fournit l'énergie du mouvement.
>
> Les deux sont nécessaires, et ils ne font pas le même travail.

---

## 5. Convertir le chimique en mécanique

![[Pasted image 20260728144440.png|640]]
*Le cycle en quatre temps. ① ATP fixé sur la tête de myosine. ② ATP hydrolysé, la tête se fixe sur l'actine. ③ Libération du Pi → **temps moteur** : la tête pivote et tire le filament. ④ Libération de l'ADP, puis un **nouvel ATP** détache la tête.*

> [!warning] Ce que fait vraiment l'ATP — souvent mal retenu
> L'ATP **ne sert pas à tirer**.
>
> - Son **hydrolyse arme** la tête de myosine, comme on arme un ressort.
> - C'est la **libération du phosphate (Pi)** qui déclenche le pivotement, donc le mouvement.
> - Et il faut une **nouvelle molécule d'ATP pour détacher** la tête de l'actine.
>
> Conséquence directe : **plus d'ATP = plus de détachement possible**. Les têtes de myosine restent accrochées et le muscle se raidit. C'est littéralement la **rigidité cadavérique**.

> 💡 **Le cycle est asynchrone.** Tant que le calcium et l'ATP sont présents, le cycle se répète — mais **tous les ponts ne se forment ni ne se détachent en même temps**. À chaque instant, une partie tient pendant qu'une autre se réarme. Sans cela, la contraction serait saccadée et le muscle lâcherait la charge à chaque cycle. C'est ce qui rend le mouvement **lisse et continu**.

---

## 6. Comment la contraction s'arrête

Un potentiel d'action ne produit pas un **état**, il produit un **événement** : une **secousse**. Le muscle se contracte, puis se relâche de lui-même — sans qu'aucun ordre d'arrêt ne soit envoyé.

> [!warning] Se relâcher est un travail actif
> La pompe **SERCA** renvoie le calcium dans le réticulum sarcoplasmique. Le calcium quitte alors la troponine, le complexe troponine-tropomyosine revient **masquer** les sites de l'actine, plus aucun nouveau pont ne peut se former, et ceux qui tiennent encore se détachent.
>
> Et ce repompage **consomme de l'ATP** — le calcium est renvoyé contre son gradient. Se décontracter n'est pas « lâcher », c'est ranger, et ça coûte cher.

**Le fait qui rend tout le reste intelligible : les deux événements n'ont pas la même durée.**

| Événement | Ordre de grandeur |
|---|---|
| Le potentiel d'action | quelques **millisecondes** |
| La secousse mécanique qu'il déclenche | **50 à 200 ms** selon le type de fibre |

![[fig-pa-vs-secousse.svg|680]]
*Les deux tracés partagent le même axe de temps. Le pic électrique est fini depuis longtemps quand la contraction qu'il a déclenchée atteint seulement son maximum.*

> 💡 **Le signal électrique est bien plus court que sa conséquence mécanique.** Le temps qu'une secousse monte et retombe, il y a largement la place d'envoyer dix autres potentiels d'action.
>
> D'où la conséquence directe : si le signal suivant arrive **avant que la pompe ait fini son travail**, le calcium cytosolique n'est jamais redescendu. Il s'accumule, plus de sites restent démasqués, plus de ponts tiennent — et les secousses **s'empilent au lieu de se succéder**.
>
> C'est la **sommation**, et c'est ce qui fait de la fréquence de décharge un levier de force à part entière.

![[fig-sommation-tetanos.svg|760]]
*Trois régimes sur un même tracé. **À gauche**, les stimulations sont espacées : chaque secousse retombe complètement, le calcium a le temps d'être repompé. **Au centre**, elles se rapprochent : les secousses repartent d'un niveau plus haut et la force monte en escalier. **À droite**, elles s'enchaînent trop vite pour que le calcium redescende : les secousses fusionnent en un plateau lisse — le tétanos.*

> 💡 **Remarque ce que la figure ne montre pas.** Les traits de stimulation, en haut, ont tous exactement la même hauteur. Rien n'est envoyé « plus fort » — seule leur cadence change. La force est pourtant multipliée par quatre. → développé dans *[[02_Théorie de l'unité motrice (v2)|Théorie de l'unité motrice]]*.

> 💡 **Une conséquence en aval.** Les fibres rapides ont une secousse **plus brève**. Il faut donc les stimuler à plus haute fréquence pour empiler leurs secousses — c'est pourquoi elles fusionnent en tétanos plus tard que les lentes.

---

## 7. Du sarcomère au muscle entier

Les filaments **ne raccourcissent pas**. Ils **glissent** les uns sur les autres. Le sarcomère raccourcit, ses filaments gardent leur longueur.

![[Pasted image 20260728144614.png|640]]
*Trois états croissants. À gauche le sarcomère, au centre le muscle entier, à droite la force produite. Plus les filaments se chevauchent, plus le muscle est court, et plus la force est élevée.*

> [!warning] La phrase qui explique le reste du cours
> **La force produite = le nombre de ponts actine-myosine formés à cet instant.**
>
> Toute la mécanique musculaire de la semaine 2 découle de cette seule idée :
> - **Relation force-longueur** — il existe une longueur optimale, celle où le chevauchement permet le plus grand nombre de ponts. Trop étiré ou trop raccourci, il y en a moins, donc moins de force.
> - **Relation force-vitesse** — plus la vitesse de raccourcissement est élevée, moins il y a de ponts attachés à un instant donné, donc moins de force.

**À l'échelle du muscle**, le raccourcissement se cumule dans les deux directions : les sarcomères sont montés **en série** (leurs raccourcissements s'additionnent en longueur) et **en parallèle** (leurs forces s'additionnent en tension). Le résultat macroscopique est une tension aux extrémités du muscle, qui produit soit un mouvement, soit le maintien d'une posture.
