---
title: Électromyographie (EMG)
---
# Électromyographie (EMG)

> Les deux chapitres précédents décrivent une chaîne : **électrique → chimique → mécanique**. L'EMG vient brancher un capteur sur le **premier maillon**, et sur lui seul.
>
> C'est tout l'intérêt de la technique, et toute sa limite : elle observe la **commande** envoyée au muscle, pas la force qu'il produit.

> Source : MOOC *Le mouvement humain*, Université de Nantes — semaine 1.
> Précédents : *[[01_Couplage excitation-contraction (v2)|Couplage excitation-contraction]]*, *[[02_Théorie de l'unité motrice (v2)|Théorie de l'unité motrice]]*.

---

## Ce que l'EMG mesure vraiment

Quand une unité motrice décharge, un potentiel d'action parcourt la membrane de chacune de ses fibres. Ce courant se propage aussi dans les tissus environnants — jusqu'à la peau, où il devient détectable.

> [!warning] Définition
> L'**électromyographie** enregistre la **somme des potentiels d'action** des unités motrices situées sous l'électrode.
>
> Elle mesure donc une **activation**, pas une force, pas une tension, pas un raccourcissement.

![[Pasted image 20260728145307.png|640]]
*Chaque couleur sur le ventre du muscle est une unité motrice différente, avec sa propre bouffée de potentiels — et elles ne sont pas synchronisées entre elles. La bulle translucide autour des électrodes est le **volume de détection** : l'électrode ne « voit » que ce qui s'y trouve, pas le muscle entier.*

**Deux façons de poser le capteur.** L'EMG **intramusculaire** utilise des aiguilles ou des fils implantés dans le muscle : précis, mais invasif, donc peu utilisé. L'EMG **de surface** utilise des électrodes collées sur la peau — c'est la méthode standard.

**Le protocole de pose, en quatre temps :**

1. Repérer le muscle et ses contours par **palpation**.
2. **Préparer la peau** (l'impédance de la peau dégrade le signal).
3. Poser l'électrode sur le ventre du muscle, **alignée dans l'axe des fibres**.
4. Vérifier la **qualité du signal** par quelques contractions test.

---

## Lire un tracé

Le principe de lecture est simple : ligne plate = muscle silencieux, bouffée dense = muscle actif. Plus la contraction est intense, plus le signal est ample.

> 💡 **Pourquoi le signal grossit — le lien avec le chapitre précédent.** Les deux leviers de la force augmentent tous les deux la quantité de potentiels d'action captés : le **spatial** ajoute des unités motrices, le **temporel** les fait décharger plus souvent. L'EMG monte dans les deux cas. C'est aussi pourquoi il ne permet pas de distinguer lequel des deux leviers est à l'œuvre.

![[Pasted image 20260728145608.png|660]]
*Course à pied, sept foulées. En haut le **vaste latéral** : une bouffée brève et discrète, calée sur le contact talon. Au milieu le **gastrocnémien** : une bouffée bien plus longue et bien plus ample, immédiatement après, sur toute la phase d'appui. En bas le signal de contact au sol, qui sert de repère temporel. Entre deux appuis, les deux muscles se taisent.*

Le tracé se lit comme une répartition des rôles dans le temps :

- **Vaste latéral** → actif au moment de la **pose du pied** : il freine, il amortit la réception.
- **Gastrocnémien** → actif juste après, pendant tout l'appui : il **propulse**.

> 💡 **Ce que l'EMG apporte ici.** Pas une valeur de force, mais une **datation**. Savoir *quel muscle travaille, à quel instant du geste, et à quelle intensité relative*. C'est ce qui en fait un outil d'analyse du mouvement — en performance comme en rééducation.

Les deux tracés sont exprimés en **pourcentage de l'activation maximale**, et non en volts. Sans cette normalisation, comparer deux muscles entre eux n'aurait aucun sens.

---

## Ce que l'EMG ne dit pas

C'est la partie utile à retenir, parce qu'on croise beaucoup d'arguments qui s'appuient sur des EMG mal lus.

> [!warning] Quatre limites qui invalident la plupart des comparaisons hâtives
> - **L'amplitude n'est pas la force.** Le lien entre les deux existe, mais il n'est ni linéaire ni stable — il dépend du muscle, de l'angle articulaire et du type de contraction.
> - **Le signal dépend du montage.** Position exacte de l'électrode, préparation de la peau, épaisseur de tissu adipeux : comparer deux personnes, ou deux séances, sans normaliser ne veut rien dire.
> - **Crosstalk.** L'électrode capte aussi les muscles voisins. Sur des muscles superficiels et proches, une partie du signal attribué à l'un vient de l'autre.
> - **La fatigue déforme le signal.** À force constante, l'amplitude EMG **augmente** avec la fatigue, parce qu'il faut recruter davantage pour maintenir le même niveau. Une hausse du signal ne signifie donc pas forcément une hausse de la force.

> 💡 **En pratique.** Quand une vidéo affirme qu'un exercice « active 30 % de plus » tel muscle qu'un autre, la question à se poser n'est pas le chiffre mais le protocole : normalisé comment, sur combien de sujets, avec quelle position d'électrode, et à charge réellement équivalente ? L'EMG est un bon outil de **chronologie** et un mauvais outil de **classement d'exercices**.
