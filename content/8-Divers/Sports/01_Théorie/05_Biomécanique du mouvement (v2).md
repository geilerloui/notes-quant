---
title: Biomécanique du mouvement
---
# Biomécanique du mouvement

> Les chapitres précédents restaient à l'intérieur du muscle. Celui-ci en sort et regarde le corps entier bouger dans l'espace.
>
> La biomécanique pose deux questions distinctes. **Décrire** le mouvement — où est le corps, à quelle vitesse, avec quelle accélération : c'est la **cinématique**. **Expliquer** le mouvement — quelles forces l'ont produit : c'est la **dynamique**.

> 💡 **Et il existe un pont entre les deux.** C'est le vrai enjeu du chapitre. En dérivant deux fois la position, on obtient l'accélération ; et l'accélération multipliée par la masse donne la force. Autrement dit : **filmer un mouvement suffit à calculer les forces qui l'ont produit**, sans aucun capteur à l'intérieur du corps.

> Source : MOOC *Le mouvement humain*, Université de Nantes — semaine 2, sections B et C.
> Précédent : *[[04_Mécanique musculaire (v2)|Mécanique musculaire]]*.

---

## 1. Cinématique — réduire l'athlète à un point

Pour décrire un déplacement, on simplifie brutalement : le corps entier est ramené à son **centre de masse**, situé approximativement au niveau du nombril. On repère sa position par rapport à un point de référence — la ligne de départ, à 0 m.

![[Pasted image 20260728154120.png|420]]
*Le coureur entier réduit à un point unique, dont on ne retient que la position le long de la piste.*

La succession de ces positions dans le temps donne la **trajectoire**, dont la forme dépend entièrement de l'activité.

![[Pasted image 20260728154032.png|560]]
*Quatre familles. **Rectiligne** pour le sprint, **parabolique** pour le saut en longueur, **circulaire** pour le lancer de disque, **aléatoire** pour un footballeur suivi pendant un match.*

---

## 2. Cinématique — vitesse et accélération

La **vitesse moyenne** est la variation de position divisée par l'intervalle de temps :

$$\bar{V} = \frac{\Delta x}{\Delta t}$$

Sur un 100 m couru en 10 s, cela donne 10 m·s⁻¹. Sauf que ce chiffre décrit un coureur qui n'existe pas.

![[Pasted image 20260728154350.png|660]]
*Trois niveaux de finesse sur la même course. La **ligne grise** à 10 m·s⁻¹ est la moyenne globale. Les **paliers noirs** sont les moyennes par tranche de 10 m. La **courbe bleue** est la vitesse instantanée mesurée au radar toutes les 0,01 s — ses oscillations sont les foulées elles-mêmes. Le pic entouré en rouge marque la **V max**, autour de 13,5 m·s⁻¹ vers 50 m.*

> [!warning] Pourquoi la moyenne ment
> Regarde où passe la ligne grise. Sur les 10 premiers mètres elle est **au-dessus** de la réalité — le coureur n'est qu'à environ 5 m·s⁻¹. Sur tout le reste elle est **en dessous** — il tient 11 à 12,5 m·s⁻¹.
>
> La moyenne ne correspond donc à aucun instant de la course. Pour obtenir la vraie vitesse, il faut faire tendre l'intervalle vers zéro, c'est-à-dire dériver :
>
> $$V = \lim_{\Delta t \to 0} \frac{\Delta x}{\Delta t} = \frac{dx}{dt}$$

**L'accélération** est à la vitesse ce que la vitesse est à la position — sa dérivée :

$$\bar{a} = \frac{\Delta V}{\Delta t} \quad \text{(en m·s}^{-2}\text{)}$$

![[Pasted image 20260728154449.png|560]]
*Deux athlètes. Ils atteignent **la même vitesse maximale**, environ 12,6 m·s⁻¹, et finissent au même niveau. Mais le bleu y arrive en 5 s et le rouge en 7 s.*

> 💡 **Ce que cette figure dit vraiment.** Les deux courbes se croisent vers 7 s. Tout ce qui sépare ces deux coureurs se joue **avant** — dans la pente initiale, pas dans le plafond. Deux athlètes de même vitesse de pointe peuvent être séparés par plusieurs mètres à l'arrivée, uniquement par leur accélération.
>
> On lit aussi les trois phases d'une course sur ces courbes : **montée** en vitesse, **maintien**, puis **décélération** finale — car personne ne tient sa vitesse max jusqu'au bout.

---

## 3. Cinématique — les mouvements de rotation

Le centre de masse suit une trajectoire à peu près rectiligne, mais il est déplacé par des **segments qui pivotent autour d'articulations**. À cette échelle, tout est rotation.

On mesure alors une **position angulaire** $\theta$ en radians — au goniomètre, par exemple — et on en dérive la **vitesse angulaire** :

$$\bar{\omega} = \frac{\Delta \theta}{\Delta t} \quad \text{(en rad·s}^{-1}\text{)}$$

![[Pasted image 20260728154721.png|660]]
*Le genou d'un skieur en descente, sur un cycle complet. La vitesse angulaire est **négative pendant la flexion** (le creux au début du cycle) puis **positive pendant l'extension** (le pic vers 65-70 %). Les deux courbes correspondent au même skieur avant et après fatigue.*

> 💡 **Ce que révèle la comparaison des deux tracés.** La courbe rouge est une version **aplatie** de la bleue : mêmes phases, même chronologie, mais des amplitudes nettement réduites — le creux ne descend plus aussi bas, le pic ne monte plus aussi haut.
>
> Autrement dit, la fatigue ne désorganise pas le geste, elle le **ralentit**. Le skieur fatigué fait toujours la même chose au même moment, simplement moins vite.

---

## 4. Dynamique — la force est un vecteur

On passe de la description aux causes. Modifier la vitesse d'une masse — l'accélérer, la freiner, la faire tourner — suppose d'appliquer une **force**. Chez l'humain, ce sont les muscles qui les créent et le système tendineux qui les transmet.

Une force ne se résume jamais à un nombre. C'est un vecteur, défini par cinq caractéristiques.

![[Pasted image 20260728155210.png|620]]
*Les cinq caractéristiques sur un exemple : **origine** (le contact pied-sol, marqué par la croix), **direction** (la ligne d'action, ici la verticale en pointillés), **sens** (vers le bas), **norme** (1000 N, codée par la longueur de la flèche), **unité** (le newton).*

> 💡 **Ce que vaut un newton.** Une force de 1 N accélère une masse de 1 kg de 1 mètre par seconde, chaque seconde. Un corps de 100 kg immobile pèse donc environ 1000 N.

---

## 5. Dynamique — le principe fondamental

$$\sum \vec{F}_{ext} = m \cdot \vec{a}$$

Avant tout calcul, on établit le **bilan des forces** : on liste et on dessine toutes les forces extérieures au système étudié.

![[Pasted image 20260728155829.png|620]]
*Le système est {sujet + barre}. Poids de la barre $\vec{P_B}$ en rouge, poids du sujet $\vec{P_S}$ en vert, réaction du sol $\vec{R_N}$ en bleu vers le haut, et force de poussée $\vec{F_P}$ en noir vers le bas.*

> [!warning] Pourquoi la force de poussée est entre parenthèses
> $\vec{F_P}$ et $\vec{R_N}$ sont opposées et de même norme — c'est le principe des actions réciproques, la troisième loi de Newton.
>
> Mais elles ne s'appliquent pas au même objet. La poussée s'applique **au sol**, la réaction s'applique **au sujet**. Seule la réaction entre donc dans le bilan du système étudié. C'est la raison de la parenthèse sur la diapo, et c'est l'erreur classique du bilan des forces.

![[Pasted image 20260728155851.png|620]]
*L'application numérique. Un accéléromètre mesure $a = 5$ m·s⁻². Avec une barre de 30 kg, un sujet de 70 kg et $g = 10$ m·s⁻² :*

$$R_N = m_{B+S} \cdot a + m_B \cdot g + m_S \cdot g = (30+70) \cdot 5 + 300 + 700 = \mathbf{1500\ N}$$

> 💡 **Le résultat est le point important.** Immobile, l'ensemble pèse 1000 N. En accélérant à 5 m·s⁻², il en produit **1500 N**. La force développée contre le sol dépasse largement le poids statique — et c'est exactement ce que mesure une plateforme de force sous un athlète qui saute.
>
> Surtout, remarque le sens du calcul : on **mesure une accélération** et on en **déduit une force** qu'on n'a jamais mesurée directement. C'est le pont annoncé en introduction.

---

## 6. Dynamique — moment de force et bras de levier

Puisque le mouvement humain est fait de rotations, il faut une grandeur adaptée : le **moment de force**, produit de la force par le **bras de levier** — la distance perpendiculaire entre la ligne d'action de la force et le centre de rotation.

![[Pasted image 20260728160012.png|620]]
*Deux forces **strictement identiques** (les flèches ont la même longueur), appliquées à des distances différentes du pivot rouge. Le moment, lui, diffère.*

$$M_1 = F_1 \cdot d_1 \qquad M_2 = F_2 \cdot d_2 \qquad \text{avec } F_2 = F_1 \Rightarrow M_2 < M_1$$

Le principe fondamental se transpose alors terme à terme.

![[Pasted image 20260728160038.png|560]]
*En rotation : $\sum M_{\vec{F}_{ext}} = I \cdot \alpha$. Le **moment de force** remplace la force, le **moment d'inertie** $I$ remplace la masse (c'est la propriété du système qui résiste à la mise en rotation), et l'**accélération angulaire** $\alpha$ en rad·s⁻² remplace l'accélération linéaire.*

**Et à l'équilibre**, quand rien ne tourne, la somme des moments est nulle.

![[Pasted image 20260728160127.png|560]]
*Tenir un verre immobile. Le poids $\vec{P}$ = 5 N agit avec un bras de levier $d$ = 0,4 m. Les fléchisseurs du coude doivent produire un moment égal et opposé : $M_{\vec{F}} = 5 \times 0{,}4 = \mathbf{2\ N \cdot m}$.*

> [!warning] Tenir un verre n'est pas « ne rien faire »
> Regarde les deux bras de levier sur le schéma. Celui du poids est **long** — toute la longueur de l'avant-bras. Celui du biceps est **minuscule** : son tendon s'insère à quelques centimètres du coude.
>
> Comme les deux moments doivent être égaux, la force musculaire est nécessairement **bien supérieure** à la charge. Avec un bras de levier de biceps d'environ 4 cm, il faut $2 / 0{,}04 = 50$ N pour tenir un verre qui n'en pèse que 5 — **dix fois la charge** *(ordre de grandeur calculé, la diapo ne donne pas cette valeur)*.
>
> Le corps humain est mécaniquement désavantagé en force, et il l'est volontairement : ce qu'il perd en force, il le gagne en **amplitude et en vitesse** au bout du segment.

---

## 7. Le pont entre les deux

![[Pasted image 20260728154940.png|660]]
*La chaîne complète. $V = dx/dt$, puis $a = dV/dt$, puis $\sum \vec{F} = m\vec{a}$. À droite, deux niveaux de modélisation : le **modèle simple** — un saut vertical décrit par sa distance de poussée $h_{PO}$ et son temps de vol $h$ — et la **modélisation musculo-squelettique** complète.*

> 💡 **C'est ce qui rend la biomécanique utilisable.** On ne peut pas mettre un capteur de force dans le tendon d'un athlète en compétition. Mais on peut **filmer** son mouvement, en tirer les positions, dériver deux fois, et remonter aux forces.
>
> Un système optoélectronique et un modèle suffisent donc à estimer ce qui se passe à l'intérieur du corps, à partir de ce qu'on voit à l'extérieur.

> 💡 **Où tu croises déjà ça** *(déduction, pas contenu du MOOC)*. Les applications qui mesurent ta détente avec la caméra d'un téléphone appliquent exactement cette chaîne : elles mesurent un **temps de vol**, en déduisent la vitesse au décollage, et de là la force et la puissance développées.
>
> C'est aussi la méthode du chapitre suivant, qui détermine la relation force-vitesse d'un sportif à partir d'une série de sauts chargés.
