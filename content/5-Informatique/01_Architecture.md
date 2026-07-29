---
title: Architecture des Ordinateurs
---
# Architecture des Ordinateurs

> [!info] Objectif de ce fichier
> Cours "Architecture des Ordinateurs" (ADO) suivi en licence — codage, architecture de Von Neumann, assembleur, processeurs actuels, mémoire. Volontairement allégé par rapport au cours original (voir tri ci-dessous) : l'objectif n'est pas la maîtrise de bas niveau (concevoir un circuit, écrire de l'assembleur x86), mais une culture d'ingénieur suffisante pour comprendre ce qu'il se passe "sous" le code qu'on écrit, et pour poser les bases qui expliquent pourquoi un GPU calcule les maths plus vite qu'un CPU.
>
> Un second fichier, [[02_Stanford]], couvrira les slides Systems du cours Stanford LLM (déjà présentes en copier-coller non digéré dans [[07_LLM]]) — les deux seront fusionnés une fois que ce fichier-ci aura posé les fondations CPU/mémoire.

---

## A. Codage — représentation des nombres en machine

### Entiers relatifs — complément à 2

Pour représenter un entier négatif en machine, on utilise le **complément à 2 (C2)** : on inverse tous les bits puis on ajoute 1.

> [!warning] Complément à 2
> Pour $-N$ sur $m$ bits : $C2(N) = \overline{N} + 1$

**Exemple.** $122 + (-7)$ sur 8 bits : $-7$ s'écrit `1111 1001` en C2. En additionnant `0111 1010` (122) et `1111 1001` (-7), on obtient `1 0111 0011` — la retenue de gauche est perdue (capacité de la machine sur 8 bits), il reste `0111 0011` = 115, le résultat correct.

### Nombres à virgule flottante — norme IEEE754

C'est la partie la plus importante de ce chapitre : c'est elle qui explique directement pourquoi le choix de précision (fp32, fp16, bf16) change les performances d'un modèle de deep learning.

> [!warning] Décomposition IEEE754
>
> | Décomposition | Signe | Exposant | Mantisse |
> |---|---|---|---|
> | Simple précision (32 bits, **fp32**) | 1 | 8 | 23 |
> | Double précision (64 bits, fp64) | 1 | 11 | 52 |
>
> $$\text{Nombre} = (-1)^{\text{signe}} \times 1{,}\text{mantisse} \times 2^{(\text{exposant} - \text{biais})}$$
>
> biais = 127 (simple précision)

L'exposant contrôle la **plage de valeurs représentables** (à quel point on peut représenter des nombres très grands ou très petits), la mantisse contrôle la **précision** (le nombre de chiffres significatifs).

> [!info] 💡 Lien direct avec le deep learning ([[07_LLM]])
> **bf16** (bfloat16) a été conçu avec **8 bits d'exposant** (comme fp32 — même plage de valeurs, donc pas de risque de dépassement/underflow) mais seulement **7 bits de mantisse** (beaucoup moins précis que fp32). C'est exactement pourquoi le mixed-precision training fonctionne : on garde la plage dynamique de fp32 (les gradients ne "débordent" pas), on sacrifie juste des chiffres après la virgule, ce qui est suffisant pour l'entraînement d'un réseau de neurones — d'où le gain en mémoire et en vitesse de calcul sans perte de stabilité numérique.

**Addition et multiplication en IEEE754** — le principe suffit : pour additionner, on aligne les deux nombres sur le même exposant puis on additionne les mantisses ; pour multiplier, on additionne les exposants et on multiplie les mantisses. Le détail bit-à-bit du calcul manuel n'est pas utile à retenir ici.

---

## B. Architecture de Von Neumann

### B.1 Le problème de départ

Un ordinateur doit exécuter un **programme** : une suite d'instructions, à traiter une après l'autre ("additionne ces deux nombres", "range le résultat", "va à l'instruction suivante"...). Deux besoins en découlent immédiatement :

- un endroit où **stocker** le programme et les données → la **mémoire**.
- un composant qui **lit** les instructions une par une et les **exécute** → le **processeur**.

> [!tip] 💡 La marque de fabrique de Von Neumann
> Instructions et données partagent la **même** mémoire physique (juste rangées à des adresses différentes). Regarde le schéma B.3 plus bas (`ch2_chemin_donnees`) : la mémoire y est dessinée en deux zones "Programme" / "Données", mais c'est une seule et même puce.

Mémoire et processeur sont deux puces séparées → il leur faut des fils électriques partagés pour communiquer, les **bus** : bus d'adresses ("je veux le contenu de la case 42") et bus de données (le contenu, qui transite dans un sens ou dans l'autre) — déjà vus plus haut.

![[images/5-Informatique/Architecture/ch2_bus_architecture-02.png]]
*Le schéma le plus large : processeur, mémoire centrale et périphériques (clavier, écran, disque) connectés aux mêmes bus partagés.*

### B.2 Comment le processeur sait où il en est

Le processeur exécute les instructions dans l'ordre — il lui faut donc un moyen de savoir laquelle vient ensuite. C'est le rôle du **PC** (*Program Counter*, aussi appelé IP ou Compteur Ordinal) : un registre qui contient toujours l'adresse de la **prochaine** instruction à exécuter. Après chaque instruction, le PC avance tout seul (il s'incrémente) — sauf en cas de saut/branchement, où on le force à pointer ailleurs.

### B.3 Ce qu'il faut en plus pour traiter une instruction

Une fois l'instruction récupérée depuis la mémoire (via le PC), il faut encore : la comprendre (**décoder** quel type d'opération c'est), effectuer le calcul (**ALU** — Unité Arithmétique et Logique), et un endroit où stocker les valeurs manipulées le temps du calcul (les **registres** — des cases mémoire minuscules mais très rapides, directement dans le processeur, à ne pas confondre avec la RAM).

Les registres, concrètement, c'est juste une poignée de cases numérotées (souvent notées R0, R1, R2... ou avec des noms historiques comme AX, BX en x86) :

![[images/5-Informatique/Architecture/ch2_registres_banc.png|299]]
*Le banc de registres : quelques cases numérotées, chacune peut être lue (vers l'ALU) ou écrite (résultat d'un calcul). AX n'a rien de spécial — c'est juste le nom donné à un registre en particulier (ici R0), utilisé par convention comme "accumulateur" (le registre par défaut pour les résultats de calcul en x86).*

> [!warning] Ne pas confondre registres et pile
> Le schéma du cours original (`ch2_chemin_donnees`) dessine "AX accu" et "Pile" collés dans la même boîte, et c'est ce qui prête à confusion. Ce sont deux choses différentes :
> - Les **registres** (dont AX) : une poignée de cases fixes, directement dans le processeur, adressées par leur nom (R0, R1...). C'est ce qu'on vient de voir.
> - **La pile** (déjà vue en section C, PUSH/POP) : une zone de la **mémoire** (RAM), pas des registres — elle sert à empiler des valeurs de taille variable, notamment pour les appels de fonction. Elle est repérée par un registre spécial, le **SP (Stack Pointer)**, qui contient juste l'adresse mémoire du sommet de la pile — mais la pile elle-même vit en RAM, pas dans le banc de registres.

C'est seulement maintenant que le schéma détaillé du processeur devient lisible : chaque bloc qu'on vient de motiver (PC, registre d'instruction, registres de calcul, ALU) y apparaît, connecté à la mémoire par les mêmes deux bus. (Le schéma est plus bas, en B.4 — on va s'en servir directement pour suivre un exemple pas à pas.)

> [!info] Vocabulaire
> L'ensemble ALU + registres + connexions qui effectue le calcul s'appelle le **chemin de données**. Le composant qui pilote et séquence tout ça (quand lire, quand écrire, quand calculer) s'appelle l'**unité de contrôle**.

### B.4 Le cycle complet, en 5 étapes

En reprenant les briques posées ci-dessus dans l'ordre où elles interviennent :

> [!warning] Cycle d'exécution d'une instruction
> 1. **Chargement** — le PC indique l'adresse de la prochaine instruction ; elle est lue en mémoire et copiée dans le registre d'instruction ; le PC est incrémenté.
> 2. **Décodage + lecture des registres** — l'instruction est décodée, les registres opérandes sont lus.
> 3. **ALU** — l'opération arithmétique/logique est exécutée.
> 4. **Accès mémoire** — lecture/écriture en mémoire, seulement si l'instruction est un `load`/`store`.
> 5. **Écriture registre** — le résultat est rangé dans le registre destination.

**Exemple concret, fil rouge.** Garde le schéma ci-dessous sous les yeux en lisant les 5 étapes — chaque étape pointe vers un bloc précis du dessin. Instruction `ADD R0, R1, R2` (= "R0 reçoit R1 + R2"), stockée à l'adresse 100, avec R1 = 5 et R2 = 3 :

![[images/5-Informatique/Architecture/ch2_chemin_donnees-03.png]]
*Repères pour les étapes ci-dessous : PC (en haut à gauche) → Registre Instruction (au centre, "Registre" sur le schéma) → Registres de calcul (à gauche, "AX accu" et la pile de registres) → ALU (en bas à gauche, le triangle) → Mémoire (à droite, RAM).*

1. **Chargement** (bloc **PC** → **Mémoire** → **Registre Instruction** sur le schéma) — PC = 100 → le processeur lit l'instruction en mémoire (les bits `0001 001 010 000`) → elle est copiée dans le registre d'instruction (IR) → PC passe à 104 (l'instruction suivante).
2. **Décodage + lecture des registres** (bloc **Registre Instruction** → **Registres de calcul**) — le décodeur (un circuit qui lit les champs de bits de l'instruction) comprend : opération = ADD, source 1 = R1, source 2 = R2, destination = R0. Les valeurs actuelles de R1 (5) et R2 (3) sont lues dans les registres.
3. **ALU** (bloc **ALU**, le triangle en bas) — reçoit 5 et 3 en entrée depuis les registres, calcule 5 + 3, produit 8 en sortie.
4. **Accès mémoire** (bloc **Mémoire**, à droite) — ADD n'est ni un `load` ni un `store` → cette étape ne fait rien ici, le flux ne va pas jusqu'à la mémoire.
5. **Écriture registre** (retour vers le bloc **Registres de calcul**) — 8 est rangé dans R0, en écrasant son ancienne valeur.

----

Exemple complet, minute par minute, pour l'instruction `ADD R0, R1, R2` (= "R0 reçoit R1 + R2") stockée à l'adresse 100 :

1. PC = 100 → le processeur va chercher ce qu'il y a à l'adresse 100 en mémoire → il lit une suite de bits, disons `0001 001 010 000` → PC passe à 104 (l'instruction suivante).
2. Décodage : le circuit décodeur lit `0001` = "c'est un ADD", `001` = "source 1 = R1", `010` = "source 2 = R2", `000` = "destination = R0". Il lit ensuite les valeurs actuelles de R1 (disons 5) et R2 (disons 3) dans les registres.
3. ALU : reçoit 5 et 3 en entrée, calcule 5+3, sort 8.
4. Pas d'accès mémoire (ADD ne lit/écrit pas la RAM, juste les registres).
5. Écriture : 8 est rangé dans R0, en écrasant ce qu'il y avait avant dedans.

-----

**Le rôle, pas le type.** Une adresse c'est bien un nombre (comme une donnée), mais ce qui distingue les deux bus c'est **à quoi sert** ce nombre, pas sa nature :

- Le **bus d'adresses** transporte un nombre qui désigne **où** aller chercher/écrire quelque chose ("je veux la case n°42").
- Le **bus de données** transporte le nombre qui est **le contenu réel** stocké à cette adresse (la valeur elle-même, pas un emplacement).

Deuxième différence importante, plus concrète : le bus d'adresses est **unidirectionnel** PAS CE QUIL YA SUR LE SCHEMA EN HAUT (seul le processeur envoie des adresses, la mémoire ne fait qu'écouter — la mémoire n'a jamais besoin de dire "je veux une adresse" au processeur). Le bus de données, lui, est **bidirectionnel** (parfois ça va mémoire→processeur en lecture, parfois processeur→mémoire en écriture).

**Sur ta phrase "quand je décode l'instruction je récupère des données via le bus" — pas tout à fait.** Le décodage (étape 2) ne touche pas les bus du tout. Voici où les bus interviennent vraiment dans le cycle qu'on a vu :

- **Étape 1 (chargement)** : _seule étape qui utilise les deux bus_. Le PC envoie une adresse sur le bus d'adresses ("va chercher l'instruction à l'adresse 100"), la mémoire répond en envoyant les bits de l'instruction sur le bus de données. C'est ce qui remplit le registre d'instruction (IR).
- **Étape 2 (décodage + lecture registres)** : aucun bus impliqué ! Le décodage lit juste les bits déjà présents dans l'IR (donc à l'intérieur du processeur), et la lecture des registres R1/R2 se fait aussi en interne — les registres sont câblés directement à l'ALU, pas connectés par un bus externe.
- **Étape 4 (accès mémoire)** : _seule autre étape qui peut réutiliser les bus_, et uniquement si l'instruction est un `load` ou un `store` (pas notre `ADD`) — même mécanisme qu'à l'étape 1 : adresse envoyée sur le bus d'adresses, donnée transportée sur le bus de données.
----

Retiens surtout ce cycle en 5 étapes : c'est la base de tout le chapitre D (pipelining) — un processeur moderne ne les exécute pas une par une pour une seule instruction avant de passer à la suivante, il les fait chevaucher entre plusieurs instructions à la fois.

---

## C. Processeur — la pile et les appels de fonction

> [!info] Pourquoi ce bout d'assembleur et pas le reste
> Le cours original détaille tout le jeu d'instructions 80x86 (MOV, arithmétique, logique, sauts conditionnels, boucles, interruptions DOS/BIOS) — c'est de la programmation en mode réel 8086, obsolète depuis les années 90. Un seul concept mérite d'être gardé : **la pile**, parce que c'est ce qui explique concrètement ce qu'est un appel de fonction (et pourquoi une récursion trop profonde fait un stack overflow).

### C.1 La RAM, trois compartiments

Vue de haut, la RAM est un seul grand espace d'adresses continu — mais par convention (organisée ainsi par le compilateur/l'OS), elle est découpée en zones :

- **Programme** (le "code") : les instructions, celles que le PC parcourt.
- **Données** : les variables globales/statiques, taille fixe et connue à l'avance.
- **Pile** : grandit et rétrécit dynamiquement au fil de l'exécution — c'est elle qui nous intéresse ici.

### C.2 Le problème que la pile résout

Les registres (section B.3) sont un nombre **fixe et très limité** de cases (8 à 32 selon le processeur). Mais un programme peut appeler une fonction, qui appelle une autre fonction, qui s'appelle elle-même récursivement — sans limite connue à l'avance sur la profondeur de ces appels imbriqués. Il faut donc un espace qui peut **grandir et rétrécir** à volonté, contrairement aux registres qui sont un nombre fixe de cases : c'est la pile, une zone réservée en RAM. Elle fonctionne comme une pile d'assiettes : on ne peut poser (**PUSH**) ou retirer (**POP**) que sur le dessus, jamais au milieu — une structure dite **LIFO** (*Last In, First Out*, "le dernier arrivé est le premier reparti").

> [!warning] PUSH / POP
> **PUSH** empile une valeur au sommet de la pile (et décrémente le pointeur de pile **SP**, Stack Pointer) ; **POP** dépile la valeur du sommet (et incrémente SP). Le SP n'est qu'un registre qui contient l'adresse mémoire du sommet actuel de la pile — la pile elle-même vit en RAM, pas dans les registres (à ne pas confondre avec le banc de registres vu en B.3).

![[images/5-Informatique/Architecture/ch3_push_pop-09.png]]
*PUSH AX empile le contenu de AX au sommet de la pile ; POP AX le retire.*

### C.3 À quoi ça sert concrètement : un appel de fonction

Imagine `main()` en train de tourner, PC pointant sur l'instruction 50, qui appelle une fonction `carre(x)`. Le processeur doit sauter ailleurs en mémoire pour exécuter `carre`, mais il doit se souvenir de "revenir à l'instruction 51 dans `main` une fois `carre` terminée" :

1. Avant de sauter dans `carre`, l'adresse de retour (51) est **empilée** (PUSH) sur la pile.
2. Si `carre` a des variables locales, elles sont empilées aussi le temps de son exécution.
3. `carre` termine son calcul, ses variables locales sont dépilées (nettoyées).
4. L'adresse de retour est **dépilée** (POP) — le PC est remis à 51, l'exécution reprend dans `main` juste après l'appel.

Si `carre` appelait elle-même une autre fonction, une deuxième adresse de retour serait empilée par-dessus la première — d'où l'image de la pile : chaque appel ajoute une couche par-dessus, chaque retour retire la couche du dessus.

> [!tip] Le lien avec le stack overflow
> Une fonction récursive qui ne s'arrête jamais (condition d'arrêt oubliée) empile une adresse de retour à chaque appel, sans jamais dépiler — la pile grandit indéfiniment jusqu'à épuiser l'espace qui lui est réservé. C'est exactement l'erreur *stack overflow*.

---

## D. Processeurs actuels — pipelining, prédiction de branchement, caches, superscalaire

C'est le chapitre le plus important pour comprendre pourquoi un CPU et un GPU sont conçus si différemment — le tradeoff **latence vs débit** qui explique tout.

### D.1 Pipelining

**Idée.** Découper l'exécution d'une instruction en étapes (Fetch, Decode, Execute...), et faire avancer plusieurs instructions en même temps, chacune à une étape différente — comme une chaîne de montage.

![[images/5-Informatique/Architecture/ch4_pipeline_deroulement-03.png]]
*Sans pipeline : 3 instructions traitées en 9 cycles. Avec pipeline : les mêmes 3 instructions traitées en 5 cycles (le temps de traitement d'une instruction seule reste inchangé, mais le débit global augmente).*

**Aléas (hazards).** Un aléa survient quand une instruction ne peut pas avancer normalement dans le pipeline.

> [!warning] Les 3 types d'aléas
> - **Structurels** : deux instructions ont besoin de la même ressource (ex. mémoire) → solution : caches séparés instructions/données.
> - **De données** : une instruction a besoin du résultat de la précédente avant qu'il soit disponible → solution : le **forwarding** (court-circuiter l'écriture registre, transmettre directement le résultat entre étages du pipeline).
> - **De contrôle** : un branchement bloque le pipeline le temps de connaître l'adresse de destination → solution : la **prédiction de branchement**.

![[images/5-Informatique/Architecture/ch4_alea-07.png]]
*Un aléa crée des "bulles" (NOP) dans le pipeline — les étages en aval restent vacants pendant que l'instruction bloquée attend.*

![[images/5-Informatique/Architecture/ch4_forwarding-11.png]]
*Forwarding : un chemin direct entre l'étage MEM/EX d'une instruction et l'étage EX de la suivante évite d'attendre l'écriture registre (WB).*

### D.2 Prédiction de branchement

En moyenne, une instruction sur 5 est un branchement — sans prédiction, le pipeline se viderait sans arrêt. On utilise une table (**Branch Target Buffer**) indexée par le PC, qui stocke les adresses de destination des branchements déjà rencontrés, pour deviner l'adresse de destination *avant* de la calculer réellement.

![[images/5-Informatique/Architecture/ch4_branch_prediction-14.png]]
*Sans prédiction d'adresse : 4 cycles de pénalité avant de charger la bonne instruction. Avec prédiction : le pipeline continue sans interruption.*

### D.3 Caches

**Le problème.** Le temps de cycle du processeur a baissé beaucoup plus vite que le temps d'accès à la mémoire — l'écart entre les deux ne cesse de grandir.

> [!warning] Temps d'accès moyen avec cache
> $$t_{glob} = t_c + (1-h)\, t_m$$
> où $t_c$ = temps d'accès au cache, $t_m$ = temps d'accès à la mémoire, $h$ = taux de succès (*hit rate*).

Un cache atteint typiquement 80-90% de taux de succès, grâce à deux propriétés de **localité** : temporelle (une donnée récemment utilisée a de bonnes chances d'être réutilisée bientôt) et spatiale (une donnée à l'adresse A a de bonnes chances qu'une donnée à une adresse voisine soit utilisée bientôt aussi).

![[images/5-Informatique/Architecture/ch4_cache_perf-18.png]]
*Principe du cache : une mémoire rapide et petite placée entre le processeur et la mémoire principale, invisible du point de vue du processeur (hit = renvoi direct, miss = va chercher en mémoire principale puis stocke une copie dans le cache).*

![[images/5-Informatique/Architecture/ch4_cache_associativite-23.png]]
*Cache associatif (n-way) : pour limiter les conflits de cache, la table est divisée en n bancs — une donnée peut se placer dans n'importe lequel des n bancs.*

![[images/5-Informatique/Architecture/ch4_cache_hierarchie-25.png]]
*Un processeur moderne n'a pas un seul cache mais une hiérarchie de caches (L1 instructions/données séparés → cache commun L2 → cache commun off-chip L3), dont la taille augmente et la vitesse décroît à mesure qu'on s'éloigne du processeur.*

### D.4 Processeurs superscalaires

Un processeur **superscalaire** exécute plusieurs instructions simultanément. Ça demande de détecter quelles instructions peuvent s'exécuter en parallèle (analyse des dépendances), de disposer de plusieurs unités de calcul, et de gérer le **renommage de registres** (registres logiques du programme vs registres physiques du processeur, via un **ROB — ReOrder Buffer** qui garantit que les instructions terminent dans l'ordre du programme malgré une exécution interne désordonnée).

![[images/5-Informatique/Architecture/ch4_superscalaire-26.png]]
*Plusieurs instructions progressent en parallèle dans le pipeline, chacune avec un léger décalage.*

![[images/5-Informatique/Architecture/ch4_rob-29.png]]
*Le ROB (FIFO) mémorise l'ordre des instructions du programme et fournit l'équivalent de registres physiques additionnels — les instructions sortent dans l'ordre du programme même si elles s'exécutent dans le désordre en interne.*

> [!info] 💡 Le lien avec le GPU (à développer dans [[02_Stanford]])
> Tout ce chapitre D explique pourquoi un cœur CPU est "cher" : prédiction de branchement, exécution superscalaire dans le désordre, renommage de registres — beaucoup de circuiterie dédiée à accélérer **une seule** instruction séquentielle (optimisé pour la **latence**). Un cœur GPU fait l'inverse : il renonce à presque tout ça (pas de prédiction sophistiquée, pas d'exécution dans le désordre) pour rester minuscule et être répété par milliers, tous exécutant la même instruction sur des données différentes (optimisé pour le **débit**). C'est exactement le tradeoff qui rend le GPU adapté à la multiplication matricielle (des millions d'opérations identiques et indépendantes) et mauvais pour du code séquentiel avec plein de branchements.

---

## E. Mémoire

### Notion de hiérarchie mémoire

Plus une mémoire est rapide, plus elle est petite et chère ; plus elle est grande, plus elle est lente et bon marché.

![[images/5-Informatique/Architecture/ch5_hierarchie_pyramide-19.png]]
*Pyramide de la hiérarchie mémoire : registres (le plus rapide, le plus petit) → cache → mémoire principale → mémoire d'appui → mémoire de masse (le plus lent, le plus grand).*

![[images/5-Informatique/Architecture/ch5_hierarchie_table-21.png]]
*Ordres de grandeur concrets : registres < 1 ns / > 50 Go/s / < 100 octets ; cache 2-5 ns / 5-20 Go/s / 100 Ko-1 Mo ; mémoire centrale 20 ns / 1 Go/s / 256 Mo-4 Go ; disque dur 1-10 ms / 300 Mo/s / 50-500 Go.*

### SRAM vs DRAM

> [!warning] Les deux familles de RAM
> **SRAM (statique)** : un bit = une bascule (4-6 transistors). Plus rapide, plus chère, moins dense → utilisée pour les caches et les registres.
>
> **DRAM (dynamique)** : un bit = une charge électrique stockée dans un condensateur (1 transistor). Plus dense, moins chère, mais fuit et doit être rafraîchie régulièrement (plus lente, lecture destructive) → utilisée pour la mémoire centrale (RAM classique).

![[images/5-Informatique/Architecture/ch5_sram_dram-09.png]]
*Structure d'une cellule SRAM (bascule à transistors) vs DRAM (un transistor + un condensateur).*

> [!info] 💡 Lien avec le GPU ([[07_LLM]])
> C'est exactement la même distinction que dans la hiérarchie mémoire d'un GPU : les registres et la *shared memory* (proche des cœurs, minuscule, très rapide) sont l'équivalent SRAM ; la mémoire globale du GPU (HBM — High Bandwidth Memory, plusieurs Go, mais bien plus lente d'accès) est l'équivalent DRAM. Le *tiling* et *FlashAttention* (vus dans [[07_LLM]]) existent précisément pour maximiser la réutilisation des données déjà chargées en SRAM/shared memory avant de devoir retourner chercher en HBM/DRAM.

### Mémoires mortes (ROM)

Contrairement à la RAM (volatile, perd son contenu hors tension), la **ROM** conserve l'information même sans alimentation. Il existe plusieurs variantes (ROM figée en usine, PROM programmable une fois, EPROM effaçable aux UV, EEPROM/Flash effaçables électriquement) — le détail de fabrication de chacune (transistors à grille flottante, tension d'effacement...) est de l'ingénierie de circuit, pas utile ici. Ce qu'il faut retenir : c'est le même principe que la mémoire Flash de nos SSD/clés USB aujourd'hui — programmable et effaçable électriquement, comportement non volatile.

---

> [!note] Points de vigilance
> Ce fichier est déjà dense malgré le tri — si en le relisant tu trouves encore des sections trop détaillées par rapport à ce que tu en fais vraiment, n'hésite pas à couper davantage. Mieux vaut un fichier plus court que tu relis vraiment, qu'un fichier exhaustif qui prend la poussière.
