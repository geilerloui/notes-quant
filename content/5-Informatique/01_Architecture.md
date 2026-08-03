## A. Architecture de Von Neumann

### A.1 Le problème de départ

Un ordinateur doit exécuter un **programme** : une suite d'instructions, à traiter une après l'autre ("additionne ces deux nombres", "range le résultat", "va à l'instruction suivante"...). Deux besoins en découlent immédiatement :

- un endroit où **stocker** le programme et les données → la **mémoire**.
- un composant qui **lit** les instructions une par une et les **exécute** → le **processeur**.

> [!tip] 💡 La marque de fabrique de Von Neumann
> Instructions et données partagent la **même** mémoire physique (juste rangées à des adresses différentes). Regarde le schéma B.3 plus bas (`ch2_chemin_donnees`) : la mémoire y est dessinée en deux zones "Programme" / "Données", mais c'est une seule et même puce.

Mémoire et processeur sont deux puces séparées → il leur faut des fils électriques partagés pour communiquer, les **bus** : bus d'adresses ("je veux le contenu de la case 42") et bus de données (le contenu, qui transite dans un sens ou dans l'autre) — déjà vus plus haut.

![[images/5-Informatique/Architecture/ch2_bus_architecture-02.png]]
*Le schéma le plus large : processeur, mémoire centrale et périphériques (clavier, écran, disque) connectés aux mêmes bus partagés.*

On a les deux briques de base (mémoire, processeur) et le lien entre les deux (les bus). Reste à voir, sur un exemple concret, comment un code qu'on écrit devient réellement des instructions exécutées.

### A.2 Le cycle complet




![[Pasted image 20260802181933.png|595]]


En comparaison 

![[Pasted image 20260802180957.png|502]]

Peut être préciser aussi que c'est le OS qui va charger le binaire compilé de CPython dans la RAM 

Quand on lance la commence ```python X.py``` ça va charger en mémoire le programme CPython qui sera dans la partie "Programme" de la RAM ainsi que notre fichier X.pyc qui lui sera dans la partie Données de la RAM. Le PC et le IR vont pointer vers le programme CPython (plein de fichier .c compilé)

Ensuite le programme CPython va effectuer deux étapes en boucle jusqu'à la fin du contenu du fichier .pyc:
1. Lire ligne courante du fichier .pyc dans "Données"
2. Faire de l'interprétation de bytecode ie il va mapper la "donnée" (qui est incompréhensible pour le décodeur) faire vers une valeur compréhensible dans son code C qui est constitué de ```switch``` et ce sera cette ligne qui sera envoyé pour le cpu. 
3. Repeat 1 jusqu'à fin du fichier

Attention, c'est pas qu'il va tranasformer le code à la volée, il fait pas ça, il va juste mapper la "donnée" vers "l'instruction" déjà stocké dans le programme.

![[Pasted image 20260802181800.png]]

Attention: c'est que python fct pour son pseudo code assembleur cvomme étant des instructions pour la machine virtuelle = python virtual machine 

Attention 2:
* Code compilé = Programme; donc en python on dit "j'exécute un script python" on va pas avoir un programme python car c'est pas compilé
* Une application = un ensemble de fichiers compilés ou pas c'est agnostique 
* Projet = Application = en fait c'est juste que le terme Application c'est une vue produit genre une API web, script d'automatisation de tâches etc


#### Étape 1: La phase de traduction

Un programme n'est jamais exécuté tel qu'on l'écrit — il passe par une chaîne de traductions. Le chemin diffère selon le langage, comme le montre la comparaison C / Python :

| | Traduction | S'arrête où | Qui exécute au final |
|---|---|---|---|
| **C** | C → assembleur → binaire | Binaire natif (le bout du chemin) | Le CPU directement, sans intermédiaire |
| **Python** | Python → bytecode | Bytecode (format intermédiaire) | L'interpréteur (lui-même du binaire natif), qui relit le bytecode et simule son exécution |

Ce code fait trois choses : `counter` est une variable globale, à une adresse RAM fixe (5000) ; `add_2` l'incrémente puis retourne `x + 2` ; `main` appelle `add_2` avec `5` et récupère le résultat.

```c
int counter = 0;              // globale, adresse fixe en RAM : 5000

int add_2(int x) {
    counter = counter + 1;
    return x + 2;
}

int main() {
    int result = add_2(5);
    return result;
}
```

Le **compilateur** traduit ce code, ligne par ligne, en instructions assembleur — il choisit une convention (ici : le paramètre `x` et la valeur de retour passent par le registre `R0`, l'adresse de retour et les variables locales passent par la pile) et l'applique systématiquement :

```nasm
main() — à partir de 1000
1000: LOAD  R0, #5        ; R0 = 5  (argument x pour add_2)
1001: CALL  2000          ; empile l'adresse de retour (1002), PC = 2000
1002: PUSH  R0             ; empile le résultat reçu dans R0 → "result" vit sur la pile
1003: POP   R0             ; return result → recharge result dans R0
1004: RET                 ; fin de main

add_2(x) — à partir de 2000
2000: LOAD  R1, [5000]    ; R1 = counter
2001: ADD   R1, R1, #1    ; R1 = counter + 1
2002: STORE [5000], R1    ; counter = R1   (écrit la nouvelle valeur en RAM)
2003: ADD   R0, R0, #2    ; R0 = x + 2     (valeur de retour, écrase R0)
2004: RET                 ; dépile l'adresse de retour (1002), PC ← 1002
```

Ce texte assembleur n'est pas encore exécutable. Un second outil, l'**assembleur** (différent du compilateur), le traduit en binaire — et surtout, il résout les adresses : les labels `1000`, `2000`, `5000` utilisés ici pour la lisibilité humaine deviennent de vraies adresses numériques. S'il y a plusieurs fichiers, le **linker** les assemble en un seul exécutable. Au lancement, c'est l'**OS** qui copie ce binaire en RAM, aux adresses prévues — c'est seulement à ce moment que le programme existe physiquement en mémoire, prêt à être exécuté.

> [!tip] La chaîne complète, en une ligne
> `.c` → **compilateur** → assembleur (texte) → **assembleur** (l'outil) → binaire → **linker** → exécutable → **OS** charge en RAM → **CPU** exécute.

#### Étape 2: La phase d'exécution

Le programme est maintenant en RAM, en binaire. Reste à voir comment le CPU l'exécute concrètement, instruction par instruction — les composants suivants interviennent à chaque étape :

![[Pasted image 20260730162546.png]]
*Schéma récapitulatif de l'étape d'exécution : PC, IR, décodeur, ALU et registres, connectés à la RAM par les bus.*

**Exemple concret** — reprenons `1000: LOAD R0, #5` (la première instruction de `main`) et suivons les 5 étapes du cycle :

1. **Chargement** — PC = 1000 ; l'instruction est lue en RAM et copiée dans l'IR ; PC passe à 1001.
2. **Décodage** — le décodeur lit l'IR : opération = LOAD, destination = R0, valeur immédiate = 5.
3. **ALU** — rien à calculer, `#5` est une valeur **immédiate** (encodée directement dans l'instruction), pas besoin de l'ALU.
4. **Accès mémoire** — rien non plus : `#5` n'est pas une adresse RAM à aller lire (contrairement à `LOAD R1, [5000]` un peu plus loin, qui elle ira vraiment chercher en mémoire).
5. **Écriture registre** — 5 est rangé dans R0.

Résultat : R0 = 5, PC = 1001 — prêt pour `1001: CALL 2000`.

#### Lexique des composants du CPU

| Composant | Signifie | Rôle |
|---|---|---|
| **PC** | *Program Counter* (Compteur Ordinal) | Contient l'adresse de la **prochaine** instruction à exécuter. S'incrémente tout seul après chaque instruction, sauf en cas de saut (`CALL`, `RET`, branchement). |
| **IR** | *Instruction Register* (Registre d'Instruction) | Contient l'instruction **en cours**, en binaire, une fois qu'elle vient d'être chargée depuis la RAM. |
| **Décodeur** | — | Lit les bits de l'IR et génère des **signaux de contrôle** (électriques, pas du texte) qui pilotent le reste du processeur : quelle opération faire, quels registres lire/écrire. |
| **ALU** | *Arithmetic Logic Unit* (Unité Arithmétique et Logique) | Effectue le calcul (addition, comparaison...) sur les valeurs qu'on lui donne — ne décide jamais rien elle-même, elle obéit au décodeur. |
| **Registres** (R0/AX, R1...) | — | Poignée de cases fixes, très rapides, directement dans le processeur — stockent les valeurs manipulées le temps d'un calcul. Pas à confondre avec la RAM. |

La section suivante (B) revient sur chacun de ces composants avec plus de détail : comment le décodeur génère ses signaux, ce qui distingue un registre de la pile, etc.


### A.3 La spécificité des langages interprétés (Python)

To do 

## B. Processeurs actuels (Calcul Sérielle)

![[Pasted image 20260730162347.png]]
*Représentation simplifiée des éléments constituant le processeur (l'organisation physique des éléments ne correspond pas à la réalité) :*

### B.1 Pipelining

#### Reprenons `add_2` comme exemple

```nasm
2000: LOAD  R1, [5000]    ; R1 = counter
2001: ADD   R1, R1, #1    ; R1 = counter + 1
2002: STORE [5000], R1    ; counter = R1
2003: ADD   R0, R0, #2    ; R0 = x + 2
2004: RET                 ; retour à l'appelant
```

**Sans pipelining** : chaque instruction fait ses 5 étapes en entier avant que la suivante démarre. `2000` occupe les cycles 1-5, puis seulement `2001` démarre (cycles 6-10), etc. Pour ces 5 instructions × 5 étapes : **25 cycles** au total.

**Avec pipelining** : dès que `2000` a fini son étape 1 (Fetch) et passe à l'étape 2 (Decode), `2001` démarre *son* Fetch au cycle suivant — chaque étape est un poste de travail séparé, occupé par une instruction différente à chaque cycle :

| Cycle        | 1       | 2       | 3       | 4       | 5       | 6       | 7       | 8       | 9       |
| ------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| `LOAD R1, [5000]`  | Fetch   | Decode  | ALU     | Mem     | WB      |         |         |         |         |
| `ADD R1, R1, #1`   |         | Fetch   | Decode  | ALU     | Mem     | WB      |         |         |         |
| `STORE [5000], R1` |         |         | Fetch   | Decode  | ALU     | Mem     | WB      |         |         |
| `ADD R0, R0, #2`   |         |         |         | Fetch   | Decode  | ALU     | Mem     | WB      |         |
| `RET`              |         |         |         |         | Fetch   | Decode  | ALU     | Mem     | WB      |

*Fetch = chargement de l'instruction, Decode = décodage, ALU = calcul, Mem = accès mémoire (lecture/écriture RAM), WB (*Write Back*) = écriture du résultat dans le registre destination — ce sont les 5 mêmes étapes vues en A.2, juste sous leur nom anglais habituel.*

Même résultat pour chaque instruction individuellement (toujours 5 étapes), mais le tout tient en **9 cycles** au lieu de 25 — c'est le débit global qui augmente, pas la vitesse d'une instruction seule.

#### C'est quoi un cycle, concrètement

Chaque colonne du tableau ci-dessus est **un cycle d'horloge**. L'horloge est un signal qui alterne 0/1 en continu ; à chaque **front montant** (passage de 0 à 1), tous les circuits font avancer leur travail d'un cran — c'est ce qui synchronise les 5 étages du pipeline entre eux. Concrètement, sur le cycle 5 du tableau : `2000` termine son écriture registre (W) pendant que `2001` fait son calcul ALU (E), que `2002` décode (D) et que `2003` est chargée (F) — tout ça au même top d'horloge, dans 4 circuits différents.

#### Le détail d'une seule instruction

Zoomons juste sur `2000: LOAD R1, [5000]`, cycle par cycle :

| Cycle | Étape | Ce qui se passe |
|---|---|---|
| 1 | **Fetch** | PC = 2000 ; l'instruction est lue en RAM, copiée dans l'IR ; PC passe à 2001. |
| 2 | **Decode** | Le décodeur lit l'IR : opération = LOAD, destination = R1, adresse = 5000. |
| 3 | **Execute** | Rien à calculer ici (l'adresse 5000 est déjà connue directement, pas besoin de l'ALU). |
| 4 | **Mem** | Accès RAM à l'adresse 5000, lecture de la valeur de `counter`. |
| 5 | **WB** | Cette valeur est rangée dans R1. |

### B.2 Unité de calcul en virgule flottante (FPU)

Le **FPU** (*Floating Point Unit*) est une unité de calcul séparée de l'ALU, dédiée à l'arithmétique sur les nombres à virgule flottante (les floats IEEE754, détaillés ci-dessous) — l'ALU classique ne sait faire que de l'arithmétique entière. Sur le schéma du processeur vu en B.1, le FPU apparaît à côté de l'ALU : les deux unités partagent l'accès aux registres, le décodeur aiguillant chaque instruction vers l'une ou l'autre selon qu'elle porte sur des entiers ou des flottants.

#### Entiers relatifs — complément à 2

Pour représenter un entier négatif en machine, on utilise le **complément à 2 (C2)** : on inverse tous les bits puis on ajoute 1.

> [!warning] Complément à 2
> Pour $-N$ sur $m$ bits : $C2(N) = \overline{N} + 1$

**Exemple.** $122 + (-7)$ sur 8 bits : $-7$ s'écrit `1111 1001` en C2. En additionnant `0111 1010` (122) et `1111 1001` (-7), on obtient `1 0111 0011` — la retenue de gauche est perdue (capacité de la machine sur 8 bits), il reste `0111 0011` = 115, le résultat correct.

#### Nombres à virgule flottante — norme IEEE754

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

### B.3 Caches (L1 et L2)

Le cache utilise la technologie **SRAM** (rapide, chère), contrairement à la RAM principale qui est en **DRAM** (dense, moins chère) — la distinction complète est détaillée plus loin, dans le chapitre Mémoire (section SRAM vs DRAM). Ici on se concentre sur son fonctionnement : comment il sait s'il a la donnée ou non.

| Mémoire | Taille typique | Emplacement |
|---|---|---|
| RAM (une barrette) | 8-32 Go | Puce séparée |
| Cache L2 | 100 Ko - 1 Mo | Sur la puce du CPU |
| Cache L1 | ~10-64 Ko | Sur la puce du CPU, collé à chaque cœur |

> [!info] Comment le cache sait s'il a la donnée : hit / miss
> Le cache stocke chaque donnée avec un **tag** (l'adresse RAM à laquelle elle correspond). Quand le processeur demande une adresse :
> - le tag est présent → **hit** : la valeur est renvoyée directement, sans toucher la RAM.
> - le tag est absent → **miss** : la donnée est allée chercher en RAM, puis une copie est enregistrée dans le cache (en écrasant une entrée existante si le cache est plein).

**Le problème.** Le temps de cycle du processeur a baissé beaucoup plus vite que le temps d'accès à la mémoire — l'écart entre les deux ne cesse de grandir.

> [!warning] Temps d'accès moyen avec cache
> $$t_{glob} = t_c + (1-h)\, t_m$$
> où $t_c$ = temps d'accès au cache, $t_m$ = temps d'accès à la mémoire, $h$ = taux de succès (*hit rate*).

Un cache atteint typiquement 80-90% de taux de succès, grâce à deux propriétés de **localité** : temporelle (une donnée récemment utilisée a de bonnes chances d'être réutilisée bientôt) et spatiale (une donnée à l'adresse A a de bonnes chances qu'une donnée à une adresse voisine soit utilisée bientôt aussi).

![[Pasted image 20260730182818.png]]
*Principe du cache : une mémoire rapide et petite placée entre le processeur et la mémoire principale, invisible du point de vue du processeur (hit = renvoi direct, miss = va chercher en mémoire principale puis stocke une copie dans le cache).*

### B.4 Processeurs superscalaires

Un processeur **superscalaire** exécute plusieurs instructions simultanément. Ça demande de détecter quelles instructions peuvent s'exécuter en parallèle (analyse des dépendances), de disposer de plusieurs unités de calcul, et de gérer le **renommage de registres** (registres logiques du programme vs registres physiques du processeur, via un **ROB — ReOrder Buffer** qui garantit que les instructions terminent dans l'ordre du programme malgré une exécution interne désordonnée).

![[Pasted image 20260802135314.png|163]]
caption. Eg de processer superscalaire de 1 coeur; superscalaire = comme tu le vois au lieu d'avoir que un seul ALU on en a 4

OK

![[Pasted image 20260730182758.png]]
*Plusieurs instructions progressent en parallèle dans le pipeline, chacune avec un léger décalage.*

![[Pasted image 20260730182739.png]]
*Le ROB (FIFO) mémorise l'ordre des instructions du programme et fournit l'équivalent de registres physiques additionnels — les instructions sortent dans l'ordre du programme même si elles s'exécutent dans le désordre en interne.*

> [!info] 💡 Le lien avec le GPU (à développer dans [[02_Stanford]])
> Tout ce chapitre D explique pourquoi un cœur CPU est "cher" : prédiction de branchement, exécution superscalaire dans le désordre, renommage de registres — beaucoup de circuiterie dédiée à accélérer **une seule** instruction séquentielle (optimisé pour la **latence**). Un cœur GPU fait l'inverse : il renonce à presque tout ça (pas de prédiction sophistiquée, pas d'exécution dans le désordre) pour rester minuscule et être répété par milliers, tous exécutant la même instruction sur des données différentes (optimisé pour le **débit**). C'est exactement le tradeoff qui rend le GPU adapté à la multiplication matricielle (des millions d'opérations identiques et indépendantes) et mauvais pour du code séquentiel avec plein de branchements.

---

## C. Mémoire

### Notion de hiérarchie mémoire

Plus une mémoire est rapide, plus elle est petite et chère ; plus elle est grande, plus elle est lente et bon marché.

![[Pasted image 20260730182705.png|436]]
*Pyramide de la hiérarchie mémoire : registres → cache → mémoire principale → mémoire de masse (vitesse et coût par bit décroissants, capacité croissante de haut en bas).*

| Niveau | Techno | Rôle |
|---|---|---|
| **Registres** | — | Quelques octets, dans le CPU — la plus rapide. |
| **Cache** (L1, L2...) | SRAM | Copie rapide d'une petite partie de la RAM (vu en B.3). |
| **Mémoire principale** | DRAM | La RAM classique — où vit le programme en cours d'exécution. |
| **Mémoire de masse** | Flash / disque dur | Stockage permanent, gros volume, lent (SSD, HDD). |

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

## D. Calcul Parallèle
### D.1 GPU (Graphical Processing Unit)



### E. TPU (Tensor Processing Unit)

