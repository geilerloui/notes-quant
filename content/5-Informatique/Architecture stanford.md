

https://www.youtube.com/watch?v=izZba4UA7iY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=5

https://cs336.stanford.edu/


il dit que appaarmnt pr coder des algos faut connaitre gpu ; et quand tu fais un algo genre flashattention etc faut connaitre comment le système fonctionne (? je suppose). IL dit qu'il est pas un system person (apparament c le nom du domaine). 

## 1. Hardware model

Point de départ : comprendre la différence entre CPU et GPU. On observe empiriquement qu'avec plus de compute, on obtient de meilleurs modèles de langage.

**Le compute**, c'est la quantité totale de calcul dépensée pour l'entraînement, mesurée en PetaFLOP/s-jours. Un FLOP (*floating point operation*) est une opération arithmétique en virgule flottante (une addition ou une multiplication). Un PetaFLOP/s-jour correspond à faire tourner $10^{15}$ opérations par seconde pendant un jour entier — c'est une unité de calcul cumulé (débit × durée).

En pratique, le compute total d'un entraînement se calcule approximativement par :
$$C \approx 6 \times N \times D$$
où $N$ est le nombre de paramètres du modèle et $D$ le nombre de tokens vus pendant l'entraînement (le facteur 6 vient du nombre de FLOPs nécessaires par paramètre et par token, forward + backward).

![[Pasted image 20260801095918.png|343]]
*Figure 1. Scaling laws (Kaplan et al., Neural Scaling Laws) : la validation loss décroît en loi de puissance avec le compute (PetaFLOP/s-jours). Chaque courbe correspond à une taille de modèle fixe (couleur = nb de paramètres) ; l'enveloppe donne la meilleure loss atteignable pour un budget de compute donné.*

C'est pour ça qu'une grande partie de la recherche vise à ajouter toujours plus de compute : plus de compute = meilleure performance des LLM.

Dans les années 90, on cherchait à faire du calcul en série de plus en plus vite en augmentant la fréquence d'horloge (clock) des CPU — plus la clock est rapide, plus les instructions s'exécutent vite. Cette tendance s'est poursuivie dans les années 2000 : le nombre de transistors a continué d'augmenter (Moore's Law), mais réduire leur taille n'accélérait plus la clock proportionnellement. Vers le milieu des années 2000, la fréquence a atteint un plafond physique (fin du Dennard scaling) : au-delà, la consommation et la chaleur deviennent ingérables.

L'axe y de ce graphe est en échelle logarithmique et superpose plusieurs métriques : nombre de transistors (milliers), performance single-thread (SpecINT ×10³), fréquence d'horloge (MHz), puissance typique (Watts), et nombre de cœurs logiques. Le nombre de transistors continue de croître exponentiellement tout du long (loi de Moore, jamais interrompue), mais avant 2005 cette croissance se traduisait directement par une clock plus rapide (Dennard scaling) — donc fréquence et performance single-thread montaient avec les transistors. Après 2005, ce lien casse (zone "End of Dennard Scaling" sur le graphe) : on ne peut plus monter la clock sans faire exploser la consommation/chaleur, donc fréquence et performance single-thread plafonnent. Les transistors supplémentaires servent alors à ajouter des cœurs plutôt qu'à accélérer un cœur unique — c'est exactement ce que confirme la bannière en bas du graphe, qui découpe la timeline en trois ères d'architecture : CISC (jusqu'à ~1985), RISC (~1985-2003), puis Multi-core à partir de ~2003, au moment même où la fréquence plafonne.

> [!note]- Qu'est-ce qu'un cœur logique ?
> Un cœur logique (*logical core*), c'est le nombre de cœurs vus et utilisables par le système d'exploitation, par opposition au cœur physique réellement gravé sur la puce.
>
> La différence vient de l'hyper-threading (Intel) / SMT (AMD) : chaque cœur physique peut être dédoublé en 2 cœurs logiques qui partagent les mêmes unités de calcul mais gèrent chacun leur propre jeu de registres et flux d'instructions. Idée : quand un thread attend (ex. une donnée en mémoire), le cœur physique bascule sur l'autre thread au lieu de rester inactif, ce qui améliore le débit global sans dupliquer tout le matériel.
>
> Un CPU à 8 cœurs physiques avec hyper-threading affiche ainsi 16 cœurs logiques.

![[Pasted image 20260801095956.png]]
*Figure 2. 42 Years of Processor Data (Hennessy & Patterson, 2018) : la fréquence et la performance single-thread stagnent après 2005, les gains viennent ensuite du nombre de cœurs (multi-core).*


Face au plafond de fréquence des CPU, l'industrie n'a eu d'autre choix que de basculer vers le parallélisme — le GPU s'inscrit dans ce même paradigme : au lieu de tout faire en série en accélérant la clock, on fait des calculs en parallèle.

Cette slide résume l'état du scaling parallèle : la performance d'un GPU seul a été multipliée par plus de 1000x en 10 ans (K20X 2012 → H100 2022). Ce gain vient de plusieurs leviers cumulés : représentation numérique plus légère (FP32 → FP16 → Int8, ~16x), instructions dédiées au calcul matriciel (tensor cores : HMMA, IMMA, ~12.5x), miniaturisation du process (28nm → 5nm, ~2.5x), et sparsity structurée (~2x). Le scaling des GPU a ainsi porté l'essentiel du scaling de compute qu'on a observé : à partir de 2017 les tensor cores apparaissent, puis viennent la sparsity structurée et les formats de nombres plus bas en précision comme moteurs principaux des gains de FLOPs.

![[Pasted image 20260730105128.png]]
*Figure 3. Single-Chip Inference Performance — gain de plus de 1000x en 10 ans (NVIDIA), décomposé par levier : représentation numérique, instructions complexes, process, sparsity.*


> [!note] Latence vs throughput
> - **Latence** : le temps que met une seule tâche pour être terminée, du début à la fin. Exemple : combien de temps ça prend pour qu'un thread produise son résultat.
> - **Throughput** (débit) : la quantité totale de travail accompli par unité de temps, en comptant tout ce qui est traité en parallèle. Exemple : combien de threads sont terminés par seconde, en tout.
>
> Le point important : optimiser l'un ne veut pas dire optimiser l'autre, et il y a souvent un compromis entre les deux.

Quelle différence fondamentale entre CPU et GPU ? Le CPU est conçu pour des applications séquentielles rapides : il a une grosse unité de control et peu d'ALU (unités de calcul), avec pour objectif la faible latence — le temps entre l'entrée d'une instruction et l'obtention du résultat doit être le plus court possible.

Le GPU, à l'inverse, optimise pour le throughput (débit agrégé) plutôt que la latence individuelle. Le diagramme de droite illustre bien la différence de philosophie : sur GPU, un thread donné peut mettre du temps à être traité — pendant qu'il attend (par exemple des données), le processeur bascule sur d'autres threads — mais l'agrégat sur l'ensemble des threads traités est bien plus élevé. C'est possible grâce à un très grand nombre de cœurs légers ("lightweight cores") : le GPU a des centaines d'unités de calcul capables de s'exécuter en parallèle, contrairement au CPU.

Dans le schéma de gauche (grille GPU), chaque ligne comporte un seul petit bloc de control (à deux tons, jaune/orange — c'est une unité unique, pas deux) qui pilote toute une rangée d'ALU (les carrés verts, ~16 par ligne dans ce schéma). Les carrés verts sont des ALU individuelles, pas des SM — le ratio 1 control pour ~16 ALU par ligne illustre le modèle SIMT (une seule unité de control envoie la même instruction à tout un groupe d'ALU qui l'exécutent en parallèle, cf. plus bas). C'est un schéma pédagogique simplifié, pas un vrai diagramme d'architecture NVIDIA : le nombre exact de lignes/colonnes n'a pas de valeur "officielle", il sert juste à illustrer le déséquilibre control/ALU par rapport au CPU (1 gros control pour seulement 4 ALU).

![[Pasted image 20260730105301.png]]
*Figure 4. CPUs optimize for a few, fast threads while GPUs optimize for many many threads : le CPU privilégie la latence (peu d'ALU, gros control, beaucoup de cache), le GPU privilégie le throughput (grille de nombreuses petites ALU, peu de cache).*

Pour visualiser le hardware réel : l'unité de base du GPU, c'est le **SM (Streaming Multiprocessor)** — un peu comme un cœur, une unité de calcul indépendante avec ses propres composants, connectée à de la mémoire partagée. Un GPU complet contient des dizaines voire des centaines de SM : le GA100 (A100) en a 128, tous capables d'accéder à la mémoire globale du GPU.

Chaque SM est lui-même divisé en 4 **sub-partitions** (appelées aussi *processing blocks*), chacune avec son propre warp scheduler, dispatch unit et register file — ce qui lui permet de gérer un warp (32 threads) indépendamment des 3 autres sub-partitions.

À l'intérieur de chaque sub-partition, chaque petite case colorée (INT32, FP32 ou FP64) est un **SP (Streaming Processor)**, aussi appelé "CUDA core" — une ALU scalaire qui exécute l'opération arithmétique d'un seul thread par cycle. Une sub-partition contient 16 SP INT32, 16 SP FP32 et 8 SP FP64, soit ~40 SP au total, en comptant les 4 sub-partitions un SM a donc 64 SP FP32 + 64 SP INT32 + 32 SP FP64.

Le **Tensor Core** n'est pas un SP : c'est une unité complètement séparée, à côté des SP dans chaque sub-partition, dédiée exclusivement au calcul matriciel (elle prend en entrée de petites matrices et fait la multiplication-accumulation en une seule opération matérielle).

![[Pasted image 20260730105625.png]]
*Figure 5. Structure d'un SM (NVIDIA GA100) : 4 sub-partitions avec warp scheduler, register file, SP (INT32/FP32/FP64), Tensor Core, et mémoire partagée L1. À droite : le GA100 complet avec 128 SM, cache L2 partagé, contrôleurs HBM2 et liens NVLink.*

On a vu le compute, mais la mémoire compte tout autant : il faut comprendre quel type de mémoire existe sur un GPU et où elle vit physiquement (détail des latences en cycles dans la Table IV de l'image).

Plus la mémoire est proche du SM, plus elle est rapide. Le L1 cache et la shared memory sont à l'intérieur même du SM. Le L2 cache est sur la puce (die) mais partagé entre tous les SM. La mémoire globale (HBM), elle, est physiquement en dehors du compute — ce sont des puces mémoire séparées, à côté du GPU sur la carte.

Pourquoi ne pas construire une puce entièrement en shared memory pour que tout soit rapide ? Parce que la SRAM (utilisée pour le cache/shared memory) coûte environ 100x plus cher par bit que la DRAM (utilisée pour la mémoire globale) et consomme bien plus d'énergie. D'où la hiérarchie mémoire : un compromis vitesse/coût/consommation. (NVIDIA a par ailleurs racheté Grok pour ses puces à très grande quantité de SRAM, mais la plupart des accélérateurs restent organisés en hiérarchie classique.)

![[Pasted image 20260730105807.png]]
*Figure 6. Hiérarchie mémoire du GPU (NVIDIA GA100) : latences d'accès (Table IV), die shot montrant les SM, partitions L2 et contrôleurs HBM2, et carte physique avec VRAM séparée du GPU.*

## 2. Software model (ou execution model)

Après le hardware, voyons le modèle logiciel du GPU — il y a 3 acteurs importants dans ce modèle d'exécution.

**Threads** : ils font le travail en parallèle. Sur GPU, les threads suivent le modèle **SIMT** (Single Instruction, Multiple Threads) : tous les threads exécutent exactement la même instruction, mais avec des inputs différents. C'est le compromis programmabilité vs efficacité du GPU.

**Blocks** : un groupe de threads. Chaque bloc s'exécute sur un seul SM, avec accès à sa propre shared memory — ce sera important plus tard quand on parlera de réutilisation des blocs (tiling).

**Warp** : l'unité de scheduling. Les threads d'un bloc sont toujours exécutés par groupes de 32 threads consécutifs (un warp). Sur l'exemple de la slide : un bloc de 256 threads est divisé en 8 warps (256/32), et chaque SM a 4 warp schedulers qui distribuent les instructions prêtes aux colonnes d'ALU correspondantes (INT32 ou FP32).

![[Pasted image 20260730110146.png]]
*Figure 7. Modèle d'exécution CUDA : un programme est découpé en blocs, chaque bloc assigné à un SM, chaque bloc divisé en warps de 32 threads, chaque warp scheduler distribuant les instructions aux ALU disponibles.*

Le code device (exécuté sur GPU) peut accéder à cinq types de mémoire, du plus local au plus global :
- **Registers** : lecture/écriture par thread — la plus rapide, chaque thread a les siens.
- **Local memory** : lecture/écriture par thread.
- **Shared memory** : lecture/écriture par bloc — utilisée pour communiquer entre threads d'un même bloc.
- **Global memory** : lecture/écriture par grid (tout le kernel) — accessible par tous les threads, mais lente.
- **Constant memory** : lecture seule par grid — peu utilisée en pratique.

Le code host (CPU), lui, peut transférer des données vers/depuis la mémoire globale et la mémoire constante du GPU — c'est le mécanisme d'*offload* pour faire entrer/sortir des données du GPU.

Chaque thread accède à son propre register, et à la shared memory de son bloc. Dès qu'on a besoin de faire transiter de l'information entre blocs différents, il faut passer par la mémoire globale — beaucoup plus lente. **C'est le point clé** : dès qu'on sort de la shared memory, tout devient lent — donc regrouper intelligemment les blocs pour maximiser l'usage de la shared memory est l'enjeu central de ce cours.

![[Pasted image 20260730110422.png]]
*Figure 8. Hiérarchie mémoire côté logiciel CUDA : registers et local memory par thread, shared memory par bloc, global memory et constant memory par grid, avec transferts host ↔ GPU.*

On parle brièvement des TPU même si l'essentiel du cours porte sur GPU — le TPU est dans la même famille d'accélérateurs.

Structure d'un TensorCore TPU (schéma abstrait) : le **Scalar Unit** joue le rôle d'un control, il dispatch les instructions vers le VPU et le MXU. Le **VPU (Vector Unit)** fait les opérations élément par élément (ex. activations) et charge les données dans le MXU. Le **MXU (Matrix Multiply Unit)** fait les multiplications matricielles — c'est lui qui pilote les FLOP/s de la puce (équivalent du Tensor Core côté GPU). La mémoire suit la même logique de hiérarchie : Smem/Vmem proches et rapides, puis HBM (High Bandwidth Memory) qui stocke poids, activations, états de l'optimiseur, nouveau batch de données.

**Structure commune aux deux** : un control léger, une grosse unité de matmul rapide, et de la mémoire rapide proche du compute.

**Différences principales** : (1) le networking entre accélérateurs (vu dans le cours sur le parallélisme) — c'est la plus grosse différence ; (2) pas de notion de warp côté TPU, seulement des blocks, avec des arbitrages différents entre opérations matmul et non-matmul. À noter aussi qu'un GPU a plus de SM qu'un TPU n'a de Tensor Cores, mais les deux atteignent une performance matmul similaire.

![[Pasted image 20260802141526.png]]
*Figure 9. Architecture abstraite d'un TensorCore TPU : Scalar Unit (control), VPU + Vmem, MXU (matrix multiply), reliés à la HBM.*

Pour chaque concept côté GPU, il existe un équivalent côté TPU (SM ↔ Tensor Core, Warp Scheduler ↔ VPU, CUDA Core ↔ VPU ALU, SMEM ↔ VMEM, Tensor Core ↔ MXU, HBM ↔ HBM) — la circuiterie est assez proche, la différence principale est le nombre d'unités. Le GPU a beaucoup plus d'unités mais plus petites (ex. H100 : 132 SM, 528 Tensor Cores), alors que le TPU a des unités bien plus grosses mais moins nombreuses (ex. TPU v5p : 2 Tensor Cores, 8 MXU) — le TPU vise moins de flexibilité mais du matmul massif par unité.

Même logique de mémoire des deux côtés : hiérarchie entre mémoire rapide proche du compute et mémoire lente (HBM), avec une unité de multiplication matricielle dédiée. Les concepts qu'on va voir (tiling, etc.) se transposent donc sans problème de l'un à l'autre.

![[Pasted image 20260730110809.png]]
*Figure 10. Correspondance terminologique GPU ↔ TPU, et comparaison chiffrée H100 vs TPU v5p.*


Le GPU est très efficace car il permet un scaling facile de la multiplication matricielle : ajouter plus de SM augmente directement le throughput, tant que la bande passante mémoire suit.

Il est relativement facile à programmer grâce au modèle SIMT : une seule instruction est dispatchée (par l'Instruction Decoder / Warp Scheduler) à de nombreux CUDA cores en parallèle — on n'a pas besoin de coder chaque thread individuellement les uns après les autres, contrairement à une approche purement séquentielle (on retrouve un peu l'esprit de la programmation fonctionnelle).

Les threads sont "légers" et peuvent être arrêtés et redémarrés à volonté : le scheduler peut décider quels threads traiter à chaque instant, ce qui permet de facilement basculer d'un job à un autre — utile notamment quand un thread est bloqué (stalled) en attente de données. C'est ce qu'illustre encore le diagramme GPU (High Throughput) vs CPU (Low Latency) déjà vu en Figure 4.

![[Pasted image 20260730111146.png]]
*Figure 11. Scalabilité (ajout de SM), modèle SIMT (1 instruction → plusieurs threads via l'Instruction Decoder/Warp Scheduler), et flexibilité du scheduling des threads GPU.*

On a vu les avantages du hardware model du GPU. Historiquement, avant même l'arrivée des Tensor Cores, ce massif parallélisme était déjà utile pour le calcul scientifique — un papier ancien ("Fast Matrix Multiplies Using Graphics Hardware") montrait qu'on pouvait programmer des shaders pour faire de la multiplication matricielle "à la main", en détournant le hardware graphique.

Aujourd'hui ce n'est plus nécessaire : depuis le V100, NVIDIA intègre un circuit spécialisé — le Tensor Core — dédié exactement à ce calcul, qui est ensuite devenu l'opération reine du deep learning. Le graphe ci-dessous montre l'écart de throughput qui en résulte entre les opérations parallélisables mais non-matmul et les opérations matmul : avant le V100 les deux courbes sont confondues, après le V100 le matmul décolle et se détache complètement (jusqu'à ~1000 TFLOP/s sur H100, contre ~60 TFLOP/s en non-matmul). C'est pour ça que les futures architectures ML auront toujours une unité de multiplication matricielle plus de 10x plus rapide que ce qu'on peut faire en floating point "générique" sur CPU.

![[Pasted image 20260730204639.png|275]]
*Figure 12. Matmul vs. non-matmul FLOPS across GPUs : le décollage du matmul à partir du V100 (introduction des Tensor Cores).*

Différents composants scalent à des rythmes très différents dans le temps. Ce graphe (source : [AI and Memory Wall](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8)) le montre bien : les FLOPS matériels (en gris) augmentent de 60000x en 20 ans (×3 tous les 2 ans) — un rythme largement plus rapide que Moore's Law classique. Mais la bande passante mémoire DRAM (en vert) n'augmente que de 100x sur la même période (×1.6 tous les 2 ans), et la bande passante d'interconnect entre GPU (en bleu, NVLink/PCIe) encore moins, 30x (×1.4 tous les 2 ans).

Le problème : cet écart entre la vitesse de calcul et la vitesse de la mémoire ne cesse de grandir. Beaucoup de code qu'on va voir dans ce cours est donc bottleneck par la mémoire et la communication plutôt que par le compute — ce qui va rendre le choix et l'optimisation du hardware de plus en plus compliqué à mesure que cet écart se creuse.

![[Pasted image 20260730204934.png|453]]
*Figure 13. Scaling des FLOPS matériels vs bande passante mémoire (DRAM) et interconnect sur 20 ans : le compute croît bien plus vite que la mémoire — le "memory wall".*

Avant le V100, il n'y avait pas de circuit dédié à la multiplication matricielle — c'est le Tensor Core qui a introduit ça. Avant, il fallait coder ce calcul "à la main" (via des shaders détournés, cf. plus haut) ; maintenant il existe un circuit spécialisé exprès pour ça.

Il évoque aussi la distance physique et la SRAM : plus une donnée doit voyager loin physiquement sur la puce (ou entre puces), plus l'accès est lent — c'est un rappel du principe "plus proche du SM = plus rapide" vu en Figure 6.

Le paysage du hardware pour l'inférence est particulièrement complexe aujourd'hui. Certains systèmes utilisent la "prefill/decode disaggregation" : la phase de prefill (très gourmande en multiplication matricielle) tourne sur une puce, et la phase de decode (limitée par la bande passante mémoire) tourne sur une autre puce, chacune optimisée pour son propre profil de calcul. D'autres modèles (ex. un modèle chinois "Step") vont jusqu'à répartir l'attention sur une puce et le MLP sur une autre. En résumé : plus l'écart entre mémoire et compute se creuse, plus ces architectures hétérogènes deviennent nécessaires — et c'est encore pire en inférence qu'à l'entraînement.

## Tricks

Voyons maintenant les tricks pour rendre le code GPU rapide — les composants de base pour optimiser un workload.

Pour se motiver, un des objectifs de cette section est d'expliquer ce plot : en x, la taille de la matrice carrée multipliée ; en y, le throughput obtenu (TFLOP/s). Globalement le throughput monte quand la matrice grandit (plus de mémoire disponible pour amortir le calcul), mais il y a aussi des motifs étranges qu'on va chercher à comprendre — les sauts entre bandes de courbes (liés au tiling) et la dispersion à droite (liée à la wave quantization).

![[Pasted image 20260730210657.png|425]]
*Figure 14. FLOPs achieved for square matmuls : throughput en fonction de la taille de matrice, avec les effets de tiling (sauts entre bandes) et de wave quantization (dispersion à droite) annotés.*

### The roofline model

Le plot précédent ressemble à ce modèle classique. Jusqu'à un certain seuil d'intensité opérationnelle, on est **memory-bound** : peu importe le compute disponible, le throughput ne peut pas dépasser ce que la bande passante mémoire permet (partie diagonale). Au-delà de ce seuil, on devient **compute-bound** : le throughput plafonne au maximum que les ALU peuvent fournir (partie plate).

Plus la mémoire utilisée est rapide (registers > shared memory > main memory), plus le plafond est haut et plus le seuil memory-bound/compute-bound est atteint tôt. Pour que du code GPU tourne bien, il faut être sur la partie plate (throughput max) et éviter la partie diagonale — donc augmenter l'intensité de calcul (le nombre d'opérations par byte chargé) pour atteindre ce seuil. C'est exactement l'objectif des 6 tricks qu'on va voir.

Les deux points bleus sur le graphe illustrent bien le principe avec des cas réels : la **multiplication de matrice dense** (diamant) a une intensité opérationnelle très élevée car chaque valeur chargée en mémoire est réutilisée énormément de fois dans les calculs — le point se situe donc loin à droite, dans la zone plate, proche du throughput maximal des ALU. La **multiplication de matrice sparse** (rond) a une intensité bien plus faible : une grande partie des valeurs chargées sont des zéros ou nécessitent de la logique d'indexation supplémentaire, sans apporter beaucoup de calcul utile en retour — le point reste dans la zone diagonale (memory-bound), avec un throughput réel bien en dessous du plafond malgré un volume de "travail" potentiellement plus important. C'est pour ça que le sparse matmul est notoirement difficile à accélérer efficacement sur GPU : le hardware est optimisé pour des opérations à haute intensité comme le dense matmul.

![[Pasted image 20260730210647.png]]
*Figure 15. Roofline model : throughput (GFLOPs) en fonction de l'intensité opérationnelle (FLOPs/byte), avec les plafonds pour registers, shared memory, main memory GPU et CPU. Le dense matmul (diamant) atteint le plafond compute-bound ; le sparse matmul (rond) reste memory-bound.*

### Trick n°1 : control divergence

Rappel du modèle SIMT : tous les threads d'un warp exécutent exactement la même instruction en même temps. Que se passe-t-il si on écrit un `if` dans du code GPU ? Contrairement au CPU, où un thread choisit simplement une branche et l'exécute, le GPU ne peut pas faire ça : quand les threads d'un warp se séparent entre `if` et `else`, le warp **diverge**. Le matériel exécute alors les deux branches l'une après l'autre (via du masquage), pendant qu'une partie des threads reste inactive en attendant son tour, jusqu'à ce que tout le warp reconverge.

Ça simplifie beaucoup la programmation (pas besoin de gérer explicitement le parallélisme), mais le coût est réel : il y a des moments où une partie du GPU ne fait rien. C'est pour ça qu'il faut éviter les `if` dans le code GPU chaud. Exemple typique : un ReLU peut se coder comme une multiplication par 0 ou par max(x,0) plutôt que par un `if`, car un branchement peut coûter l'équivalent de 2 cycles d'horloge perdus.

![[Pasted image 20260730210631.png]]
*Figure 16. Control divergence : quand un warp rencontre un `if`/`else`, les threads divergent et exécutent les deux branches séquentiellement avant de reconverger.*

### Trick n°2 : lower precision computation

NVIDIA passe énormément de temps sur ce levier : baisser la précision numérique (FP32 → FP16/BF16 → INT8 → FP8...). L'idée est simple — si on a 2x moins de bits à manipuler par valeur, on résout mécaniquement une partie du problème mémoire (moins de bytes à charger/écrire pour la même donnée). C'est le levier "Number Representation" de la Figure 3, qui à lui seul apporte ~16x de gain.

Exemple concret avec un ReLU sur un vecteur de taille $n$ : en FP32, chaque élément nécessite 1 lecture + 1 écriture (si $x<0$) à 4 bytes, pour 1 comparaison + 1 FLOP — soit une intensité de 8 bytes/FLOP. En FP16, la mémoire par élément est divisée par 2 (2 bytes), donc l'intensité tombe à 4 bytes/FLOP :

```
(Float 32)
Memory access: 1 read (x), 1 write (if x<0), 4 bytes/valeur
Operations: 1 comparaison, 1 FLOP
Intensité: 8 bytes / FLOP

(Float 16)
Memory access: 1 read (x), 1 write (if x<0), 2 bytes/valeur
Operations: 1 comparaison, 1 FLOP
Intensité: 4 bytes / FLOP
```

(Note : l'intensité en bytes/FLOP est l'inverse de l'"operational intensity" flops/byte vue dans le roofline model — ici on veut la minimiser, pas la maximiser.)

En pratique, le low precision est un vrai art compliqué. Pour la multiplication matricielle, le Tensor Core fonctionne en low precision de la façon suivante : on downcast les opérandes avant de faire le produit, puis la somme (accumulation) se fait en pleine précision, et le résultat est renvoyé en FP32. La difficulté n'est pas juste "baisser la précision partout" — c'est de savoir quelle opération peut se permettre quelle précision, et dans quel format (un softmax en FP32 ne se comporte pas comme une multiplication matricielle en FP16, par exemple). Il a fallu plusieurs années de recherche pour arriver à un entraînement stable en low precision, et une grande partie du travail consiste justement à déterminer quelles métriques/opérations peuvent être downcastées sans casser l'entraînement.

![[Pasted image 20260730210619.png]]
*Figure 17 (= Figure 3 réutilisée). Gains de performance par levier : la représentation numérique (FP32→FP16→INT8) apporte à elle seule ~16x.*

**Comment le Tensor Core gère le mix de précisions.** Deux inputs 16-bit (FP16/BF16) entrent dans une multiplication en pleine précision (*full precision product*), puis la somme s'accumule avec un accumulateur FP32 (*more products* qui s'ajoutent), pour ressortir en FP32. C'est ce qui permet de stocker/transférer en 16-bit (économie mémoire) sans perdre en précision sur l'accumulation, qui est la partie la plus sensible aux erreurs d'arrondi.

Toutes les opérations ne tolèrent pas le même niveau de précision :
- **Peuvent tourner en 16-bit (FP16/BF16)** : les multiplications matricielles, la plupart des opérations pointwise (ReLU, tanh, add, sub, mul).
- **Ont besoin de plus de précision (FP32/FP16)** : additionner de petites valeurs à de grandes sommes peut créer des erreurs d'arrondi importantes — donc les opérations de réduction (sum, softmax, normalisation).
- **Ont besoin de plus de range/plage dynamique (FP32/BF16)** : les opérations pointwise où $|f(x)| \gg |x|$ (exp, log, pow), et les fonctions de loss.

![[Pasted image 20260802143255.png]]
*Figure 18. Fonctionnement d'un Tensor Core en précision mixte : produit en pleine précision, accumulation FP32, et classification des opérations selon la précision qu'elles tolèrent.*

**Les frontières du low precision.** Dès qu'on peut couper la précision en 2, les gens le font. FP8 a été la frontière suivante après FP16/BF16 — et là, il n'y a pas de format canonique unique : E4M3 (4 bits d'exposant, 3 de mantisse) privilégie la précision, E5M2 (5 bits d'exposant, 2 de mantisse) privilégie la plage dynamique.

En FP8 "classique", toutes les activations sont en 8 bits avec un seul scaling factor FP32 global pour éviter l'underflow (avec si peu de bits d'exposant, on sort vite de la plage représentable sans ce facteur d'échelle). Le problème : une seule matrice peut contenir des valeurs de magnitudes très différentes, donc un seul scaling factor global est sous-optimal. D'où **MXFP8** (format introduit avec Blackwell) : au lieu d'un seul scaling factor, on en a plusieurs, un par sous-matrice colorée — chaque scaling factor est lui-même stocké en 8 bits (format E8M0, qui n'a qu'un exposant, donc des puissances de 2 pures).

Le problème que ça introduit : la **transposée**. Avant, transposer une matrice était trivial. Avec des scaling factors par bloc, la transposée n'a pas le même pattern de scaling — il faut requantizer toute la matrice pour respecter les 32 patterns de blocs. En pratique, MXFP8 crée donc deux copies quantizées : une pour la matrice originale, une pour sa transposée.

![[Pasted image 20260802143333.png]]
*Figure 19. Formats FP16/BF16/FP8 (E4M3, E5M2) bit à bit, et principe des scaling factors multiples MXFP8 (Blackwell), avec le problème de la transposée qui nécessite deux copies quantizées (forward/backward).*

**En pratique, l'entraînement en FP8** est fait par beaucoup de frontier labs, mais demande d'être très prudent : on ne quantize que certaines couches jugées "sûres", et trouver lesquelles reste largement empirique (trial and error) — dans le schéma d'un bloc Transformer, seules certaines parties (Q/K/V, projection, FC1, FC2) sont converties en MXFP8, le reste restant en BF16. Le gain est réel côté matrices (20-30% d'économie), mais on n'obtient pas un speedup de 2x à cause du coût de toute cette quantization/requantization.

![[Pasted image 20260802143345.png]]
*Figure 20. Application de MXFP8 dans un bloc Transformer : seules certaines couches (Q/K/V, projections, FC) sont quantizées, poids et activations quantizés séparément pour forward (FPROP) et backward (DGRAD, WGRAD).*

**La prochaine frontière, c'est MXFP4** : seulement 16 valeurs représentables au total (structured block de 4 bits), avec un scaling factor E4M3 partagé par groupe de 16 valeurs. Un papier a montré un entraînement possible en FP4, mais pas encore de vrai modèle de production sur ce format — même si les prochaines générations de modèles s'orientent clairement vers FP4.

![[Pasted image 20260802143406.png]]
*Figure 21. MXFP4 : les 16 valeurs représentables (structured block 4 bits), avec 1 scaling factor E4M3 par groupe de 16.*

Le choix des scaling factors dépend des librairies utilisées (parfois basé sur le max, parfois sur le min de la sous-matrice). Il y a aussi du travail en cours du côté de la sparsity structurée, notamment avec les architectures MoE (Mixture of Experts).

### Trick n°3 : operator fusion

Idée conceptuelle simple. On peut voir le GPU comme une usine : un entrepôt de mémoire d'un côté, une usine de compute de l'autre, reliés par un tapis roulant qui fait des allers-retours. On est bottleneck par ce tapis roulant (la bande passante mémoire), pas par l'usine elle-même.

![[Pasted image 20260802143747.png]]
*Figure 22. Le GPU comme une usine : mémoire ↔ compute reliés par un tapis roulant. Plus l'usine grossit (plus de compute), plus le tapis roulant devient le goulot d'étranglement s'il ne scale pas au même rythme.*

Le problème s'aggrave si on enchaîne plusieurs opérations : chaque aller-retour au tapis roulant consomme de la bande passante mémoire dans les deux sens (duplex bidirectionnel). Mieux vaut une seule "giant factory" qui fait tout d'un coup, pour ne payer le coût de bande passante que deux fois (un aller, un retour) au lieu d'une fois par opération.

![[Pasted image 20260802143800.png]]
*Figure 23. Naïve (non-fused) : chaque opération fait un aller-retour séparé vers la mémoire. Fused kernel : une seule lecture initiale et une seule écriture finale, tout le reste se passe dans le compute.*

C'est exactement ce qui se passe avec un calcul comme $\sin^2(x) + \cos^2(x)$ : naïvement, chaque étape (sin, cos, pow, pow, add) est un "kernel" séparé qui lit/écrit en mémoire globale à chaque appel — 5 lancements de kernel CUDA, donc beaucoup de lectures/écritures inutiles.

![[Pasted image 20260802143812.png]]
*Figure 24. Calculer sin²(x) + cos²(x) naïvement lance 5 kernels CUDA distincts, chacun avec son propre aller-retour mémoire.*

La solution : écrire du code GPU qui lit une seule fois depuis la mémoire globale, fait toutes les opérations à l'intérieur du SM, puis réécrit le résultat final en mémoire globale — un seul kernel qui fait tout. C'est la **fusion d'opérateurs** (*operator fusion*) : le graphe de calcul est "compressé" en un seul appel CUDA. Les compilateurs comme `torch.compile` ou JAX font ça automatiquement.

![[Pasted image 20260802143824.png]]
*Figure 25. Avant/après fusion (TorchInductor) : les 5 opérations pointwise sont fusionnées en un seul kernel CUDA.*

### Trick n°4 : recomputation

Rappel : pendant le forward pass, on stocke les activations, et pendant le backward pass on les réutilise pour calculer les gradients (backpropagation classique — on stocke les valeurs "forward" $f_i$ en jaune, et on calcule les dérivées "backward" $g_i$ en vert).

![[Pasted image 20260802143839.png]]
*Figure 26. Backpropagation classique : on stocke les activations (jaune) au forward, on calcule les Jacobiens (vert) au backward.*

Ici on ne regarde plus ça du point de vue des maths, mais du point de vue système : comment minimiser la mémoire utilisée ? Exemple avec 3 sigmoïdes empilées : le forward pass classique fait 1 lecture + 3 écritures (on stocke chaque activation intermédiaire s1, s2), et le backward fait l'inverse (3 lectures + 1 écriture) — soit 8 accès mémoire au total pour très peu de calcul réel, une intensité arithmétique très faible.

![[Pasted image 20260802143900.png]]
*Figure 27. 3 sigmoïdes empilées : le forward stocke chaque activation intermédiaire (3 écritures), le backward les relit (3 lectures) — 8 accès mémoire au total, très mauvais pour la perf.*

Alternative : ne rien stocker au forward (juste 1 lecture + 1 écriture), et au backward, **recalculer** à la volée les activations intermédiaires à partir de $x$ au moment où on en a besoin. Contre-intuitif — jeter le résultat d'un calcul pour le refaire plus tard — mais dans un monde où le compute est bon marché et la mémoire est chère, c'est justement optimal : on passe à 5/8èmes des accès mémoire d'origine.

![[Pasted image 20260802143914.png]]
*Figure 28. Recomputation : le forward ne stocke rien (1 lecture, 1 écriture), le backward recalcule les sigmoïdes à la volée à partir de x — 5/8èmes des accès mémoire de la version classique.*

### Trick n°5 : memory coalescing

La mémoire globale (DRAM) est structurée pour être lue par "bursts" : accéder à une seule adresse ne renvoie pas qu'une seule valeur, mais tout un bloc de mémoire contigu d'un coup (une "burst section", typiquement 128 bytes ou plus en pratique). C'est une conséquence du fonctionnement physique de la DRAM : activer l'amplificateur d'une ligne/colonne renvoie toute la ligne/colonne en une fois.

![[Pasted image 20260802143939.png]]
*Figure 29. La DRAM (mémoire globale) est lue par bursts : accéder à une adresse renvoie tout le bloc contigu (burst section) auquel elle appartient.*

Un accès mémoire est dit **coalescé** si tous les threads d'un warp accèdent à des adresses qui tombent dans la même burst section — dans ce cas, une seule requête DRAM suffit pour tout le warp, au lieu d'une requête par thread. Rappel : un warp est un groupe de 32 threads consécutifs qui s'exécutent ensemble, donc leurs accès mémoire ont intérêt à être groupés.

![[Pasted image 20260802143957.png]]
*Figure 30. Accès coalescé : les threads T0-T3 d'un warp accèdent tous à la même burst section — une seule requête DRAM suffit.*

Application concrète aux matrices : pour du matmul, on veut lire par gros blocs contigus. Pour une matrice stockée en row-major, des threads qui avancent le long des **lignes** ne sont **pas** coalescés (chaque thread saute d'une ligne à l'autre, donc des burst sections différentes à chaque itération) — alors que des threads qui avancent le long des **colonnes** peuvent l'être, car ils accèdent à des éléments contigus en mémoire.

![[Pasted image 20260802144010.png]]
*Figure 31. Pour une matrice row-major, un accès par colonnes (threads T0-T3 lisant chacun une colonne) reste dans la même burst section à chaque itération — un accès par lignes ne l'est pas.*

### Trick n°6 : tiling (le plus gros des tricks)

Le tiling a un impact majeur sur la performance. L'idée : grouper et ordonner les accès mémoire au maximum, en passant par la shared memory plutôt que par la mémoire globale à chaque fois.

Reprenons la multiplication de matrice. Dans l'approche naïve, chaque élément de $M$ et $N$ est relu $N$ fois depuis la mémoire globale (une fois par thread qui en a besoin) — les accès ne sont ni coalescés, ni dédupliqués : $M_{0,0}$ et $N_{1,0}$ par exemple sont lus plusieurs fois par différents threads.

![[Pasted image 20260802144022.png]]
*Figure 32. Multiplication de matrice naïve : chaque élément est relu plusieurs fois depuis la mémoire globale par différents threads (ex. M0,0 relu par thread(0,0) et thread(0,1)), sans coalescing.*

Le tiling consiste à découper les matrices en petites **tuiles** (*tiles*), à charger chaque tuile une seule fois dans la shared memory, puis à faire tous les calculs qui en ont besoin depuis cette copie rapide, avant de charger la tuile suivante. Le calcul se fait "par phases" : (1) charger les tuiles $M_{0,0}$ et $N_{0,0}$ en shared memory (SHM), (2) calculer les sommes partielles pour $P$ avec cette tuile, (3) charger les tuiles suivantes ($M_{0,0}$ et $N_{2,0}$), et ainsi de suite. Avantage : les lectures répétées se font depuis la shared memory (rapide) et non la mémoire globale (lente), et les accès peuvent être coalescés.

![[Pasted image 20260802144034.png]]
*Figure 33. Tiling : chaque tuile est chargée une fois en shared memory (SHM), puis réutilisée pour calculer toutes les sommes partielles qui en ont besoin, par phases successives.*

Vue plus abstraite : la boucle externe itère sur les tuiles, la boucle interne itère sur les éléments à l'intérieur d'une tuile, pour produire une tuile temporaire du résultat.

![[Pasted image 20260802144049.png]]
*Figure 34. Structure en double boucle du tiling : boucle externe sur les tuiles (taille T), boucle interne sur les éléments, matrices de taille N.*

**Les maths du tiling.** Sans tiling, chaque élément d'entrée est lu $N$ fois depuis la mémoire globale. Avec tiling de taille $T$, chaque élément d'entrée n'est lu que $N/T$ fois depuis la mémoire globale (et $T$ fois depuis la shared memory, rapide) — soit une réduction d'un facteur $T$ des accès à la mémoire globale.

**Complexité n°1 : le tile sizing.** Le tiling introduit des subtilités. Exemple : avec des tuiles de 128×128 sur une matrice 256×256, tout se divise parfaitement en 4 tuiles pleines (cas (a)). Mais si on augmente la dimension de la matrice de seulement 1 (257×257), on se retrouve avec des tuiles fines quasi vides sur les bords (cas (b), *tile quantization*) : 6 blocs de threads sont lancés, dont 2 gaspillent l'essentiel de leur travail. Le bon choix de taille de tuile dépend de plusieurs facteurs (accès mémoire coalescé, taille de la shared memory disponible, divisibilité des dimensions de la matrice) et n'a pas de réponse universelle — c'est pour ça que PyTorch propose un compilateur "max-autotune" qui benchmark plusieurs tailles de tuiles pour trouver la plus rapide empiriquement.

![[Pasted image 20260802144112.png]]
*Figure 35. Tiling avec des tuiles 128×128 : (a) cas parfait, matrice 256×256 divisée en 4 tuiles pleines ; (b) tile quantization, matrice 257×257 génère 6 blocs dont 2 gaspillent l'essentiel de leur travail.*

**Complexité n°2 : l'alignement avec les bursts.** Si la taille des tuiles ou de la matrice ne s'aligne pas avec les burst sections de la DRAM, tenter de lire une tuile déclenche beaucoup plus de requêtes mémoire que nécessaire (layout "aligné" = 1 seule tuile propre en une requête ; layout "non aligné" = 2 tuiles "sales" nécessitant plusieurs requêtes partielles). La solution dans ce cas est le padding, pour faire coïncider les frontières des tuiles avec celles des burst sections.

![[Pasted image 20260802144150.png]]
*Figure 36. Aligned vs unaligned layout : une tuile mal alignée avec les burst sections de la DRAM nécessite plusieurs requêtes mémoire au lieu d'une seule.*

**Retour au plot de départ.** On a maintenant tout ce qu'il faut pour l'expliquer : il faut suffisamment de travail par lecture mémoire pour saturer le compute (intensité arithmétique suffisante). En colorant les courbes selon la divisibilité de la taille de matrice par $K$ (2, 8, 16, 32), on voit clairement que les tailles non divisibles par de grandes puissances de 2 ont un moins bon throughput — les tailles divisibles par 16 ou 32 performent nettement mieux, car elles s'alignent avec les burst windows et permettent des lectures coalescées. C'est le tiling qui a un impact majeur ici, à travers cet alignement.

![[Pasted image 20260802144207.png]]
*Figure 37. FLOPs achieved for square matmuls, coloré par divisibilité de la taille par K : les tailles divisibles par 16 ou 32 (violet/rouge) atteignent un bien meilleur throughput grâce à l'alignement mémoire.*

**Wave quantization.** Reste le comportement périodique observé (la courbe orange qui zigzague). Ça se produit par exemple entre les tailles 1792 et 1793 : avec une tuile de taille 256×128, une matrice 1792×1792 se découpe exactement en $1792/256 \times 1792/128 = 7 \times 14 = 98$ tuiles. Mais en passant à 1793×1793, il faut désormais $8 \times 15 = 120$ tuiles (une tuile de plus dans chaque dimension à cause de l'arrondi). Or un A100 n'a que 108 SM : il ne peut pas exécuter les 120 tuiles en une seule "vague" (*wave*), il lui en faut une seconde — d'où une chute de throughput chaque fois qu'on passe ce genre de seuil, avant que ça remonte progressivement jusqu'au prochain saut.

![[Pasted image 20260802144234.png|419]]
*Figure 38. Wave quantization : franchir un seuil de taille de matrice (ex. 1792→1793) ajoute une tuile supplémentaire dans chaque dimension, ce qui peut dépasser le nombre de SM disponibles (108 sur A100) et forcer une seconde vague d'exécution — d'où le comportement en dents de scie du throughput.*

## FlashAttention

FlashAttention (Dao et al.) est l'exemple qui rassemble tout ce qu'on vient de voir : on a maintenant tous les outils pour le comprendre.

FlashAttention accélère radicalement l'attention par rapport à une implémentation PyTorch naïve. Le gain est spectaculaire : le graphe de gauche montre qu'on peut traiter des séquences de plus en plus longues, le tableau du milieu montre que le gain vient essentiellement d'une réduction massive du volume de données transférées (HBM R/W : 40.3 GB → 4.4 GB, soit ~9x moins, pour un runtime divisé par ~5.7). D'après le papier, deux techniques déjà vues permettent ce résultat : **tiling** et **recomputation**, combinées pour atteindre une complexité sous-quadratique en accès HBM.

![[Pasted image 20260802144258.png]]
*Figure 39. FlashAttention vs PyTorch naïf sur GPT-2 : un seul kernel fusionné remplace 5 étapes séparées (matmul, dropout, softmax, mask, matmul), avec 9x moins d'accès HBM et un runtime ~5.7x plus rapide.*

Rappel du calcul d'attention : 3 multiplications de matrices (Q, K, V) avec un softmax entre les deux — $\mathrm{softmax}(XQK^\top X^\top) \cdot XV$.

![[Pasted image 20260802144309.png]]
*Figure 40. Calcul de l'attention : 3 multiplications matricielles avec un softmax intermédiaire, produisant un output de taille n×d.*

FlashAttention applique le tiling : la Figure 1 du papier montre que le calcul KQ se fait tuile par tuile, chunk by chunk, en écrivant le résultat dès qu'une tuile est prête. La difficulté : le softmax est une opération **globale** (il faut normaliser sur toute la ligne), donc on ne peut pas naïvement le calculer tuile par tuile sans "recoller" les morceaux entre eux.

![[Pasted image 20260802144334.png]]
*Figure 41. La Figure 1 du papier FlashAttention est littéralement du tiling appliqué à la multiplication matricielle KQV. Mais comment gérer le softmax, qui est une opération globale ?*

Le trick clé est l'**online softmax**. Le softmax classique (à gauche) soustrait le max global de tous les éléments puis normalise — il faut donc connaître tous les éléments avant de commencer. L'online softmax (à droite) calcule un softmax normalisé au fur et à mesure : dès qu'un nouvel élément plus grand apparaît, on met à jour le max (*swap out the max*) et on réajuste l'accumulateur en conséquence (somme télescopique). L'important : ça permet de calculer le softmax tuile par tuile, en sauvegardant seulement des résultats partiels en mémoire globale, sans jamais avoir besoin de toute la ligne d'un coup.

![[Pasted image 20260802144409.png]]
*Figure 42 (Milakov & Gimelshein, 2018). Pour garder trace du max en cours de route, on le met à jour incrémentalement et on construit une somme télescopique — ce qui permet de calculer le softmax tuile par tuile.*

FlashAttention 2 illustre bien comment tout s'assemble : les blocs en pointillés sont en SRAM, les blocs pleins en mémoire globale (HBM). Le produit scalaire KQ se fait par tiling matriciel dans la SRAM, puis on calcule l'exponentielle et le softmax "courant" (running softmax) au sein de cette même tuile, en gardant les sommes partielles dans la shared memory/registres. Une fois tout ça calculé, il devient facile de multiplier par V et de diviser par l'accumulateur — ce qui permet de traiter l'opération d'attention en petits chunks de tuiles, avec fusion des opérations, et de recalculer (recomputation) ce qui est nécessaire pendant le backward pass plutôt que de tout stocker.

![[Pasted image 20260802144451.png|569]]
*Figure 43 (Dao, 2023). Le forward pass complet de FlashAttention : (i) calcul tuile par tuile des produits scalaires (S), (ii) fusion de l'opérateur exponentiel, (iii) calcul tuile par tuile du softmax via la somme télescopique online. Le backward pass (non détaillé ici) recalcule les activations tuile par tuile de la même façon.*

