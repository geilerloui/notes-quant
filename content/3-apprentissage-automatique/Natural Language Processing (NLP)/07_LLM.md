

# Introduction : ce que couvre ce fichier

L'essentiel de la recherche académique sur les LLMs se concentre sur deux axes : l'**architecture** et l'**algorithme d'entraînement / la loss** — les deux sont couverts dans [[06_Représentations contextuelles]] (Transformer, BERT, GPT). Mais construire un LLM de bout en bout, en pratique, demande trois autres briques largement ignorées par la recherche académique : la **donnée**, l'**évaluation**, et les **systèmes** (l'infrastructure de calcul).

| Axe | Couvert par la recherche académique | Couvert dans ce fichier |
|---|---|---|
| Architecture | ✅ | — (voir 06) |
| Training algo / loss | ✅ | — (voir 06) |
| Data | peu | ✅ |
| Evaluation | peu | ✅ |
| Systems | peu | ✅ |

Le cycle de vie d'un LLM se découpe en deux grandes phases, chacune avec ses propres composantes :

| Phase | Résultat | Composantes |
|---|---|---|
| **Pretraining** | Un modèle de base type GPT-3 | Task & loss, Data, Evaluation, Scaling laws |
| **Post-training** | Un assistant type ChatGPT | Task, SFT (data & loss), RLHF (data & loss), Evaluation |

Ce fichier suit ce découpage : la partie Pretraining d'abord (Data, Scaling laws, Systems), puis Post-training (SFT, RLHF).

# I. Large Language Models
## A. Pretraining (GPT4)

### i. Task & loss et Evaluation

**Task & loss** : c'est exactement ce qui a été détaillé dans [[06_Représentations contextuelles]], section III, "Decoder-only (GPT family)" — GPT-3 est un decoder-only, la task de pretraining est la prédiction causale du token suivant, et la loss est la cross-entropy calculée à chaque position. Rien de nouveau à ajouter ici, ce fichier renvoie directement à cette section plutôt que de la dupliquer.

**Evaluation** : deux niveaux coexistent, un hérité des language models classiques et un propre aux LLMs.

- **Évaluation intrinsèque (classique)** : la **perplexité**, calculée sur un jeu de test non vu pendant l'entraînement. C'est directement liée à la cross-entropy — la perplexité est $e^{\mathcal{L}}$ où $\mathcal{L}$ est la cross-entropy moyenne par token (cf. la loss du Transformer/GPT en 06). Plus elle est basse, mieux le modèle prédit du texte naturel. Cette évaluation existait déjà pour n'importe quel language model, bien avant les LLMs.
- **Évaluation extrinsèque / downstream (propre aux LLMs)** : des benchmarks de tâches (MMLU, HellaSwag, GSM8K, HumanEval...), évalués en zero-shot ou few-shot **par prompting**, sans fine-tuning dédié. C'est ça qui est spécifique aux LLMs : un modèle comme BERT ne pouvait être évalué sur une tâche donnée qu'après un fine-tuning avec une tête dédiée (cf. 06, III, BERT). Les LLMs suffisamment grands (GPT-3 et au-delà) peuvent résoudre ces tâches directement via le prompt, sans aucun réentraînement — c'est l'in-context learning, une capacité qui émerge avec l'échelle.

En résumé : la perplexité mesure la qualité du modèle de langage en tant que tel (identique à avant les LLMs) ; les benchmarks downstream mesurent des capacités pratiques, rendues accessibles sans fine-tuning grâce à l'échelle — c'est cette deuxième couche qui est propre aux LLMs.

### ii. Data

**Idée de départ** : utiliser tout l'internet propre. **Problème** : l'internet est sale et pas représentatif de ce qu'on veut. En pratique, le pipeline de collecte ressemble à ça :

1. **Télécharger tout l'internet.** Common Crawl : 250 milliards de pages, plus d'1 PB (plus d'1e6 GB).
2. **Extraire le texte du HTML** (difficile : formules mathématiques, boilerplate).
3. **Filtrer le contenu indésirable** (NSFW, contenu toxique, PII).
4. **Dédupliquer** (URL / document / ligne) — les en-têtes, pieds de page et menus de forums reviennent systématiquement identiques d'une page à l'autre.
5. **Filtrage heuristique** : retirer les documents de mauvaise qualité (nombre de mots, longueur des mots, tokens aberrants, tokens "sales").
6. **Filtrage par modèle** : prédire si une page a des chances d'être référencée par Wikipédia (proxy de qualité).
7. **Data mix** : classer les données par catégorie (code / livres / divertissement...) et repondérer chaque domaine via les scaling laws pour maximiser la performance downstream.

Autres pratiques : *learning rate annealing* sur des données de haute qualité en fin d'entraînement, *continual pretraining* avec un contexte plus long.

L'image ci-dessous est une page HTML brute — difficile à lire, mais on distingue un peu de contenu utile tout en bas : ça illustre à quel point l'étape 2 (extraction) est complexe.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/im5.png]]

**Pourquoi c'est si central** :
- Bien collecter la donnée est probablement la partie la plus critique d'un LLM en pratique.
- Beaucoup de recherche encore à faire : comment traiter efficacement à grande échelle ? Comment équilibrer les domaines ? Utiliser de la donnée synthétique ? De la donnée multimodale ?
- Beaucoup de secret industriel autour de ce sujet — dynamique concurrentielle, risques de responsabilité liés au copyright.

**Datasets académiques courants** :
- C4 (150B tokens, 800GB)
- Dolma (3T tokens)
- The Pile (280B tokens)
- FineWeb (15T tokens)

Composition de The Pile par catégorie :
![[images/3-Apprentissage automatique/06_Natural language processing/LLM/im1.png|537]]

Datasets propriétaires (ordres de grandeur) : LLaMA 2 (2T tokens), LLaMA 3 (15T tokens), GPT-4 (~13T tokens estimés).

### iii. Scaling laws

**Intuition générale** (à vérifier / creuser) : pour améliorer un modèle, trois paramètres doivent augmenter ensemble :

1. $N$ — nombre de paramètres
2. $D$ — taille du dataset
3. $C$ — compute

Si $N$ augmente mais $D$ reste fixe, il n'y a pas d'amélioration : le modèle finit par surapprendre sur une donnée insuffisante pour sa taille. Un point notable est que les scaling laws observées à ce jour n'atteignent pas de plateau — c'est ce qui justifie la construction de clusters GPU gigantesques : investir dans plus de compute continue de payer, sans rendement décroissant visible pour l'instant.

**Empiriquement** : plus de données et des modèles plus grands → meilleure performance. Contre-intuitivement, des modèles plus grands n'impliquent pas nécessairement plus d'overfitting.

**Idée centrale** : prédire la performance d'un modèle à partir de la quantité de données et du nombre de paramètres, avant même de l'entraîner en entier.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/im2.png|495]]

**Scaling laws pour le tuning d'hyperparamètres** — le problème concret : on dispose de 10K GPUs pour un mois, quel modèle entraîner ?

- **Ancien pipeline** : tuner les hyperparamètres directement sur de gros modèles (ex. 30 modèles testés), garder le meilleur — le modèle final est entraîné aussi longtemps que chacun des modèles écartés (ex. 1 jour chacun).
- **Nouveau pipeline** : trouver des "recettes" de scaling (ex. le learning rate décroît avec la taille), tuner les hyperparamètres sur des petits modèles de tailles variées (ex. moins de 3 jours), extrapoler via les scaling laws vers les grandes tailles, puis entraîner le modèle final géant une seule fois (ex. plus de 27 jours).

**Exemple — Transformers vs LSTM** : question type, faut-il utiliser des Transformers ou des LSTM ? Les scaling laws montrent que les Transformers ont une meilleure constante *et* une meilleure pente (taux de progrès avec l'échelle) que les LSTM.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/im3.png|381]]

**Exemple — Chinchilla** : comment allouer optimalement les ressources d'entraînement entre taille du modèle et taille des données ? La réponse de Chinchilla (Hoffmann et al., 2022) : environ 20 tokens par paramètre. Mais ce ratio ne prend pas en compte le coût d'inférence — en pratique, les modèles actuels utilisent des ratios bien plus élevés (>150:1), car un modèle plus petit mais entraîné sur plus de données coûte moins cher à faire tourner ensuite, même si l'entraînement en lui-même est légèrement sous-optimal.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/im4.png]]

**Questions auxquelles les scaling laws permettent de répondre** :
- Allocation de ressources : entraîner plus longtemps ou entraîner un modèle plus gros ? Collecter plus de données ou acheter plus de GPUs ?
- Données : répéter les données sur plusieurs epochs ? Comment pondérer le mélange de domaines ?
- Algorithme : LSTM ou Transformer ? Largeur ou profondeur du réseau ?

**The Bitter Lesson** (Sutton, 2019) : les modèles s'améliorent avec l'échelle et la loi de Moore — *"la seule chose qui compte sur le long terme, c'est de tirer parti du calcul."* Conclusion pratique : ne pas perdre de temps à sur-complexifier — faire des choses simples et les faire passer à l'échelle. [Lien](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

**Entraîner un modèle SOTA** (SOTA ≈ "frontier model") — exemple avec LLaMA 3 405B, entraîné à ~40 tokens/paramètre (compute-optimal) :

- **Données** : 15.6T tokens
- **Paramètres** : 405B
- **FLOPs** : $6ND = 6 \times 15.6\text{e}12 \times 405\text{e}9 = 3.8\text{e}25$ FLOPs (environ 2x moins que le seuil de l'executive order américain sur l'IA)
- **Compute** : 16K GPUs H100, débit moyen de 400 TFLOPS
- **Temps** : $3.8\text{e}25 / (400\text{e}12 \times 3600) \approx 26\text{M}$ GPU-heures, soit $26\text{e}6 / (16\text{e}3 \times 24) \approx 70$ jours (le papier original annonce ~30M GPU-heures)
- **Coût** : compute loué + salaires ≈ \$2/h × 26M h + \$500k/an × 50 employés ≈ \$52M + \$25M ≈ \$75M (fourchette \$65–85M)
- **Émissions carbone** : 26M h × 0.7 kW × 0.24 kg/kWh ≈ 4400 tCO2eq, soit environ 2000 allers-retours JFK–LHR
- **Et le modèle suivant ?** : environ 10x plus de FLOPs

### iv. Systems

**Problème central** : tout le monde est bottleneck par le compute. Pourquoi ne pas simplement acheter plus de GPUs ? Parce qu'ils sont chers et rares, et qu'il existe des limites physiques (communication entre GPUs) — d'où l'importance de bien allouer les ressources (scaling laws) et d'optimiser les pipelines.

**Massivement parallèle** : une même instruction est appliquée sur tous les threads, mais sur des inputs différents → optimisé pour le débit (throughput), pas la latence individuelle.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im1.png|435]]
![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im2.png|438]]

**Multiplication matricielle rapide** : des cœurs spécialisés (tensor cores) sont plus de 10x plus rapides que les autres opérations flottantes.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im3.png|262]]

**Compute > mémoire & communication** : il est difficile de garder les processeurs alimentés en données assez vite.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im4.png|450]]

o

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im5.png|454]]

**Hiérarchie mémoire** : plus proche des cœurs = plus rapide mais moins de mémoire ; plus loin des cœurs = plus de mémoire mais plus lent.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im6.png|338]]

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im7.png|364]]

**Métrique : Model FLOP Utilization (MFU)** — ratio entre le débit observé et le débit théorique maximal du GPU. Atteindre 50% est déjà considéré comme très bon.

#### Précision réduite

Moins de bits → communication plus rapide et consommation mémoire plus faible. Pour le deep learning, la précision décimale importe peu, sauf pour les exposants et les mises à jour de poids.

- Les multiplications matricielles peuvent utiliser bf16 au lieu de fp32.
- Pour l'entraînement : **Automatic Mixed Precision (AMP)**.
  - Poids stockés en fp32, convertis en bf16 juste avant le calcul.
  - Activations en bf16 → gain mémoire principal.
  - Multiplications matricielles (uniquement) en bf16 → gain de vitesse.
  - Gradients en bf16 → gain mémoire.
  - Poids maîtres mis à jour en fp32 → précision complète conservée là où elle compte.

#### Fusion d'opérateurs

**Problème** : la communication (lecture/écriture mémoire) est lente.

```
x1 = x.cos()   # lit x depuis la mémoire globale, écrit x1
x2 = x1.cos()  # lit x1 depuis la mémoire globale, écrit x2
```

Chaque nouvelle ligne PyTorch déplace des variables vers la mémoire globale. **Idée** : communiquer une seule fois — c'est ce que fait `torch.compile`, en fusionnant les opérations.

![[images/3-Apprentissage automatique/06_Natural language processing/LLM/systems/im8.png|426]]

#### Tiling

![[Pasted image 20260715154231.png|559]]

Exemple : si un thread ne peut garder que 8 valeurs en mémoire, il doit relire toutes les valeurs à chaque fois (aucun cache hit).

**Idée** : grouper et ordonner les threads pour minimiser les accès à la mémoire globale (lente). Exemple avec la multiplication matricielle : calculer le produit par sous-phases pour réutiliser la mémoire déjà chargée.

1. Charger les tuiles $M_{00}$ et $N_{00}$ dans la mémoire partagée (SM)
2. Calculer les sommes partielles pour $P$
3. Charger $M_{00}$ et $N_{20}$ dans la SM
4. ...

→ réutilisation des lectures (effet de cache), réduction du nombre de lectures en mémoire globale.

#### FlashAttention

**Idée** : fusion de kernels, tiling, et recomputation appliqués spécifiquement à l'attention → 1.7x de speed-up de bout en bout.

![[Pasted image 20260715154440.png|561]]

#### Parallélisation

**Problème** : le modèle est trop gros pour tenir sur un seul GPU, et on veut utiliser autant de GPUs que possible. **Idée** : répartir mémoire et calcul entre plusieurs GPUs.

**Contexte** : entraîner naïvement un modèle à $P$ paramètres demande au moins $16P$ GB de DRAM :
- $4P$ GB pour les poids du modèle
- $2 \times 4P$ GB pour l'optimiseur
- $4P$ GB pour les gradients

Exemple : un modèle à 7B paramètres nécessite déjà 112 GB.

**Data parallelism naïf** :
1. Copier le modèle et l'optimiseur sur chaque GPU
2. Répartir les données
3. Communiquer et réduire (sommer) les gradients

Avantage : parallélisation du calcul. Inconvénient : aucun gain de mémoire — chaque GPU garde une copie complète.

![[Pasted image 20260715154539.png|316]]

**Sharding** : objectif = répartir aussi la *mémoire*, pas seulement le calcul. Idée : chaque GPU met à jour un sous-ensemble des poids, puis les GPUs se synchronisent avant l'étape suivante.

![[Pasted image 20260715154745.png|544]]

**Problème** : le data parallelism ne fonctionne que si la taille du batch ≥ nombre de GPUs. **Idée alternative** : chaque GPU prend en charge des paramètres spécifiques plutôt que de tous les mettre à jour.

- **Pipeline parallel** : chaque GPU héberge des couches différentes du réseau.

![[Pasted image 20260715154832.png|624]]
*(GPipe, Huang et al. 2018)*

- **Tensor parallel** : une seule matrice est découpée entre plusieurs GPUs, qui calculent chacun une somme partielle.

![[Pasted image 20260715154911.png|460]]
*(Megatron-LM, Shoeybi et al. 2019)*

#### Sparsité architecturale

**Idée** : les modèles sont énormes, mais chaque exemple n'a pas besoin de traverser tous les paramètres. Exemple : **Mixture of Experts** — une couche de sélection choisit quels paramètres sont "actifs" pour un input donné, gardant le même nombre de FLOPs total malgré un nombre de paramètres bien plus élevé.

![[Pasted image 20260715155006.png|697]]
*(Sparse Expert Models, Fedus et al. 2021)*

## B. Post training (ChatGPT)

Le pretraining apprend au modèle à faire du language modeling — mais ce n'est pas ce qu'on veut au final. Un modèle purement pré-entraîné comme GPT-3 brut produit un texte étrange, "à côté", quand on lui envoie un prompt de type instruction :

![[Pasted image 20260715172145.png]]

Ce qu'on veut à la place, c'est une tâche qu'on appelle **alignment** : que le LLM suive les instructions de l'utilisateur, dans le respect des choix du designer (ex. modération). Le principe est de repartir du LLM pré-entraîné et de le fine-tuner sur une petite quantité de données ciblées — c'est le **post-training**.

### i. Supervised Finetuning (SFT)

#### Méthode 1 — annotation humaine (OpenAssistant)

Les données prennent la forme de paires `"User: ... \n Assistant: ..."`. Le fine-tuning est exactement le même mécanisme de next-token prediction que le pretraining GPT (06, III) — même architecture, même chain rule, même projection + softmax sur le vocabulaire à chaque position. La seule différence est **où on calcule la loss** : la cross-entropy n'est comptée que sur les tokens de la réponse de l'assistant, jamais sur ceux du prompt utilisateur. C'est le même principe de sélection que la loss MLM de BERT (06, III) — sauf qu'ici on ne sélectionne pas des positions masquées mais les positions appartenant à la réponse.

**Problème** : il faut qu'un humain écrive à la main la question et la réponse pour chaque exemple — énorme travail. Papier : OpenAssistant, Köpf et al. 2023.

![[Pasted image 20260715172406.png|518]]

#### Méthode 2 — self-instruct (Alpaca)

Idée : utiliser un LLM pour démultiplier la collecte de données. On part d'un petit jeu de données écrit à la main (~200 exemples, le "seed dataset"), puis on demande à un modèle plus puissant (ex. GPT-4) de générer des milliers de questions/réponses dans le même style — c'est ce qu'on appelle la **distillation de modèle**. Une fois ces données obtenues, le fine-tuning se fait exactement comme avec la méthode OpenAssistant.

![[Pasted image 20260715172511.png|575]]
*(Alpaca, Taori et al. 2023)*

#### Combien de données pour le SFT ?

Très peu — de l'ordre de quelques milliers d'exemples suffisent. La figure ci-dessous montre que la qualité du modèle stagne quasiment dès les premiers milliers d'exemples, malgré l'augmentation du dataset d'entraînement.

![[Pasted image 20260715172602.png|345]]
*(LIMA, Zhou et al. 2023)*

### ii. RL from Human Feedback (RLHF)

#### Pourquoi le SFT ne suffit pas

Le SFT est du **behavior cloning** : le modèle imite les réponses humaines qu'on lui montre, ce qui pose trois problèmes.

1. **Borné par les capacités humaines** : les humains peuvent préférer des réponses qu'ils ne sont pas capables de générer eux-mêmes.
2. **Hallucination** : cloner une réponse "correcte" écrite par un humain apprend au modèle à produire des affirmations avec assurance, même sur des sujets qu'il ne connaît pas réellement — s'il ne connaît pas un fait, il apprend quand même à générer quelque chose de plausible plutôt qu'à exprimer son incertitude.

![[Pasted image 20260715172719.png]]

Exemple typique : si le modèle ne connaît pas une référence bibliographique donnée (ex. Bivens 2013), le SFT peut malgré tout lui apprendre à inventer des références à consonance plausible.

3. **Coût** : collecter des réponses idéales écrites par des humains est cher.

#### L'idée du RLHF : maximiser la préférence plutôt que cloner le comportement

Pipeline général :

1. Pour chaque instruction, générer deux réponses avec un modèle déjà correct (le modèle SFT).
2. Demander à des labellers humains de choisir la réponse qu'ils préfèrent.
3. Fine-tuner le modèle pour qu'il génère davantage de réponses du type préféré.

Concrètement : on part du modèle SFT, on lui donne une instruction ("explique-moi la bourse"), il génère deux réponses A et B, un labeller humain indique laquelle il préfère. Cette préférence sert à entraîner un petit réseau — le **reward model** — à prédire ce qu'un humain choisirait.

#### Le reward model

Le reward model est un réseau de neurone $r_\phi(x, \hat y)$ prend en entrée l'instruction $x$ concaténée à une réponse $\hat y$, et prédit un score scalaire. Pour deux réponses $\hat y_i$ et $\hat y_j$ (la A et la B) à la même instruction, la probabilité que la réponse $i$ soit préférée à la réponse $j$ suit un modèle de Bradley-Terry :

$$
p(i>j)=\frac{\exp \left(r_\phi\left(x, \hat{y}_i\right)\right)}{\exp \left(r_\phi\left(x, \hat{y}_i\right)\right)+\exp \left(r_\phi\left(x, \hat{y}_j\right)\right)}
$$

Cette expression se réécrit avec un sigmoïde, ce qui donne la loss d'entraînement du reward model — une cross-entropy binaire sur la différence des deux scores :

$$
\mathcal{L} = -\log\big(\sigma(r_\phi(x,A)-r_\phi(x,B))\big)
$$

Algorithme d'entraînement du reward model :

```
répéter {
    r_A = r_phi(x, A)
    r_B = r_phi(x, B)
    L = -log(sigma(r_A - r_B))    # si A est la réponse préférée par le labeller
    backpropagation
}
```

#### Les trois réseaux du RLHF

Une fois le reward model entraîné, le pipeline RLHF fait intervenir trois réseaux :

1. **Reward model** $r_\phi$ — figé, entraîné à l'étape précédente.
2. **Reference model** $p_{\text{ref}}$ — figé, c'est le modèle SFT tel quel.
3. **Policy model** $p_\theta$ — entraînable, initialisé à partir du modèle SFT.

Le policy model est entraîné avec **PPO** (Proximal Policy Optimization), en maximisant :

$$
\mathbb{E}_{\hat{y} \sim p_\theta(\hat{y} \mid x)}\left[r_\phi(x, \hat{y})-\beta \log \frac{p_\theta(\hat{y} \mid x)}{p_{\text{ref}}(\hat{y} \mid x)}\right]
$$

**Calcul de $p_\theta(\hat y \mid x)$** : on donne l'instruction $x$ au policy model, qui génère une phrase complète $\hat y$ mot par mot. Par la chain rule (déjà vue en 06, "Language model") :

$$
p_{\theta}(\hat{y} \mid x) = p(t_1 \mid x)\, p(t_2 \mid t_1, x)\, p(t_3 \mid t_2, t_1, x)\, \ldots
$$

Ce calcul repose exactement sur le mécanisme de masked self-attention causale de GPT (06, III) : le policy model **est** un decoder-only Transformer. Pour une séquence donnée (instruction + réponse déjà générée), chaque position produit une ligne de softmax sur tout le vocabulaire — exactement le tableau `Logits` vu en 06. Calculer $p(t_2 \mid t_1, x)$ revient simplement à lire, dans cette matrice, la probabilité softmax attribuée à $t_2$ sur la ligne correspondant à la position de $t_1$ dans la séquence $[x, t_1]$ donc par exemple $p(chat \mid le, x)$. 

**Le terme de régularisation** $\beta \log \frac{p_\theta}{p_{\text{ref}}}$ (une pénalité KL) est nécessaire car la fonction de reward peut attribuer des scores anormalement élevés à des réponses dégénérées — très longues, répétitives, absurdes — que le reward model n'a jamais vues pendant son propre entraînement. Sans cette contrainte, le policy model apprendrait à exploiter ces failles du reward model plutôt qu'à produire de bonnes réponses (*reward hacking*). La pénalité force le policy model à rester proche du modèle de référence.

**Limite pratique** : le RL est simple en théorie mais compliqué en pratique (clipping, rollouts, boucles imbriquées...) — voir AlpacaFarm, Dubois et al. 2023, qui documente ces difficultés d'implémentation.

#### DPO — Direct Preference Optimization

Innovation clé : plus besoin de reward model séparé, ni de boucle RL. Seulement deux réseaux :

1. **Reference model** $\pi_{\text{ref}}$ — figé, c'est le modèle SFT.
2. **Policy model** $\pi_\theta$ — entraînable, initialisé à partir du modèle SFT.

Les données de préférence sont notées avec une réponse gagnante $y_w$ (préférée, plus proche de ce qu'écrirait un expert) et une réponse perdante $y_l$ (plus vague ou erronée), pour une instruction $x$

Contrairement à PPO qui est du reinforcement learning, DPO est du **supervised learning** (apprentissage contrastif) : pour chaque triplet $(x, y_w, y_l)$, on calcule $\pi_\theta(y_w\mid x)$, $\pi_{\text{ref}}(y_w\mid x)$, $\pi_\theta(y_l\mid x)$ et $\pi_{\text{ref}}(y_l\mid x)$ — toutes par chain rule, comme ci-dessus — et on entraîne directement le policy model à augmenter la probabilité de la réponse gagnante et diminuer celle de la perdante, relativement au modèle de référence :

car il n'y a plus de reward function car ils ont montré que 

$$
r_{\phi}(x,y) = \beta \cdot log \frac{\pi_{\theta}(y \mid x)}{\pi_{ref}(y \mid x)} + cst
$$


$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_\theta ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_w, y_l\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_w \mid x\right)}{\pi_{\mathrm{ref}}\left(y_w \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_l \mid x\right)}{\pi_{\mathrm{ref}}\left(y_l \mid x\right)}\right)\right]
$$

![[Pasted image 20260715173557.png]]
*(DPO, Rafailov et al. 2023)*

DPO est approximativement équivalent à RLHF/PPO sous certaines conditions — les deux formulations partagent le même optimum global. DPO est beaucoup plus simple à implémenter que PPO (pas de boucle RL, pas de reward model séparé) tout en obtenant des performances comparables, ce qui en a fait le standard dans la communauté open source.

**Collecte des données** : *human in the loop*. Le modèle génère deux versions différentes d'une réponse (ex. via des paramètres de température différents), puis un humain — ou un modèle comme GPT-4 — lit les deux réponses et indique laquelle est meilleure.

#### Résultats et limites

Les papiers de ce domaine comparent souvent PPO et DPO via des courbes de **win rate** (fraction de fois où la réponse générée est jugée meilleure qu'une réponse de référence, par des humains ou un juge LLM) en fonction du budget de divergence KL par rapport au modèle de référence — l'idée étant de montrer que DPO atteint un compromis reward/KL similaire ou meilleur que PPO, avec une méthode plus simple.

![[Pasted image 20260715173712.png]]
*(à gauche : Learn to Summarize, Stiennon et al. 2020 — à droite : AlpacaFarm, Dubois et al. 2023)*

**Difficulté du jugement humain** : même avec des guidelines détaillées, il reste difficile pour un labeller de choisir entre deux réponses générées par IA quand les deux sont de qualité proche.

![[Pasted image 20260715174009.png|352]]

**Biais de longueur** : juger la correction d'une réponse est difficile, et les évaluateurs ont tendance à se laisser influencer par des éléments moins pertinents comme la forme ou la longueur. Le papier "A Long Way to Go" (Singhal et al. 2024) montre que plus on applique de RLHF, plus les réponses du chatbot deviennent verbeuses — un effet de reward hacking indirect via le biais des évaluateurs humains.

![[Pasted image 20260715174136.png]]

**Biais de distribution des annotateurs** : le papier "Whose Opinions Do Language Models Reflect?" (Santurkar et al., 2023) montre que la distribution démographique/politique des labellers utilisés influence directement les opinions reflétées par le modèle final — la question de "quel humain veut-on représenter" est centrale.

![[Pasted image 20260715174245.png]]

**Éthique du crowdsourcing** : les labelleurs doivent être exposés à beaucoup de contenu toxique dans le cadre de leur travail, ce qui pose un problème de bien-être. Une tendance actuelle (comme dans AlpacaFarm) consiste à remplacer une partie du travail humain par des juges LLM, qui présentent moins de variance que les humains dans leurs jugements — mais la norme dans la communauté open source reste d'utiliser un mélange d'humains et de LLMs pour améliorer les données de préférence.

![[Pasted image 20260715174424.png|385]]
*(AlpacaFarm, Dubois et al. 2023)*


## C. Inference

To do one day

# II. Agentic AI

Pk on dit Agentic c bizarre comme mot non ? 

## 0. LLM Reasoning

il dit pretraining on a appris a faire de l'auto complétition; finetuning on a appris au modèle à faire qu'ils répondent aux questions on prépare safety data (high quality curated dataset ) pr dire au modele comment il doti se comporter il est tune pour une task spécifique. et en fin on a vu une troisieme étape: preference tuning step le but est d'aligne le modele avec human preferences on a vu RLHF qui est divisé en deux étapes (a) apprend avec human data (b) RL stage qui sera utile today

![[Pasted image 20260715192834.png]]



relation entre RL et LLM: l'enviroenemtn c un ensemble de token qu'il peut générer, et l'action qu'il peut faire c générer un nouveau token en fct de la distribution, on peut obtenir des préférences humaines pour chaque completion. 

![[Pasted image 20260715192913.png]]


la il te parle de PPO-Clip

$$
\begin{aligned}
&\text { PPO-clip }
&L^{C L I P}(\theta)=\hat{\mathbb{E}}_t\left[\min \left(r_t(\theta) \hat{A}_t, \operatorname{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\right]
\end{aligned}
$$

et la PPO-KL penalty

$$
\text { PPO-KL penalty } \quad L^{K L P E N}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t-\beta \operatorname{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right]
$$


il dit on a vu vanilla LLM pour l'instant il prenne un prompt et te donne une answer elle connaisse structure of the text, they debug code, generate code, generate essays...  but some weakness they have limited reasoning typically if you have some sophisticated maths problem it will not come up a solution maybe it will get lost in the way since until now our model use next token prediction only it's a limitation. 2nd weakness the LLM we have has been pretrained on a huge quantity of data so it's knowledge is static limited to cutoff date eg si j'entraine le modele avant l'élection et que je demande maintenant qui sont les candidat il pourra pas y répondre. 3e pb he cannot perform actions . 4e prob: contrairement au NLP models classqiue LLM génère du texte et c'est dure à évaluer eg translation word tu utilises rule based metric like BLUE to evaluate your output 

et en gros dans ce cours on va se concentrer sur le problème n°1: limited reasoning - il dit c'est des papiers de 2024 et 2025 donc recherche très récente

--- 

on va voir c'est quoi reasoning models et on va voir comment ils sont entrainés 

reasoning = ability to solve a problem; pr résoudre ça il faut un multi step reasoning pb, eg qd ta un pb tu le décomposes en sous étapes avant de trouver la solution finale. il dit eg 

| **Not reasoning**                                                  | **Reasoning**                                          |
| ------------------------------------------------------------------ | ------------------------------------------------------ |
| "What is the course code of Stanford's Transformers & LLMs class?" | "The bear was born in 2020. How old is this bear now?" |

on va voir comment créer un modèle qui faita ce reasoning, on a vu la méthode CoT - 

Strategy. Teach model to explain its reasoning before answering (Chain of Thought)

llm trained with next token prediction objective, si le pb est difficile c peu probable qu'il soit dans le training set, il faut donc que le llm décompose le pb en tractable once et qu'il utilise les patterns sur lequel il a appris . un peu comme quand t'es étudiant tu décomposes un pb en le reliant avec des choses que ta appris avant. 

donc jusq'ua maintenant on avait vanillz LLM qui avait un input et output 

et maintneant on veut une qst en input on veut pas générer une answer direct on veut d'abord penser via une reasoning chain puis on donne la réponse donc LLM output est: Reasoning + Answer

![[Pasted image 20260715191836.png|517]]

![[Pasted image 20260715191851.png]]

Les deux blocs tout à droite: Output = Reasoning + Answer


Il dit les Reasoning models a commencé à apparaitre septembre 2024 avec OpenAI puis après les gens se sont demandés comment OpenAI a fait . Or DeepSek a réussi à rattraper la performance jan 2025 

![[Pasted image 20260715194514.png]]

on va regarder maintenant pour savoir comment un modèle un reasoning model: il dit quand on a le "Thinking" et le temps pour faire le reasoning chain ce temps de thinking et à droite on a le summary de toute la réflexion qu'il a fait, c'et un summary pour éviter les distillation par d'autre boite

![[Pasted image 20260715194823.png]]

on va parler des benchmarks pour les reasoning models

i) Coding. en gros j'ai un pb, la solution cest du code et pour vérifier on a un ensemble de fonction test peut etre test unitaire ou je sais pas quoi 

HumanEval, CodeForces, SWE-bench

ii) Maths. il dit on va juste parser le string de sortie genre le 5 que tu vois et tu le compares avec la solution qu'on cherche le ground truth

AIME, GSM8K - il dit les benchmark sont des pb pas trivial en maths et AIME maths exam pour les US maths olympiads

iii) Metrics pour quantify ces metrics

$$
pass@k=\text{"Probability that at least 1 of k attemps succeeds"}
$$

par exemple blabla

il dit y'a trois metrics:
* pass@k
* pass@1
* cons@k

----

Scaling with RL

comment construire un reasoning model ; on va donc motiver le modèle à raisonner avant de répodnre, le pb avec ça est que l'écriture des reasoning chains cest une tâche difficile surtout pour des long reasoning chains. donc si on regarde les techniques qu'on a vu jusqu'à maintenant ? SFT ? pb faut écrire a la main d'humain ces reasoning chains donc trop galère. La solution 2 est que la façon dont le modèle raisonne peut être différent de comment on raisonne . 3e fait ces reasoning tasks ont peut faire du RL en demandant "est ce qu'il a réussi à résoudre le pb ?" 

pour rappel on veut apprendre a notre modèle de résoudre ces pb compliqués et on veut qu'il raisonne avant pour faire ça; on doit avoir le reward qui dit si le reasoning chain est la; par exemple à droite on regarde les "think end token"??? 

![[Pasted image 20260715200958.png]]

et second reward check solution qu'il produit marche via du code verification comme ce qu'on a déjà discuté juste avant avec le pass@k
![[Pasted image 20260715201011.png]]



il dit training de deepseek R1-zero, il dit si on motive le système à reason on voit qu'il peut réussir 
* avec nos deux rewards qu'on a défini réussite du pb et que ce soit suffisament bon en terme d'explication

![[Pasted image 20260715201122.png|429]]

problem : not all prompts are equel certains demande de la régflexion et d'autre non; et y'a eu du travail sur comment controler la quantité de thinking que le modele does

ideas to control thinking: (chaque bullet point = un papier)
* dynamic budget : coment s'assurer que le modèle overthink a des qsts qui demande pas trop de thinking il suffit d'avoir un calssifier sur le prompt pour dire est ce que c'est un prompt d'un pb facile ou difficile
* context awareness: les llm ont une limited context window donc qd il pense il doit savoir combien de context length est restant
* budget forcing: si tu veux que ton modèle continu à apprendre tu rajoutes des tokens pour le forcer à ,réfléchir plus . ou dire que on est arrivé au temps max faut qu'il donne une réponse
* continuous thoughts: peut etre les modèles n'apprene pas sur language space mais sur un autre space genre une hidden representation qui est plus compressé


---

GRPO = Group Relative Policy Optimization

RL algorithm qui doit faire deux choses:
* Maximize advantages ie ça te dit si ce que tu produis comme réponse est better than what you would expecte
* on veut pas trop dévié du vieu modèle ie celui de l'itération précédent ou le base model celui a sft stage 

après y'a une key difference c'est le calcul du "advantage" generalized avantage simulation on avait vu cette méthode mais ça impact de avoir une value function donc c la merde

mais GRPO dit on fait autre chose, on calcule chaque reward qu'on compare avec le average of reward eg on a un pb de maths on génère plusieurs compleition for that same prompt et pour chaque prompt et compleition on va mesurer le relative measure of the reward et tt les reward des autres cdts , le bénéfice c qu'on utilise pas value functions, juste on sample plusieurs compleition pr calcule le average of the reward of the group 


GRPO a été crée y'a un an et ça prend en compte les motivations des LLM alors que DPO ça a été inventé y'a longtemps et ça marche pas frocément aussi bien dans ce contexte spécifique de LLM.

Explication: GRPO: on prend la query (q) on passe par le policy model (llm) on veut générer pas un mais plusieurs compleitions disons g compleition o_1,..,o_G on les passe via le reward model qui te donne G rewards on veut calculer le advantage 

$$
A_i=\frac{r_i-\operatorname{mean}\left(\left\{r_1, r_2, \cdots, r_G\right\}\right)}{\operatorname{std}\left(\left\{r_1, r_2, \cdots, r_G\right\}\right)}
$$

l'advnatge et pour thune le policy model et KL diveregence pour pas trop dévier du modèle intiial 

![[Pasted image 20260715202714.png]]

Pour PPO en comparaison et le GAE on va pas voir 

![[Pasted image 20260715202749.png]]

PUIS LA faut rajouter les deux blocs jaune et bleu: le jaune juste pour dire que PPO update deux blocs différents



GRPO:
$$
\begin{aligned}
& \mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(O \mid q)\right] \\
& \frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left\{\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta| | \pi_{r e f}\right]\right\}
\end{aligned}
$$

et 

$$
\begin{aligned}
&\text { PPO }\\
&\mathcal{J}_{P P O}(\theta)=\mathbb{E}\left[q \sim P(Q), o \sim \pi_{\theta_{o l d}}(O \mid q)\right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[\frac{\pi_\theta\left(o_t \mid q, o_{<t}\right)}{\pi_{\theta_{o l d}}\left(o_t \mid q, o_{<t}\right)} A_t, \operatorname{clip}\left(\frac{\pi_\theta\left(o_t \mid q, o_{<t}\right)}{\pi_{\theta_{o l d}}\left(o_t \mid q, o_{<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) A_t\right]
\end{aligned}
$$


graphe on voit que la réponse du LLM est de plus en plus longue c à caus ede la reasoning chain qui est de plus en plus sophisticated quand on le compare aux performance du modèle c'est quasiment pareil sauf tout à gauche où les perfs du modèles stabilisé et la length continue d'augmenter 

![[Pasted image 20260715203320.png]]

et donc les gens ont essayé de comprendre pk on a cet effet chelou il dit faut regarder le GRPO loss 

$$
\begin{aligned}
& \mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(O \mid q)\right] \\
& \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{\left|o_i\right|}\left|\frac{1}{\left|o_i\right|}\right|\left(\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta| | \pi_{r e f}\right]\right\}
\end{aligned}
$$


en gros il dit si ta une short sentence tes poids sont plus grand que dans une grande phrase ; si on est dans le cas que le token est dans un output de A>0 ça veut dire qu'on veut up weight ces tokens bcp plus pr les short sentences et en même temps on veut que lorsque A<0 soit ultra down weight ; il dit c le pb car ça veut dire que : if you have a short bad sentence it is worst que si tu as un mauvais long sentence. donc en gros dividing by the length of the input ça réduit encore plus le poids 

![[Pasted image 20260715203731.png|564]]


les gens ont donc focus sur le $1/|o_i|$ et ils ont crée DAPO et Dr. GRPO; et maintenant 

si on compare sur le graphe reward as a fct of output length le modele va commencer à increase its length again and again 
![[Pasted image 20260715203922.png|323]]

ok
![[Pasted image 20260715203939.png|220]]![[Pasted image 20260715203955.png|227]]


il dit autre modif qu'ils font y'a ça : bias linked to level of difficulty 

$$
\hat{A}_{i, t}=\frac{R\left(\mathbf{q}, \mathbf{o}_i\right)-\operatorname{mean}\left(\left\{R\left(\mathbf{q}, \mathbf{o}_1\right), \ldots, R\left(\mathbf{q}, \mathbf{o}_G\right)\right\}\right)}{\operatorname{std}\left(\left\{R\left(\mathbf{q}, \mathbf{o}_1\right), \ldots, R\left(\mathbf{q}, \mathbf{o}_G\right)\right\}\right)}
$$
et encourage diversity: car on veut le ratio que $-\varepsilon \le \pi / \pi_{old} \le 1+\varepsilon$ mais on veut une asymétrie entre lower et upper bound :

$$
\operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \quad \longrightarrow \quad \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon_{\text {low }}, 1+\varepsilon_{\text {high }}\right)
$$

et y'en a encore d'autre 

----

Applications:
on a le base model -> traditional model c'est que je fais du RL dessus comme on a vu dans le LLM vanilla
on a le base model -> R1 c'est le full reasoning model

on va regarde la recette pour le R1-Zero ils utilisent l'architecture MoE = Mixture of Experts et ils utilisnet un trick qui s'appelle Multi latent attention LMA de deepsek v2 on voit l'architectre prenorm, transform block .. 

(i) archi  V3-base

![[Pasted image 20260715205418.png]]

(ii) au lieu de faire alignement strategy avec SFT il commence avec le pretrained model du next token prediction et ils appliquent sur reasoning data ils regardent le reward, ils augment le reward ;;;; le papier donne le template de formatting 

ce qu'on voit en rogue c'est ce qu'on remplace par le sample prompt 

![[Pasted image 20260715205524.png]]

ils ont montré que reasoning based benchmark c'est amélioré

![[Pasted image 20260715201122.png|429]]

on a pas de SFT et on a les meilleur résultat possible ? en fait les mecs ont vu des pb sur le reasoning chains, desfois le modele mix languages et on va voir comment il le résolve

| **Benefits**                        | **Challenges**                                             |
| ----------------------------------- | ---------------------------------------------------------- |
| Reasoning abilities without any SFT | Chains of reasoning have formatting and readability issues |
maintenant qu'on a vu R1 zero, on va regarde comme ajuster les pb dans une pipeline complete qu'ils appellent R1 

(i) au début dedans ta V3-base le basique model
(ii) au lieu de faire le RL stage qu'on a vu, ils vont l'aligned it pr virer les pb qu'on a vu; ils vont faire "small scale" SFT with reasoning data ils ont utilisé les humains pour réécrire et prompt comme data pour entrainer SFT; 

(iii) GRPO with reasoning data: same RL process as with R1-zero, ils ont rajouté qq chose de nouveau à cause de la poor readibility un language consistency reward donc on a trois reward je suppose ? c en gros pr l'empecher de mélanger les langues 

formatting + accuracy + language consistency

(iv) large scale SFT with reasoning and non reasoning data environ 200k pairs general data mostly reuse V3 SFT data et +600k pairs maths, coding, logic rejection sampling of R1 so far, responses via rules + V3 judge ; en gros ils prennent des prompt pour ce reasoning based fields. en gros rejection sampling c filtering out answer qui sont pas parfaite 

(v) GRPO with reasoning and non reasoning data: R1 

il dit reasoning data c ce qu'on a vu = reward = formating + accuracy sur maths, lcoding, logic
general data mostely use v3 rl data ; reward = helpfulness + harmlessneesss

---- 

les résutlats qu'ils ont eu 

XXXXXXXXXX


Lecture 7. on va voir practical technique pour que le LLM interagisse avec le monde ext ou d'autre systèmes, pr l'instant on a vu coment il peut raisonner etc maintenant on veut l'utiliser dans un contexte de systèmes. Aujourd'hui on va voir les RAG, TOol Calling et Agents

pour rappel on avait vu 4 limitations des LLM:
* Limited reasoning -> qu'on a résolu avec la méthode d'avant
* Knowledge is static -> today
* Cannot perform actions -> today
* Hard to evaluate

--- 

disons qu'on a entrainé un modèle mais que les données sont au max y'a 1 an; et maintenant on veut avoir des info sur le winner de l'éléection or il a pas l'info; eg sur l'image on voit que la dernière fois que le modèle a appris c'était le sep 30; enfin le base model 

aussi on voit que 400,000 content window,  tokens

![[Pasted image 20260715214231.png]]


il dit si on mets tout dans le contexte, c'est que les gens ont remarqué que si tu mets des irrelevant information dans le llm ça perf va diminuer; eg qui est le gagnant de la dernière élection et tu lui mets des info par relevant le llm va être confus, les gensont fait un needle in the haystack truc bidule pour analyser les données. dans cette slide heatamap pour gpt4 la personne ;; document length vs length of the prompt; pour les prompt trop grand le llm avait du mal à récupérer l'information et surtout lorsque le fact est dans le first half of the prompt. il dit qu emême si context length est ilimited t'aurais des pb

![[Pasted image 20260715214420.png]]


Pricing is per input/output token dans l'image du haut on a $1.25 par million de token un truc comme ça 

donc pour toute ces raisons ont a besoin d'une autre approche, ce qu'il faut faire c de récupérer l'info est le mettre dans le prompt

## A. Retrieval-Augmented Generation (RAG)

Idea. Augment prompt with relevant pieces of information

high levels: questions as input (Q) faut fetch the relevant piece of information D pour output the answer (A). 


**(i) Retrieve** relevant document via similarity operation across the knowledge base

on a le prompt et on veut récupérer une relevant piece of info; donc faut récupérer des documents intéressants eg tt les doc sont dans une knowledge base

![[Pasted image 20260715214942.png|530]]

**(ii) Augment** prompt with retrieved information; une fois qu'on a les données on augmente le prompt ie je mets les infos supplémentaire au début du prompt

![[Pasted image 20260715214931.png|542]]


**(iii) Generate** reponse

![[Pasted image 20260715214919.png|612]]

---

Retrievla part:

première étape est de nettoyer les documents qu'on peut avoir besoin (ie une knowledge base) ie on collecte les documents et une fois qu'on les a on les divise en chunks ie un chunk c'est un susbset du document qui a un maximum de length ie 100 tokens max; l'idée est de calculer embedding pour chacun de ces chunks. 

Des qu'on a le knowledge base y'a des hyper paramètres qu'on doit tweak eg size of the embedding ie bigger size pour complex document en général c'est 1k-5k puis on a le chunk size on veut pas que ce soit trop small sinon le texte serait out of context en général 500 tokens ; Dernier hyperamatre combien de overlap on veut avec nos chunk 

pour retrieve les documents:
step 1: candidate retrieval le but est d'aller dans tt les chunks et de prendre qu'un mini set on veut maximizer recall 

step 2: qui est défois optional et d'avoir les top documents c'est le ranking step based on the list of potentiel relevant document les ranké ;; durant cette étape on utilise une méthode plus compute intensive car on a plus petit set 

----

on dvp plus les deux étapes

faut rajouter comment les mecs ont défini les embeddings pas compris 

step 1: semantic search using embeddings-based similarity; il dit compliqué car knowlkedge base peut etre huge donc approximate neareste neighbor method peut etre utilisé ici ça peut faire sens en gros tu partitiones lees emnbeddings afin qui'l évite de faire un naive linear search 


![[Pasted image 20260715215955.png|493]]

il dit aussi que ces embeddings sont "bi-encoder" tu passes query via un encoder il dit c un BERT like model pour encode those documents 

![[Pasted image 20260715220132.png]]

il dit qu'il ercommande de lire : sentence-BERT: sentence embeddings using siamese BERT networks reimers 2019; il dit c des embeding fait expres pour similairty operations en gros haut cosine pr les truc proche et faible sinn 

en gros il te dit que par défaut c'est cosine similarity ; alors que le sentence bert c autre chose 

Méthode 2: BM25

Méthode 3: hybrid combination of semantics and BM25 - 

---

il dit si tu as un embedding pour la qst et pour le document et que tu as le même encodeur comme en haut c'est problématique donc une solution qu'il y' a : precise zero-shot dense retrieval without relevance labels Gao et al. 2022


----
papier : introduction contextual retrieval, Anthropic team, 2024
Contextualize document chunks: prepend des morceau de texte pour qu'on comprenne les chunks , comment faire ça ? eg tu as le whole document et le chunk que tu veux contextualize tu peux ajouter du prompt caching 

![[Pasted image 20260716102613.png|566]]

il dit si tu regardes model pricing page il y'a un prix pour la partie input et cached input token ici c'est 1 tenth of the price 

![[Pasted image 20260716102951.png|376]]

#### Ranking

c appelé ranking vorie re ranking car a la première étape on a déjà une sorte de ranking. Ce qu'on fait c'est que au lieu de faire la similairty operation entre embeddings on va utiliser qq chose de plus sophistiqué au lieu de considérer query et chunk séparament; la on va mettre les deux dans le encoder et on obtient un relevance score. 

![[Pasted image 20260716103122.png|436]]

Cross encoders, SBERT.net

puis tu fais ça sur tout tes chunks tu auras un score par chunk puis tu obtiens le ranking 

![[Pasted image 20260716103237.png]]

mais il faut une métrique pour mesurer le ranking après pour savoir si c'est bien en gros c'est pareil que les domaines du search ou recommendation.


(a) Normalized Discounted Cumulative Gain at k (NDCG@k)

$$
\mathrm{DCG} @ k=\sum_{i=1}^k \frac{\mathrm{rel}_i}{\log _2(i+1)} \quad \text { with } \operatorname{rel}_i \in\{0,1\}
$$

avec

$$
\mathrm{NDCG} @ k=\frac{\mathrm{DCG} @ k}{\mathrm{IDCG} @ k}
$$

(b) Reciprocal Rank at k (RR@k)

$$
RR = \frac{1}{rank}
$$

(c) Recall at $k$ : out of all the relevant dlcument which are the ones which are actually predicted as relevant, basically which one are in the top k 

$$
\text { Recall@ } k=\frac{\mid \text { relevant in top } k \mid}{\mid \text { relevant } \mid}
$$

(d) Precision at $k$ : then you have the precision equivalent

$$
\text { Precision@k }=\frac{\mid \text { relevant in top } k \mid}{k}
$$




## B. Tool Calling

RAG on a que des mots donc unstructeured et on veut fetch les documents pr notre prompt; maintenant grosse différence on veut des données strucutré par exemple on peut reframe ce pb dans un **Function Calling**. dans le mon de tool calling et function calling on utilise souvent python car c'est facile à lire mais rien n'empêche d'avoir du Tool Calling dans un autre langage.  

Eg. tu cherche un teddy bear autour de toi, sans tools le llm serait pas ce qu'il y'aurait autour de toi, mais avec l'utilisation du **Tool Calling** il faut réfléchir à ce qu'on peut rajouter dans le preambule du prompt

![[Pasted image 20260716104336.png]]

il dit on prends un full exemple d'une fonction, ici fct definition called "fine_teddy_bear.py" il appelle un api et récupère les potentials candidates.

=> puis il te dit tu mets le fct API donc la fonction avec la doc dans le preambule ; 2nd stage il prends tes arguments et il l'exécute tu obtiens une réponse qui est compréhensible qui t'informe sur le return... name location etc. Puis tu fit cette réponse de retour dans le LLM 

![[Pasted image 20260716104835.png]]

---

Maintenant comment on entraine le modèle à être comme ça ? 

bon j'ai pas trop compris le mec a l'air de training de deux façons ? Tool prediction puis Response generation il dit c'est des SFT pairs les deux; il dit le 2e paire link all conversation history so far comme ça il sait que la premiere query cherche un teddy bear et il sait que le résultat c'est le Tool prediction. ++ il te dit vu que c'est SFT on aura plus que un exemple 

![[Pasted image 20260716132844.png]]


il dit y'a une autre méthode 2: il save manipuler python code very well, donc faut il maper une query a une fonction call ? this days you can forgo sft training et plutot faire que du training, ici dans l'exemple au lieu d'écrire sft et réécrire le modèle tu pourrais le remplacer uniquement avec une explication. une méthode c few shot learning, dans le context window tu mets des exemples de input/output ; mais si je te dis que few shot learning a des challenge pour la généralisation car faut donner des spécifique point pour input/output donc ça généralise pas sufisament a tout le langage humain. il dit read the prompt of the reasoning il dit tu prends les SFT pairs qu'on a vu dans la méthode 1 et tu les utilises comme des evaluations sets et avec un set of pairs tu run a un reasoning model pour qu'il te done l'explication. c'est une façon pour éviter de faire le hardwork. 

![[Pasted image 20260716133229.png]]

Examples of common use cases:

Information
- Web/database search
- Weather, stocks, and any other tracker
- Codebase

Computation
- Calculator
- Code execution (often in Python)
- ..and many more!

Action
- Send emails/messages and other in-computer action
- anything else within the domain of an **assistant**


----

in practice dans le context tu peux avoir plusieurs tools pas que un seul genre plein de fonctions en input quoi car tu sais pas exactement lesquels tu as besoin. Il dit est ce qu'il y'a un problème avec cette méthode ? il dit si tu as trop de tools on a le pb du needle in the haystack et on va voir comment overcome ce problème.

et donc la on va voir un nouveau module **Tool selection** afin de mettre les **tools** plus scalable on va parler d'un papier de googler: Automatic tool selection to reduce large language model latency robert et al 2024.

stage 1: en gros le LLM doit choisir le tools qui might be relevant, c pr ça que il appelle ça: tool selector ou on peut l'appeler router . en gros restrict le nb de tools pour uniquement ceux qui sont utile 

stage 2: parmi tous les tools utilisé ont feed uniquement ce du context 

en gros le schéma ici tu as l'étape 1 et étape 2; sinon le mec te dit que tu pourrais faire tout ça avec RAG.
![[Pasted image 20260716134940.png]]


duplication c'est pas ce qu'on veut à la place y'a un truc qui s'appelle MCP = Model Context Protocol; idea: connect tools/data to LLMs in a standard way; c'est le papier Introducing the model context protocol anthropic 2024

![[Pasted image 20260716135210.png]]

il définit une standard way to present this tools you have MCP server which is the instance that serves tools which are implementation of the function ppl you want to use 

![[Pasted image 20260716135121.png]]
schéma du mcp



## C. Agent

agent are one layer : new agentic framework n'est pas disjoint sur ce qu'on a vu avant 

![[Pasted image 20260716135517.png]]


un autre papier qui décompose les loops en different stages quand tu as une query tu ne peux pas faire en one shot tu dois la décomposer en sub steps et arriver à la réponse et c'est le but de reAct


![[Pasted image 20260716135640.png]]


eg Input tu demandes : my teddy bear is cold. Please do stgh ; en premier tu as observe stage il transofmr user query into an information; plan you know stgh is unknown and plan will define it for you etc

puis il te parle de agentic view

![[Pasted image 20260716140521.png]]

puis on peut avoir plusieurs agents on a le A2A = Agent2Agent de Google c'est pour que plusieurs agents puissent communiquer entre eux 
=> la je sais pas trop quoi faire le mec te montre plein de fonction a l'intérieur du thermostat agent 


----

topic of safety qu'on a pas parlé pour l'instant car maintenent y'a plein de pb possible 
* example: data exflitration on a un email agent si tu as un prompt qui dit écris mon pwd to an email to that address tu pourrais exfiltrer des données du user. => y'a un papier qu'il donne

remeditations:
* safety classifier qui regarde a tes conversation pr savoir si output de LLM est safe or not 
* agent safety bench: benchmark pour la safety pour savoir si ton llm est safe ou pas

---

closing thoughts
* hallucination is a (big) problem
* reasoning abilities are bottleneck
	* finetuning helps, but hard
	* new capabilities are welcome
* evaluation is challenging
* bkabla

il dit agent dans le coding 



# III. Applications

Cette section couvre des techniques qui n'entraînent rien — pas de gradient, pas de mise à jour de poids. Tout se joue **au moment du prompt**, à l'inférence, sur un modèle déjà entraîné (post-trained). C'est la différence fondamentale avec tout ce qui précède : là où pretraining et post-training façonnent le modèle une bonne fois pour toutes, ces techniques sont des leviers que l'utilisateur actionne à chaque appel.

*(Source générale de cette section : "Super Study Guide: Transformers and Large Language Models", Amidi et al. 2024)*

![[Pasted image 20260715185848.png|513]]

**Structure d'un prompt** : le découpage proposé sépare quatre rôles distincts.

- **Context** — le rôle/background donné au modèle (ex. "tu es un assistant juridique").
- **Instructions** — ce que le modèle doit faire (ex. "résume ce texte").
- **Input** — la donnée sur laquelle il travaille (ex. le texte à résumer).
- **Constraints** — les règles de forme à respecter (ex. "en 3 phrases, en français").

C'est un gabarit pratique de rédaction, pas un concept qui nécessite de mécanisme sous-jacent à comprendre — l'essentiel est de garder ces 4 rôles distincts en tête au moment d'écrire un prompt, plutôt que de tout mélanger dans un seul bloc de texte.

## Longueur de contexte : ordres de grandeur, et "context rot"

Avant de choisir une stratégie de prompting, il faut avoir en tête l'échelle des choses : la quantité de tokens que représentent différents types d'input (un mail, un article, un livre, un repo de code) varie sur plusieurs ordres de grandeur, tout comme la taille de contexte supportée par les différents modèles.

![[Pasted image 20260715190236.png]]

Point de vigilance : la fenêtre de contexte annoncée par un modèle (ex. 128k, 1M tokens) ne garantit pas une performance stable sur toute cette longueur. Le **context rot** (Hong et al., 2025) désigne la dégradation de performance du LLM à mesure que le nombre de tokens en entrée augmente — même bien en-deçà de la limite technique du modèle. Plus il y a de texte à traiter, plus le modèle a de mal à repérer et exploiter correctement l'information pertinente.

## In-Context Learning (ICL)

Le prompt lui-même sert de mécanisme d'apprentissage, sans toucher aux poids du modèle. Deux variantes :

| Zero-shot learning | Few-shot learning |
|---|---|
| La question est posée sans exemple | Le prompt contient des exemples input/output |
| Performance très dépendante des capacités brutes du modèle | Généralement meilleure performance |

Montrer des exemples est généralement bénéfique, mais ça a un coût : effort de construction des exemples, complexité de calcul (et donc coût), latence accrue.

*(Brown et al. 2020, "Language Models are Few-Shot Learners" — le papier GPT-3)*

## Chain-of-Thought (CoT)

Idée : expliciter le raisonnement dans le prompt améliore la performance de la réponse finale. Concrètement, l'exemple donné en few-shot ne montre pas juste la réponse, mais tout le raisonnement qui y mène — le modèle imite ce style et "réfléchit à voix haute" avant de conclure.

![[Pasted image 20260715190037.png]]

Contrepartie : plus de tokens générés → coût et latence plus élevés. En échange : meilleure performance et interprétabilité — on voit où le raisonnement dérape si la réponse finale est fausse.

*(Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models")*

En pratique, le CoT sert dès qu'une réponse nécessite plusieurs étapes de raisonnement qu'un modèle a tendance à sauter s'il doit répondre d'un coup. Quelques cas concrets où ça compte vraiment :

- **Calculs multi-étapes** : "Ce projet coûte 40k€, on a un budget de 65k€ et 3 sous-traitants à 8k€/mois sur 2 mois, combien il reste ?" — sans CoT, le modèle balance souvent un chiffre faux parce qu'il essaie de "deviner" la réponse plutôt que de dérouler le calcul.
- **Debugging de code** : demander au modèle de tracer l'exécution étape par étape avant de dire où est le bug, plutôt que de cracher direct "le bug est ligne 12" (souvent faux sans ce détour).
- **Décisions avec plusieurs contraintes** : "Quel créneau propose ce planning sachant que X est indispo le mardi, Y ne peut qu'après 14h, et la salle B est prise le matin ?" — faire lister les contraintes une par une avant de conclure évite les erreurs d'oubli.
- **Raisonnement juridique/logique** : appliquer une règle à un cas précis nécessite souvent de vérifier plusieurs conditions dans l'ordre.

## Self-Consistency

Idée : un seul raisonnement CoT peut se planter sur une étape particulière. Solution : générer plusieurs chemins de raisonnement indépendants pour la même question (en samplant avec de la température), puis agréger les réponses finales — par exemple à la majorité.

![[Pasted image 20260715190111.png|547]]

Contrepartie : le coût est multiplié par le nombre de chemins générés. En échange : robustesse accrue face à une erreur de raisonnement isolée sur un seul chemin.

*(Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning in Language Models")*

# Outlook

Sujets non couverts dans ce fichier :
- Architecture : MoE & SSM
- Décodage & inférence
- Interface & outils : ChatGPT
- Multimodalité
- Usages malveillants
- Taille de contexte
- "Data wall" — probablement le fait qu'il n'y ait plus assez de données disponibles, la majorité du web étant déjà récupérée
- Légalité de la collecte de données
