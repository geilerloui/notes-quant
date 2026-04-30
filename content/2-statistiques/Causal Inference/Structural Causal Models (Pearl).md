## 🧭 Big picture

**Le but du framework Pearl.** Comme Rubin, on veut estimer des effets causaux. Mais Pearl le fait via un autre langage : les **graphes causaux (DAGs)** et l'opérateur **`do(·)`**, plutôt que les *potential outcomes (résultats potentiels)*. Les deux frameworks sont mathématiquement équivalents, mais Pearl est plus adapté pour **raisonner sur la structure** d'un système complexe.

**L'idée centrale.** On modélise le monde comme un **graphe orienté acyclique (DAG)** où :
* les nœuds sont des variables (X, Y, Z, …)
* les flèches encodent les **relations causales directes** ($X \to Y$ veut dire "X cause directement Y")

Ce DAG, augmenté de **mécanismes structurels** (les fonctions qui décrivent comment chaque nœud dépend de ses parents), s'appelle un **Structural Causal Model (SCM)**.

> 💡 **Un SCM contient strictement plus d'information qu'une distribution jointe $P(X, Y, Z)$.** Plusieurs SCMs différents peuvent donner la même distribution observationnelle. C'est pour ça qu'on ne peut pas faire de causalité avec juste des stats classiques — il faut injecter des hypothèses structurelles.

**La Ladder of Causation (Pearl).**

| Niveau          | Question                                      | Outil mathématique       | Exemple                                                   |
| --------------- | --------------------------------------------- | ------------------------ | --------------------------------------------------------- |
| 1. Association  | Si je *vois* X, à quoi m'attendre pour Y ?    | $P(Y \mid X)$            | "les fumeurs ont plus de cancer"                          |
| 2. Intervention | Si je *force* X, qu'arrive-t-il à Y ?         | $P(Y \mid \text{do}(X))$ | "si je fais arrêter de fumer, le cancer baisse-t-il ?"    |
| 3.Contrefactuel | Si X *avait été* différent, qu'aurait-on vu ? | $P(Y_x \mid X', Y')$     | "ce patient mort, aurait-il survécu sans le traitement ?" |

* **Niveau 1** = stats classiques.
* **Niveaux 2-3** = inférence causale, *inaccessibles* sans hypothèses structurelles.

**Statistical model vs Causal model (le schéma de Peters).**

![[Pasted image 20260429192838.png|482]]

Le modèle causal **subsume** le modèle probabiliste : si tu as le SCM, tu peux dériver $P$. L'inverse est faux.

**Les outils principaux du framework Pearl.**
* **DAG** : représentation graphique du modèle causal
* **`do(X = x)`** : opérateur d'intervention — *forcer* X à la valeur $x$, briser ses flèches entrantes
* **d-séparation** : règle graphique pour lire les indépendances conditionnelles dans un DAG
* **Backdoor criterion** : règle graphique pour identifier les confondeurs à ajuster
* **do-calculus** : ensemble de 3 règles pour transformer $P(Y \mid \text{do}(X))$ en quantités estimables à partir d'observations

**Rubin vs Pearl.**

| | Rubin (potential outcomes) | Pearl (DAGs + do-calculus) |
|---|---|---|
| Outil principal | $Y_i(0), Y_i(1)$ | DAG + `do(·)` |
| Questions naturelles | ATE, ATT, LATE | Identifiabilité, ajustement, effets directs/indirects |
| Force | Estimation, inférence statistique | Raisonnement structurel, modélisation de systèmes complexes |
| Communauté | Économétrie, biostat | CS / IA, épidémiologie |
| Quand l'utiliser | Tu as un design (RCT, IV, RDD…) et tu veux estimer | Tu as un système avec plein de variables et tu veux savoir *quoi ajuster* |

Les deux frameworks sont équivalents mathématiquement. Pearl a démontré qu'on peut traduire n'importe quel problème de l'un vers l'autre. En pratique, on utilise les deux selon le contexte.

**Le fil rouge à garder en tête.**
* Avec Rubin tu poses la question : *"si on randomisait X, quelle serait la moyenne de Y ?"*
* Avec Pearl tu poses la question : *"étant donné comment le monde est structuré (DAG), peut-on identifier $P(Y \mid \text{do}(X))$ à partir de données observationnelles ? Et si oui, comment ?"*

Pearl est donc plus puissant pour **réfléchir à ce qui est identifiable avant même d'estimer**.

---

## I - L'opérateur `do(·)` et la G-formula

### A. L'intervention vs l'observation

**Différence fondamentale.** Conditionner sur un événement (ce qu'on fait en stats classiques) ≠ intervenir sur une variable (ce que veut la causalité).
* $P(Y \mid X = x)$ → "parmi les gens *où on observe* $X = x$, distribution de Y"
* $P(Y \mid \text{do}(X = x))$ → "si on *force* $X = x$, distribution de Y"

Les deux sont différentes dès qu'il y a un confondeur. Exemple : *parmi les gens qui prennent le médicament*, ils sont peut-être déjà malades (confondeur). Si je *force* tout le monde à le prendre, j'efface ce biais.

**Définition formelle.** L'intervention `do(X := x)` consiste à :
1. Effacer toutes les flèches **entrant** dans X dans le DAG
2. Forcer X à la valeur x
3. Laisser le reste du DAG inchangé

Le DAG modifié s'appelle le **mutilated graph**.

### B. La G-formula (covariate adjustment)

**Formule centrale.** Si $\mathbf{Z}$ est un *valid adjustment set* pour $(X, Y)$ :

$$
p^{\text{do}(X := x)}(y) = \int_{\mathbf{z}} p(y \mid x, \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}
$$

(somme à la place de l'intégrale dans le cas discret).

**Lecture intuitive.** Pour chaque "tranche" de confondeurs $\mathbf{z}$ :
* on calcule $P(Y \mid X = x, \mathbf{Z} = \mathbf{z})$ — ce qu'on *voit* dans cette tranche
* on pondère par $P(\mathbf{Z} = \mathbf{z})$ — la fréquence de cette tranche dans la population

Et on agrège. C'est exactement le même principe que la G-formula vue chez Rubin, mais ici Pearl la *dérive* du DAG.

> 💡 **Lien avec la régression linéaire vue chez Rubin.** La formule ci-dessus est une *identité mathématique* qui ne dit pas comment estimer ses ingrédients. En pratique :
> * Si **$\mathbf{Z}$ est discret avec beaucoup de données** (ex: kidney stones, $S \in \{\text{small}, \text{large}\}$), on prend juste les fréquences empiriques pour $P(y \mid x, \mathbf{z})$ et $P(\mathbf{z})$.
> * Si **$\mathbf{Z}$ est continu** (ex: l'âge dans l'exemple Rubin formation/salaire), on ajuste un modèle paramétrique — typiquement une régression linéaire — pour $\mathbb{E}[Y \mid X, \mathbf{Z}]$, et on moyenne ses prédictions sur les valeurs observées de $\mathbf{Z}$.
>
> C'est *la même G-formula*, juste deux **estimateurs** différents pour les mêmes ingrédients.

### C. Valid adjustment set : quoi ajuster ?

C'est la question pratique : **quelles variables mettre dans $\mathbf{Z}$ ?**

Réponse partielle (suffisante) : **les parents de X dans le DAG sont toujours un valid adjustment set**.

$$
\mathbf{Z} = \mathbf{PA}_X \implies \text{ajustement valide}
$$

⚠️ Ce n'est pas le seul. Le **backdoor criterion** (qu'on verra plus tard) donne la condition complète. Mais en pratique, ajuster sur les parents marche.

> 💡 C'est là où Pearl ajoute de la valeur par rapport à Rubin : **il te donne un critère graphique** pour savoir *quoi* ajuster, là où Rubin te demande juste "ajuste sur tous les confondeurs" sans dire comment les identifier.
è
### D. Exemple : kidney stones et le paradoxe de Simpson

**Setup (DAG).**

```
        S (taille du calcul)
       / \
      ↓   ↓
      T → R
   (traitement) (récupération)
```

S est un confondeur : il influence à la fois le choix du traitement et la récupération.

**Données.**

| | Treatment A | Treatment B | Total |
|---|---|---|---|
| Small stones | 81/87 = 0.93 | 234/270 = 0.87 | 357/700 = 0.51 |
| Large stones | 192/263 = 0.73 | 55/80 = 0.69 | 343/700 = 0.49 |
| **Total** | 273/350 = **0.78** | 289/350 = **0.83** | |

**Lecture naïve** (sans ajustement) :
* Treatment B paraît meilleur : 0.83 > 0.78
* Pourtant **dans chaque sous-groupe** A est meilleur : 0.93 > 0.87 (small) et 0.73 > 0.69 (large)
* C'est le **paradoxe de Simpson**

**Application de la G-formula** (ajustement par S) :

$$
\begin{aligned}
P^{\text{do}(T := A)}(R = 1) &= \sum_S P(R = 1 \mid T = A, S) \, P(S)\\
&= 0.93 \times 0.51 + 0.73 \times 0.49 = \mathbf{0.832}
\end{aligned}
$$

$$
P^{\text{do}(T := B)}(R = 1) = 0.87 \times 0.51 + 0.69 \times 0.49 = \mathbf{0.782}
$$

**Conclusion : Treatment A est meilleur** (0.832 > 0.782). Le naïf 0.83 vs 0.78 était trompeur parce que B était plus souvent prescrit pour les small stones (qui guérissent mieux toutes choses égales par ailleurs).

**Calcul de l'ATE.** L'effet causal moyen du traitement A par rapport à B :
$$
\text{ATE} = P(R = 1 \mid \text{do}(T = A)) - P(R = 1 \mid \text{do}(T = B)) = 0.832 - 0.782 = +0.05
$$
En moyenne, choisir A plutôt que B augmente la probabilité de guérison de **5 points de pourcentage**. C'est exactement la quantité que Rubin notait $\mathbb{E}[Y(A) - Y(B)]$ — même nombre, vocabulaires différents.

> ⚠️ **Comparaison avec la lecture naïve.** Si on avait calculé l'"ATE" à partir des fréquences brutes :
> $$
> P(R=1 \mid T=A) - P(R=1 \mid T=B) = 0.78 - 0.83 = -0.05
> $$
> On aurait conclu que A est *pire* que B de 5 points. Le signe lui-même est inversé. C'est la dangerosité du paradoxe de Simpson : sans ajustement, on prend la mauvaise décision clinique.

**Intervention ≠ Conditionnement.** Ici :
* $P(R = 1 \mid T = B) = 0.83$ (observation)
* $P(R = 1 \mid \text{do}(T = B)) = 0.782$ (intervention)

C'est la signature d'un confondeur : **les deux quantités diffèrent**.

### E. Placeholder exemple mais en continu ?

## I bis - Le SCM formel

### A. Définition

Un **Structural Causal Model (SCM)** est un couple $\mathcal{S} = (\mathbf{S}, P_\mathbf{N})$ où :
* $\mathbf{S} = \{S_1, \dots, S_d\}$ est un ensemble de **$d$ équations structurelles** :
$$
S_j : \quad X_j := f_j(\mathbf{PA}_j, N_j), \quad j = 1, \dots, d
$$
* $P_\mathbf{N} = \prod_j P_{N_j}$ est la distribution jointe des bruits, **mutuellement indépendants**.

Avec $X_j$ variable endogène, $\mathbf{PA}_j$ ses parents causaux dans le DAG, $f_j$ mécanisme déterministe, $N_j$ bruit exogène. Le DAG induit s'obtient en traçant $X_i \to X_j$ ssi $X_i \in \mathbf{PA}_j$, et l'**acyclicité** est imposée.

> ⚠️ **Notation : $X_j$ ≠ treatment.** Ici $X_j$ désigne *n'importe quelle* variable du DAG. Différent des sections I/II où $X$ tout court désigne le treatment.

> 💡 **Le `:=` n'est pas une égalité.** C'est une **affectation** (comme en programmation). Asymétrique : on ne peut pas inverser. C'est *là* qu'est encodée la causalité.

**D'où vient le DAG ?** Deux cas de figure :
* **Connaissance du domaine** (cas standard) : tu poses le DAG à la main avec la théorie économique, médicale, physique. C'est ce qu'on fait dans 99% des cas.
* **Causal discovery** (cas exceptionnel) : trop de variables pour théoriser à la main (ex : 10 000 gènes), donc on essaie de récupérer le DAG depuis les données via LiNGAM/ANM/ICP. Sous des hypothèses fortes, avec peu de garanties pratiques.

> 💡 **Slogan de Pearl : "No causes in, no causes out".** Sans hypothèse causale apportée par toi, rien n'est causal. Les données te donnent des distributions ; le DAG vient de toi.

### B. Les 3 mondes du coefficient causal

C'est *exactement* ce qui perd tout le monde quand on voit un SCM linéaire $Y := \alpha X + N$.

**Monde 1 — La nature.** Le vrai mécanisme du monde a une certaine valeur $\alpha^*$. Personne ne la connaît.

**Monde 2 — Le bac à sable pédagogique.** Quelqu'un (Peters dans son notebook) *invente* un SCM avec coefficients explicites pour simuler des données. Les coefficients sont **posés à la main** pour pouvoir vérifier que la méthode marche.

**Monde 3 — La vraie vie.** Tu as un dataset. Tu **postules un DAG** (la structure des flèches, par théorie). Tu **n'inventes pas les coefficients** — tu les **estimes** à partir des données via régression / matching / IV / etc.

| Quoi | D'où ça vient |
|---|---|
| Structure du DAG | Toi (théorie du domaine) |
| Vrais coefficients | La nature (inconnus) |
| Coefficients de Peters | Posés dans son bac à sable |
| Coefficients estimés $\hat\alpha$ | Tes données + une méthode |

### C. Équation structurelle vs régression linéaire (LE piège)

Cas linéaire bivarié.

**Régression linéaire** (descriptive) : $Y = \alpha X + \varepsilon$, avec $\alpha = \text{Cov}(X,Y)/\text{Var}(X)$, $\varepsilon$ orthogonal à $X$ par construction. **Toujours calculable**, **symétrique**, **aucune notion de causalité**.

**Équation structurelle** (générative) : $Y := \alpha X + N_Y$. $\alpha$ est un **paramètre causal** fixé par le mécanisme du monde. $N_Y$ est un **bruit exogène** qu'on **suppose** indépendant de $X$. **Asymétrique** : $X$ cause $Y$.

> ⚠️ **Même tête mathématique, sémantiques radicalement différentes :**
> * Régression : *en aval* des données. Le coefficient est calculé.
> * Équation structurelle : *en amont* des données. Le coefficient est un paramètre du monde.

**Quand est-ce qu'elles coïncident ?** Si le SCM est $Y := \alpha X + N_Y$ avec $X \perp N_Y$, alors `lm(Y ~ X)` estime $\alpha$ correctement. **Mais s'il y a un confondeur caché** ($X \not\perp N_Y$), la régression naïve donne un coefficient biaisé. C'est exactement le cas où il faut faire une G-formula ou un IV.

> 💡 **Mémo.** OLS est un *outil d'estimation neutre*. Selon les variables qu'on y met, il peut estimer (a) une corrélation aveugle, ou (b) un effet causal. *La régression elle-même ne sait pas.* C'est *le DAG* qui te dit comment l'utiliser.

## II - d-séparation et backdoor criterion

### A. Pourquoi on a besoin de ça

Dans la section I on a dit "les parents de X sont un valid adjustment set". C'est *suffisant* mais pas *nécessaire* — il y a souvent d'autres choix possibles, et parfois les parents ne sont pas observables (c'est le cas typique).

La **d-séparation** est l'outil graphique qui permet de :
1. **Lire les indépendances conditionnelles** dans un DAG sans calculer
2. **Décider quoi conditionner** pour bloquer les chemins de confusion
3. **Justifier formellement** un valid adjustment set via le **backdoor criterion**

> 💡 La promesse de Pearl : **toute question d'identification causale se réduit à un problème de connectivité dans un graphe**.

### B. Les 3 motifs élémentaires

Tout chemin dans un DAG est une succession de 3 motifs. Comprendre ces 3 motifs = comprendre toutes les indépendances conditionnelles.

**Motif 1 — Chain (chaîne)**
```
X → Z → Y
```
$Z$ "transmet" l'information de $X$ vers $Y$. Sans conditionner sur $Z$, $X$ et $Y$ sont dépendants. **Conditionner sur $Z$ bloque le passage** : $X \perp Y \mid Z$.

> Ex : tabac → goudron → cancer. Si on connaît le niveau de goudron, savoir si la personne fume n'apporte plus rien sur le cancer.

**Motif 2 — Fork (fourche / cause commune)**
```
X ← Z → Y
```
$Z$ est une cause commune. $X$ et $Y$ sont **corrélés sans qu'aucun ne cause l'autre** (corrélation spurieuse). **Conditionner sur $Z$ bloque le passage** : $X \perp Y \mid Z$.

> Ex : Z = "richesse du quartier" cause à la fois X = "nombre d'arbres" et Y = "faible criminalité". Sans conditionner sur Z, on voit une corrélation arbres↔criminalité qui n'est pas causale.

**Motif 3 — Collider (collision / cause commune d'effets)**
```
X → Z ← Y
```
$Z$ est un effet commun. **Sans conditionner**, $X$ et $Y$ sont **indépendants** : $X \perp Y$.
**Conditionner sur $Z$ ouvre le passage** : $X \not\perp Y \mid Z$. C'est le fameux *selection bias* / *Berkson's paradox*.

> Ex : X = talent, Y = beauté, Z = "est célèbre" (acteur célèbre = talentueux OU beau). Dans la population générale, talent et beauté sont indépendants. Mais *parmi les acteurs célèbres*, ils deviennent négativement corrélés (si t'es pas talentueux, t'es là parce que t'es beau).

> ⚠️ **C'est l'inverse de l'intuition stats classique** : ici conditionner *crée* de la dépendance au lieu de la réduire.

**Récap.**

| Motif | Sans conditionnement | Avec conditionnement sur Z |
|---|---|---|
| Chain $X \to Z \to Y$ | dépendants | **bloqué** |
| Fork $X \leftarrow Z \to Y$ | dépendants | **bloqué** |
| Collider $X \to Z \leftarrow Y$ | **indépendants** | ouvert (crée une dépendance) |

### C. d-séparation : la règle complète

**Définition.** Deux nœuds $X$ et $Y$ sont **d-séparés** par un ensemble $\mathbf{Z}$ si **tous les chemins** entre $X$ et $Y$ sont **bloqués** par $\mathbf{Z}$.

Un chemin est **bloqué** par $\mathbf{Z}$ si :
* il contient une **chain** $\to W \to$ ou une **fork** $\leftarrow W \to$ avec $W \in \mathbf{Z}$, OU
* il contient un **collider** $\to W \leftarrow$ avec $W \notin \mathbf{Z}$ ET aucun descendant de $W$ dans $\mathbf{Z}$

**Théorème fondamental.** d-séparation dans le DAG ⟹ indépendance conditionnelle dans la distribution.
$$
X \perp_d Y \mid \mathbf{Z} \quad \text{(graphique)} \implies X \perp Y \mid \mathbf{Z} \quad \text{(probabiliste)}
$$

C'est ce qui te permet de **lire les indépendances directement sur le graphe**, sans calculer aucune probabilité.

### D. Backdoor criterion : le quoi-ajuster définitif

**Idée.** Pour estimer l'effet causal de $X$ sur $Y$, on veut bloquer tous les **chemins de confusion** (= chemins qui passent par "derrière" $X$, via une flèche entrant dans $X$). On ne veut **pas** bloquer les chemins causaux directs $X \to \dots \to Y$.

**Définition (Backdoor criterion).** Un ensemble $\mathbf{Z}$ satisfait le critère du backdoor pour $(X, Y)$ si :
1. **Aucun nœud de $\mathbf{Z}$ n'est un descendant de $X$** (sinon on bloque l'effet causal)
2. **$\mathbf{Z}$ bloque tous les chemins backdoor** entre $X$ et $Y$ (= chemins qui partent de $X$ par une flèche entrante)

Si $\mathbf{Z}$ satisfait le backdoor criterion, alors :
$$
P(Y \mid \text{do}(X = x)) = \sum_{\mathbf{z}} P(Y \mid X = x, \mathbf{Z} = \mathbf{z}) \, P(\mathbf{Z} = \mathbf{z})
$$

C'est la **G-formula justifiée graphiquement**.

> 💡 **Les parents de $X$ satisfont toujours le backdoor criterion** (c'est ce qu'on disait en I.C). Mais ce n'est pas le seul ensemble valide — d'où l'intérêt du critère général.

> 💡 **Lien avec l'unconfoundedness de Rubin.** Chez Rubin on suppose $\{Y(0), Y(1)\} \perp X \mid \mathbf{Z}$ ("on a capturé tous les confondeurs en conditionnant sur $\mathbf{Z}$"). C'est une hypothèse *qu'on pose sans la justifier*. Le backdoor criterion est la **traduction graphique exacte** de cette hypothèse : si $\mathbf{Z}$ satisfait le backdoor criterion dans le DAG, alors l'unconfoundedness est *prouvée*, pas supposée. C'est ce qui rend Pearl plus puissant : il transforme une hypothèse invisible en condition vérifiable sur un graphe.

### E. Exemple : retour aux kidney stones

```
        S (taille du calcul)
       / \
      ↓   ↓
      T → R
```

**Question.** Quels ensembles satisfont le backdoor criterion pour $(T, R)$ ?

**Chemins backdoor de T vers R** (qui partent de $T$ par une flèche entrante) :
* $T \leftarrow S \to R$ — c'est le seul

**Vérification de $\mathbf{Z} = \{S\}$ :**
1. $S$ n'est pas descendant de $T$ ✅
2. $\{S\}$ bloque le chemin backdoor $T \leftarrow S \to R$ (motif fork, $S$ dans $\mathbf{Z}$ → bloqué) ✅

→ $\mathbf{Z} = \{S\}$ est valide. C'est ce qu'on a fait.

**Vérification de $\mathbf{Z} = \emptyset$ :**
1. ✅ trivialement
2. ❌ le chemin $T \leftarrow S \to R$ n'est pas bloqué

→ Sans ajuster, on a un biais de confusion (= ce qu'on observait avec le 0.83 vs 0.78 trompeur).

### F. Front-door criterion (TODO)

*À compléter plus tard. En bref : alternative au backdoor quand le confondeur n'est pas observable mais qu'un médiateur l'est. Exemple canonique : effet du tabac sur le cancer sans observer la génétique, mais en observant le goudron dans les poumons.*

### G. Do-calculus : les 3 règles (TODO)

*À compléter plus tard. En bref : généralisation rigoureuse qui dit *quand* $P(Y \mid \text{do}(X))$ est identifiable et comment le calculer. Backdoor et front-door en sont des cas particuliers. C'est le résultat de complétude de Pearl.*

## III - Causal Discovery (modèles restreints)

### A. Le problème de l'identifiabilité

**Question.** Étant donnée une distribution observationnelle $P(X, Y)$, peut-on retrouver le DAG qui l'a engendrée ?

**Théorème (non-identifiabilité générale).** *Sans hypothèses sur la classe de SCM*, c'est impossible. Pour toute distribution $P(X, Y)$, il existe un SCM avec $X \to Y$ ET un SCM avec $Y \to X$ qui induisent exactement la même distribution observationnelle.

> 💡 **Conséquence philosophique.** Le DAG n'est PAS dans les données. Pour faire de la causal discovery, il faut **injecter une hypothèse** qui contraint la classe de SCM autorisée. Sans hypothèse, la causalité n'est jamais identifiable à partir d'observations seules.

**Lien avec ce qu'on a vu.** Sections I et II supposaient le DAG **connu**. Ici on essaie de le **récupérer**. C'est un problème *plus* dur.

### B. La stratégie : restreindre la classe de SCM

L'idée : si on contraint $f_j$ et $P_N$, alors les deux directions $X \to Y$ et $Y \to X$ ne sont *plus* indistinguables, et on peut tester laquelle des deux est compatible avec les données.

> 💡 **Deux grandes familles de stratégies.**
> * **Single-environnement** (C, D, E ci-dessous) : on a *un seul* jeu de données et on impose des contraintes sur la forme du SCM (linéarité, additivité du bruit, non-gaussianité, etc.) pour briser la symétrie entre directions.
> * **Multi-environnements** (F) : on a *plusieurs* jeux de données issus de conditions différentes (interventions, régimes, périodes), et on exploite le fait que les vraies relations causales sont **invariantes** à travers ces environnements.

| Restriction | Identifiable ? |
|---|---|
| Aucune | ❌ |
| Linéaire + bruit gaussien | ❌ (cas dégénéré) |
| Linéaire + bruit **non-gaussien** | ✅ (LiNGAM) |
| Non-linéaire additif $Y = f(X) + N$ | ✅ presque toujours (ANM) |
| Modèle post-non-linéaire | ✅ avec quelques exceptions |

> 💡 **Intuition.** La gaussianité est *trop symétrique* — elle ne contient aucune asymétrie qui pourrait révéler la direction causale. Toute déviation par rapport à la gaussianité (skewness, kurtosis, non-linéarité de $f$) brise cette symétrie et rend la direction identifiable.

---
#### Approche 1 : Single-environnement
---

### C. LiNGAM (Linear Non-Gaussian Acyclic Model)

**Modèle.** SCM linéaire avec bruits indépendants non-gaussiens :
$$
X_j = \sum_{i \in \text{PA}_j} \alpha_{ji} X_i + N_j, \quad N_j \text{ non-gaussien, indépendants}
$$

**Théorème (Shimizu et al. 2006).** Le DAG est **identifiable à partir de $P(X)$**.

**Pourquoi ça marche (cas bivarié).** Suppose $Y = \alpha X + N_Y$ avec $X \perp N_Y$, $N_Y$ non-gaussien.

Si on essaie de fitter le modèle inverse $X = \beta Y + N_X'$, on calcule $\beta$ par moindres carrés et on regarde le résidu $N_X' = X - \beta Y$.
* Si la vraie direction est $X \to Y$ : $N_X'$ est *corrélé* à $Y$ (donc à la cause supposée).
* Si la vraie direction est $Y \to X$ : $N_X'$ est *indépendant* de $Y$.

→ Test simple : on fit dans les deux sens, on regarde dans quelle direction le résidu est indépendant de la cause supposée. C'est ça la direction causale.

**⚠️ Le piège du cas gaussien.** Si $N_Y$ est gaussien, **les deux directions donnent un résidu indépendant** (théorème Darmois–Skitovich). C'est pour ça que le cas gaussien est non-identifiable.

**Visualisation.** Pour des bruits uniformes, le nuage de points $(X, Y)$ a des **bords nets** dans la direction causale. L'exercice 1 du notebook illustre ça : tu génères 1000 points dans un sens et dans l'autre, et tu vois la différence de forme à l'œil nu.

### D. ANM (Additive Noise Models)

**Modèle.** Généralisation non-linéaire :
$$
Y = f(X) + N_Y, \quad X \perp N_Y
$$
avec $f$ non-linéaire. Le bruit $N_Y$ peut maintenant être gaussien.

**Théorème (Hoyer et al. 2009).** Sauf cas très particuliers (comme $f$ linéaire + $N_Y$ gaussien), le DAG est **identifiable**.

**Pourquoi ça marche.** Même argument que LiNGAM : si on fit le modèle inverse $X = g(Y) + N_X'$ avec $g$ non-linéaire, le résidu $N_X'$ ne sera *pas* indépendant de $Y$ — sauf dans la vraie direction causale.

> 💡 **C'est puissant.** ANM couvre quasi tous les cas pratiques : bruit gaussien et $f$ non-linéaire (la situation la plus fréquente en sciences expérimentales). LiNGAM était le cas spécial linéaire.

### E. Test d'indépendance résiduelle (HSIC)

**Le problème pratique.** Comment teste-t-on numériquement "le résidu est indépendant de $X$" ?

**Solution.** **HSIC** (Hilbert-Schmidt Independence Criterion). C'est un test d'indépendance non-paramétrique qui :
* mesure la dépendance entre deux variables aléatoires de manière complètement non-linéaire,
* fonctionne sans avoir à supposer une forme paramétrique,
* a une distribution sous $H_0$ qu'on peut calculer.

**Pipeline causal discovery (cas bivarié).**
1. Fitter $\hat Y = \hat f(X)$ (régression non-linéaire, ex: GAM ou GP)
2. Calculer le résidu $\hat N_Y = Y - \hat f(X)$
3. Tester $\hat N_Y \perp X$ via HSIC. Si p-value élevée → direction $X \to Y$ plausible.
4. Refaire dans l'autre sens.
5. La direction "gagnante" est celle où l'indépendance résiduelle est la plus forte.

C'est exactement ce que fait Peters dans le notebook avec la fonction `dHSIC` du package R.

### F. Au-delà du bivarié : algorithmes pour $d > 2$ variables

Jusqu'ici on a regardé le cas **bivarié** ($X$ vs $Y$). Avec $d$ variables, le nombre de DAGs candidats explose ($O(d!)$), et il faut des algos qui passent à l'échelle. Deux grandes familles.

**Independence-based approach** (PC, FCI)

Idée : tester systématiquement les **indépendances conditionnelles** $X_i \perp X_j \mid \mathbf{Z}$ et reconstruire le graphe à partir de ces tests.
1. **Squelette** : on commence avec un graphe complet. On supprime l'arête $(X_i, X_j)$ s'il existe un $\mathbf{Z}$ tel que $X_i \perp X_j \mid \mathbf{Z}$.
2. **Orientation** : on oriente les arêtes via les **v-structures** (colliders détectables) puis via des règles de Meek qui propagent l'orientation.

→ Output : **CPDAG** (graphe partiellement orienté) — toutes les arêtes ne sont pas forcément orientables sans hypothèse supplémentaire.

✅ Pas besoin de modèle paramétrique
❌ Beaucoup de tests d'indépendance, sensible à la qualité des tests

**Score-based approach** (GES, NOTEARS)

Idée : définir un **score** pour chaque DAG candidat, et chercher le DAG qui maximise ce score.
* **Score classique** : log-vraisemblance pénalisée (BIC) sous un modèle paramétrique (typiquement gaussien)
* **Score utilisé en pratique pour ANM** : $\sum_j \log \text{Var}(\hat N_j)$ (le **log-variance score** dont parle le notebook). Sous une approximation gaussienne, minimiser ce score équivaut à maximiser la log-vraisemblance.
* **Recherche** : algorithme glouton ajout/suppression d'arêtes (GES) ou optimisation continue avec contrainte d'acyclicité (NOTEARS)

✅ Critère d'optimalité bien défini, plus efficace
❌ Hypothèses paramétriques, optimum local possible

> 💡 **Lien avec HSIC.** Pour le cas bivarié, le `log Var` du notebook joue le même rôle que HSIC : c'est un score qui te dit dans quelle direction le résidu est "le plus indépendant". HSIC = test rigoureux non-paramétrique. `log Var` = approximation gaussienne rapide. Les deux convergent vers la même réponse en grande dimension.

**Functional / asymmetry-based approach** (DirectLiNGAM, RESIT)

Idée : étendre au cas multivarié la logique du bivarié vue en C/D/E — exploiter une **asymétrie fonctionnelle** (non-gaussianité du bruit pour LiNGAM, non-linéarité de $f$ pour ANM) pour identifier la direction de chaque flèche via tests d'indépendance résiduelle.

* **DirectLiNGAM** (Shimizu et al. 2011) : généralisation multivariée de LiNGAM. Algo glouton qui identifie itérativement la **variable la plus exogène** (= celle dont les résidus, après régression sur toutes les autres, sont les plus indépendants). On la place en tête de l'ordre causal, on la retire du dataset (en régressant tout le monde dessus), on recommence sur les variables restantes. À la fin on a un ordre causal complet, dont on déduit le DAG.
* **RESIT** (Peters et al. 2014) : équivalent pour ANM. Même logique itérative, mais avec régression non-paramétrique au lieu d'OLS et test HSIC pour l'indépendance résiduelle.

→ Output : **DAG complet** (pas un CPDAG), parce que l'asymétrie fonctionnelle oriente *toutes* les arêtes — c'est le gros avantage par rapport aux deux familles précédentes.

✅ DAG complètement orienté, pas de classe d'équivalence
❌ Hypothèses fortes (linéarité + non-gaussianité, ou bruit additif), s'effondre si elles sont violées

> 💡 **Le piège de DirectLiNGAM en pratique.** L'algo te rendra *toujours* un DAG complet, même quand les hypothèses sont violées (variables binaires, non-linéarités fortes, confondeurs cachés). Contrairement à PC qui peut renvoyer une arête non-orientée pour signaler son ignorance, DirectLiNGAM tranche systématiquement — parfois à tort. À utiliser avec un diagnostic préalable (Shapiro sur les variables, scatter plots pour la linéarité).


### XX. Independence based approach

Algo "PC"  AUthors Date + PC stands for what ? 

On commence par un complete (edge between all pairs of variable ie there is no independence) undirected graph. Then
1. Identify the skeleton
2. Identify the v-structure (immoralities) and orient them
3. Orient qualifying edges that are incident on colliders

![[Pasted image 20260430210234.png|177]]

Pour ce qui est du squelette on a deux boucles for :
Première boucle on trovue que $A \perp B \mid \{ \}$ 


(1) Étape 1 : Identify the skeleton

| Graphe complet                            | Fin de 1ère itération                     | Fin de 2e itération                       |
| ----------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260430210313.png\|204]] | ![[Pasted image 20260430210410.png\|191]] | ![[Pasted image 20260430210506.png\|186]] |



puis $\forall$ other pairs $(X,Y), X \perp Y \mid \{C\}$ 


En gros on effectue un test statistique d'indépendance sur nos données :

Première itération $|Z|=0$ :
On commence avec $Z=\emptyset$ puis pour chaque arête du graphe complet, on lance un test stats sur les données : 
* $\text { Arête } A-B \text { : test stat } \text { → } A \perp B \text { ?" } \text { → } \text { oui } \text { → } \text { on coupe. }$
* $\text { Arête } A-C \text { : test } \text { → } " A \perp C \text { ?" } \text { → } \text { non } \text { → } \text { on garde. }$
* etc
=> Résultat à la fin de cette itération : on a coupé une seule arête $A-B$. Le graphe de rtavail a maintenant 9 arêtes au lieu de 10.

$\text { Deuxième itération: }|Z|=1$
On va maintenant ajouter un conditioning set de taille 1. 
* Arête $A-C$ : on teste " $A \perp C \mid B$ ?" non, " $A \perp C \mid D$ ?" non, " $A \perp C \mid E$ ?" non → on garde.
* $\text { Arête } A-D: \text { on teste " } A \perp D \mid C \text { ?" } \text { → } \text { oui } \text { → } \text { on coupe. }$
* etc
=> Résultat à la fin de cette itération : On a coupé plusieurs arêtes 

Étape 2 : 2. Identify the v-structure (immoralities) and orient them

On va récupérer les sepset (separating set) en en gros c'est sur quelle variable j'ai conditioné durant mes deux itérations on a pour $|Z|=0$ : $\operatorname{sepset}(A, B)=\emptyset$ puis pour $|Z|=1$ : on a $\operatorname{sepset}(A, D)=\{C\}, \operatorname{sepset}(A, E)=\{C\}$ etc.  

$$
\begin{array}{|c|c|c|c|}
\hline
\text{Triplet } (X, Z, Y) & \text{sepset}(X, Y) & Z \in \text{sepset}? & \text{Action} \\
\hline
(A, C, B) & \emptyset & C \notin \emptyset & \text{oriente } A \rightarrow C \leftarrow B \\
\hline
(A, C, D) & \{C\} & C \in \{C\} & \text{rien} \\
\hline
(A, C, E) & \{C\} & C \in \{C\} & \text{rien} \\
\hline
etc & ... & ... & ... \\
\hline
\end{array}
$$


| Fin de 2e itération                       | Fin de l'étape 2                          |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260430210506.png\|186]] | ![[Pasted image 20260430211616.png\|242]] |

Etape 3 : Orient qualifying edges that are incident on colliders - il va propager les arêtes restées non-orientées. La règle générale : on oriente une arête si ne pas l'orienter dans un sens créer une contradiction


| Fin de l'étape 2                          | Fin de l'étape 3                          |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260430210506.png\|186]] | ![[Pasted image 20260430212020.png\|188]] |




### XX. Functional / asymmetry-based approach


---
#### Approche 2 : Multi-environnements
---

### G. ICP (Invariant Causal Prediction)

**Idée centrale.** Les vraies relations causales sont **invariantes** à travers les environnements. Si on dispose de plusieurs jeux de données issus de conditions différentes (interventions, régimes, périodes), on peut identifier les parents causaux de $Y$ comme étant le sous-ensemble $\mathbf{S}$ de prédicteurs tel que $P_e(Y \mid X^{\mathbf{S}})$ reste **identique** dans tous les environnements $e$. Les interventions n'agissant pas directement sur $Y$ ne modifient pas le mécanisme qui génère $Y$ depuis ses parents — donc la conditionnelle est invariante. Formellement (cas linéaire) il existe $\beta$ tel que dans tous les environnements : $Y_i = \mu + X_i^{\text{pa}(Y)} \beta + \varepsilon_i$ avec $\varepsilon_i \perp X_i^{\text{pa}(Y)}$.

> 💡 **Visuellement.** Si tu plottes deux samples du même SCM $X \to Y$ — un naturel, un sous shift intervention sur $X$ — la **droite de régression** $Y = f(X)$ est *identique* (conditionnelle invariante), mais le **nuage des $X$** est décalé (marginale non-invariante). C'est cette asymétrie qu'ICP exploite.

**Algorithme.**
1. Pour chaque sous-ensemble candidat $\mathbf{S} \subseteq \{1, \dots, d\}$ : régresser $Y$ sur $X^{\mathbf{S}}$ dans chaque environnement, et tester si la conditionnelle est invariante.
2. Garder tous les $\mathbf{S}$ qui passent le test.
3. **Output** = intersection des $\mathbf{S}$ acceptés.

C'est une **estimation conservative** : on préfère renvoyer un sous-ensemble *strict* des vrais parents (avec garantie de couverture $\geq 1 - \alpha$) plutôt que de risquer un faux positif. Si aucun sous-ensemble ne passe, l'output est vide — ICP refuse de conclure plutôt que d'inventer.

**Lien avec ML moderne.** Cette idée d'invariance à travers environnements est *exactement* ce que l'**IRM** (Invariant Risk Minimization, Arjovsky et al. 2019, Facebook AI) et tous les travaux récents sur la **généralisation hors distribution** essaient de formaliser dans un cadre ML neuronal. ICP est l'ancêtre statistique propre de cette ligne de recherche.

## IV - Contrefactuels (TODO)

*Niveau 3 de la ladder of causation — à compléter plus tard.*

**Idée en bref.** Un contrefactuel répond à : *"Étant donné ce qui s'est réellement passé, qu'est-ce qui se serait passé si...?"*

* **Intervention** ($P(Y \mid \text{do}(X = x))$) : prospectif, sur une population aléatoire — "si on force $X = x$, quel $Y$ ?"
* **Contrefactuel** ($P(Y_x \mid X = x', Y = y')$) : on conditionne *sur l'observé* ET on intervient sur le passé — "Bob a pris le médicament et est mort. Aurait-il vécu sans le prendre ?"

**Pourquoi c'est plus dur.** On mélange deux mondes (réel + contrefactuel). Pour calculer ça, il faut un **SCM complet**, pas juste un DAG. La procédure de Pearl en 3 étapes : **abduction** (mettre à jour la distribution des bruits via l'évidence), **action** (intervenir sur le SCM modifié), **prediction** (calculer $Y$ dans ce monde modifié).

**Pourquoi c'est le niveau 3.** Le niveau 2 suffit pour des questions de *politique publique* ("faut-il vacciner ?"). Le niveau 3 est nécessaire pour des questions de *responsabilité* ("le vaccin a-t-il causé la mort de Bob ?"), de *blame*, ou d'*explication individuelle*.

**Lien avec Rubin.** Rubin gère les contrefactuels *par construction* avec ses potential outcomes $Y_i(0), Y_i(1)$. Ce que Rubin appelle "contrefactuel" depuis 1974, Pearl l'a formalisé dans le langage des SCM des décennies plus tard. Les deux frameworks sont *équivalents* sur les contrefactuels — juste des notations différentes.

**À couvrir plus tard :** procédure formelle abduction-action-prediction, mediation analysis (direct/indirect/total effect à la Pearl), effects of treatment on the treated, probabilities of causation (PN, PS, PNS).

---

## Annexe — Tableau récap des algos de causal discovery

Pour la section III. Trois grandes familles + une approche multi-environnements. Liste *non* exhaustive — choix des algos canoniques à connaître, pas exhaustivité.

| Algo | Famille | Année | Auteurs | Hypothèse clé | Output |
|---|---|---|---|---|---|
| **PC** | Independence-based | 1991 | Spirtes & Glymour | Faithfulness, sufficiency | CPDAG |
| **FCI** | Independence-based | 2000 | Spirtes, Meek, Richardson | Faithfulness (sans sufficiency) | PAG (gère les confondeurs cachés) |
| **GES** | Score-based | 2002 | Chickering | Modular score (BIC), faithfulness | CPDAG |
| **NOTEARS** | Score-based (continu) | 2018 | Zheng, Aragam et al. | Linéarité, contrainte d'acyclicité différentiable | DAG |
| **ICA-LiNGAM** | Functional / asymmetry | 2006 | Shimizu et al. | Linéaire, bruits non-gaussiens indépendants | DAG complet |
| **DirectLiNGAM** | Functional / asymmetry | 2011 | Shimizu et al. | Idem, version stable et reproductible | DAG complet |
| **RESIT** | Functional / asymmetry | 2014 | Peters et al. | Bruit additif non-linéaire (ANM) | DAG complet |
| **ICP** | Multi-environnements | 2016 | Peters, Bühlmann, Meinshausen | Invariance à travers environnements | Sous-ensemble des parents de Y |

**Lecture rapide.**
* **CPDAG** = Completed Partially Directed Acyclic Graph. Toutes les arêtes ne sont pas orientables sans hypothèse supplémentaire (classe d'équivalence de Markov).
* **PAG** = Partial Ancestral Graph. Généralisation du CPDAG qui gère les confondeurs latents.
* **DAG complet** = toutes les arêtes orientées. Possible uniquement avec les approches functional / asymmetry-based (ou avec multi-environnements).

**Mon top 3 à connaître.**
1. **PC** — l'ancêtre, le plus cité, le mieux étudié. À savoir expliquer à l'oral (squelette → orientation des v-structures).
2. **GES** — représentant canonique du score-based, basé sur BIC.
3. **DirectLiNGAM** — extension multivariée du bivarié de la section C, exploite la non-gaussianité.

**À connaître de nom.**
* **NOTEARS** — la version "deep learning friendly" (optimisation continue), beaucoup citée depuis 2018.
* **FCI** — quand on n'a pas la sufficiency (confondeurs cachés possibles).
* **ICP** ⭐ — particulièrement pertinent en finance (régimes de marché = environnements), lien direct avec l'IRM d'Arjovsky pour la généralisation OOD.
