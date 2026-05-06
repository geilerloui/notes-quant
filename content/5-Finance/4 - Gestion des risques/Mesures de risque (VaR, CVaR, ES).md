---
title: Mesures de risque - VaR, CVaR, Expected Shortfall
---
# Mesures de risque : VaR, CVaR, Expected Shortfall

> Une **mesure de risque** est une fonction qui transforme la distribution des pertes d'un portefeuille en un nombre, supposé résumer son risque. Elle sert dans trois contextes : la **réglementation** (calcul des fonds propres bancaires sous Bâle III/IV), la **gestion interne** (limites de risque imposées aux gérants), et l'**optimisation de portefeuille** (où on peut soit minimiser une mesure de risque, soit la contraindre). Cette note couvre les trois mesures les plus utilisées en pratique : la VaR (Value-at-Risk), historiquement standard mais avec des défauts mathématiques sérieux, et le CVaR (Conditional Value-at-Risk, alias Expected Shortfall), sa version "corrigée" qui répare ces défauts et qui domine désormais le cadre réglementaire.

## I. Préliminaires probabilistes

Avant la VaR, il faut être à l'aise avec deux objets : la **fonction de répartition** $F_X$ et son inverse, le **quantile** $F_X^{-1}$. Toute la VaR n'est qu'une lecture appropriée de ces deux objets.

### A. Fonction de répartition

> [!warning] Définition
> Soit $X$ une variable aléatoire (pour nous : un rendement, un P&L, une perte). La **fonction de répartition** (cumulative distribution function, CDF) est
> 
> $$F_X(x) = \mathbb{P}(X \leq x).$$
> 
> Elle répond à : *"quelle est la probabilité que $X$ soit inférieure ou égale à $x$ ?"*

> [!example] Lecture concrète
> Si $X$ est la taille des adultes en France et $F_X(1.70) = 0.60$, ça veut dire : **60% des gens font 1m70 ou moins**.

**Cas continu.** Si $X$ a une densité $f_X$, alors $F_X(x) = \int_{-\infty}^x f_X(u)\,du$. Graphiquement, $F_X(x)$ est l'aire sous la courbe de densité à gauche de $x$.

**Cas discret.** Si $X$ prend les valeurs $x_1, x_2, \ldots$ avec probabilités $p_1, p_2, \ldots$, alors

$$F_X(x) = \sum_{x_i \leq x} p_i, \qquad p_i = \mathbb{P}(X = x_i).$$

C'est une fonction en escalier qui saute de $p_i$ à chaque $x_i$. La fonction de masse $p_i$ et la fonction de répartition $F_X$ sont les deux représentations équivalentes d'une loi discrète.

### B. Quantile

> [!warning] Définition
> Le **quantile** d'ordre $\alpha \in (0, 1)$ est l'inverse de la fonction de répartition :
> 
> $$F_X^{-1}(\alpha) = \inf\{x : F_X(x) \geq \alpha\}.$$
> 
> Lecture : *"quelle est la valeur $x$ telle qu'une fraction $\alpha$ de la population soit inférieure ou égale à $x$ ?"*

**Lien avec la fonction de répartition.** $F_X$ et $F_X^{-1}$ sont deux lectures de la même information :

- $F_X$ va de $x$ vers une probabilité ("60% sont sous 1m70" → $F_X(1.70) = 0.60$).
- $F_X^{-1}$ va d'une probabilité vers $x$ ("le seuil sous lequel se trouvent 60% des gens est 1m70" → $F_X^{-1}(0.60) = 1.70$).

> 💡 **Cas particuliers à connaître par cœur.** $F_X^{-1}(0.5)$ est la **médiane**. $F_X^{-1}(0.05)$ est le **5%-quantile** : la valeur en dessous de laquelle se trouvent 5% des observations — donc une valeur "extrême basse". C'est précisément cette idée qui définit la VaR.

> [!note]- Convention sur les rendements vs pertes
> Selon les sources, on travaille tantôt sur les **rendements** $R$ (où le mauvais cas est $R$ très négatif), tantôt sur les **pertes** $L = -R$ (où le mauvais cas est $L$ très positif). Cela change le signe des formules. Dans cette note, on travaille sur les rendements (ou P&L), et donc le mauvais cas est la queue **gauche** de la distribution. La VaR est définie comme un nombre **positif** (un montant à risque), d'où le signe moins dans la définition ci-dessous.

## II. Value-at-Risk (VaR)

### A. Définition

L'idée intuitive de la VaR : *"dans le pire des $\alpha\%$ scénarios, je perds au moins combien ?"*. C'est le quantile bas de la distribution des rendements.

> [!warning] Value-at-Risk
> Pour un niveau de confiance $\alpha \in (0, 1)$ (typiquement $\alpha = 0.05$ ou $0.01$),
> 
> $$\text{VaR}_\alpha(X) = -F_X^{-1}(\alpha).$$
> 
> Le signe moins traduit le fait qu'on rapporte un *montant à risque* (positif) à partir d'un quantile bas (négatif).

> [!example] VaR à 5% sur un P&L journalier
> On dispose d'une distribution empirique du P&L journalier sur 250 jours (1 an). On calcule
> 
> $$F_X^{-1}(0.05) = -10\,000\,\text{€}.$$
> 
> Lecture : 5% des journées ont un P&L $\leq -10\,000$ €. Autrement dit : *"il y a 5% de chance que je perde 10K ou plus en une journée"*. La VaR à 5% vaut donc
> 
> $$\text{VaR}_{0.05}(X) = +10\,000\,\text{€}.$$
> 
> Convention : la VaR est un montant positif, qu'on lit comme **"montant qu'on risque de perdre dans 5% des cas"**.

### B. Le défaut de la VaR : non-convexité

La VaR a un problème mathématique sérieux : **elle n'est pas convexe** comme fonction des poids du portefeuille. Concrètement, ça signifie qu'elle peut **violer la sous-additivité** :

$$\text{VaR}(X + Y) > \text{VaR}(X) + \text{VaR}(Y) \quad \text{(possible !)}.$$

C'est anti-intuitif : la diversification (combiner deux portefeuilles) *devrait* réduire le risque, ou au pire le laisser inchangé. Une mesure qui peut **augmenter** sous diversification n'est donc pas une bonne mesure de risque.

> 💡 **Conséquence pour l'optimisation.** Minimiser la VaR sous contraintes de portefeuille n'est pas un problème convexe — il peut avoir plusieurs minima locaux, les solveurs standards (CVXPY, MOSEK) ne peuvent pas garantir l'optimum, et le calcul devient lourd. C'est précisément ce défaut qui motive l'introduction du CVaR.

### C. Autre défaut : la VaR ignore l'ampleur des pertes au-delà du seuil

La VaR est un quantile : elle dit **où** se trouve le seuil de perte, mais **rien sur ce qu'il y a au-delà**. Deux portefeuilles peuvent avoir la même VaR avec des comportements catastrophiques très différents.

> [!example] Deux portefeuilles, même VaR mais profils opposés
> $\text{VaR}_{0.05} = 10\,000$ € pour les deux portefeuilles A et B. Mais en regardant les 5% pires journées :
> 
> | Portefeuille | Comportement dans les 5% pires cas |
> |---|---|
> | **A** | pertes concentrées entre 10K et 12K — *des pertes "petites" autour du seuil* |
> | **B** | pertes typiquement entre 10K et 15K, mais 1 jour de krach à 100K |
> 
> Même VaR, mais le portefeuille B est manifestement plus risqué. La VaR ne le voit pas.

C'est cette limite qui motive l'**Expected Shortfall** : au lieu de regarder uniquement la position du quantile, on regarde la **moyenne des pertes au-delà**.

## III. Expected Shortfall (ES) / Conditional Value-at-Risk (CVaR)

> [!warning] Expected Shortfall (alias CVaR)
> $$\text{ES}_\alpha(X) = \frac{1}{\alpha} \int_0^\alpha \text{VaR}_u(X)\,du \;\geq\; \text{VaR}_\alpha(X).$$
> 
> Lecture : c'est la **moyenne des VaR à tous les niveaux $\beta \in (0, \alpha]$** — autrement dit la moyenne des pertes dans la queue gauche de la distribution. Les noms **Expected Shortfall** (ES) et **Conditional Value-at-Risk** (CVaR) désignent la même quantité, le second étant plus utilisé en optimisation.

**Formulation alternative en espérance conditionnelle.** L'ES peut s'écrire comme une espérance conditionnelle aux pires scénarios :

$$\text{ES}_\alpha(X) = \mathbb{E}\big[\,-X \;\big|\; -X \geq \text{VaR}_\alpha(X)\,\big].$$

Lecture : *"sachant qu'on est dans les $\alpha\%$ pires cas (où la perte dépasse la VaR), quelle est la perte moyenne ?"*

> [!example] ES à 5% sur le P&L journalier
> Reprenons l'exemple de la VaR. On a $\text{VaR}_{0.05} = 10\,000$ €, et on regarde les 5% pires journées sur 1 an (soit 12 jours, puisque 5% × 250 ≈ 12). Imaginons que ces 12 jours se décomposent ainsi :
> 
> - **8 jours** où on a perdu entre 10K et 15K
> - **3 jours** où on a perdu autour de 20K
> - **1 jour de krach** où on a perdu 100K
> 
> La VaR s'est arrêtée au seuil 10K. L'ES, elle, fait la **moyenne des pertes sur ces 12 jours** :
> 
> $$\text{ES}_{0.05} = \frac{\text{somme des pertes des 12 pires journées}}{12} \approx 25\,000\,\text{€}.$$
> 
> L'ES capture donc le krach que la VaR ignorait totalement.

> 💡 **Le contraste fondamental VaR vs ES.**
> 
> - **VaR** = *"montant minimal que je risque de perdre dans les pires $\alpha\%$ cas"* — un seuil.
> - **ES** = *"perte moyenne attendue dans les pires $\alpha\%$ cas"* — une moyenne.
> 
> L'ES est toujours **supérieur ou égal** à la VaR (puisque c'est la moyenne de quantiles tous $\geq \text{VaR}_\alpha$), avec égalité seulement dans des cas dégénérés.

### A. ES discrimine les portefeuilles que la VaR confond

Reprenons l'exemple V.B avec deux portefeuilles A et B de même VaR à 10K :

> [!example] Même VaR, ES très différents
> 
> | Portefeuille | $\text{VaR}_{0.05}$ | $\text{ES}_{0.05}$ | Lecture |
> |---|:---:|:---:|---|
> | **A** | 10K € | **12K €** | Pertes typiques autour du seuil — risque "normal" |
> | **B** | 10K € | **80K €** | Pertes typiques modérées, mais des krachs massifs |
> 
> L'ES voit ce que la VaR ne voit pas : le portefeuille B a des pertes **bien plus catastrophiques** dans les pires cas, même si sa VaR est identique à celle de A.

### B. La propriété décisive : convexité

Contrairement à la VaR, l'ES (et donc le CVaR) est **convexe** comme fonction des poids du portefeuille :

$$\text{ES}(\lambda x + (1-\lambda) y) \leq \lambda\,\text{ES}(x) + (1-\lambda)\,\text{ES}(y).$$

Cela implique en particulier la **sous-additivité** : $\text{ES}(X + Y) \leq \text{ES}(X) + \text{ES}(Y)$, qui est la propriété attendue d'une bonne mesure de risque (la diversification ne peut qu'aider).

> 💡 **Conséquence pour l'optimisation.** Minimiser le CVaR ou le contraindre dans un programme d'optimisation produit un problème **convexe**, résolvable par les solveurs standards. C'est cette propriété qui en a fait la mesure de référence dans les standards modernes (Bâle FRTB en 2019 a remplacé la VaR par l'ES) et l'objet privilégié pour l'optimisation de portefeuille robuste.

> [!note]- Pourquoi VaR n'est pas convexe et ES l'est : intuition
> La VaR est un *quantile* — un point sur la fonction de répartition. Elle ne dépend que du seuil, pas de la masse au-delà. Combiner deux portefeuilles peut déplacer ce point dans une direction qui semble pire (la position du quantile peut "sauter" à cause d'événements rares dont les distributions interagissent).
> 
> L'ES est une *espérance* — une intégrale lisse sur la queue. Les espérances sont linéaires, donc l'opération "moyenne des pires cas" hérite naturellement de la convexité par l'inégalité de Jensen appliquée à la queue.

### C. Cadre des mesures de risque cohérentes

L'analyse formelle vient d'Artzner, Delbaen, Eber & Heath (1999) qui ont défini les **axiomes d'une mesure de risque cohérente** : monotonie, invariance par translation, homogénéité positive, et **sous-additivité**. Ils ont montré que :

- La VaR ne satisfait *pas* la sous-additivité — elle n'est donc *pas* cohérente.
- L'ES (CVaR) satisfait les quatre axiomes — elle *est* cohérente.

Ce résultat théorique a été un tournant historique : il a justifié rigoureusement le passage progressif de la VaR vers l'ES dans la réglementation bancaire et la gestion d'actifs.

## IV. Utilisation en optimisation de portefeuille

Trois manières d'incorporer le CVaR dans un programme d'optimisation. Toutes les trois donnent des problèmes **convexes** (contrairement à la VaR), donc résolvables par CVXPY, MOSEK, Gurobi.

### A. Minimisation pure du CVaR

$$\min_x\;\;\text{CVaR}_\alpha(R_x) \qquad \text{s.c.}\;\; x \in \mathcal{X}.$$

On cherche directement les poids $x$ qui rendent **la moyenne des pires pertes** la plus basse possible. C'est l'analogue model-free du portefeuille de variance minimale : pas de notion de rendement attendu, juste minimiser le risque downside.

> 💡 **Lien avec le portefeuille de variance minimale.** Le programme $\min_x x^\top \Sigma x$ (variance minimale) correspond au cas $\lambda \to \infty$ dans Markowitz : un agent infiniment averse au risque, qui ne se soucie plus du rendement. Même logique pour $\min_x \text{CVaR}(R_x)$, mais avec une mesure de risque qui ne pénalise plus que le downside. Dans les deux cas : portefeuille peu volatile, rendement faible.

### B. Critère pénalisé (analogue Markowitz)

$$\max_x\;\; \mathbb{E}[R_x] - \lambda\,\text{CVaR}_\alpha(R_x) \qquad \text{s.c.}\;\; x \in \mathcal{X}.$$

Même structure que le critère de Markowitz $E[R] - \lambda \text{Var}$, mais avec le CVaR à la place de la variance. C'est le programme de référence pour la **mean-CVaR optimization**.

> 💡 **Pourquoi remplacer Var par CVaR ?** Voir [[Markowitz#V. Limites du modèle|Markowitz §V — Limites du modèle]] : la variance pénalise hausses **et** baisses (symétrie), alors que le CVaR ne pénalise que les pertes extrêmes. Le critère est donc beaucoup plus aligné avec la perception intuitive du risque.

### C. CVaR comme contrainte

$$\max_x\;\; \mathbb{E}[R_x] \qquad \text{s.c.}\;\; \text{CVaR}_\alpha(R_x) \leq c, \;\; x \in \mathcal{X}.$$

On maximise le rendement attendu **sous contrainte** que la perte moyenne dans les cas de crise ne dépasse pas un seuil $c$ (typiquement défini en montant absolu, ex. 1 million d'euros pour un portefeuille de 100M).

> [!example] Lecture concrète
> *"Je veux maximiser mon rendement espéré, mais je m'impose que ma perte moyenne en cas de crise (sur les 5% pires scénarios) ne dépasse pas 1 million d'euros."*
> 
> C'est typiquement comme ça qu'un risk manager institutionnel pose la contrainte : il fixe un budget de risque exprimé en euros (ou en pourcentage de l'AUM), et il laisse le gérant maximiser le rendement à l'intérieur de cette enveloppe. Voir aussi [[Markowitz#A.4 Contraintes de risque|Markowitz Annexe A.4]] qui mentionne cette contrainte parmi les contraintes de risque standard du programme.

---

## Résumé

| Concept | Idée en une ligne |
|---|---|
| **Fonction de répartition** $F_X$ | $F_X(x) = \mathbb{P}(X \leq x)$ — probabilité d'être en dessous de $x$ |
| **Quantile** $F_X^{-1}(\alpha)$ | seuil sous lequel se trouve une fraction $\alpha$ de la population |
| **VaR** $_\alpha = -F_X^{-1}(\alpha)$ | montant qu'on risque de perdre dans les $\alpha\%$ pires cas (un **seuil**) |
| **Défauts VaR** | non-convexe, pas sous-additive, ignore l'ampleur des pertes au-delà du quantile |
| **ES / CVaR** $_\alpha$ | moyenne des pertes au-delà du quantile $\alpha$ (une **moyenne**) |
| **Pourquoi CVaR domine** | convexe → optimisation tractable ; cohérent au sens d'Artzner et al. (1999) |
| **3 usages en optim** | (A) min CVaR ; (B) max E[R] − λ·CVaR ; (C) max E[R] s.c. CVaR ≤ seuil |
