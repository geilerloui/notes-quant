---
title: Information Theory
description: De la compression optimale aux applications en machine learning
---

# Information Theory

> De Shannon à l'apprentissage automatique : comment quantifier et exploiter l'information.

---

## I. Introduction et motivation

### Qu'est-ce que l'information ?

La théorie de l'information, développée par Claude Shannon en 1948, répond à une question fondamentale : **comment quantifier l'information** ? L'intuition de Shannon : l'information d'un événement est inversement proportionnelle à sa probabilité. Un événement rare (probabilité faible) apporte beaucoup d'information ; un événement prévisible en apporte peu.

Cette idée simple révolutionne deux domaines :
- **Communication** : comment transmettre efficacement de l'information ?
- **Machine Learning** : comment mesurer l'incertitude et optimiser les prédictions ?

### Le problème central : compression optimale

Imagine que tu veuilles transmettre la phrase "dog cat fish bird" de la manière la plus efficace possible. Comment encoder chaque mot pour minimiser la taille du message ? C'est exactement le problème que résout la théorie de l'information.

L'enjeu dépasse la simple compression : en machine learning, **minimiser la cross-entropy** revient à **trouver l'encodage optimal** des classes. Les concepts sont intimement liés.

---

## II. Une seule variable aléatoire (cas discret)

### A. Le problème de l'encodage

#### Codes à longueur fixe

Considérons l'alphabet $\mathcal{A} = \{\text{dog, cat, fish, bird}\}$. La solution naïve : attribuer un code binaire de longueur fixe à chaque symbole.

![[Pasted image 20260420232052.png]]

Avec 4 symboles, nous avons besoin de $\lceil \log_2 4 \rceil = 2$ bits par symbole (où $\lceil \cdot \rceil$ est la fonction plafond). On construit le **dictionnaire de codes** :
- $\text{code["dog"]} = 00$
- $\text{code["cat"]} = 01$  
- $\text{code["fish"]} = 10$
- $\text{code["bird"]} = 11$

Autrement dit : $\text{dog} \rightarrow 00$, $\text{cat} \rightarrow 01$, $\text{fish} \rightarrow 10$, $\text{bird} \rightarrow 11$.

Pour transmettre "dog cat fish bird", on concatène : **00011011** (8 bits total).

**Limitation** : cette approche ignore complètement la fréquence d'utilisation des symboles. Si $p(\text{dog}) = 1/2$ (50%) et $p(\text{bird}) = 1/8$ (12.5%), pourquoi leur attribuer la même longueur de code ?

#### Codes à longueur variable

L'idée révolutionnaire : attribuer des codes **courts aux symboles fréquents** et longs aux symboles rares.

![[Pasted image 20260420232145.png]]

Supposons les probabilités suivantes :
- $p(\text{dog}) = 1/2$ (50%)
- $p(\text{cat}) = 1/4$ (25%)  
- $p(\text{fish}) = 1/8$ (12.5%)
- $p(\text{bird}) = 1/8$ (12.5%)

Un encodage optimal pourrait être :
- $\text{code["dog"]} = 0$ (1 bit)
- $\text{code["cat"]} = 10$ (2 bits)
- $\text{code["fish"]} = 110$ (3 bits)  
- $\text{code["bird"]} = 111$ (3 bits)

La phrase "dog cat fish bird" devient : **010110111** (9 bits), mais la **longueur moyenne attendue** par symbole chute à :
$\text{Longueur moyenne} = \sum_x p(x) \cdot L(x) = \frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 = 1.75 \text{ bits}$

**Visualisation intuitive** :

![[Pasted image 20260420200532.png|421]]
*Figure : Code à longueur fixe - chaque symbole occupe exactement 2 bits*

![[Pasted image 20260420200627.png|413]]  
*Figure : Code à longueur variable optimal - l'aire totale représente l'entropie (1.75 bits)*

Dans le code optimal, chaque "rectangle" a une aire $p(x) \times L(x)$ qui correspond à sa contribution à la longueur moyenne. L'**aire totale** donne l'entropie — le minimum théorique de compression.

#### La propriété de préfixe

![[Pasted image 20260420200637.png|331]]

**Problème crucial** : comment decoder sans ambiguïté ? Si dog = 0 et cat = 01, le message "01" peut être lu comme "dog-cat" ou "cat". 

**Solution** : la **propriété de préfixe**. Aucun code ne peut être le préfixe d'un autre. Dans notre exemple :
- dog = 0 bloque l'accès aux codes commençant par 0
- Cela "coûte" la moitié de l'espace des codes possibles

**Règle fondamentale** : plus un code est court ($L(x)$ petit), plus il "bloque" de l'espace. La contrainte de préfixe impose :
$$\sum_x \frac{1}{2^{L(x)}} \leq 1$$

Cette inégalité (inégalité de Kraft) relie directement longueur des codes et probabilités. À la limite :
$$p(x) = \frac{1}{2^{L(x)}} \iff L(x) = \log_2\left(\frac{1}{p(x)}\right)$$

#### Fractional bits

Dans la pratique, $L(x) = \log_2(1/p(x))$ n'est pas toujours entier. Par exemple, si $p(x) = 1/3$, alors $L(x) = \log_2(3) \approx 1.58$ bits.

Comment encoder 1.58 bits ? Plusieurs stratégies :
- **Huffman coding** : approximation par des entiers
- **Arithmetic coding** : encodage de séquences entières
- **Asymptotic equipartition** : sur de longs messages, la moyenne converge vers l'entropie

### B. Entropie : la limite absolue de compression

#### Définition et interprétation

L'**entropie** d'une variable aléatoire $X$ relie directement compression optimale et probabilités :

$\boxed{H(X) = \sum_{x} p(x) \cdot L(x) = \sum_{x} p(x) \log_2\left(\frac{1}{p(x)}\right) = -\sum_{x} p(x) \log_2 p(x)}$

où $L(x) = \log_2(1/p(x))$ est la longueur optimale du code pour le symbole $x$.

**Interprétations multiples** :
1. **Compression** : longueur moyenne d'un code optimal (première égalité)
2. **Information** : quantité d'information moyenne apportée par $X$
3. **Surprise** : incertitude moyenne avant d'observer $X$  
4. **Limite théorique** : minimum de bits nécessaires pour encoder $X$

Sur notre exemple :
$H(X) = \frac{1}{2} \cdot 1 + \frac{1}{4} \cdot 2 + \frac{1}{8} \cdot 3 + \frac{1}{8} \cdot 3 = 1.75 \text{ bits}$

C'est exactement la longueur moyenne de notre code optimal ! L'entropie **n'est pas une abstraction** — c'est littéralement le coût de compression.

#### Propriétés fondamentales

![[images/1-Mathématiques/E_Optimal transport/im2-3.png|364]]
*Figure : Entropie d'une distribution binaire*

**Minimum** : $H(X) = 0$ si et seulement si $X$ est déterministe (une seule valeur possible)

**Maximum** : $H(X) = \log_2 |\mathcal{X}|$ si et seulement si $X$ suit une loi uniforme

**Concavité** : $H$ est une fonction concave des probabilités

#### Théorème de Shannon (noiseless coding)

**Théorème fondamental** : L'entropie $H(X)$ est la **borne inférieure** de la longueur moyenne de tout code satisfaisant la propriété de préfixe.

Plus précisément : pour tout dictionnaire de codes $\text{code}[\cdot]$ donnant des longueurs $L(x)$,
$\text{Longueur moyenne} = \sum_x p(x) \cdot L(x) \geq H(X)$

avec égalité si et seulement si $L(x) = \log_2(1/p(x))$ pour tout $x$ (codes "fractional bits").

**Conséquence pratique** : aucun algorithme de compression ne peut faire mieux que l'entropie en moyenne. C'est la **limite théorique absolue**.

### C. Cross-entropy et divergence KL

#### Le problème d'Alice et Bob

Supposons qu'Alice et Bob aient des distributions différentes sur le même alphabet.

| Distribution | dog | cat | fish | bird |
|-------------|-----|-----|------|------|
| Alice (p)   | 0.5 | 0.25| 0.125| 0.125|
| Bob (q)     | 0.4 | 0.4 | 0.1  | 0.1  |

Alice optimise son code pour sa distribution $p$ : $H(p) = 1.75$ bits/mot.
Bob optimise son code pour sa distribution $q$ : $H(q) = 1.92$ bits/mot.

**Comparaison visuelle** des deux dictionnaires :

| Colonne 1                                 | Colonne 2                                 |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260421100037.png\|308]] | ![[Pasted image 20260421095859.png\|305]] |

**Question** : que se passe-t-il si Alice utilise le code de Bob ?

#### Cross-entropy

La **cross-entropy** $H_p(q)$ mesure le coût moyen d'encoder la distribution $p$ avec le code optimal pour $q$ :

$$\boxed{H_p(q) = \sum_{x} p(x) \log_2\left(\frac{1}{q(x)}\right) = -\sum_{x} p(x) \log_2 q(x)}$$

Dans notre exemple :
$$H_p(q) = 0.5 \cdot \log_2(1/0.4) + 0.25 \cdot \log_2(1/0.4) + 0.125 \cdot \log_2(1/0.1) + 0.125 \cdot \log_2(1/0.1)$$
$$= 0.5 \cdot 1.32 + 0.25 \cdot 1.32 + 0.125 \cdot 3.32 + 0.125 \cdot 3.32 = 2.32 \text{ bits}$$

Alice paie un **surcoût** de 2.32 - 1.75 = 0.57 bits par mot en utilisant le mauvais code.

**Propriété clé** : $H_p(q) \neq H_q(p)$ (non-symétrie)

**Récapitulatif des quatre possibilités** :

![[Pasted image 20260421102706.png|497]]

- Bob utilisant son propre code : $H(p) = 1.75$ bits
- Alice utilisant le code de Bob : $H_p(q) = 2.25$ bits
- Alice utilisant son propre code : $H(q) = 1.75$ bits  
- Bob utilisant le code d'Alice : $H_q(p) = 2.375$ bits

#### Applications en Machine Learning

**Cross-entropy loss** en classification :

En apprentissage supervisé, on cherche à approximer la vraie distribution $p(y|x)$ par un modèle $q_\theta(y|x)$. La cross-entropy loss est :

$$\mathcal{L}(\theta) = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log q_\theta(y_c|x_i)$$

où $y_{i,c}$ est l'indicatrice de la vraie classe.

📌 **Interprétation** : on minimise le "coût de compression" de la vraie distribution avec notre modèle. Plus le modèle est proche de la vérité, moins on gaspille de bits.

![[Pasted image 20260421101915.png|133]]
*Exemple : matrice de confusion comme distribution jointe*

#### Divergence KL

La **divergence de Kullback-Leibler** mesure l'écart entre deux distributions :

$$\boxed{KL(p \parallel q) = H_p(q) - H(p) = \sum_{x} p(x) \log_2\left(\frac{p(x)}{q(x)}\right)}$$

📌 **Interprétations** :
- **Compression** : surcoût moyen d'utiliser $q$ au lieu de $p$
- **Information** : information supplémentaire nécessaire pour corriger $q$ vers $p$

**Propriétés importantes** :
- $KL(p \parallel q) \geq 0$ avec égalité ssi $p = q$
- $KL(p \parallel q) \neq KL(q \parallel p)$ (non-symétrie)

**Visualisation de la non-symétrie** :

| Colonne 1                                 | Colonne 2                                   |
| ----------------------------------------- | ------------------------------------------- |
| ![[Pasted image 20260421125006.png\|229]] | ![[Pasted image 20260421125015.png\|217]] |

*Ces images illustrent que $KL(p \parallel q) \neq KL(q \parallel p)$ selon la direction de la divergence.*

#### Applications : Maximum Likelihood et approximation

**Lien avec MLE** :

Minimiser la KL divergence entre la vraie distribution $p$ et un modèle paramétrique $q_\theta$ :
$$\min_\theta KL(p \parallel q_\theta) = \min_\theta \left[ H_p(q_\theta) - H(p) \right]$$

Comme $H(p)$ ne dépend pas de $\theta$, cela équivaut à minimiser $H_p(q_\theta)$, soit maximiser la vraisemblance !

**En pratique** : avec un dataset $\{x_i\}_{i=1}^n$, on approxime l'espérance par la moyenne empirique :
$$KL(p \parallel q_\theta) \approx \frac{1}{n} \sum_{i=1}^n \log q_\theta(x_i) + \text{constante}$$

C'est exactement la log-vraisemblance négative.

---

## III. Plusieurs variables : communication avec bruit

### A. L'exemple du canal bruité

Passons maintenant au cas de **deux variables** $X$ (émetteur) et $Y$ (récepteur) reliées par un canal de communication potentiellement bruité.

#### Le scénario

$X$ (émetteur Bob) veut transmettre des mots à $Y$ (récepteur Alice) à travers un canal de communication potentiellement bruité. Voici la distribution jointe observée :

$$p(X,Y) := \begin{array}{l | ccc || c}
 & x = \text{Dog} & x = \text{Cat} & x = \text{Fish} & p(y) \\
\hline
y = \text{Dog}  & 0.25  & 0     & 0.125 & 0.375 \\
y = \text{Cat}  & 0     & 0.125 & 0     & 0.125 \\
y = \text{Fish} & 0.125 & 0.25  & 0.125 & 0.500 \\
\hline \hline
p(x)            & 0.375 & 0.375 & 0.250 & 1.000
\end{array}$$

Cette table représente la probabilité jointe $p(X=x, Y=y)$ d'observer simultanément Bob qui envoie $x$ et Alice qui reçoit $y$.

#### Distribution conditionnelle du canal

| **Table des probabilités conditionnelles** | **Graphique du canal** |
|---|---|
| $p(Y \mid X) := \begin{array}{l \| ccc} & y = \text{Dog} & y = \text{Cat} & y = \text{Fish} \\ \hline x = \text{Dog}  & 0.667 & 0     & 0.333 \\ x = \text{Cat}  & 0     & 0.333 & 0.667 \\ x = \text{Fish} & 0.5   & 0     & 0.5   \\ \end{array}$ | ![[Pasted image 20260422092206.png\|313]] |

*Le tableau montre $p(Y=y\|X=x)$ : la probabilité qu'Alice reçoive $y$ sachant que Bob envoie $x$. Le graphique illustre ce canal de transmission.*

📌 **Interprétation concrète** : quand Bob envoie $X = \text{"Cat"}$, Alice reçoit $Y = \text{"Fish"}$ dans $p(Y = \text{Fish} | X = \text{Cat}) = 0.667$ soit 67% des cas ! Le canal est donc très bruité.



### B. Entropies jointes et conditionnelles

#### Entropie jointe

L'**entropie jointe** mesure l'incertitude totale du système :

$$H(X, Y) = \sum_{x,y} p(x,y) L(x,y) =\sum_{x,y} p(x,y) \log_2\left(\frac{1}{p(x,y)}\right)$$

![[Pasted image 20260422092720.png|189]]
Figure. Visualisation de la distribution pour l'entropie jointe.

#### Entropie conditionnelle

L'**entropie conditionnelle** $H(Y|X)$ mesure l'incertitude sur $Y$ une fois $X$ connu :

$$H(Y|X) = \sum_x p(x) \sum_y p(y|x) \log_2\left(\frac{1}{p(y|x)}\right) = \sum_{x,y} p(x,y) \log_2\left(\frac{1}{p(y|x)}\right)$$

📌 **Interprétation** : c'est le "bruit moyen" du canal. Si Bob dit "Cat", Alice hésite encore entre "Cat" et "Fish".

Dans notre exemple : $H(Y|X) = 0.938$ bits/mot.

#### Chain rule pour l'entropie

$$\boxed{H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)}$$

**Preuve intuitive** : pour encoder $(x,y)$, on peut d'abord encoder $x$ (coût $H(X)$), puis encoder $y$ sachant $x$ (coût $H(Y|X)$).

### C. Information mutuelle

#### Définition et interprétation

L'**information mutuelle** quantifie l'information partagée entre $X$ et $Y$ :

$$\boxed{I(X,Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)}$$

📌 **Interprétations** :
- **Réduction d'incertitude** : combien connaître $X$ réduit l'incertitude sur $Y$
- **Information transmise** : quelle fraction de l'information survit au canal

Dans notre exemple :
$$I(X,Y) = H(Y) - H(Y|X) = 1.41 - 0.938 = 0.47 \text{ bits/mot}$$

#### Ratio de qualité du canal

Pour évaluer la performance du canal, on peut calculer :
$$\text{Ratio} = \frac{I(X,Y)}{H(X)} = \frac{0.47}{1.16} \approx 40\%$$

**Interprétation** : environ 40% de l'information envoyée par Bob survit au canal. Les 60% restants sont perdus à cause du bruit.

#### Propriétés de l'information mutuelle

- $I(X,Y) \geq 0$ avec égalité ssi $X$ et $Y$ sont indépendants
- $I(X,Y) = I(Y,X)$ (symétrie, contrairement à KL)
- $I(X,Y) \leq \min(H(X), H(Y))$

#### Relation avec la divergence KL

$$I(X,Y) = KL(p(x,y) \parallel p(x)p(y))$$

L'information mutuelle mesure à quel point la distribution jointe s'écarte du cas indépendant.

### D. Variation of Information

La **variation of information** est une vraie distance entre variables :

$$\boxed{VI(X,Y) = H(X|Y) + H(Y|X)}$$

**Applications pratiques** :

1. **Comparer des technologies** : évaluer $VI(X, Y_1)$ vs $VI(X, Y_2)$ pour deux systèmes de transmission différents

2. **Mesurer la fidélité** : calculer $VI(Y_1, Y_2)$ entre les sorties de deux technologies pour voir si elles "racontent la même histoire"

**Avantages sur KL** :
- $VI(X,Y) = VI(Y,X)$ (symétrie)
- $VI(X,Y) = 0$ ssi $X$ et $Y$ sont déterministiquement liés
- Satisfait l'inégalité triangulaire

**Interprétation** : c'est le "carnet de réparation" — la quantité totale d'information manquante pour parfaitement prédire une variable à partir de l'autre.

#### Applications en Machine Learning

**Clustering evaluation** : comparer deux partitions $\mathcal{P}_1$ et $\mathcal{P}_2$ d'un dataset

**Feature selection** : sélectionner les features qui maximisent $I(X_i, Y)$

**Model comparison** : évaluer la cohérence entre modèles via $VI(\text{Model}_1, \text{Model}_2)$

---

## IV. Extension au cas continu

### A. Motivation : Pourquoi le cas continu ?

Dans un fil de cuivre ou dans l'air, on ne peut pas envoyer des "carrés" parfaits de 0V et 5V. La physique fait que ça bave partout. On utilise donc des **sinus** parce qu'ils se propagent très bien :
- Envoyer un "1", c'est envoyer un sinus qui vibre fort : $5 \sin(\omega t)$
- Envoyer un "0", c'est envoyer un sinus qui vibre peu : $1 \sin(\omega t)$

Quand on plot nos trajectoires, on ne regarde que le **sommet du sinus** — chaque point représente la valeur d'un sommet reçu après avoir traversé le canal. La trajectoire, **c'est l'évolution de l'amplitude de ton sinus**.

Si on enlève le bruit, cette amplitude serait une **ligne droite parfaite** à 5V. Mais le bruit (chaleur, interférences) fait que ton "5V" se transforme en une **distribution Gaussienne** centrée sur 5V.

Cela nous amène naturellement aux distributions continues et à l'entropie différentielle.

### B. Entropie différentielle

#### Définition

Pour une variable continue $X$ de densité $p(x)$ :

$\boxed{h(X) = -\int_{-\infty}^{\infty} p(x) \log p(x) \, dx}$

**Attention cruciale** : contrairement au cas discret, $h(X)$ peut être **négative** !
[]
#### Exemple : loi gaussienne

Pour $X \sim \mathcal{N}(\mu, \sigma^2)$ :
$h(X) = \frac{1}{2} \log(2\pi e \sigma^2)$

- Si $\sigma^2 < 1/(2\pi e) \approx 0.058$, alors $h(X) < 0$
- Plus la variance est faible, plus l'entropie est négative

C'est la formule spécifique à la distribution Gaussienne. En théorie de l'information on utilise cette distribution car pour une puissance donnée, c'est elle qui possède l'entropie la plus élevée. C'est le "pire" bruit possible.

#### Problème des unités

L'entropie différentielle **dépend des unités** ! Si on mesure une longueur en mètres puis en centimètres :
$h(X_{\text{cm}}) = h(X_{\text{m}}) + \log(100) = h(X_{\text{m}}) + 4.61 \text{ nats}$

**Conséquence** : on ne peut pas comparer directement des entropies différentielles de variables dans des unités différentes.

### C. Information mutuelle continue

Heureusement, l'**information mutuelle reste bien définie** :

$I(X,Y) = \iint p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right) dx \, dy$

**Propriété clé** : $I(X,Y)$ est **invariante par transformation monotone** — elle ne dépend pas des unités.

#### Applications en ML

**Feature selection** : maximiser $I(X_i, Y)$ pour sélectionner les features les plus informatives

**Independent Component Analysis (ICA)** : minimiser $I(S_1, S_2, \ldots, S_n)$ entre les composantes

### D. Exemple concret : Canal de communication

#### Modélisation du bruit

Dans la réalité d'un câble, le bruit agit de manière continue. On résout une **Équation différentielle stochastique** (EDS). On utilise un processus d'Ornstein-Uhlenbeck :
$dV_t = -\theta V_t dt + \sigma dW_t$

- **$dW_t$** : C'est le mouvement brownien (le bruit pur)
- **$-\theta V_t dt$** : Tendance du signal à revenir vers zéro à cause de la résistance du câble

Pour simplifier, on peut utiliser le modèle discret : $V_t = V_{t-1} + \varepsilon_t$

#### Calcul de l'entropie du bruit

On utilise la formule Gaussienne :
$h(N) = \frac{1}{2} \log_2(2\pi e \sigma^2)$

Où $\sigma^2$ est la puissance du bruit. On a $\sigma_{\text{total}} = \sigma_{\text{choc}} \times \sqrt{\text{longueur canal}}$ car les bruits s'ajoutent de manière quadratique.

#### Résultats numériques

Sur un exemple concret :
- **Dispersion finale** (Sigma total) : 3.50 V
- **Entropie différentielle du bruit** : 3.85 bits
- **Taux d'erreur binaire (BER)** : 23.72%

📌 **Interprétation** : On a 3.85 bits d'incertitude "gratuite" vs 1 bit de message. Comme 3.85 > 1, le bruit recouvre largement le signal. C'est comme essayer d'entendre un murmure (1 bit) au milieu d'un réacteur d'avion (3.85 bits).

#### Visualisation du problème

![[Pasted image 20260422120921.png]]
*Figure : Modélisation du "bordel" — les cloches se chevauchent*

La zone **marron** représente l'incertitude : quand le voltage tombe dans cette zone, le récepteur ne sait plus si c'est un "0" bruité ou un "1" affaibli. Plus cette zone est large, plus l'Information Mutuelle diminue.

#### Solutions pratiques

Comment sauver cette liaison ? Trois leviers (tous touchent à l'entropie) :

1. **Augmenter la puissance ($S$)** : Passer de 5V à 20V écarte les cloches. La zone marron diminue.

2. **Réduire le bruit ($N$)** : Refroidir le composant ou mieux blinder le câble pour baisser $\sigma$.

3. **Codes correcteurs d'erreurs** : Même avec 23% d'erreurs, des bits de contrôle intelligents permettent de retrouver le message original.

#### Formule de Shannon et capacité du canal

La célèbre formule de Shannon :
$C = \log_2\left(1 + \frac{S}{N}\right)$

- $C$ : nombre de bits par seconde transmissibles
- $S/N$ : rapport Signal/Bruit

**Comment ça devient des paliers ?**
- Si $C = 3$, on a assez de "place" pour $2^3 = 8$ paliers distincts
- Si le bruit augmente et $C = 1$, il ne reste que $2^1 = 2$ paliers (0V ou 5V)

### E. Applications modernes en ML

#### KL divergence continue

$KL(p \parallel q) = \int p(x) \log\left(\frac{p(x)}{q(x)}\right) dx$

#### Applications

**Variational Autoencoders (VAE)** : la loss ELBO contient un terme KL entre l'encodeur $q_\phi(z|x)$ et le prior $p(z)$

**Normalizing Flows** : minimiser $KL(p_{\text{data}} \parallel p_{\text{model}})$ pour apprendre des distributions complexes

**Generative Adversarial Networks (GANs)** : sous certaines conditions, minimiser la divergence JS (liée à KL)

---

## Synthèse et connexions

| Concept | Cas discret | Cas continu | Application ML |
|---------|-------------|-------------|----------------|
| Entropie | $H(X) = -\sum p(x) \log p(x)$ | $h(X) = -\int p(x) \log p(x) dx$ | Mesures d'impureté (arbres) |
| Cross-entropy | $H_p(q) = -\sum p(x) \log q(x)$ | $h_p(q) = -\int p(x) \log q(x) dx$ | Loss function classification |
| KL divergence | $\sum p(x) \log \frac{p(x)}{q(x)}$ | $\int p(x) \log \frac{p(x)}{q(x)} dx$ | MLE, VAE, modèles génératifs |
| Information mutuelle | $\sum p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$ | $\iint p(x,y) \log \frac{p(x,y)}{p(x)p(y)} dx dy$ | Feature selection, ICA |

La théorie de l'information offre un cadre unifié pour comprendre compression, transmission, et apprentissage. De Shannon aux réseaux de neurones modernes, ces concepts restent au cœur de l'IA.
