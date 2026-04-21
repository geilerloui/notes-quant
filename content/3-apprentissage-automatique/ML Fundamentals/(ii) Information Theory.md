To do


https://colah.github.io/posts/2015-09-Visual-Information/

surtout ça a réécrire les trucs utiles 

---
Motivation : Quel est le but de l'information theory ?



apparament la source est soit un "sender" soit un "receiver"

# Section 1 : code à longueur fixe 


schéma: trois boites: source -> symboles -> mot de code (codewords) -> output on envoie la phrase "dog cat fish bird" on va juste concaténer : 00 + 01 + 10 = 000110

![[Pasted image 20260420232052.png]]



C'est l'alphabet I guess ? $\mathcal{A} = \{dog, cat, fish, bird\}$ en fait j'ai l'impression que ce A l'alphabet c'est une variable aléatoire


* alphabet de la source  = {dog, cat, fish, bird} un élément de l'alphabet c'est ce qu'on appelle un symbole eg "dog". 
* Le code c'est un dictionnaire eg code_1["dog"] = 00, code_1["cat] = 01

Il y'a deux grands types de codes:
* code à longueur fixe : eg 2 bit pour tout le monde on a une longueur $L(x)=2$ et $p(x)$ probabilité de chaque symbole

![[Pasted image 20260420200532.png|421]]

# Section 2 : code à longueur variable 

## Exemple du graphe

* Code à longueur variable : il permet de prendre en compte la fréquence d'utilisation des symboles eg "dog" arrive 1/2 (50% du temps), "cat" arrive (25% du temps), "fish" et "bird" arrivent 1/8 (12.5% du temps). De plus si on dit que 1 bit = 5$ on veut minimiser le coût total du message. 

![[Pasted image 20260420232145.png]]



écriture alternative






![[Pasted image 20260420200627.png|413]]


## La "prefix property"


![[Pasted image 20260420200637.png|331]]


Il dit que si code["dog"] = 0 et code["cat"] = 01 , dés que tu choisis un code de longueur $L(x)$ tu sacrifies l'accès aux bit suivants dans ce cas là pour "cat" on bloque l'accès à 2/8 bits ce qui donne $1/2^2$ ou $1/2^{L(x)}$ 

=> La règle principale à retenir c'est plus ton code est long $L(x)$ grand plus son coût est bas 

Puis on observe que que notre équations est < 1 donc on peut dire que 

$$
p(x) = \frac{1}{2^{L(x)}} \iff L(x) = log_2\Big( \frac{1}{p(x)} \Big)
$$

## Apparament faudrait parler aussi des fractional bits




# Une seule variable aléatoire
## L'entropie

**Définition (Entropie).** C'est la limite absolu de compression. Le nombre minimum de bit "câchés dans une information".

$$
H(p) = \sum_x p(x) L(x) =\sum_x p(x) log_2\Big( \frac{1}{p(x)} \Big)
$$
**Interprétation (Graphique).** Une $H(p)=0.5$ c'est la "surprise" maximale c'est comme une loi uniforme je n'ai aucune information sur qui est le vainqueur. En langage de théorie de l'information c'est qu'on a un code fixe pour chaque symbole.

![[im2-3.png|364]]
Figure X. Figure de l'entropie

**Eg (du code à longueur variable).** $H(p) = 1.75\text{ bit/mot}$


relation entre entropie et : noiseless coding theorem Shannon 1948 states that the entropy is a lower bound on the nb of bits needed to transmit the state of a random variable.

remarque 2: l'entropie c'est une espérance en fait 

remarque 3: aucune idée pk le mec parle de ça mais on peut calculer le maximum entropy en résolvant le lagrangien de l'entropie avec contrainte que la some des p(x_i)=1 enfait je crois c juste pour prouver mathématiquement que le max c une uniforme law

## La Cross-Entropy

**Définition (Cross-Entropy).** blablaba
$$
H_p(q)= \sum_x q(x) L^{Bob}(x) =\sum_x q(x) \log _2\left(\frac{1}{p(x)}\right)
$$

$L^{Bob}(x)$ c'est la longueur du dictionaire eg pour $x=dog$ : $len(dict[\text{"dog"}]) = 1$ 


Si on compare les deux où Alice a un autre code elle utilise plus souvent les mots cat -> fish -> big & dog; alors que bob c'est dog -> cat -> fish & bird

| Colonne 1                                 | Colonne 2                                 |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260421100037.png\|308]] | ![[Pasted image 20260421095859.png\|305]] |

**Propriétés (non-symétrie).** On a la propriété que $H_p(q) \neq H_q(p)$ 
So, now we have four possibilities:

- Bob using his own code (H(p)=1.75 bits)
- Alice using Bob’s code (Hp(q)=2.25 bits)
- Alice using her own code (H(q)=1.75 bits)
- Bob using Alice’s code (Hq(p)=2.375 bits)


![[Pasted image 20260421102706.png|497]]


**Exemple (Machine Learning).** Dans un problème de classification supervisé on va écrire que  $\mathcal{A} = \{dog, cat\}$ avec $dog$ en vert et $cat$ en rose.

de ce que j'ai compris la matrice de confusion peut être vue comme une probabilité jointe $(X,Y)$


![[Pasted image 20260421101915.png|133]]
Table de prédiction

Puis on a pour chaque distribution de probabilité des y et y hat : 

| Colonne 1                                 | Colonne 2                                 |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260421175425.png\|299]] | ![[Pasted image 20260421175326.png\|374]] |
La formule ce serait 

$$
H_p(q) = - \sum_x p(y\mid x) L^{ \hat{y}}(y \mid x) =- \sum_x p(y\mid x) log_2(q_{\theta}(y \mid x))
$$
Cross-Entropy $=-\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i, c} \log \left(\hat{y}_{i, c}\right)$








en général on fait 

$$
H_p(q) = -\sum_x p(x) log_2(q(x))
$$

donc si on développe $H_p(q)=-(1\cdot log(0.8) + 0 \cdot log_2(0.2)) = 0.322 \text{ bit/mots}$

> En ML plus c'est petit plus ça veut dire que ma longueur de la target eg dog est bien compressé donc beaucoup utilisé. Le but du ML est de chercher à compresser le message le plus possible pour mieux représenter la classe à prédire.


En gros j'ai l'impression que l'objectif est de trouver le meilleur "dictionaire" celui qui map un symbole à un code qui va permettre la compression la plus optimale.

En fait faudrait dire que 

$$
\mathcal{L}(??) = \underset{?}{min}~H_p(q)
$$

> Peut être faut préciser que la cross-entropy est utilisé dans le cadre discriminatif + peut etre rajouter un menu déroulant câché pour rappeler la différence entre les deux modèles.




Definition (Differentiel entropy).
$$
H(p) = \int p(x) log_2(p(x)) = \mathbb{E}_{x \sim p(x)}[log_2(p(x))]
$$




## KL divergence or Relative Entropy


Belle image qui te montrer je sais plus quoi KL divergence n'esrt pas symétrique 

| Colonne 1                                 | Colonne 2                                   |
| ----------------------------------------- | ------------------------------------------- |
| ![[Pasted image 20260421125006.png\|229]] | ![[Pasted image 20260421125015.png\|217]]\| |

Definition (KL Divergence discrète). C'est l'erreur de compression ou je sais plus quoi 

$$
KL(p || q) = D_q(p)=H_q(p)-H(p) = \sum_x p(x) log_2\Big( \frac{p(x)}{q(x)} \Big)
$$

c'est la distance entre deux distribution pour savoir how different they are


Definition (KL Divergence Continu). On en définit deux différentes
(i) Forward KL : $KL(p ||q)$ 
(ii) Reverse KL : $KL(p ||q)$ 


Exemple (ML). Consider some unknown distribution $p(\mathbf{x})$, and suppose that we have modelled this using an approximating distribution $q(\mathbf{x})$. If we use $q(\mathbf{x})$ to construct a coding scheme for the purpose of transmitting values of $\mathbf{x}$ to a receiver, then the average additional amount of information (in nats) required to specify the value of $\mathbf{x}$ (assuming we choose an efficient coding scheme) as a result of using $q(\mathbf{x})$ instead of the true distribution $p(\mathbf{x})$ is given by

la il donne la formule

We see that there is an intimate relationship between data compression and density estimation (i.e., the problem of modelling an unknown probability distribution) because the most efficient compression is achieved when we know the true distribution. If we use a distribution that is different from the true one, then we must necessarily have a less efficient coding, and on average the additional information that must be transmitted is (at least) equal to the Kullback-Leibler divergence between the two distributions.

Suppose that data is being generated from an unknown distribution $p(\mathbf{x})$ that we wish to model. We can try to approximate this distribution using some parametric distribution $q(\mathbf{x} \mid \boldsymbol{\theta})$, governed by a set of adjustable parameters $\boldsymbol{\theta}$, for example a multivariate Gaussian. One way to determine $\boldsymbol{\theta}$ is to minimize the Kullback-Leibler divergence between $p(\mathbf{x})$ and $q(\mathbf{x} \mid \boldsymbol{\theta})$ with respect to $\boldsymbol{\theta}$. We cannot do this directly because we don't know $p(\mathbf{x})$. Suppose, however, that we have observed a finite set of training points $\mathbf{x}_n$, for $n=1, \ldots, N$, drawn from $p(\mathbf{x})$. Then the expectation with respect to $p(\mathbf{x})$ can be approximated by a finite sum over these points, using (1.35), so that

$$
\begin{equation*}
\mathrm{KL}(p \| q) \simeq \sum_{n=1}^N\left\{-\ln q\left(\mathbf{x}_n \mid \boldsymbol{\theta}\right)+\ln p\left(\mathbf{x}_n\right)\right\} . \tag{1.119}
\end{equation*}
$$

The second term on the right-hand side of (1.119) is independent of $\boldsymbol{\theta}$, and the first term is the negative log likelihood function for $\boldsymbol{\theta}$ under the distribution $q(\mathbf{x} \mid \boldsymbol{\theta})$ evaluated using the training set. Thus we see that minimizing this Kullback-Leibler divergence is equivalent to maximizing the likelihood function.


> Peut être faut préciser que la cross-entropy est utilisé dans le cadre génératif + peut etre rajouter un menu déroulant câché pour rappeler la différence entre les deux modèles.


Exemple (statistique). le MLE c'est un KL 

Exemple (Modèle génératif). A compléter

# Multiple Variables


On va avoir X la source (l'émetteur) et Y la destination (le récepteur). Quand la source envoie le symbole, la destination reçoit "Dog" avec une probabilité de 25% ou "Fish" avec 25% => le canal est bruité. 

$$
p(X,Y) := ~~
\begin{array}{l | ccc || c}
 & x = \text{Dog} & x = \text{Cat} & x = \text{Fish} & p(y) \\
\hline
y = \text{Dog}  & 0.25  & 0     & 0.125 & 0.375 \\
y = \text{Cat}  & 0     & 0.125 & 0     & 0.125 \\
y = \text{Fish} & 0.125 & 0.25  & 0.125 & 0.500 \\
\hline \hline
p(x)            & 0.375 & 0.375 & 0.250 & 1.000
\end{array}
$$

On écrit également la probabilité conditionnelle ce qui veut dire que quand Bob envoie "Cat" le canal se trompe est comprends "Fish" dans 67% des cas. En calculant $H(Y \mid X)$ c'est le bruit, si Bob dit "Cat" le récepteur hésite entre "Cat" et "Fish". Si je généralise à l'ensemble des symboles on peut dire que $H(Y \mid X)=0.938 \text{ bit/mots}$  
$$
p(Y \mid X) := ~~
\begin{array}{l | ccc}
 & y = \text{Dog} & y = \text{Cat} & y = \text{Fish} \\
\hline
x = \text{Dog}  & 0.667 & 0     & 0.333 \\
x = \text{Cat}  & 0     & 0.333 & 0.667 \\
x = \text{Fish} & 0.5   & 0     & 0.5   \\
\end{array}
$$


![[Pasted image 20260421171731.png|285]]
Figure X. Canal $p(Y\mid X)$ 


Ensuite on introduit la notion de mutual information 

$$
I(X,Y) = \underbrace{H(Y)}_{\text{ce que l'on reçoit}} - \underbrace{H(Y \mid X)}_{\text{le bruit moyen}} = 0.47\text{ bits/mots}
$$
Avec une valeur de 0.47 bits / mots le canal est très mauvais.

Mais en pratique on va plutôt calculer le ratio 

$$
Ratio = \frac{I(X,Y)}{H(X)} = \begin{cases} \text{Si on a 1 c'est parfait} \\ \text{Si on a 0 c'est caca} \end{cases}
$$

Dans notre exemple on a un $Ratio=40\%$ c'est comme si sur chaque phrase envoyé, environ 60% était perdu à cause des interférences.
* Usage pratique : Tu peux dire "Mon canal a un ratio de 0.95" tout le monde comprend c'est excellent si tu envoies des photos 4k ou du texte simple.
Le problème : ça ne dit pas pourquoi ça raté. ça te dit juste que c'est raté.


L'intérêt principal de la VI c'est le "carnet de réparation" 

$$
VI(X,Y) = H(X \mid Y) + H(Y \mid X)
$$
On va l'utiliser pour comparer deux technos différentes de transmission de l'information entre $X \rightarrow Y$ En gros on aura 
* Le couple $\left(X, Y_1\right) \rightarrow$ donne la matrice $P\left(X, Y_1\right)$.
* Le couple $\left(X, Y_2\right) \rightarrow$ donne la matrice $P\left(X, Y_2\right)$.
On peut mesurer $VI(X, Y_1)$ vs $VI(X, Y_2)$ c'est mon score de qualité on veut que la distance soit la plus courte possible.

On peut également comparer la fidélité entre techno (Sortie 1 et Sortie 2) en faisant $VI(Y_1, Y_2)$ si leur distance est très faible les deux technos racontent la même 


va être de comparer plusieurs 


=> en gros ça te dit "quel pourcentage du message originel a survécu au voyage ?"


Probablement rajouté 


Variation of Information c'est 


Definition (Joint entropy). 

$$
H(X, Y) = \sum_x p(x,y)L(x,y)= \sum_x p(x,y)log_2\Big( \frac{1}{p(x,y)} \Big)
$$

TO ADD IMAGE 



Chain Rule for Entropy
$$
H(X, Y) = H(Y) + H(X | Y)
$$
ok
$$
\begin{aligned}
H(X \mid Y) & =\sum_y p(y) \sum_x p(x \mid y) \log _2\left(\frac{1}{p(x \mid y)}\right) \\
& =\sum_{x, y} p(x, y) \log _2\left(\frac{1}{p(x \mid y)}\right)
\end{aligned}
$$






En gros le $H(X,Y)$ , le $H(X|Y)$ , $I(X,Y)$ mutual information et le $V(X,Y)$ variation of information.

qq trucs intéressant

Now consider the joint distribution between two sets of variables $\mathbf{x}$ and $\mathbf{y}$ given by $p(\mathbf{x}, \mathbf{y})$. If the sets of variables are independent, then their joint distribution will factorize into the product of their marginals $p(\mathbf{x}, \mathbf{y})=p(\mathbf{x}) p(\mathbf{y})$. If the variables are not independent, we can gain some idea of whether they are 'close' to being independent by considering the Kullback-Leibler divergence between the joint distribution and the product of the marginals, given by

$$
\begin{aligned}
\mathrm{I}[\mathbf{x}, \mathbf{y}] & \equiv \operatorname{KL}(p(\mathbf{x}, \mathbf{y}) \| p(\mathbf{x}) p(\mathbf{y})) \\
& =-\iint p(\mathbf{x}, \mathbf{y}) \ln \left(\frac{p(\mathbf{x}) p(\mathbf{y})}{p(\mathbf{x}, \mathbf{y})}\right) \mathrm{d} \mathbf{x} \mathrm{~d} \mathbf{y}
\end{aligned}
$$
Using the sum and the product rules or probability, we see that the mutual information is related to the conditional entropy through 

$$
\mathrm{I}[\mathbf{x}, \mathbf{y}]=\mathrm{H}[\mathbf{x}]-\mathrm{H}[\mathbf{x} \mid \mathbf{y}]=\mathrm{H}[\mathbf{y}]-\mathrm{H}[\mathbf{y} \mid \mathbf{x}] .
$$

Thus we can view the mutual information as the reduction in the uncertainty about $\mathbf{x}$ by virtue of being told the value of $\mathbf{y}$ (or vice versa).

REMARQUE: How does this relate to KL divergence, which also gave us a notion of distance? Well, KL divergence gives us a distance between two distributions over the same variable or set of variables. In contrast, variation of information gives us distance between two jointly distributed variables. KL divergence is between distributions, variation of information within a distribution.

**Applications (Machine learning).** From a Bayesian perspective, we can view $p(\mathbf{x})$ as the prior distribution for $\mathbf{x}$ and $p(\mathbf{x} \mid \mathbf{y})$ as the posterior distribution after we have observed new data $\mathbf{y}$. The mutual information therefore represents the reduction in uncertainty about $\mathbf{x}$ as a consequence of the new observation $\mathbf{y}$.



![[Pasted image 20260421153921.png|203]]
Figure X. Venn Diagramme