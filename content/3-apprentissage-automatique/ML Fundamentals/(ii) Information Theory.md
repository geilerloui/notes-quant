To do


https://colah.github.io/posts/2015-09-Visual-Information/

surtout ça a réécrire les trucs utiles 

---
Motivation : Quel est le but de l'information theory ?


# Section 1 : code à longueur fixe 


schéma: trois boites: source -> symboles -> mot de code (codewords) -> output on envoie la phrase "dog cat fish bird" on va juste concaténer : 00 + 01 + 10 = 000110

![[Pasted image 20260420232052.png]]



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

## L'entropie

**Définition (Entropie).** C'est la limite absolu de compression. Le nombre minimum de bit "câchés dans une information".

$$
H(p) = \sum_x p(x) L(x) =\sum_x p(x) log_2\Big( \frac{1}{p(x)} \Big)
$$
**Interprétation (Graphique).** Une $H(p)=0.5$ c'est la "surprise" maximale c'est comme une loi uniforme je n'ai aucune information sur qui est le vainqueur. En langage de théorie de l'information c'est qu'on a un code fixe pour chaque symbole.

![[im2-3.png|364]]
Figure X. Figure de l'entropie

**Eg (du code à longueur variable).** $H(p) = 1.75\text{ bit/mot}$

## La Cross-Entropy

**Définition (Cross-Entropy).** blablaba
$$
H_p(q)= \sum_x q(x) L^{Bob}(x) =\sum_x q(x) \log _2\left(\frac{1}{p(x)}\right)
$$

$L^{Bob}(x)$ c'est la longueur du dictionaire eg pour $x=dog$ : $len(dict[\text{"dog"}]) = 1$ 


Si on compare les deux
![[Pasted image 20260420232415.png]]

OK

![[Pasted image 20260420232537.png]]