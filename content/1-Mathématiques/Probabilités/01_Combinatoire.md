ok
$$
\binom{n}{k}=\frac{n!}{k!(n-k)!}
$$


ok

| Type | Ordre compte ? | Répétition autorisée ? | Formule |
| --- | --- | --- | --- |
| Permutation | Oui | Non | $n!$ |
| Arrangement | Oui | Non | $\frac{n!}{(n-k)!}$ |
| Combinaison | Non | Non | $\binom{n}{k}$ |
| Tirage avec remise | Oui | Oui | $n^k$ |

## Tirage avec remise : 

exemple 1: combien y'a t-il de mots de trois lettres composés des lettres "a" et "b"

$$
2 \times 2 \times 2 = 8
$$

exemple 2: combien y'a t-il de séquence de 3 bits composés de "0" et de "1"

On peut énumérer l'ensemble des cas

000
001
010
011
100
101
110
111

ça fait huit cas ou sinon on peut juste dire que on a n=3 et k=2 donc 3^2=9

Exemple : Combien peut-on faire de mots de cinq lettres avec l'alphabet A, B, C, .., Z ? 
![[Pasted image 20260609175222.png|626]]

## Arrangement


Exemple : On a 10 chevaux, combien y-a-t'il de tiercé ? (i.e. nombre de podiums à trois gagnants)

![[Pasted image 20260609175347.png|401]]

On aura 10 chevaux initialement puis plus que 9 possible puis 8 possible

$$
10 \times 9 \times 8 = 720 \text{ combinaisons}
$$

Si on note les chevaux de $C_1, .., C_{10}$ ça revient à avoir l'ensemble des arrangements

$[\{C_1, C_2, C_3\}, \{C_1, C_2, C_4\}, \{C_1, C_2, C_5\} ... ]$


$$
A_n^p = \frac{n!}{(n-p)!} = \frac{10!}{(10-3)!} = \frac{10!}{7!}
$$


Théorie : 
![[Pasted image 20260609175929.png|508]]
Puis on multiplie les éléments
$$
\begin{aligned}
&n (n-1) ... (n-(p-1)) \\
&= \frac{n(n-1) ... (n-p+1)(n-p)}{(n-p) \times ... \times 1} = \frac{n!}{(n-p)!} = A_n^p
\end{aligned}
$$

## Permutation

Exemple : Quatre personnes font une course combien y'a t-il d'ordres d'arrivés

![[Pasted image 20260609180326.png|561]]

On peut tracer le graphe, puis en multipliants les nombres de possibilités de chaque segment on obtient:
$$
4 \times 3 \times 2 \times 1 = 4!
$$


Remarque ! Si on compare la permutation à l'arrangement on voit juste que genre dans arrangement ben la j'avais trop de chevaux par rapport à l'arrivée. Dans la permutation c'est j'ai une bijection j'ai l'impression je dois avoir autant d'input que d'output.

Exemple :  Combien y'a t-il de bijection de $\{ 1, 2, ..., n\} \rightarrow \{ 1, 2, ..., n\}$ ? on peut le tracer dans le cas $n=4$

On trace également le graphique des arbres on obtient $4 \times 3 \times 2 \times 1 = 4!$.

Ce qui nous permet de le généraliser et dire qu'il y'a $n!$ bijections


![[Pasted image 20260609180401.png|198]]
Figure de bijection la

## Les Combinaisons : L'ordre ne compte pas!


![[partition.png|245]]



### Problème des boules

Exemple : Si au lieu du tiercé qu'on a vu pour l'Arrangement (où l'ordre compte) on nous demande : Combien de trios de tête peut-on former ? 

Si on note les chevaux de $C_1, .., C_{10}$ ça revient à avoir l'ensemble des arrangements 

$$
A_n^p = \frac{n!}{(n-p)!} = \frac{10!}{(10-3)!} = \frac{10!}{7!}
$$
Le problème c'est que dans l'arrangement l'ordre compte par exemple le trois premiers chevaux peuvent arriver dans un ordre différent mais dans la Combinaisons ce n'est pas important. 

$$[\{C_1, C_2, C_3\}, \{C_3, C_2, C_1\}, \{C_3, C_1, C_2\} ... ]$$
On peut aussi avoir 

$$[\{C_{10}, C_2, C_4\}, \{C_4, C_2, C_{10}\}, \{C_4, C_{10}, C_2\} ... ]$$

C'est pour ça qu'on divise par $p!$ pour enlever tous ces cas 

$$
C_n^p = \frac{A_n^p}{p!}
$$

Exemple : Combien y'a t-il de tirage de 5 numéros au loto (50 boules) ? 
Si il y'a ordre il y'a cinq arrangements parmi 50 boules, la formule est:
$$
\frac{50!}{(50-5)!} = \frac{50!}{45!}
$$
Sauf que au loto l'ordre de tirages des boules ne compte pas, le tirage de 12345 est le même que 54321 etc. Comme pour les anagrammes on peut regrouper les tirages qui se ressemble:

$$[\{B_1, B_2, B_3, B_4, B_5 \}, \{B_5, B_2, B_3, B_4, B_1 \}, ..]$$

Trouver le nombre de paquet c'est trouver le nombre de tirage dans n'importe quel ordre. La taille d'un paquet c'est toutes les permutations de 5 boules possibles donc $5!$. Ainsi:
$$
\frac{\frac{50!}{45!}}{5!} = \frac{50!}{45!5!}
$$
### Problème de l'alphabet

Exemple Trouver les anagrammes de 

$$
\text{FLAGADA}
$$

On va déjà commencé par écrire 

$$
FLA_1GA_2DA_3
$$
puis on va pouvoir regrouper les paquets en communs en effet on peut avoir 

$$[\{F,L, A_2, G, A_1, D, A_3 \}, \{F,L, A_1, G, A_3, D, A_2 \}, ...]$$

puis

$$[\{G, A_1, D, A_2, F, L, A_3 \}, \{G, A_2, D, A_3, F, L, A_1 \}, ...]$$
On voit qu'il y'a trois façon de mélanger les A donc:
$$
\frac{7!}{3!}
$$


\textbf{Exemple applicatif:} Trouver les anagrammes de 
$$
ASSASSINS
$$
On va utiliser une autre méthode que la précédente. On a neuf lettres, on va dessiner neuf cases et on va s'intéresser aux nombres de façon de poser les lettres de ASSSASINS dans ces cases la.

On va raisonner sous forme d'arbre. On commence par choisir 5 cases parmi les 9 pour les S

![[images/1-Mathématiques/F_Probabilité/01_Combinatoire/im7.png|198]]

![[Pasted image 20260609181808.png|146]]

Puis pour les A il reste quatre cases et deux lettres : 

![[images/1-Mathématiques/F_Probabilité/01_Combinatoire/im8 (1).png|209]]

ok
![[Pasted image 20260609181903.png|304]]
Enfin pour le I il reste deux possibilités et $n$ plus que une
![[images/1-Mathématiques/F_Probabilité/01_Combinatoire/im9 (2).png|214]]


![[Pasted image 20260609181945.png|535]]


![[images/1-Mathématiques/F_Probabilité/01_Combinatoire/im10 (2).png|219]]


Ainsi le nombre total de combinaison est de:
$$
\begin{pmatrix}9 \\ 5\end{pmatrix} \times \begin{pmatrix}4 \\ 2\end{pmatrix} \times 2 \times 1 = \frac{9!}{5! \times 2!} = 1512
$$

## Le complémentaire

Dès que tu vois dans un énoncé la phrase magique : **"au moins un"** (ou "au moins une fois").

- _Exemple :_ "On tire 5 cartes dans un jeu, combien y a-t-il de tirages contenant **au moins un** As ?"

**Avec le complémentaire :** Le contraire de "au moins un As", c'est **"aucun As"**.

1. Tu calcules le nombre total de tirages possibles.
    
2. Tu calcules le nombre de tirages sans aucun As.
    
3. Tu fais : **Total - Aucun As = Au moins un As.**

Exemple (Le tirage de 3 boules) : Imagine une urne avec 10 boules : 2 rouges et 8 bleues. Tu en tires 3 . Combien de tirages ont au moins une boule rouge ?
- Total de tirages : $\binom{10}{3}=120$.
- Le complémentaire (Aucune rouge $=3$ bleues) : Tu tires 3 boules parmi les 8 bleues.
- $\binom{8}{3}=\frac{8 \times 7 \times 6}{3 \times 2 \times 1}=56$.
- Résultat : $120-56=64$ tirages avec au moins une rouge.



## Truc plus advanced

en fait c pareil que le truc des lettres mais dans un cadre différent en fait 

1. Exemple : Comité de 3 personnes ( 2 femmes, 1 homme) parmi 6F et 4H Imaginons 3 cases à remplir pour le comité : [ ] 

Étape 1: Choisir les places pour les femmes
Il y a 3 cases. On doit choisir 2 cases pour y mettre des femmes.
C'est $\binom{3}{2}=3$ façons de placer les femmes. Par exemple :
* ```[F] [F] [H] ```
* ```[F] [H] [F] ```
* ```[H] [F] [F] ```



Étape 2 : Remplir les cases
- Pour les 2 cases "Femmes" : on choisit 2 femmes parmi les $6 \rightarrow\binom{6}{2}=15$ façons.
- Pour la case "Homme" : on choisit 1 homme parmi les $4 \rightarrow\binom{4}{1}=4$ façons.

Total pour chacune des 3 configurations de cases on a $15 \times 4$ façons de remplir les sièges.


Exemple : On a un jeux de 5 cartes dans sa main on en choisit 2 pour les As 

Étape 1: Choisir les places pour les 2 As
On a 5 cases, on en choisit 2 pour les As : $\binom{5}{2}=10$ façons de placer les As dans la main.
Par exemple: ```[A] [A] [X] [X] [X]``` ou ```[A] [X] [A] [X] [X]``` etc.
Étape 2 : Remplir les cases
- Les 2 cases "As" : on choisit 2 As parmi les $4 \rightarrow\binom{4}{2}=6$ façons.
- Les 3 cases "Autres" ( $X$ ) : on choisit 3 cartes parmi les 48 non-As $\rightarrow\binom{48}{3}=17296$ façons.

Total :
On prend les 10 configurations de cases possibles, et pour chacune, on multiplie par le nombre de façons de remplir les As et les non-As :

$$
10 \times\left(\binom{4}{2} \times\binom{ 48}{3}\right)=10 \times(6 \times 17296)=1037760 \ldots \text { Attends }!
$$

## On a vu un ou c'était mixte

### Combinaison + Arrangement

C'est le cas où tu dois choisir un groupe (combinaison), puis ordonner une partie de ce groupe (arrangement).
- Exemple : "Sur 10 employés, tu en choisis 3 pour former une équipe (combinaison), et parmi ces 3 , tu en nommes un Président et un Trésorier (arrangement)."
- Le calcul :

1 Choix des 3 personnes : $\binom{10}{3}$
2 Placement des rôles (Président/Trésorier) parmi les 3 élus : $A_3^2$
3 Résultat: $\binom{10}{3} \times A_3^2$

### Combinaison + Tirage avec remise

Le problème

Tu dois créer un code de **5 caractères**. La règle est la suivante :

- Il doit y avoir **exactement 2 chiffres** (choisis parmi 0-9).
    
- Il doit y avoir **3 lettres** (choisies parmi A-Z, soit 26 lettres).
    
- Tu peux répéter les chiffres et les lettres (donc **avec remise**).
    

La décomposition en deux blocs

Pour résoudre ça, on ne cherche pas une seule formule, on construit le code étape par étape en mélangeant nos deux outils :

Bloc 1 : La structure (La Combinatoire)

Avant de mettre les chiffres et les lettres, on doit décider **où** ils vont se placer dans le code de 5 caractères. On choisit les emplacements pour les 2 chiffres parmi les 5 places disponibles :

$$
\binom{5}{2}=10 \text { possibilités de placement. }
$$
_Exemple de structure possible :_ ```[Chiffre] [Lettre] [Chiffre] [Lettre] [Lettre]```

Bloc 2 : Le remplissage (Le Tirage avec remise)
Maintenant que les places sont réservées, on remplit chaque catégorie :
- Pour les 2 chiffres : Chaque case a 10 options, et comme on a le droit de répéter, ça donne $10 \times 10=10^2=100$.
- Pour les 3 lettres : Chaque case a 26 options, avec répétition autorisée, ça donne $26 \times 26 \times 26=26^3=17576$.

Le résultat final (Le mélange)
Puisque chaque placement (Bloc 1) peut être combiné avec chaque remplissage (Bloc 2), on multiplie tout:

$$
\text { Total }=\underbrace{\binom{5}{2}}_{\text {Placements }} \times \underbrace{10^2}_{\text {Chiffres }} \times \underbrace{26^3}_{\text {Lettres }}
$$
Total $=10 \times 100 \times 17576=\mathbf{1 7 5 7 6 0 0 0}$ codes possibles.




