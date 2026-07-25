# 1. Intérêts

En général si la maturité est < 1 ans ce sera les intérêts simples si > 1 ans ce sera les intérêts composés

**Intérêts simples:** Nous prenons l’exemple d’une obligation à court terme comme un bon du Trésor à 6 mois.

- Remarque: Avec le taux d’intérêt simple, si on avait deux paiements ils ne s’accumuleraient pas à chaque gains de six mois.

![image.png](images/6-Économie/obligations/01_image.png)

**Inputs:**

- **Capital initial** $C_0=1,000€$
- **Taux d’intérêt simple annuel** $i=5\%$
- **Fréquence des paiements:** Deux paiements par an (il faut donc réajuster le taux d’intérêt) car dans notre cas on s’arrête au premier paiement n=0.5

**Intérêts composés (Livret A):** On souhaite calculer les quatre métriques du Livret A de manière composée, c'est-à-dire que les intérêts gagnés chaque année sont réinvestis et génèrent à leur tour des intérêts.

![image.png](images/6-Économie/obligations/02_image.png)

**Inputs:**

- **Capital initial** $C_0=1,000€$
- **Taux d’intérêt simple annuel** $i=3\%$
- **Durée n:** 5 ans

# 2. Obligations

## 2.1 Cours

_**(a) Les inputs:**_

- **Valeur nominal (=valeur facial)** $N$: c’est le montant que l’émetteur de l’obligation s’engage à rembourser à échéance.
    
- **Le taux de coupon $i$:** Il est soit fixe soit variable.
    
- **La valeur du coupon $C$**: est $C = N \times i$ ; ce sont des paiements périodiques constants qui rembourse l’intérêt à l’investisseur.
    
- **Durée de vie de l’obligation $n$:** C’est la période entre sa date d’émission et sa date d’échéance (maturité)
    
- **Taux de marché $r$:** C’est un instantané ****des taux à l’instant $t$ il peut changer à tout instant en fonction des conditions économiques et de la politique monétaire.
    

_**(b) Zéro-coupon (ZC) {Zero-Coupon}: = exactement pareil que l’intérêt composé**_

Le prix d’un ZC est la valeur actualisée de son unique flux de fin de vie.

$$  
P_0 := \frac{N}{(1+r)^n}  
$$

Une fois acheté un ZC, on est en mesurer de calculer r le taux d’actualisation (Yield to Maturity YTM)

$$  
r = \Big( \frac{N}{P_0} \Big)^{1/n} -1  
$$

**(c) Les Obligations:** Cette formule montre qu’une obligation peut être démembrée en une somme de ZC:

$$  
P_0 = \sum_{i=1}^{n} \frac{C}{(1+r(0,i))^i} + \frac{N}{(1+r(0,n))^n}  
$$

![Actualisation d’une obligation à taux fixe sur cinq ans](images/6-Économie/obligations/03_image.png)

Actualisation d’une obligation à taux fixe sur cinq ans

$r(0,i)$ est le taux du marché YTM du ZC de fin de vie la date i

- Use case 1:
    
    - ⇒ En fait tout le principe de la preuve c’est de dire que le taux N x i est i il est statique donc pour i est de 5% au début puis
    - Au temps t=0, disons que j’ai r(0,10)=5% c’est le taux pour une obligation à 10 ans
    - Puis disons que au temps t=3 j’ai un changement r(3,7)=3% c’est le taux d’une obligation à 7 ans.
    
    Je vais obtenir deux formules: avec N=1000E, C=50E (taux de coupon de 5%) et échéance T=5 ans
    
    $$  
    \text { Valeur de l'obligation }=\sum_{k=1}^T \frac{C}{(1+r(0, k))^k}+\frac{F}{(1+r(0, T))^T}  
    $$
    
    ⇒ En gros il faut juste faire deux scénarios un ou r(0,k) est constant il reste toujours élever à 5% un ou il est toujours bas, et un ou il y’a un changement au milieu de 5 à 3% et voir quel est le plus rentable. Normalement celui qui a le changement devrait être pas trop mal.
    
- Use case 2: exemple taux de marché fixe à 4% coupon 50 euro annuel pendant 5 ans; F =1000E
    

⇒ je voulais jute tracer une courbe pour voir l’évolution du prix d’une obligation mais ça doit pas être ça.

**Exemple 1:** Soit l'obligation de montant nominal $100 \$$, de maturité 3 ans et de taux de coupon 10%. Les taux zéro-coupon à 1 an, 2 ans et 3 ans sont de $7 \%, 9 \%$ et $10 \%$. Le prix P de l'obligation est égal à

$$  
P=\frac{10}{1+7 \%}+\frac{10}{(1+9 \%)^2}+\frac{110}{(1+10 \%)^3}=100.407 \$   
$$

<aside>  
⚠️

Ce que je comprends pas trop c’est que du coup à un instant T j’utilise ma table de courbe des taux et ça me permet de calculer tous mes taux zéro-coupon ? ou c’est pas ça ?

- Apparament le YTM c’est un taux interne de rentabilité le (TRI)  
    </aside>

|Définition|Description|
|---|---|
|Taux de rendement actuariel|yield to maturity|
|Taux de rendement courant|Current yield|
|Taux de rendement à l’échéance intermédiaire|yield to call|
|||

## 2.2 Facteurs explicatifs du niveau des taux d’intérêt le “r”

**Taux d'intérêt et impact économique**

- Rôle des banques centrales et politique monétaire.
- Comment les variations des taux d’intérêt affectent les obligations.
- Effet des cycles économiques sur le marché obligataire.

## 2.3 Les Risques d’une Obligation

- Types de risques liés aux obligations (risque de taux, risque de crédit, risque de liquidité, risque de réinvestissement).
- Système de notation des agences de crédit (S&P, Moody's, Fitch).
- Importance de la notation et implications pour les investisseurs.

## 2.4 La sensibilité du prix d’une obligation aux variations du taux d’intérêt: La duration et la convexité

Duration, sensibilité etc

$$  
P=\sum_{t=1}^n \frac{C F_t}{(1+i)^t} \quad D=\frac{1}{P} \sum_{t=1}^n \frac{t C F_t}{(1+i)^t} \quad S=\frac{d P / P}{d i}=-\frac{D}{1+i} \quad \frac{d P}{P}=S . d i  
$$

CF = Cash flow = flux de trésorerie

[https://www.youtube.com/watch?v=PrGaUHJT4dU](https://www.youtube.com/watch?v=PrGaUHJT4dU)

[https://www.youtube.com/watch?v=v0Q5-1KRvtA&t=360s](https://www.youtube.com/watch?v=v0Q5-1KRvtA&t=360s)

# 3. Reconstitution d’une courbe des taux (Yield Curve)

## 3.1 Généralités

**Reconstruction d’une courbe des taux (Yield Curve):**

**Il existe trois grands types de courbe des taux ZC:**

- _**Courbe des taux d’emprunt d’état:**_
    
    - La courbe Trésor (d’État) c’est moins de an et pour des maturités moyenne ou longue on va avoir les Obligations Assimilables du Trésor (OAT)
    - T-bills moins de 1 an; Treasury bonds de 10 à 30 ans et Treasury Notes (T-Notes) de 2 à 10 ans)
- _**La courbe interbancaire:**_ Courte durée < 1 an c’est EURIBOR; en durée moyenne et longue on a des taux interbancaires plus élevés.
    
- **La courbe corporate:** On va trouver en x-axis la maturité en y-axis les taux; chaque entreprise en fonction de son taux aura accès à une courbe de taux spécifique.
    
    ![La courbe des taux Corporate](images/6-Économie/obligations/04_image.png)
    
    La courbe des taux Corporate
    

Comment lire les courbes

- [https://www.google.com/imgres?q=courbe d'emprunt d'état&imgurl=https%3A%2F%2Fwww.lynxbroker.fr%2Fapp%2Fuploads%2Fsites%2F2%2F2022%2F03%2Fcourbe-de-rendement-courbe-des-taux-courbes-normale-inversee-plate.png&imgrefurl=https%3A%2F%2Fwww.lynxbroker.fr%2Fbourse%2Fanalyses-bourse%2Fproduits-boursiers%2Faplatissement-courbe-taux%2F&docid=RrwwiMaiksYjUM&tbnid=8veRWMCkckPQSM&vet=12ahUKEwiA16TUt8CJAxVwT6QEHaOEOccQM3oECGIQAA..i&w=913&h=512&hcb=2&ved=2ahUKEwiA16TUt8CJAxVwT6QEHaOEOccQM3oECGIQAA](https://www.google.com/imgres?q=courbe%20d%27emprunt%20d%27%C3%A9tat&imgurl=https%3A%2F%2Fwww.lynxbroker.fr%2Fapp%2Fuploads%2Fsites%2F2%2F2022%2F03%2Fcourbe-de-rendement-courbe-des-taux-courbes-normale-inversee-plate.png&imgrefurl=https%3A%2F%2Fwww.lynxbroker.fr%2Fbourse%2Fanalyses-bourse%2Fproduits-boursiers%2Faplatissement-courbe-taux%2F&docid=RrwwiMaiksYjUM&tbnid=8veRWMCkckPQSM&vet=12ahUKEwiA16TUt8CJAxVwT6QEHaOEOccQM3oECGIQAA..i&w=913&h=512&hcb=2&ved=2ahUKEwiA16TUt8CJAxVwT6QEHaOEOccQM3oECGIQAA)
- [https://www.youtube.com/watch?v=S6xEW5a4tFM](https://www.youtube.com/watch?v=S6xEW5a4tFM)

## 3.2 La Méthode Théorique

La formule permets de calculer le prix actuel (ou prix de marché) d’une obligation zéro-coupon d’une valeur de 1€, payable à la date future t.

$$  
B(0,t) = \frac{1}{(1+r(0,t))^t}  
$$

Dans cette formule:

- $B(0,t)$ c'est le **prix de marché** à la **date 0** d’une obligation zéro-coupon délivrant 1 euro à la date t. On appelle aussi $B(0,t)$ le **facteur d’actualisation** en 0 pour la maturité t.
- $r(0,t)$ est le **taux de rendement en 0** de l'obligation ZC délivrant 1 euro en t; on dit aussi que c'est aussi le taux zéro-coupon en 0 de maturité t

Le prix $V_i(t)$ d’une obligation à la date t est la somme des valeurs actualisées de chaque flux de trésorerie futur que cette obligation paiera. La formule que tu as donnée s'écrit :

$$  
V_i(t) = \sum_{i=t+1}^m \frac{F(i)}{[1+r(t,i-t)]^{i-t}} = \sum F(i) B(t,i)  
$$

Dans cette formule:

- $F(i)$ représente le **flux de trésorerie** (comme un **coupon** ou le remboursement du principal) que l'obligation versera à la date
- $r(t,i−t)$ est le **taux zéro-coupon** entre la date actuelle
- $B(t,i)$ est le **facteur d'actualisation** à la date t pour le flux versé à la date i

**Exemple :** Taux de zéro-coupon pour chaque année: $r(0,1)=2\%, r(0,2)=3\%, r(0,3)=4\%$; Flux de trésorerie F(i) pour l'obligation - À 1 an : F(1)= 5€ (coupon) - À 2 ans : F(2)= 5€ (coupon) - À 3 ans : F(3)= 105€ (coupon + valeur nominale), il suffit d’appliquer la formule $V_i(0) = \sum_{i=1}^m \frac{F(i)}{[1+r(0,i)]^{i}} = \sum F(i) B(0,i)$; il suffit de calculer $V_3(0) = 4.90€ + 4.71€+93.36€ = 102.97€$ c’est le prix de l’obligation à la date t=0 sachant que l’obligation à une échéance de trois ans d’où l’indice i=3.

**Définition (Modèle théorique)**:

1. **Vecteur des prix $P_t$**

$$  
P_t=\left(P_{t 1}, P_{t 2}, \ldots, P_{t n}\right)^T   
$$

Où $P_{ti}$ représente le prix de l’obligation $i$ à l’instant $t$. Ce vecteur regroupe les prix actuels des $n$ obligations à coupons dans le panier à la date $t$.

1. **Matrice des flux $F$:**

$$  
F=\left(\begin{array}{cccc}  
F_{t 1}(1) & F_{t 1}(2) & \ldots & F_{t 1}(n) \\  
F_{t 2}(1) & F_{t 2}(2) & \ldots & F_{t 2}(n) \\  
\vdots & \vdots & \ddots & \vdots \\  
F_{t n}(1) & F_{t n}(2) & \ldots & F_{t n}(n)  
\end{array}\right)  
$$

Où chaque $F_{ti}(j)$ représente le flux de trésorerie de l'obligation $i$ à la date $t_j$. La matrice $F$ est de taille $n \times n$ et regroupe les flux pour chaque obligation du panier, avec des dates de tombées identiques pour tous les titres.

1. **Vecteur des facteurs d’actualisation $B_t$:**

$$  
B_t=\left(B\left(t, t_1\right), B\left(t, t_2\right), \ldots, B\left(t, t_n\right)\right)^T  
$$

où $B(t,t_j)$ est le facteur d'actualisation pour une période allant de la date actuelle $t$ à la date future $t_j$. Ce vecteur sert à actualiser les flux futurs pour en obtenir la valeur actuelle à la date $t$.

Par AOA, on obtient le vecteur des facteurs d’actualisation $P_t = F \cdot B_t$ soit $B_t = F^{-1} \cdot P_t$ car F est inversible

**Exemple (Modèle Théorique):**

_**Panier de deux obligations à coupons**_

- Obligation 1 : paie des coupons de 3€ à 1 an et 3€ à 2 ans, puis rembourse le principal de 100€ à 2 ans.
- Obligation 2 : paie des coupons de 4€ à 1 an et 4€ à 2 ans, puis rembourse le principal de 100€ à 2 ans.

_**Taux zéro-coupon pour chaque année :**_

- Taux à 1 an : $r(0,1)=2\%$
- Taux à 2 ans : $r(0,2)=3\%$

La matrice $F$ regroupe les flux des deux obligations à chaque date. Puisque les dates de tombée des flux sont identiques pour les deux obligations, nous avons:

$$  
F=\left(\begin{array}{ll}  
3 & 3+100 \\  
4 & 4+100  
\end{array}\right)=\left(\begin{array}{ll}  
3 & 103 \\  
4 & 104  
\end{array}\right)  
$$

Les facteurs d'actualisation sont calculés comme suit :

- Pour 1 an : $B(0,1)=\frac{1}{1+r(0,1)}=\frac{1}{1.02} \approx 0.9804$
- Pour 2 ans: $B(0,2)=\frac{1}{(1+r(0,2))^2}=\frac{1}{1.03^2} \approx 0.9426$

Ainsi, le vecteur $B_t$ est donné par:

$$  
B_t=\binom{0.9804}{0.9426}  
$$

Le prix de chaque obligation à l'instant $t=0$ est obtenu en multipliant la matrice des flux $F$ par le vecteur des facteurs d'actualisation $B_t$ :

$$  
P_t^{theory}=F \cdot B_t=\left(\begin{array}{ll}  
3 & 103 \\  
4 & 104  
\end{array}\right) \cdot\binom{0.9804}{0.9426} = \binom{100.028}{101.952}; \quad P_t^{marché}= \binom{102.6}{102.05}  
$$

On peut comparer le prix de marché avec le prix théorique; leurs différences pourraient indiquer des ajustements liés aux conditions de marché ou à des attentes des investisseurs.

Pour extraire le vecteur des taux zéro-coupon continus à partir des facteurs d'actualisation, on utilise la relation suivante :

$$  
R\left(t, t_i-t\right)=\left[\frac{1}{B\left(t, t_i\right)}\right]^{\frac{1}{t_i-t}}-1   
$$

Où

- $R\left(t, t_i-t\right)$ est le taux zéro-coupon continu pour la période $t_i-t$,
- $B\left(t, t_i\right)$ est le facteur d'actualisation entre $t$ et $t_i$
- $t_i-t$ est la durée entre le moment actuel $t$ et la date future $t_i$.

**Exemple (Modèle théorique):** Dans cet exemple on va utiliser la formule $B_t = F^{-1} \cdot P_t$

|Titre|Coupon|Maturité (années)|Prix de marché|
|---|---|---|---|
|Titre 1|5|1|101|
|Titre 2|5.5|2|101.5|
|Titre 3|5|3|99|
|Titre 4|6|4|100|

$$  
P_t = F \cdot B_t \iff \begin{cases}101 = 105 \cdot B(0,1) \\ 101.5 = 5.5 \cdot B(0,1) + 105.5 \cdot B(0,2)\\99= 5 \cdot B(0,1) + 5 \cdot B(0,2) + 105 \cdot B(0,3)\\100= 6 \cdot B(0,1) + 6 \cdot B(0,2) + 6 \cdot B(0,3) + 106 \cdot B(0,4)\end{cases} \Rightarrow \begin{cases}B(0,1)=0.9619 \\ B(0,2)=0.9119 \\ B(0,3)=0.8536 \\ B(0,4)=0.7890 \end{cases}\Rightarrow \begin{cases}r(0,1)=3.96\% \\ r(0,2)=4.717\% \\ r(0,3)=5.417\% \\ r(0,4)=6.103\% \end{cases}  
$$

<aside>  
⚠️

Le modèle théorique n’est pas utilisable en pratique car les taux ZC plus grands que 1 année sont inconnus !!

</aside>

## 3.2 La Méthode du Bootsrap

### 3.2.1 Reconstitution d’une courbe d’Emprunts d’État

**Exemple (Bootstrap):** Le calcul nécessite deux tables qu’on récupère à un instant T (de type YYYY-MM-DD)

- il faut d’une part récupérer la table 1 pour obtenir les valeurs ZC, typiquement des ZC (maturité < 1 an) des Courbes du trésor (ou T-bills aux USA)
- La table 2 sont ceux qu’on va chercher à interpoler qui sont de maturité > 1 an eg Obligations Assimilables du Trésor (OAT) (Treasury notes {T-Notes} pour 2 à 10 ans ou Treasury bonds {T-bills} pour 10 à 30 ans)

|Maturité|ZC|
|---|---|
|Overnight|4.40%|
|1 mois|4.50%|
|2 mois|4.60%|
|3 mois|4.70%|
|6 mois|4.90%|
|9 mois|5.0%|
|1 an|5.10%|

||Coupon|Maturité (années)|Prix|
|---|---|---|---|
|Titre 1|5%|1 an et 2 mois|103.7|
|Titre 2|6%|1 an et 9 mois|102|
|Titre 3|5.50%|2 ans|99.5|

1. **Pour le segment de la courbe inférieur à 1 an [0,1]:** On extrait les taux zéro-coupon grâce au prix des titres zéro-coupon cotés sur le marché puis obtention d’une courbe continue par interpolation linéaire ou cubique.
2. **Pour le segment de la courbe allant de 1 an à 2 ans:** On va utiliser les données de ZC décomposer le prix en deux termes un connu (ZC) et une inconnu (la maturité > 1 an) $P = C B(0, t_1) + (N + C) B(0, t  
    _2) \text{ avec } t_1 \le 1 ,\&, 1< t_2 \le 2$

- Taux à 1 an et 2 mois  $103.7=\frac{5}{(1+4.6 \%)^{1 / 6}}+\frac{105}{(1+x)^{1+1 / 6}}$ soit $x=5.41 \%$ ; ici $t_1=1/6; t_2=1+1/6$=1 an et deux mois.
- Taux à 1 an et 9 mois $102=\frac{6}{(1+5 \%)^{9 / 12}}+\frac{6}{(1+x)^{1+9 / 12}}$ soit $x=5.69\%$
- Taux à 2 ans $99.5=\frac{5.5}{(1+5.1 \%)^1}+\frac{105.5}{(1+x)^2}$ soit $x=5.79\%$

1. **Pour le segment de la courbe allant de 2 ans à 3 ans:** On réitère l’opération précédente à partir des titres ayant une maturité comprise entre 2 ans et 3 ans.

- Taux à 3 ans $97.6=\frac{5}{(1+5.1 \%)^1}+\frac{5}{(1+5.79 \%)^2}+\frac{105}{(1+x \%)^3}$ soit $5.91\%$

![Courbe des taux d’emprunts d’états à l’instant T](images/6-Économie/obligations/05_image.png)

Courbe des taux d’emprunts d’états à l’instant T

Il existe aussi l’interpolation cubique …

**Interpolation linéaire:** On connaît les taux zero-coupon de maturités $t_1$ et $t_2$. On souhaite interpoler le taux de maturité t avec $\mathrm{t}_1<\mathrm{t}<\mathrm{t}_2$

$$  
R(0, t)=\frac{\left(t_2-t\right) R\left(0, t_1\right)+\left(t-t_1\right) R\left(0, t_2\right)}{\left(t_2-t_1\right)}   
$$

_**Exemple:**_ $R(0,3)=5.5 \%$ et $R(0,4)=6 \%$

$$  
R(0,3.75)=\frac{0.25 \times 5.5 \%+0.75 \times 6 \%}{1}=5.875 \%   
$$

### 3.2.2 Reconstitution d’une courbe Interbancaire

Voir les slides

### 3.2.3 Reconstitution d’une courbe Corporate

Le billet de trésorerie je suppose qu’il est ZC

[https://www.youtube.com/watch?v=DDlOosH-_wg](https://www.youtube.com/watch?v=DDlOosH-_wg)

## 3.3 Analyses: différentes formes de courbe (croissante, inversé, plate)

- Structure par terme des taux d'intérêt.
- Différentes formes de la courbe des taux (croissante, inversée, plate).
- Importance de la courbe des taux pour prévoir l’économie.