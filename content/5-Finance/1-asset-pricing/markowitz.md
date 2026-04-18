---
title: a-Markowitz
order: 1
---
# Théorie moderne du portefeuille — Markowitz

> **Prérequis :** probabilités de base, espérance, variance, covariance.

---

## (i) Fonction d'utilité $u(W)$

Une **fonction d'utilité** traduit la satisfaction d'un agent en fonction de sa richesse $W$. Elle permet de comparer des choix risqués en transformant toute situation en un nombre : si l'agent préfère $A$ à $B$, alors $u(A) > u(B)$.

La fonction est toujours croissante ($u' > 0$), et c'est le signe de $u''$ qui détermine l'attitude face au risque :

$$
u''(W) < 0 \;\Rightarrow\; \text{concave} \;\Rightarrow\; \textbf{averse au risque} \qquad
u''(W) > 0 \;\Rightarrow\; \text{convexe} \;\Rightarrow\; \textbf{amateur de risque}
$$

L'exemple canonique pour un agent averse au risque est $u(W) = \ln(W)$, avec $u' = 1/W > 0$ et $u'' = -1/W^2 < 0$.

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/markowitz/im1.png" style="max-width:75%;" alt="Fonction d'utilité concave ln(W)"/>
</div>

Figure 1. Fonction $u(W) = \ln(W)$ concave. Pour un pari 50/50 entre 100 € et 900 €, l'utilité espérée $E[u(W)] \approx 5.70$ est inférieure à $u(500\text{ €}) \approx 6.21$ : l'agent préfère 500 € certains.

---

## (ii) Prime de risque

Face à un pari risqué d'espérance $E(W)$, un agent averse au risque accepterait une somme certaine $C < E(W)$ procurant la même utilité. On appelle $C$ l'**équivalent certain** et la différence

$$
\pi = E(W) - C \;\geq\; 0
$$

la **prime de risque** : ce que l'agent sacrifie en espérance pour avoir la certitude. Par définition $U_c = u(C) = E[u(W)]$.

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/markowitz/im2.png" style="max-width:75%;" alt="Prime de risque et équivalent certain"/>
</div>

Figure 2. La droite rouge relie $u(W_1)$ à $u(W_2)$ — son milieu donne $E[u(W)]$. La courbe bleue étant au-dessus, $u(E(W)) > E[u(W)]$ : l'agent préfère le certain. $C$ est l'équivalent certain tel que $u(C) = E[u(W)]$.

La mesure formelle de l'aversion au risque est le **coefficient d'Arrow-Pratt** :

$$
\theta = -\frac{u''}{u'}
$$

Plus $\theta$ est grand, plus l'agent est averse au risque. C'est la version formelle du paramètre $\lambda$ qui apparaîtra dans la formule de Markowitz.

---

## (iii) Von Neumann & Morgenstern (1944)

**Résultat fondateur.** Tout agent rationnel face à des choix risqués agit *comme s'il maximisait* $E[u(W)]$ — l'espérance de sa fonction d'utilité appliquée à sa richesse finale. C'est le socle théorique sur lequel Markowitz construit tout.

---

## (iv) La rupture de Markowitz (1952)

Avant Markowitz, la recherche financière répondait à : *"quel titre acheter ?"*. Markowitz déplace la question vers : *"quel portefeuille construire ?"*. Il est le premier à formaliser la diversification avec des métriques précises, et à montrer qu'on peut réduire le risque sans sacrifier le rendement en combinant intelligemment des actifs.

---

## (v) Richesse finale et rendement du portefeuille

On note $x$ le **vecteur de composition** du portefeuille — les poids de chaque actif $(w_1, w_2, \ldots, w_n)$ avec $\sum_i w_i = 1$. La richesse finale vaut

$$
W^x = W_0\,(1 + R(x))
$$

où $W_0$ est la richesse initiale et $R(x)$ le rendement réalisé. On en déduit

$$
E[W^x] = W_0\,(1 + E[R(x)]), \qquad \text{Var}(W^x) = W_0^2\,\text{Var}(R(x)).
$$

$W_0$ étant une constante, le passage de $W^x$ à $R(x)$ est **transparent pour l'optimisation** : maximiser $f(E[W^x], \text{Var}(W^x))$ par rapport à $x$ est équivalent à maximiser $f(E[R(x)], \text{Var}(R(x)))$. On travaille donc directement sur les rendements.

---

## (vi) Hypothèse quadratique et score d'utilité

Travailler avec $E[u(W^x)]$ en toute généralité est trop complexe. Markowitz suppose que $u$ est **quadratique** :

$$
u(W) = W - c\,W^2
$$

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/markowitz/im3.png" style="max-width:75%;" alt="Utilité quadratique avec zone absurde"/>
</div>

Figure 3. La parabole $u(W) = W - cW^2$ croît jusqu'au sommet $W^* = 1/(2c)$ puis décroît — zone économiquement absurde. C'est le prix à payer pour la tractabilité.

<details>
<summary>Preuve que l'hypothèse quadratique mène à espérance-variance</summary>

$$
E[u(W^x)] = E[W^x - c(W^x)^2] = E[W^x] - c\,E[(W^x)^2]
$$

Or $\text{Var}(W^x) = E[(W^x)^2] - (E[W^x])^2$, donc $E[(W^x)^2] = \text{Var}(W^x) + (E[W^x])^2$. En substituant :

$$
E[u(W^x)] = E[W^x] - c\,\text{Var}(W^x) - c\,(E[W^x])^2
$$

Le dernier terme est une constante quand on compare des portefeuilles à même espérance de rendement. On obtient la forme de Markowitz :

$$
\boxed{E[u(W^x)] \;\propto\; E[R(x)] - \lambda\,\text{Var}(R(x))}
$$

</details>

On définit ainsi le **score d'utilité** d'un portefeuille $x$ :

$$
U(x) = E[R(x)] - \lambda\,\text{Var}(R(x))
$$

où $\lambda \geq 0$ est l'aversion au risque de l'investisseur (lié à $\theta$ d'Arrow-Pratt). C'est une **fonction de $x$** : on fait varier les poids, on recalcule $U$, on cherche le maximum.

---

## (vii) Pourquoi la diversification marche

Le rendement espéré du portefeuille est linéaire en les poids :

$$
E[R(x)] = \sum_i w_i\,E[R_i]
$$

La variance, elle, fait intervenir toutes les **covariances** :

$$
\text{Var}(R(x)) = \sum_i \sum_j w_i\,w_j\,\sigma_{ij}, \qquad \sigma_{ij} = \sigma_i\,\sigma_j\,\rho_{ij}
$$

Pour deux actifs $A$ et $B$ :

$$
\text{Var}(R) = w_A^2\sigma_A^2 + w_B^2\sigma_B^2 + 2\,w_A w_B\,\sigma_{AB}
$$

Le terme croisé $2\,w_A w_B\,\sigma_{AB}$ est la clé : si $\rho_{AB} < 0$ (actifs anti-corrélés), ce terme **réduit** la variance totale sans toucher au rendement espéré.

> **Exemple.** $w_A = w_B = 50\%$, $\sigma_A = \sigma_B = 20\%$, $\rho_{AB} = -0.5$.
> $$\text{Var} = 0.25 \times 0.04 + 0.25 \times 0.04 + 2 \times 0.5 \times 0.5 \times (-0.02) = 0.01 \;\Rightarrow\; \sigma = 10\%$$
> Chaque actif seul avait $\sigma = 20\%$ : la diversification a divisé le risque par deux.

---

## (viii) Allocation optimale

**Définition.** L'**allocation optimale** $x^*$ est le vecteur de poids qui maximise l'utilité de l'investisseur :

$$
x^* = \underset{x}{\arg\max}\; U(x) = \underset{x}{\arg\max}\; E[R(x)] - \lambda\,\text{Var}(R(x))
$$

C'est-à-dire : quelle pondération des actifs (30 % Apple, 50 % Total…) donne le meilleur score $U$ ?

---

## (ix) Frontière efficiente — sans actif sans risque

Chaque vecteur $x$ donne un point $(\sigma(x),\, E[R(x)])$ dans le graphe risque/rendement. Les points **non dominés** — ceux pour lesquels on ne peut pas faire mieux sans prendre plus de risque — forment la **frontière efficiente**.

**Deux formulations équivalentes :**

$$
\text{(1)}\quad \sup_x\,E[R(x)] \;\text{ s.c. }\; \text{Var}(R(x)) = \sigma^2 \qquad \text{(2)}\quad \min_x\,\text{Var}(R(x)) \;\text{ s.c. }\; E[R(x)] = m
$$

Pour chaque $\sigma$ (resp. $m$), la solution $x^*$ est un portefeuille efficient. L'ensemble de ces solutions trace la frontière.

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/markowitz/im4.png" style="max-width:75%;" alt="Frontière efficiente sans actif sans risque"/>
</div>

Figure 4. Chaque point gris est un portefeuille possible. La courbe bleue pleine est la frontière efficiente (partie haute) ; la partie en pointillés est non efficiente. Le point $\lambda$ grand correspond à un investisseur très averse au risque (à gauche), $\lambda$ petit à un investisseur moins averse (à droite).

### Coordonnées du min variance portfolio

On définit deux scalaires à partir de la matrice de covariance $\Sigma$ et du vecteur $\mathbf{1} = (1,\ldots,1)^\top$ :

$$
a = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \qquad b = E\mathbf{R}^\top \Sigma^{-1} \mathbf{1}
$$

Le **portefeuille de variance minimale** — point le plus à gauche de la frontière — a pour coordonnées exactes :

$$
\sigma_{\min} = \frac{1}{\sqrt{a}}, \qquad \mu_{\min} = \frac{b}{a}
$$

<details>
<summary>Pourquoi ces formules ? (inégalité de Cauchy-Schwarz)</summary>

On cherche $\min_x \text{Var}(R(x)) = x^\top \Sigma x$ sous contrainte $\mathbf{1}^\top x = 1$. Par Cauchy-Schwarz dans l'espace muni du produit scalaire $\langle u,v\rangle_{\Sigma^{-1}} = u^\top \Sigma^{-1} v$ :

$$
|\langle u, v\rangle_{\Sigma^{-1}}|^2 \leq \|u\|^2_{\Sigma^{-1}}\,\|v\|^2_{\Sigma^{-1}}
$$

avec égalité si et seulement si $u$ et $v$ sont colinéaires. La minimisation donne directement $\sigma_{\min} = 1/\sqrt{a}$.

</details>

Il sépare deux zones : **au-dessus**, la frontière efficiente ; **en dessous**, les portefeuilles dominés.

---

## (x) Avec un actif sans risque — droite de marché (CML)

On introduit un actif sans risque $S^0$ de rendement $r_f$ et $\sigma = 0$. Il se place en $(0, r_f)$ dans le graphe. Combiner $S^0$ et un portefeuille risqué $x$ trace une **droite** depuis $(0, r_f)$.

La nouvelle frontière efficiente est la droite tangente à la frontière courbe — la **Capital Market Line** :

$$
m(\sigma) = r_f + \frac{E[R(\xi)] - r_f}{\sigma(\xi)}\cdot\sigma
$$

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/markowitz/im5.png" style="max-width:75%;" alt="Frontière efficiente avec CML et portefeuille tangent"/>
</div>

Figure 5. La droite verte (CML) part de $r_f$ et est tangente à la frontière bleue en $\xi$. Elle domine la frontière courbe pour tout $\sigma$ : c'est l'avantage d'avoir accès à un actif sans risque.

---

## (xi) Ratio de Sharpe et portefeuille tangent

Le **ratio de Sharpe** d'un portefeuille $x$ mesure le rendement excédentaire par unité de risque :

$$
S(x) = \frac{E[R(x)] - r_f}{\sigma(x)}
$$

Géométriquement, c'est la **pente de la droite** depuis $r_f$ vers le point $(\sigma(x), E[R(x)])$.

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/markowitz/im6.png" style="max-width:75%;" alt="Ratio de Sharpe — pentes depuis r_f"/>
</div>

Figure 6. Parmi toutes les droites depuis $r_f$, celle de pente maximale est la CML. Elle touche la frontière exactement en $\xi$ : le **portefeuille tangent** est le portefeuille de Sharpe maximal.

Le portefeuille tangent $\xi$ est remarquable : il appartient à la fois à la frontière sans actif sans risque (la courbe) et à la frontière avec (la droite). C'est le seul portefeuille 100 % risqué qu'un agent rationnel voudra détenir.

---

## (xii) Théorème des deux fonds

**Théorème.** Tout agent rationnel qui suit un portefeuille efficient partage son investissement entre exactement deux fonds :

$$
\text{Portefeuille optimal} = x_0 \cdot S^0 \;+\; (1-x_0)\cdot \xi
$$

où $x_0 \in [0,1]$ dépend de $\lambda$ :
- $\lambda$ grand $\Rightarrow$ $x_0$ grand : beaucoup d'actif sans risque, peu de $\xi$.
- $\lambda$ petit $\Rightarrow$ $x_0$ petit : beaucoup de $\xi$, peu d'actif sans risque.

**Moralité.** Peu importe le profil de risque de l'investisseur, le portefeuille risqué optimal est toujours $\xi$ — seule la dose change. Il n'est pas nécessaire de construire un portefeuille risqué sur mesure pour chaque client.

---

## Résumé — fil logique complet

| Étape | Contenu |
|---|---|
| **1** | $u(W)$ concave → aversion au risque → prime de risque $\pi = E(W) - C$ |
| **2** | vNM : agent rationnel maximise $E[u(W^x)]$ |
| **3** | Markowitz : $u$ quadratique → $E[u(W^x)] \propto E[R(x)] - \lambda\,\text{Var}(R(x))$ |
| **4** | Diversification : terme croisé $\sigma_{ij} < 0$ réduit la variance |
| **5** | Allocation optimale : $x^* = \arg\max_x\, U(x)$ |
| **6** | Sans actif sans risque : frontière efficiente = courbe, min variance en $(1/\sqrt{a},\, b/a)$ |
| **7** | Avec actif sans risque : frontière = CML tangente en $\xi$ |
| **8** | $\xi$ = portefeuille de Sharpe maximal |
| **9** | Théorème des deux fonds : tout agent mélange $r_f$ et $\xi$, seul $x_0$ varie selon $\lambda$ |
