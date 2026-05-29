---
title: "c-Anomalies"
tags: [asset-pricing, portfolio-theory]
order: 3
---

# Les années 1980 — La chasse aux anomalies

## Contexte

BJS et Fama-MacBeth ont montré que le CAPM est imparfait mais ils n'avaient pas de meilleure alternative. Dans les années 1980, les chercheurs ont accès à des bases de données beaucoup plus riches (CRSP aux US, qui couvre toutes les actions depuis 1926) et à des ordinateurs suffisamment puissants pour faire des régressions sur des milliers d'actions. Ils commencent à chercher systématiquement : est-ce qu'il y a des variables, autres que le $\beta$, qui prédisent les rendements ?

Le CAPM dit non — $\beta$ est le seul facteur qui compte (H4 de Fama-MacBeth). Chaque fois qu'on trouve une variable qui prédit les rendements **au-delà du $\beta$**, c'est une **anomalie** — quelque chose que le modèle ne peut pas expliquer.

---

## (i) La méthode : portefeuilles triés

La méthode est toujours la même, inspirée de BJS. On appelle ça les **portefeuilles triés** (sorted portfolios) :

1. Trier toutes les actions selon la variable suspecte (taille, ratio valeur comptable/prix, rendement passé…)
2. Former des déciles ou quintiles
3. Observer si les rendements moyens varient systématiquement d'un groupe à l'autre
4. Vérifier que cette variation **ne s'explique pas par les différences de $\beta$** entre les groupes

C'est ce dernier point qui est crucial. L'anomalie apparaît quand les rendements varient même après avoir contrôlé pour le $\beta$ — on dit qu'il y a un **alpha non nul**, un rendement que le CAPM ne peut pas expliquer.

---

## (ii) L'effet taille — Banz (1981)

Banz trie toutes les actions du NYSE par capitalisation boursière et forme des quintiles. Pour chaque action $i$, on estime le beta par régression time-series sur 5 ans de données mensuelles :

$$R^i_t = \alpha_i + \beta_i R^M_t + \varepsilon^i_t$$

On trie ensuite par capitalisation boursière (prix $\times$ nombre d'actions) et on agrège les betas au niveau du bucket :

$$\hat\beta_{\text{bucket}} = \sum_{i \in \text{bucket}} w_i \hat\beta_i$$

Sur la période suivante, on mesure le rendement moyen de chaque bucket.



![Effet taille — Banz (1981)](images/5-Finance/C_anomalies/im1.png)



*Figure 1. Rendement moyen annuel par quintile de capitalisation boursière. Les petites caps (Q1) rapportent environ 24% contre 11% pour les grandes caps (Q5), une différence que les betas ne peuvent pas expliquer.*

Les betas des quintiles de taille sont tous relativement proches — les petites caps ont des betas légèrement plus élevés (autour de 1.4 vs 0.85), mais la différence est modeste. Le CAPM prédit donc un écart raisonnable entre Q1 et Q5. Mais l'écart observé est beaucoup plus grand. Concrètement, si Q1 a un $\beta = 1.4$ et que la prime de marché est 10% :

$$E[R^{Q1}]_{\text{CAPM}} = 5\% + 1.4 \times 10\% = 19\%$$

Mais Banz observe que Q1 rapporte en moyenne 24%. L'alpha est donc :

$$\alpha^{Q1} = 24\% - 19\% = +5\%$$

Ce n'est pas du bruit — c'est stable sur des décennies. Pour le confirmer formellement, on refait une régression Fama-MacBeth en ajoutant la taille comme variable :

$$R_{i,t} = \gamma_{0,t} + \gamma_{1,t}\hat\beta_i + \gamma_{2,t}\log(\text{Size}_i) + \varepsilon_{i,t}$$

Le coefficient $\gamma_{2}$ est **négatif et significatif** — plus la capitalisation est grande, moins le rendement est élevé, indépendamment du $\beta$. Le CAPM dit que $\gamma_2$ devrait être nul. Il ne l'est pas.

> On prend $\log(\text{Size})$ et non la taille directement parce que la distribution des capitalisations est très asymétrique. Le log linéarise la relation et évite que quelques géants écrasent statistiquement tout le reste.

---

## (iii) L'effet janvier — Keim & Roll (1983)

En regardant les données mois par mois, Keim et Roll remarquent que la surperformance des petites caps n'est pas répartie uniformément sur l'année. Elle se concentre massivement sur **les premiers jours de janvier**.



![Effet janvier — Keim & Roll (1983)](images/5-Finance/C_anomalies/im2.png)



*Figure 2. Rendement moyen mensuel des petites caps par mois de l'année. Janvier concentre environ 8% de rendement, soit près de 8 fois la moyenne des autres mois.*

C'est absurde du point de vue du CAPM : le risque systématique d'une action ne change pas selon le mois. L'effet janvier est une anomalie pure — le calendrier ne devrait avoir aucun pouvoir prédictif.

L'explication la plus crédible : les investisseurs vendent leurs positions perdantes en décembre pour réaliser des moins-values fiscales (*tax-loss selling*), ce qui déprime artificiellement les prix. En janvier, la pression vendeuse disparaît et les prix remontent. Mais même si c'est vrai, ça implique que des considérations fiscales — absentes du CAPM — déterminent les prix des actifs.

---

## (iv) L'effet value — Rosenberg, Reid & Lanstein (1985)

On trie les actions par leur ratio **book-to-market (BM)** : valeur comptable divisée par valeur de marché. Un BM élevé signifie que le marché valorise l'entreprise peu par rapport à ses actifs — c'est une action *value*. Un BM faible signifie que le marché valorise l'entreprise très au-dessus de ses actifs — c'est une action *growth*.



![Effet value — Rosenberg et al. (1985)](images/5-Finance/C_anomalies/im3.png)


*Figure 3. Rendement moyen annuel par quintile de ratio book-to-market. Les actions value (Q5, haut BM) surperforment les actions growth (Q1, bas BM) d'environ 10% par an, indépendamment de leur beta.*

La prime value — l'écart de rendement entre actions value et growth — est d'environ 10% par an sur longue période. Elle ne disparaît pas quand on contrôle pour le $\beta$. L'intuition : une action "pas chère" l'est souvent parce que le marché la considère en difficulté ou peu prometteuse. Sa surperformance pourrait compenser un risque de détresse financière que le $\beta$ ne capture pas. Mais ça, le CAPM n'en parle pas.

---

## (v) Le momentum — Jegadeesh & Titman (1993)

Le momentum est peut-être l'anomalie la plus dérangeante de toutes, parce qu'elle viole directement un principe fondamental : l'**efficience des marchés**. L'idée est simple — les actions qui ont bien performé dans le passé récent continuent à bien performer dans le futur proche.

### La méthodologie

Jegadeesh et Titman trient les actions par leur rendement cumulé sur les **6 à 12 derniers mois** (période de formation). Ils forment deux portefeuilles extrêmes :

- **Winners** : le décile des actions ayant le mieux performé sur la période de formation
- **Losers** : le décile des actions ayant le moins bien performé

Ils observent ensuite les rendements sur les **6 mois suivants** (période de détention). La stratégie momentum consiste à acheter les winners et vendre les losers — c'est un portefeuille **long-short**.



![Momentum — Jegadeesh & Titman (1993)](images/5-Finance/C_anomalies/im4.png)



*Figure 4. Stratégie momentum : acheter les winners (rendement passé élevé) et vendre les losers. La prime momentum est d'environ 13% par an sur 1965–1989, non expliquée par les différences de beta entre les deux groupes.*

### Résultats

La stratégie génère environ **1% par mois** (~13% par an) sur la période 1965–1989. Les winners n'ont pas un beta significativement plus élevé que les losers — l'alpha est donc massif et robuste.

Ce résultat est particulièrement gênant pour deux raisons. D'abord pour le CAPM — les rendements passés ne devraient pas prédire les rendements futurs si $\beta$ explique tout. Mais aussi pour l'**efficience des marchés** : si les prix reflètent toute l'information disponible, les rendements passés (information publique par excellence) ne devraient avoir aucun pouvoir prédictif.

### Deux familles d'explications

**Explication rationnelle.** Le momentum compenserait un risque que le $\beta$ ne capture pas, lié à l'exposition aux cycles économiques. Cette explication peine à convaincre quantitativement — la prime est trop grosse pour être entièrement une prime de risque.

**Explication comportementale.** Les investisseurs sous-réagissent aux bonnes nouvelles (*underreaction*) : quand une entreprise publie de bons résultats, le marché ne l'intègre pas instantanément et la correction se fait progressivement sur les mois suivants. C'est une friction cognitive, pas un risque.

---

## (vi) Synthèse — ce que le CAPM ne peut pas expliquer

| Anomalie | Auteurs | Variable | Ce qu'on observe |
|---|---|---|---|
| Effet taille | Banz (1981) | Capitalisation boursière | $\alpha > 0$ pour les petites caps, non expliqué par $\beta$ |
| Effet janvier | Keim, Roll (1983) | Mois de l'année | Rendements anormaux en janvier uniquement |
| Effet value | Rosenberg et al. (1985) | Ratio book-to-market | $\alpha > 0$ pour les actions value, non expliqué par $\beta$ |
| Momentum | Jegadeesh & Titman (1993) | Rendement passé 6–12m | +13%/an pour la stratégie long-short winners/losers |

Chaque anomalie dit la même chose : il existe des dimensions du risque — ou des frictions de marché — que le CAPM ignore complètement. Ce qui manque c'est un modèle qui intègre ces variables directement. C'est exactement ce que Fama et French vont faire en 1992 : si la taille et le ratio book-to-market prédisent les rendements au-delà du $\beta$, autant les traiter comme des facteurs de risque à part entière.
