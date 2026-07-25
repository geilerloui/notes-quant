# La faillite du Hedge Fund LTCM - part1

Le plus gros Hedge Funds des années 90 du livre When Genius failed de Roger Lowenstein; on va expliuqer certains concepts avant de voir l’histoire du hedge fund;

hedge fund peut utiliser n ‘importe quel méthode pour gagner de l’argent, arbitrage c’est la technique, soit deux maisons ds le même quartier une nouvelle et une ancienne; la maison nouvelle est plus cher que l’ancienne de 220 à 90; le spéculateur classique va parier sur la hausse ou la baisse de l’immobilier, l’arbitragiste va aller voir dans le détails si il peut pas faire un paris plus précis eg parier sur le faite que l’écart de valeur entre les maisons augmente ou diminue. Pk ça fait sens ce truc ? analyse te dit que la maison est en deux critères quartier et design on décompose la valeur des deux maisons, pour produit dérivée et obligataire la distinction est très nette (pour maison la c’es tpour l’exemple); ici seul le design peut expliquer la différence de prix , car la valeur quartier doit être la même en fait ça te dit que deux vieilles maisons vaut une maison neuve. Sur les marchés FI le traail d trouver la valeur de référence combien de produit B doivent être utilisé pour comparer un produit A , il faut des modèles mathémtiques pour avoir des approimations. Ici on voit que avec deux maisons vieilles y’a 40 de différence de designs, on va donc parier sur la valeur design , il dit que design va baisser pour la neuf car logique. On parie sque l’écart entre les valeurs design va se baisser . On parie à la baisse sur la valeur de la maison, on emprunte la maison A pour la vendre ⇒ on a une obligation de rachat qui marche comme une dette & on touche l’argent de la vente qui est la valeur de la position; pour l’instatnat on a rien gagné et rien perdu.

il dit que on peut avoir la valeur quartier qui monte et le design baisse et la je perds de la thune ⇒ il nous faut donc un hedge contre la valeur quartier la solution c’est d’avoir deux vieilles maisons , seul les variations de la valeur design nous font perdre ou gagner de l’argent.

On doit investier 180 pour acheter les deux maisons pr l’instant on a rien gagné ou perdu ; pk on utilise les 220 de la maisons récente ? pas possible histoire de collatérol les 220 ont utilisé pr autre chose.

IMG

maintenant valeur maison à bouger, la valeur quartier a explosé x2, maintenatn on doit racheter une maison de 300 à gauche on perds 80 et à droite les maisons on gagne 96 ben à la fin j’ai gagné +16 ⇒ la valeur design nous a fait gagné 20 et on a perdu 4 à gauche la différence ⇒ trade d’arbitrage sont compliqué et implique bcp de modèles, si on avait la variable baie vitré en plus ce serait compliqué faudrait une autre maisona vec une baie vitré ⇒ il trouver des produits financiers qui bouge dans le même sens pour isoler des variables c’est pas simple

![image.png](images/6-Économie/ltcm/01_image.png)

donc si je veux parier sur la variable 3 de mon produit il va falloir que je trouve d’autres produits qui vont me permettre de me hedger contre les autes variables.

Qui dit arbitrage dit souvent effet de levier, c’est 4% mais il laisse le 8.9% ; effet de levier arbitrageiste parie sur petit écart de valeur, qui veut investir dans une stratégie d’arbitrage qui rapporte quasi rien, alors qu’on pourrait acheter immobilier et se laisser porter par le marché ? on aurait pu se tromper et l’écart de la valeur design aurait pu exploser.

![image.png](images/6-Économie/ltcm/02_image.png)

fond investiseemenet classique est un fond dictionrel il parie sur la hausse ou la baisse il utilisé à 100% l’argent de ses clients; et à gauche il a un rendements de 8.9% donc il dit que effet de levier permets de rattraper une stratégie qui a pas une rentabilité élevé; c dangereux pr fond directionel tous les fonds doivent tomber à 0 pour tout perdre. Pour à gauche si tu perds 1/5 de sa valeur les produits tu perds tout , surtout effet de levier est risqué mais si inexistant l’artebitrage rapporterait rien.

![image.png](images/6-Économie/ltcm/03_image.png)

Pour Lehman on a vu ratio 1/30 pour l’effet de levier, est ce que banque font aussi arbitrazge ? trader market maker gagne l’argent en fournissant liquidté aux clients les traders achète quand lcient veulent vendre et achter quand cliente veulent acheter; donc ils doivent se hedger contre leurs positions mais on ne peut pas tout contrôlé y’aura toujours des variables non hedgées le hedge parfait n’existe pas. En général l’expo des banques aux variations des marchés FI résulte du faite que (a) que hedge parfaite existe pas (b) les modèles sont imparfaites (c) les trzaders interprétent les modèles et choissent de garder certaines expositions et pas d’autres.

donc trader des banques font arbitrage ou pas ? oui car le hedge parfait n’existe pas, mais chez les hedge fund ils savent parfatitement ce sur quoi ils doivent parier alors que les choix d’abitrage des banques sont moins précis ça dépend de ce que font les clients et des choix des traders market makers.

IMG ou pas

coment fiare pr un hedge fund de ne pas faire faillite car une perte de qq % peut amener à une chiute, solution est d’utiliser la formule de Black & Scholes , il y a une variable de perte de risque diffiicile à estimer et black & scholes ont trouvé coment l’estimer

![image.png](images/6-Économie/ltcm/04_image.png)

eg le dow jones : perdu 13% le 19 octobre 87 c’est le black friday loup de wall street, baisse de 8% octobre 89, 7% en octobre 97 de 6% en octobre 98; on constate que les journées de forte hausse ou baisse sont rare

![image.png](images/6-Économie/ltcm/05_image.png)

on se demande sur les 3771 journées de trading entre 1895 et 2000 combien on vu l’indice américain bougé de + de 0% mais moins de 0.1% c’”est 218 ; puis combien de jouréne au dessus de 0.1% mais en dessous de 0.2% etc (en gros il explique la logique de comment on construit l’histogramme). Cette distributoion est gaussienne, on voit deux choses (a) jouréne de trading se répartisse autour de 0% de performance ie la moyenne de l’ensemble des performance sde la période qui est de0.06% on a 1888 journée avec une perf inférieur a 0.06% ou perf supérieur de 1883 journée donc quasi autant de chose que ça tombe d’un côté que l’autre; les journée de trading est plus proche de la moyenne que du bord. on peut estimer la volatilité qui set 146% donc si je trace une loi normal autour de 0.06% avec écart type de 14.6% on obtient la courbe continue en noir.

![image.png](images/6-Économie/ltcm/06_image.png)

pour mieux comprendre la volatilité on voit une courbe en cloche plus écrasé car la volatilité est plus importante. les marchés n’ont aucune mémoire ça veut dire que chaque nouvelle journée est indépendante d’une autre c’est une bille qu’on mets dans la distribution aléatoirement.

![image.png](images/6-Économie/ltcm/07_image.png)

la gausseinne sera utile pour le risque eg journée de cotation avec -1.5% je compte le nb de jouréne sous ce seuil 171/3771 = 4.5% (en gros je prends la taile gauche de la distribution) et la loi normale te donne 173/3771=4.6% on parle de VaR(95.5)=-1.5% ; on a aussi que VaR(99)=-2.1% distribution normale puis en historique VaR(99)=-2.5%. En utilisant la VaR tu supposes que chaque jouréne de cotation est indépendant et qu’elle peut tomber à droite ou à gauche. Puis chaque nouvelle jouréne de cotation change performance moyenne et la volatiliét ce qui modifie la distribution normale. la distribtuon historique devrait tendre à ressembler à la distribution, Vu que les marchés financiers sont de plus en plus efficient elle supprime les risques. Il dit que VaR se calcule sur plusieurs jouréne eg sur les 10 dernières journée consécutive et certiutde de 99% quels sont les pires performances auquel on doit s’attendre si on a tout misé sur la hausse du dow jones. Il dit hedge fund et banque on peu de capital mais bcp de dette ainsi la VAR(99.9) permets d’estimer les pire perte , il dit que 10j - Var(99.9) = -2.8%; et le 10j - VaR(99.9)=-6.9% perte moyenne

![image.png](images/6-Économie/ltcm/08_image.png)

donc si le on a un levier de 1/5 ie 20% de capital pour 100% de produit investit dans la hausse du dow jones on peut dire que on devrait pas perdre plus ; on pourrait calculer sur trente jourss, trois mois.. mais le plus important c court terme ce qui laisse pas le temps au financier pour réagir , au dela ils auront le temps de pre,ndre des assurance ou réfléchir.

# La faillite du Hedge Fund LTCM - part2

en 1979 Ekcstein & co société de courtage doit vendre une position de arbitrage qui lui coute de plus en plus cher qui va la faire couller; et de l’autre y’a des T bills remboursement de <1an on peut décomposer la valeur d’une obligation: le taux d’intérêt + risque de défaut.

rappel : obligation a un principal son prix puis y’a un coupon divisé en deux taux + défaut qui définisse le niveau du coupon et c le rendemetn de l’obligation au début de sa vie, le rendement de l’obligation c la prime de risque. Mais coupon est un paiement obligatoire contractuel comme si les risques des deux variables taux & défaut peut varir dans le temps (dépende de l’entreprise et BC qui défini le tau directeur) ains investisseur achéterai l’obligation A suite au chgt de taux de la banque centrale, la hausse des taux réduit le rendement.

A l’inverse si risque de défaut monte a 10% les investisseurs la vente pour que celle de défaut soit vraiment prise en compte. En gros si les deux partie du coupon taux + défaut sont différent dans la vraie a un instant t par rapport à l’instant t=0 alors la S/D change le prix car le coupon est fixe. On rappelle que prix et rendement évolue en sens inversne. Pour un t-bill le défuat = 0% par contre le taux dépend de la BC.

![image.png](images/6-Économie/ltcm/09_image.png)

Puis il y’a des futur un contrat à terme d’acheter des T bills livré dans un futur proche ⇒ il en parle pas plus

![prix du futur 6 mois avant livraison](images/6-Économie/ltcm/10_image.png)

prix du futur 6 mois avant livraison

![image.png](images/6-Économie/ltcm/11_image.png)

Truc des maisons: sur les futurs, si je possede une maison si je veux hedgé la oslution est de vendre une maison similaire ici la logique est que plutot que vendre une vraie maison, tu vends une mais en construction qui sera terminé que dans six mois. tu remplaces la maison par une obligation c pareil. La maison futur devrait être moins cher que maintenant, obligation future est moins cher car tu auras des intérêts en retour que dans six mois. = écart entre le T bills et le contrat futur, mais plus je me rappelle de la date de livraison plus l’écart entre les deux produits se réduit. Donc un investissuer qui achete 6 mois à l’avance paye moins cher que un qui achete a 1 mois donc d’où l’arbitrage, le resserrement de valeur entre les deux produits financiers ⇒ emprunte T bills pour les vendre puis acheter future de T bill et on attends que ça se réduise.

En 1979 les futurs coute plus cher que les T bills pour une raison pas clair , donc l’entreprise Eckstein parie à l’inverse que ça va baisser elle va acheter des T bills et vendre des futurs pour aprier sur le retour à la normal car y’a aucune raison économique qu’un futur est plus cher qu’un T bill c une anomalie de marché ça ne va pas durer. Le probleme la situation poursuit et la valeur de futur en fait augmente encore plus elle doit faire des gros appels de marge qu’elle paye à une chambre de compensation . Or ils ont plus les fonds ils se retourne ver une banque d’investissement Salomon Brothers pour racheter leur trade, ils leur disent la logique est pertinente faut juste payer les appels de marge jusqu’à ce que le prix revienne a sa logique ⇒ Sauf que chez Eckstein pb d’appel de marge voire pb de liquidité . Il dit que Bond arbitrage fait partie de salomon brothers les perte finissent par se finir en gain au bout de qq semaines, meriwhether est confiant avant que perte devienne des gains, les associés de salmon brothers ont peur mais meriwehther reste conscient

![image.png](images/6-Économie/ltcm/12_image.png)

---

pour lui obligation est plus attractif que action, c plus mathématiques : BC augmente ou baisse taux directeur + risque de défaut donc risque de non remboursement. Variables = Risque de taux , risque de défaut; alors que action c plein de truc profits futurs, ventes futures, charisme du PDG… ont peut justifier n’improte suel prix.

![image.png](images/6-Économie/ltcm/13_image.png)

A l’époque les traders c’est surtout des commerciaux qui se base sur l’instinct plutôt que des modèles mathématiques, il va recruter des profils scientifiques dans son équipe, équipe surnommé les professeurs; toujours meme stratégie : ils téléchargent tt les historiques de prix des obligation regarde les fluctuations de prix il regarde les niveaux de volatilité, et il regarder les obligations qui suivent toujours la même direction et qq fois mais qui récemment ce sont mis à divergerou converger pour parier sur un retour à la n ormal vu que arbitrage produit complexe y’a plusieurs financiers …

![image.png](images/6-Économie/ltcm/14_image.png)

![image.png](images/6-Économie/ltcm/15_image.png)

Exemple:

- swap : entreprise a un prêt de 100m$ à taux variable : 3% fixe + 2% taux directeur de la FED (en fait taux interbancaire) ⇒ donc si fed monte le taux l’entreprise payera plus d’intérêt et inverse
- imagine que l’entreprie n’aime pas l’insécéurité de taux : l’entreprise aurait préfré un taux fixe, elle veut un investisseur pour payer à sa place, l’entreprise payerait 5%+0.75%=5.75% de taux fixe à cet investisseur pour éviter d’avoir un taux variable. elle s’est débarrassé du risque de variation du taux de la fed, en gros l’investisseur partie que le taux de la fed passera pas au dessus de 2.75% et l’entreprise parie l’inverse on dit que 0.75% c’est le spread de swap le prix d’une assurance sur la variation du taux directeur

![image.png](images/6-Économie/ltcm/16_image.png)

la il dit que le futur c’est pas pareil que le swap la pour ce projet de la hausse de la hausse du taux directeur ? ⇒ dans les deux cas l’objectif est de se protéger contre une hausse du taux directeur

- mais futur: protege celui qui prête celui qui possède une obligation contre la perte de valuer de cette derniè_re
- le swap (=échanger) protege celui qui emprunte à taux variable contre l’augmentation du taux à payer. ⇒ on échange taux variable contre taux swap

⇒ ce spread de swap évolue en fonction de la perception du risque de taux et est différent en fonction des matutirés de prêt comme on peut le voir sur ce gra^phique, on peut donc parier que le spread de swap de 5 ans et 1 an vont converger et diverger

![image.png](images/6-Économie/ltcm/17_image.png)

![image.png](images/6-Économie/ltcm/18_image.png)

Les professeurs veulent exploiter les failles dans le marché activité spéculative purement, pour les économistes néoclassiques des professeurs, la monnaie st neutre donc la création monétaire ne finance pas l’ééconomie la seule solution pour croitre est de rediriger l’argent de l’épargne vers les projets d’avenirs via banque de dépot et marchéés financiers, pour que les MI fonctionne bien faut toujours que ce soit liquide pour que la longue chaine d’acheteur vendeur se retrouve utile à l’entreprise, donc les professeurs aident les marchés.

Sauf que la monnaie n’est pas neutre, les prof ne veulent pas comprendre que leurs activité spéculative extrait de la monnaie de l’économie réelle pour faire des b ulles spéculatives plutot qu’aider l’économie réelle

![image.png](images/6-Économie/ltcm/19_image.png)

Les professeurs font des weekends ensemble ils restent que entre eux dans l’entreprise, ils sont coupés du monde sans recule, dvp un comportement paranoiaque les professeurs ne se prêtent pas leurs trades et idées. Meriwether est plus manager que trader, un trader vient le voir si il peut doubler sa position il dit oui sans justification. Merithewether devient chef de toute l’activité obligataires de la banque. Les professeurs sont surtout des traders sur compte propre pas des market maker enfin la frontière entre les deux métiers c’est assez floud asn les années 80, tant qu’ils gagnent de l’argent ça leur suffit.

ROnald Perelman essaye de racheter Salomon Brothers, mais buffet prends contrôle de la banque mais il aime pas les professeurs ils sont très arrogants ils ont perdu 30% krack bopuriser de 1987 san,s rien remettre en question et buffet croit pas aux produits dérivés il participoe actiement aux entreprises ou il investit en tant que acitonnaire. Les prof ont rapporté bcp plus d’argent avant même si il en est pas fan. Mais en 1989 les prof sont très gourmands car certains veulent plus de salaire à la fin 15% des profits iront aux professsurs y’en a un qui gagné 40 millions de $. CE qui isole encore plus les proefesseurs. Paul Mozer un trader vient trouvé meriwther il dit qu’il utilise une technique illégale pour acheter des obligatins d’état, il fazit remonté l’information à la hiérarchie et aux autorités ce qui coute cher il est viré meriwether maisa les professeurs restent . ET meriwther veut créer un hedge fund pour répliquer les stratégie des professeurs sans les contraintes de la banque.

## Episode 3

fond sd’investissement clasasique sont tres regulier ils doivent dire leurs méthodes aux épargnants; alor que hedge fund ont très peu de contrainte c’est pour les épargnants les plus riches qui peuvnt investir des centaines de milliers de $; c’est les années 90 les hedge fund sont à la mode pour les plus aisés. Meriwether veut exploiter cette tendance il contacte des banque ou fonds d’investisseement qui ont des gens fortunés pour avoir une expositin différente via un hedge fund spécialisé dans un arbitrage. Donc fond d’investissement invests dans un hedge fund pour la diversification. Son but est de collecter 2,500 million de $ soit x100 que ce que commence un hedge fund au début. Puis il prévoit de bien payer les équipes deux façons de se rémunérer:

- Commission fixe 1% de l’argent fixe investit par les clients
- bonus qui est en environ 20% des bénéfices engendré

Le fonds de meritherwt aura un 2% et 25% ce qui est au dessus des clients, il force aussi les clients a rester pendant 3 ans car il a déjà vécu la situation précédente avec l’arbitrage ou fallait attendre. LTCM long terme capital management.

toute l’équipe des professeurs les rejoigne et aussi Merton et Scholes le rejoingne. Merton croit dure comme fer à la théorie des marchés efficients et des agesn économiques calculateur de la théorie économie orthodoxe.

efficient = chaque prix sur les marchés financiers intégre toute les informations existante ⇒ eg si tu as le cours du spx qui est à 2900 mais qui semble suivre une tendance haussière tu ne pourras pas l’exploiter, la forme de la trajectoire de la courbe est lié à un enchanement d’informations aléatoire. toutes les informations intéressantes ont été pris en compte par le marché. Pour lui les marchés ne sont pas complétement efficients mais qui tende à le devenir. Pour merton marché FI sont surtout composé d’agent calculcateur qui utilise que qq infos pertinente. La rumeur non éteillé de la faillite de l’entreprise A ne fera pas bcp bouger le prix de A, les agents rationnels vont comprendre au bout d’un moment qu’une rumeur ne justifie pas une baisse de prix.

![image.png](images/6-Économie/ltcm/20_image.png)

![image.png](images/6-Économie/ltcm/21_image.png)

Scholes est une star de la finance, il est pas aussi convaincu que merton sur l’efficience des marchés. Merton est un théoricien et scholes un pragmatique qui veut toujours tester les théories.

![image.png](images/6-Économie/ltcm/22_image.png)

en janvier 94 greespan directeur de la fed augmente le taux directeur qui passe de 3.05% en janvier à 6.05% en avril une tel hausse pren les marchés par surprise et bcp d’investisseur obligataire qui n’avait pas hedgé voit une perte de leurs obligation. Donc baisse des prix donc perte de valeur

![image.png](images/6-Économie/ltcm/23_image.png)

FOnds classique n’utilise pas d’effet de levier ils snt donc plus résilient. Mais on peut faire fiallite pour manque de cash et ça c’est demande de retrait du client du fonds qui panique et qui veulent retirer leurs fonds. LTCM utilise cette panique pour faire un trade de convergence avec arbitrage , certains produits ont eu des baisse assez forte on a donc des écarts de valeurs importants il parie sur le resserrement il parie sur un retour à la normal de la volatilié et après 2 mois le fonds décolle. Mais ils arrivent avec des milliards sur un marché qui se disloque les professeurs savhent que un évt inconnu peu les frapper à tout moment et que les hedge qu’ils mettent en place les laisse peuvent ne pas aller dans leur sens, pour ce protéger il se repose sur un vision statistique si ils investissent à différents endroits de la planète le risque qu’ y’a une crise globale est peu probable ? en fait non en 94 il y’a eu une crise globale. mais acant d’expliquer ça on va voir comment LTCM devient une grande machine à milliards de wall street.

# episode 4

eg stratégie arbitrage de ltcm, tous les six mois le govt américain émets des obligations qui seront remboursés dans trente ans, donc obli 30 ans févrir et obli 30 août ce sont les obligations les plus surs au monde, puis apres six mois de cotation les obligations 30 ans ont tendance àdevenir moins liquide, car au début de l’émission il y’a un système de lotterie ce qui fait que les investisseurs ne sont jamais sur de la quantitéd’obligation qu’il sohaite certaine en ont trop et d’autre pas assez donc pendant 6 mois les investisseurs s’échangent leurs obligation, puis apres il les garde en atendant 29 ans et demi;

CErtians fond d’investissement veulent jamais avoir l’obligation ranger dans les tirroirs, ils ont trop peur que si ils deviaent tout vendre d’un coup ils ne trouveraint pas preneur, ils veulent que celle de 30 ans qui s’échange massivement. DOnc à cause de ces fonds l’obligations des six mois est toujours à un prix légérement supéeieur à celle qui ont été rangé. Supérieur car les fonds appuie sur la demande la différence est de 1% et dés que l’obligations du moment est remplacé par la suivante elle perds le surplus de 1%. Donc on veut obligation les plus cher du moment donc les 30a-fev-95 et on acheète les plus cher 30a-aou-94 (car initialement celle du 30a-fev-95 et à droite elle a les 1% de delta) mais j’attends qu’elle devienne has been puis elle chute vers celle des 2e etc

![image.png](images/6-Économie/ltcm/24_image.png)

pour rendre ça plus rentable les preofesseurs s’endette pour augmente le rendement pour 1$ du fonds ils emprunte 29$ ce qui multiplie le rendement par x30, effet de levier, avec toutes cette dette faut payer des intérêts poiru imapcer les bénéfices du fonds ?

![image.png](images/6-Économie/ltcm/25_image.png)

en fait non, LTCM achèete les obligations les moins cher et vendre les plus cher;

- les professseurs emprunte pour 2 milliards d’obligations a une banque d’invest pour les vendre
- la banque qui prête les obligation réclame un vesrement d’intérêt et le dépôt d’un montant de collatéral en échange du prêt. Le collatéral corresponds à la valeur des obligations prêté

Explication sur le collatéraol: si tu me prête ton vélo à $150 tu me demande de te verser un intérêt et de te verser 150 euros en garantie (le collatéral) que je rends si tu me rends mon vélo.

![image.png](images/6-Économie/ltcm/26_image.png)

- donc tcm récupérer 2 milliard d’oblig mais il doit passer 2bo$ en échange du prêt , cpdt LTCM peut avoir les obligations maintenant et passer le collatéraol maintennat, ainsi il vends direct les obligations obtient les 2mds pour acheter les obligations de l’autre pack du trade
- puis ils prêtent obligations qu’ils ont acheté a une autre banque en échange du colatéraol du 2mds$ en échange puis ils le prêtent à l’autre

⇒ donc rien à payer juste un écart d’intérêt mais assez faible. Il dit ensuite que ils vendent les obli les plus cher pour ensuite acheter les moins cher donc les montants sur les deux pates peuvent pas être les mêmes ? si il suffit d’acheter un peu plus d’obligations pas cher par rapport à celle qui sont cher. Sur la pate vendeuse la banque demande un Haircut ie un peu plus élevé que la somme du collatérol donc 2.02Mds$ ce haircut vient du capital du fonds les 20 millions. Sauf que LTCM négocie avec les banques et supprime les haircuts donc aucun haircut les professeurs peuvent investir sans limite. Car LTCM est le hedge fund le plus prestigieux les banques acceptent.

![image.png](images/6-Économie/ltcm/27_image.png)

pk pas levier de x100 alors ? car le niveau de levier vient du modele de risque de Merton & Scholes, si tu te rappelles de la VaR, si je connais les performances passé de l’obliation A et B, je peux donc créer distribution hybride de A-B ce qui me donne sa vol, moyenne… différence de deux gaussienne, dés que je la vol et moenne je peux calculer la VaR de 10 jours à 99 indice de confiance = 3% donc pour 100$ d’oblig et avec certitude de 99.9% risque d eperte en 10 jours de 3$ au maximum dit autrement tu peux investir jusqu’à x33 ton capital avec un risque de 0.1% de perde plus que ce capital. DOnc pour un capital 1,250 millions la qt d’argent que les prof peuvent parier est e 1.25 * 33 = 41,667 millions de $. En gros les prof définisse au début manuellement un niveau de certitude et un horizon de temps vAR(99.9).. a 10 jours, 30 jours; ensuite leur modele calcul la volatilité du portefeuille ce qui est pas simple car on combine plusieurs produits financier y’a de la cov et bcp de données. Ils ont ensuite une courbe en cloche puis on insere la volatilité dans l’équation de la VaR pour connaitre la perte attendu pour définir le levier optimal du fond.

il dit courbe cloche symétrique donc avec faible volatilité il gagne quasi rien et perde quasi rien car symétrique. Cette volatilité est supposé être constante or c’est faux. Cependant leur modele corresponds plutot à être proche de la rélaité. ie c une gaussienne vs histogramme; la gaussienne est utilisé pour anticiper les pertes; la histogramme est pour anticiper leurs gains. Pour repéréer les gains ils se disent que la véritable réalité va converger vers la théorie des équations. Pour merton un jours le modèle va converger avec la réaliét et y’aura autant de chance que de perdre.

Mais on voit ques les profits ont explosé , les commissions leurs permettent d’être ultra riche, mais ils réinvesitssent une grande partie dans le fonds et hillibrand faire un effet de levier sur son salaire pour réinvestir dans le fonds. Mais les prof sont vus commes des génies mais leurs arrogances bcp eg les banques car y’a jamais de compromis et les banques ne savent pas ce qu’ils font avec leurs positions. Et aussi car ils divisent les positions du trade sur plusieurs banques une partie pour le pari et l’autre pour le hedge ; les banques comprennen rien et cmme ça ça évite qu’ils comprennent leurs trades. Il dit que les techniques d’arbitrage sont monnaie courante donc la seule force de LTCM c’est ça puissance de négotiation avec les banques en réduisant les frais et haircut alors que y’a pas vraimnt de stratégie original. Mais donc vu que leur stratégie est connu tout le monde va faire pareil donc le fonds va avoir de moins en moins d’argent

![image.png](images/6-Économie/ltcm/28_image.png)

![image.png](images/6-Économie/ltcm/29_image.png)

LTCM trade sur toute la planète pour la diverisification il gère 7.4Mds$ quasi x6 de son lacnement et avec un effet de levier x20 ils ont des positions de 140 milliards de $ au total pour leurs positions en comparaison tu as lehman brother de 5000 employés et le hedge fund de 200 employés et qq prix nobel (Merton et Scholes en 1997)

![image.png](images/6-Économie/ltcm/30_image.png)

![image.png](images/6-Économie/ltcm/31_image.png)

On refait de la théorie, on voit la courbe de vol de IBM, cette courbe devrait être une ligne chez black & scholes, sauf que réalité est différente car investisseur on peur si trader était que rationel avec les infos alors la volatilité devrait ête constante, les variabilité de vol sont des décisions irrationels due à la peur viscérale de perdre de l’argent qui passe devant le sang froid des traders. Il faut supprimer la peur des traders pour rendre la volatilité constante. Les raisons pr laquel les prix d’un produit financier varie sont infini mais la conséquence d’une trop forte variation est toujours la même : une perte d’argent trop importante et si il était possible de supprimer la peur avec des produits dérivés pour se hedger contre les variations de prix. Si les marchés vont pas dans le bon sens pas besoin de vendre tu prends une ascenceur . Le 2e truc qui fait peur au trader c’est de vouloir acheter/vendre et pas trouver de vendeur/acheteru, il faut toujours deux contrepartie qui se décide sur un prix c’est le risque de liquidité qd tt les traders sont d’accord entre eux tous acheteur ou tous vendeurs. Mais vec des produits dérivés on crée des nouveau type de paris quels sont les chances que tt les traderes veulents faire la même chose si ils font des paris différent. C pk merton défend l’idée de marché effciient tout ces produits créer des produits szans peur grace aux hedge et risque de liquidité duex aux avis divergents comme ça ils peuvent étudier calmement les marchés. finis les piques de volatilité les marchés peuvent devenir la machine de la théorie. SI les marchés sont efficient ie si les risques de pertes et de liquidité disparaisse petit à petit alors les variables de risques doivent converger

![image.png](images/6-Économie/ltcm/32_image.png)

ooo

![image.png](images/6-Économie/ltcm/33_image.png)

ooo

![image.png](images/6-Économie/ltcm/34_image.png)

variable de risque qui converge : eg actions et son produit dérivé qui lui sert d’assurance une option, quand les grands fonds d’investissement ont peur de voir la valuer des actions qu’ils détiennent chuter ils prennent des assurance qui sont des options de vente (put), la valeur de l’action dépends de tout un tas de varibale difficile à cerner (bénéfices, vents.. ) = variables actions. La variable de l’options dépends des même variables que l’action ainsi que la volatilité de l’action qui est lié à au risque de l’entreprise et les duex pertes supplémentaires. Or si les deux rouges disparaisse donc la volatilité va diminuer pour converger vers sa valeur naturel et c quoi cette valeur ? a rien c une hypothese du modele on dit que chaque produit financier a une volatilité invariable qui lui est propre mais rien ne dit que c vrai c une croyance basé sur un modèle mathématiques.

![image.png](images/6-Économie/ltcm/35_image.png)

En plus de ce paris sur la réduction de la volatilité, LTCM mise sur la réduction de la réduction des spread de swap ; le modele économique standard prédit une économie stable toujours en équilibre et dont le taux de croissance est lié au progres technique avec des marché sfinancier pour allouer les ressources monétaires et les porjets il devrait y’avoir de moins en moins de perturbation. Or la politique monétaire de BC sert à relancer l’économie qd marché ne l’alimente plus ou la calmé quand les marchés l’alimente trop, les marché efficient fera disparaitre cette technique donc moins bouger le taux directeur donc moins de risque de variation donc police d’assurance qui coute moins cher donc spread de swap qui baisse. ⇒ prof pense diversifié mais ils font le même parie partout dasn le monde : réduction du risque de liquidité et de perte massive, réduction du risque de taux, finalement réduction de tout ce qui sert à mesurer le risque ? le paris sur la volatilité et spread du swap y’a d’autes paris qu’on a pas dit ; y’a pas assez de diversification.

En juillet 1997 crise financière asiatique, LTCM est pas trop affecté les paris sur less baisses de taux swap ont pas été mis en place en asie ; les prof s’insuiqété d’autre chose ils n’arriven tplus à trouer de nouveaux paris 140/7.4=18.9 d’effet de levier c’est trop faible faut revenir à 30 mais faut trouer des opportunités d’arbitrage qu’il pense que c’est impossible. la solution set de rende la mise au client pour avoir un meilleur rendement on passe de 140/4.6=30 avec un fonds de 26% de rendement en 97. Les prof ne sont pas satisfiait si quasi tt les clients doient récupérer un peu d’argent , les 3 passe à 40% dans le fonds ils ont touché 155 millions de $ en 97 ils espérent doubler leurs gains.

## Episode 5

deux évts qui déclenche : nouveau trade d’arbitrage sont difficile à trouver car tte les variables de risques sont au mimjimum car tous les arbitragistes font pareil preuve que le tout wall street réplique leurs statégies; les arbitragistes se rendents compte que y’a plus de marge de manoeuvre dans ce domaine. Ils vendent leurs positions eg Salomon brothers warren buffet le vend à traveler’s group, l’ancien département arbitrage fait les mêmes stratégie que LTCM. or sandy weill le boss n’apprécie pas ses traders arbitragistes, il trouve qu’ils sont peu éthique et plus risqué que ce que dise les modèles et qui ont des énormes bonus masi pas de malus quand ils se trompent il ferme le département. Ainsi les positions vont être vendus donc baisse du cours mais avec le hedge ont en est proétégé ? En fait on dit plutôt liquidé les positions. On ne parle pas d’une baisse mais plutôt d’une hausse, les arbitragistes ont acheté A et emprunté puis vendu B pour parier sur la baisse de l’écart de valeru de la composante risque. Mais pour sortir, liquidité leurs positions ils doivent faire l’inverse ce qui a été emprunté et vendu doit être racheté et rendu et ce qui a été acheté doit être vendu. DOnc vendre A baisse la valeur, et rachter B hausse la valeur donc écart de valeur entre les variables augmente. Or LTCM n’est pas hedgé sur ces variables car ils parient dessus.

![image.png](images/6-Économie/ltcm/36_image.png)

perte se font sentir mais leurs plus grosse perte à pas dépassé 99% et les modèles de Var 30j (99.99) = 40% ce qui donne une volatilité de 15% donc arive que les 500 millions de milliard d’année, cpdt pb du modele marche pas en période de crise, chaque jouréne est indépendante de la précédente, mais en cas de crise chaqque journée de perte est de perte et le jour d’après aussi les marchés ont une mémoire plus un tirage aléatoire

![image.png](images/6-Économie/ltcm/37_image.png)

mais on avait dit que la panqieu amène à des variations non rationnels qui p résente des opportunités pour les traders rationnels ? en théorie oui, mais quand la baisse irationel démarre, le trader rationel sait que la baisse n’est pas justifié donc pr lui c une opoortnité d’achat, mais faut que suffisament de trader fasse comme lui, suffisament de gens valide cette analyse. Les russes doivent emprunter eds $ pr importer marchendise du reste du monde car peu de pays accepte de payer en rouble, et le prix du pétrole chute, en juin 1998 ils emprunten 1.25Mds$ à 12% de taux et les prof pensent que la russie ne peuvent pas faire défaut car ils ont l’arme nucléaire. LTCM en achetant à 12% mais ils hedgé il dit que si la russie ferait défaut elle serait dans un état précaire seul solution est de dévaluer leurs monnaies. Remarque : value du rouble par rapport au $ est fixé par les autorités russe et pas par les marchés financiers c pour _a qu’on dit dévaluation sinon faudrait dire dépréciation. Donc si russie fait défuat elle dévalue leur monnaie donc parie sur la baisse de valeur du rouble par rapport au $. SI ça se passe bien la monnaie est pas dévalué il perds rien sur le hedge et préléve intérêt de 12% si pb il perde argent prêté et intérêt maias amortisse la perte grace au paris sur la dévaluation.

⇒ le govt russe fait défaut sur sa dette souveraine mais pas sur les obli que les prof ont acheté; il suspendu les intérêts en $ donc la valeur des oblig plonge, mais le rouble chute donc le hedge doit compenser la perte sauf que la banque russe fait faillite donc le hedge ne fonctionne pas donc 430 millions de $ de perdu pour ltcm. Tous les paris de LTCM prennent l’eau les prof ont parié sur la réduction de tte les variations de risque c l’inverse qui se passe : la volatilité décolle et les spread de swap aussi l’idée est simple prof on parié sur

1. la baisse de valeurs des produits dérivés permettant de s’assurer contre la perte de valeur des actions
2. parié sur réduction de l’écart de rendement entre les produits obligataires les plus risqués et les moins risqués

OR à cause de la panqieu les assureurs classiques sont très en demande d’assurance contre la perte de valeur des actions donc le prix des options augmnte. Côté obligataire ils vendent les produits les plus risqués ce qui fait baisser leurs prix donc augmenter leurs rendements et achete les produits obli les moins risqués ce qui fait monter leur prix dont baisser leurs montant c fly to quality. la théorie des marché efficient dit que c que temporaire c cette opportunité que ltcm avait utilisé à ses débuts quand greenspan avait monté le taux directeur de la fed; le pb c que faut de l’argent pr faire sauf que en crise on a difficement plus de capital et l’effet de levier étant déjà maximale la dette ne peut pas plus augmenter

![image.png](images/6-Économie/ltcm/38_image.png)

![image.png](images/6-Économie/ltcm/39_image.png)

ccl arbitagiste peuvent pas lutter ocntre la panique en faisant des paris ils doivent attendre pdt la tempête, donc arbitragiste craignant plus le risque de faillite, liquide leurs positions ce qui accentue la tempête, les vendeurs trovuent pas acheteur.. panique liés à une crise de liquidité ils n’arrive plus à liquider leurs positions risqués. La soit disant diversification des paris de LTCM n’existe pas, les prof ont toujours misé sur la même variable et face à une tel crise leur ancienne force devienne une faiblesse = ils sont trop gros 100 milliards de $ de positiion globale. D’autrers grosses avec leurs desk d’arbitrage ont chité aissi sauf que banque d’investissement à d’autres activité que arbitage pour se calmer.

![image.png](images/6-Économie/ltcm/40_image.png)

![image.png](images/6-Économie/ltcm/41_image.png)

ils ont perdu 550 millions alors que leur odele avait prédit 100 millions avec leurs modele avec ceftitude de 99.9% le prof veulent réduire leurs expositions mais ils ne trouvent pas preneurs ou il ferait trop bouger les prix ce qui détruirait leurs modèles. encore une réalité que le modele de merton & scholes ne prenne pas en compte. si le prix d’un produit est de 10 hier le prix passera par 9.9 , 9.7.. ce qui laisse le temps de vendre mais en réalité ça peut sauter d’une valeur à l’autre et être massif , une courbe de prix n’a pas de raison d’être conteinu

![image.png](images/6-Économie/ltcm/42_image.png)

il dit que leurs capital chute trop ils vont chercher des fonds à soros, goldman sachs, warren buffet suite au défaut russe, qd tout revient dans l’autre les bénéfices vont être massif faut injecter qq milliards ⇒ find août 98 les prof ont perdu 2.35 miliards de $ la moitié du fonds.

Il demande l’aide de Vinny Mattone qui lui dit qu’il est finis, puis Fed de NYC doit intervenir car LTCM pèse en aout 98 100 md$s pr un capital de 2m$ donc les 98mds$ sont les créditeurs sont 50 banques d’investissement qui doivent récupérer les prêts qu’ils lui ont fait. banque consciente de se pb les banques essaye de s’auto sauver. patron des grands banques d’investissement fotn des batailles d’égo. et risque effet domino une banque coule puis une autre .. car elle se prête toute entre elle. Puis le 23 septembre 1998 apres négotiation LTCM qui ne vaut que 400 millions de $ est racheté , les prof perdent toute leurs mise dans la boite. seul larry hilliband est compéltement ruiné il a une dette de 24 millions de $. Puis après racahat (rouge) le fonds continue à perdre de l’argent jusqu’à ce que les taux re augmente.

![image.png](images/6-Économie/ltcm/43_image.png)

Quand tu regarde sles deux principaux paris du fonds, si LTCM aura re gagné de l’argent à partir de 2003 soit 5 ans plus tard

que retenir ? modele de black & scholes es tun modlee mathematique a des hypotheses qu’il faut comprendre

- chaque jouréne de trading aune performance aléatoire qui tourne à gauche ou droite de la performance moyenne, marché n’ont pas de mémoire
- chaque acteur sur les marchés financier est un robot il les analyse, réfléchit à ses pertes et gains et prends une décision
- seul le calcul de proba rationnelle sont utilisé spr prendre des décisions ; la peur de perte potentiel ne l’emporte pas sur la prise de décision sur l’espoirt de la prise de gain potentiel
- chaque produit financier a un niveau de vol invariable on ne sait pas pk elle est invariable ou elle fait référence à quoi. en gros ça dit que la distribution des performances futurs sera pareil que celle du passé. Donc le passé est représentatif de ce que l’avenir nous réserve
- les prix évolue de façon continue pas de saut
- marché liquide y’a toujours un vendeur en face d’un acheteur
- marché efficients : chaque prix est issu d’une réflexion rationnelel à partir de toute les info dispon y’a pas de tendance, forme de courbe qui te dit comment vont évoluer les prix

voila pk ltcm s’est planté, en pensant que le monde doit se comporter comme un modèle ils oublietn que c l’inverse, le modèle qui doit représenter le monde. Pusi le modèle a été perfectionné gaussienne minimise fortement les performances très négatives et les évts qui doivent à partir les X milliards d’annéees arrive en fait tous les 3 ans.

Mais en fait, le moee de black scholes doit se comprendre dans le contexte de la théorie économique orthodoxe ou il se calque car l’épargne est le seul moyen de financer les entreprises qui investisent dans léconomie réelle pour permetrte la croissance . le marché est une machine intermédiaire obligatoire , marché qui doit pas bloquer il doit toujours rester liquide d’où l’importance des produits dérivés et des stratégie d’arbitrages qui permettent la couverture des risques et la poursuite d’intérêts multiples ce qui permets d’avoir toujours un vendeur devant un acheteur . La justification de l’existence des produits dérivés, de l’effet de levier et de la spéculation à outrance se trouve dans la théorie économique mais qui n’intégre ni la création monétaire par les banques, ni le financement de l’investissement par les entreprises par la réutilisation de leur propre bénéfice ni le danger des bulles spéculatives, ni que la maximisation de l’utilité sur la base d’un calcul mathématiquemenr ationnel ne coreponds pas au comportemetn des gens. Or leurs modele a été construit sur ces hypothese comme tt le théorie financière. Pour éviter les crises il faut remettre en qst le rôle d’intermédiaire edes marchés financier, il est faux de dire que seul l’épargne peut financer l’investissement . face au réinvesitseemtn dfes bénéfices et a la crétion monétaire les marchés FI ne sont important que dans certains cas particuier, les acteurs des marhés et régulateurs doivent comprenedre que la liquidité ne justifie pas ttes la spéculation produit dérivé.. car a toujours checher la liquidité on arrive à des bulles Pour la théorie orthodoxe les bulles spéculatives est due au hasard pas au pb de fct des marchés financiers. la théorie économique orthodoxe doit se faire évoluer.

![image.png](images/6-Économie/ltcm/44_image.png)

aaa