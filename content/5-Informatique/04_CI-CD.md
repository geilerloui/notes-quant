ensemble de pratique pr accélré le rythme de déploiement des applications:
CI = continuous integration = intégration continu
CD = Continuous depluent = déploiement continu

le rythme de mise en prod est de 2 à 3 version majeur par an historiquement, c'est très long, processus séquentiel entre équipe de dév et d'ops ie de production. sur ces quatre principales étapes chacune est géré par une équipe en effet à chaque étape faut synchroniser les différentes acteurs.. Le CI cheche à automatiser les opérations autour du dvp, et le CD cherche à automatiser les opérations de déploiement; l'implémentation d'un pipeline CI/CD ou chaine CI/CD cherche à supprimer les activités humaines chronophage afin d'avoir un dvp jusqu'à la mise en production 

![[Pasted image 20260731105918.png|432]]

les applications modernes sont décompoéss en petit service indépendant aka micro service, chaque dév travaille simultanément sur différente partie d'une application, or lorsqu'un dév fait des modifs ça peut entrer en conflit avec les autres dev; l'outil de gestion de code source le SCM va permettre de centraliser au fur et a mesure les modifs de code et gérer les évolutions, GIt et gitlab eg. Le serveur d'intégration récupère ce code (il a un service orchestration comme jenkins va lancer des opérations , puis il appelle service pour compile ie créer un build ie une version exécutable de l'application,), ensuite il fait des testes rapides , tests automatisés eg test unitaire codé par le développeur, test d'intégration système test fonctionnelle pr vérifier le bon fonctionement dans son ensemble, ensuite controle qualité un 1er service vérifie que la qualité du code n'a pas été dégradé comme sonarcube, un 2nd type vérifie que les dépendances n'ont pas de faille connu genre jfrog.. si y'a un pb la chaine CI/CD s'arrête sinon le nouveau build est validé est archivé dans un dépot d'artefact ou repository , il peut donc après être déployé.

![[Pasted image 20260731110243.png]]

puis déploiement continu: nouveau test vont etre fait sur les différentes environeement: test de recette avec selenium pr tester des scénarios complet, test pour tester l'écosystème complet , ce type de teste est prise en charge avec UFT eg, aussi test de charge et perf pour tester qu'on a pas une dégradation des temps d'accès de l'application et temps utilisateur instantané aussi; on teste non régression ie pour chaque type de test on revérifie des scénarios ; pr le déploiement de l'appli sur un environement hisotioruqement c sur un serveur ou alors sur appli plus moderne on déploie via le conteneur via une image docker il est aussi possible d'utilisb infrastructure as code a partir de fichier 

![[Pasted image 20260731110452.png]]

ccl: grace a ce processus automatisé c plus simple et rapide de tester mettre en production une nouvelle fonctionalité, le principe est de repérer une erreur rapidement dans les étapes et d'avoir des iétérations plus petite permet de mieux les gérer et les erreurs ont moins d'impact donc. 

un des objectifs du devops ets de réduire le time to market (délai entre idée initial et appli sur le marché) il répond à ça avec l'automatisation ; avec CI/CD on peut faire des petit itération rapidement accessible aux utilisateurs ; en ccl 

![[Pasted image 20260731110904.png|516]]

## Serveur

le rôle du serveur eg retaurant: t'es un client tu comandes ton repas au serveur et son rôle c'est d'apporter ce que tu veux comme nouriture; serveur informatique c'est pareil sauf qu'ils fournissent service autour des données eg fournir accès au page web, service courier électronique, stocké vos données en base de donnée, faire tourner application; donc spotify, gmail, youtube ou instagram c'est gérére et hébérgé sur des serveurs et tout ces services sont accessible via des requête que font les client on parle de relation client-serveur 

![[Pasted image 20260731111214.png|416]]

![[Pasted image 20260731111445.png|384]]

caption le client peut etre soit humain soit un autre serveur eg serveur web demande au serveur bdd des info pour afficher des données

il est dans les logo de HP les serveurs sont dans une baie il a ram, processeur ... mais en bcp plus performant, rien n'empêche d'utiliser un ordinateur comme un serveur mais c'est pas la level. les entreprises hébergent leurs esrveurs dans des datacenters site d'hébergement qui offre des services eg service télécom, réseau, refroidissement des serveurs, sécurité incendie... 

### serveur d'application

c un middleware qui est entre couche OS et application, pou etre plus précis c un 

![[Pasted image 20260731113752.png|284]]

eg l'accès aux base de donneés, gestion de transaction, connexion au SI, sécurité, clustering eg 

![[Pasted image 20260731113835.png|347]]


### middleware

couche technique entre le OS et couche applicative, mais c aussi un logiciel qui tourne sur un OS sauf que sont ^role est d'aider les applications à interagir ensemble 

![[Pasted image 20260731114101.png|197]]

il permet de faire circuler les données d'une apli à l'autre snas que ça a été prévu pour, cest pour que les dév se taff que sur le côté business. 




## comprendre micro services

une application est dividé en plusieurs petit service, chaque service est spécialisé en une seule tâche, service = service métier = fonctionalité métier;

![[Pasted image 20260731112136.png|303]]

eg site de vente en ligne, micro service, panier, historique, inventaire... ensuite l'archi a été inventé pr les pb des applications monolithe ie ambition de traiter toutes les requêtes possible sauf qu'elles grossisent de plus en plus mais en supprimant les anciennes fonctionalités et avec le temps les blocs devinennyt interdépendante la quantité de code augmente et c hyper compliqué 

![[Pasted image 20260731111900.png|293]]

la micro dit on découpe l'application en module fonctionelle ie microservice chaque partie dvp une partie unique de l'application et ça peut être appeler par l'API du microservice correspondant , comme tech c'est souvent couplé à docker en gros un conteneur par unité de microservice pour hébérgé son code, si un service est plus demandé lors d'un pic de charge on peut générer plus de conteur, et a contrario on garde que le minimum de micro service alors que en monolithe faudrait tout modifier.

![[Pasted image 20260731112205.png|355]]


donc avec une application en micro service c plus facile de faire des v2 et supprimer des parties. 

![[Pasted image 20260731112229.png|377]]

![[Pasted image 20260731112205.png|418]]


## ETL/ ELT

rôle d'un ETL est de consolidé des données a travers trois actions: extraire ie extraire données depuis plusieurs soruces, données, applications, bdd ... transformer: nettoyer puis standardiser les données en définissant le stockage les doublons, données inutilisable en gros transformer les données en format exploitable; load= l'etl charge ses données vers une destination cible genre datawarehouse ou les données seront accessible via des donnéees pour faire des requêtes

![[Pasted image 20260731112642.png|457]]

on a aussi le ELT: cloud fans la partie load espace de stockage quasi illimité ce qui permet d'ingérer les données dés que dispo même sans savoir ce qu'on va en faire, c souvente dans un data lake,stockage intermédiaire dans le cloud ; la transformation est réalisé que quand c'est nécessaire eg machine learning, eg talent et apache kafka 

![[Pasted image 20260731112854.png|570]]
## API (application programming interface)

interface de programmation d'application; permetatnt d'accédé a un service comme donnée ou fonctionalité fournit par un système tier (on dit qu'il expose une api) en gros c pour faire dialoguer des applications une consommatrice de service et l'autre productrice de service. 

eg uber , elle fait appel a d'autre appel fournit , qd j'entre l'adresse ou je veux aller dans l'appli il calcul ta distance optimal, uber fait appel au service de localisation et carte de google maps puis une fois le trajet effecetué c'est facturé via le api applepay l'appli uber utiliser donc deux entreprises ext et en contrepartie apple paye apple et google 

![[Pasted image 20260731113223.png|560]]

api rest a été construit en suivant les standard du web ie basé sur le HTTP qui est un protocle de référence qui défini les communications sur le web, l'objectif est d'exploiter pleienemnt le protocle; toute les api ne sont pas rest mais ça permet une meilleur communication avec les éléments du web

![[Pasted image 20260731113532.png]]


# Virtualisation

faire fonctioner sur une même machine physique plusieurs systèmes comem s'il fonctioné sur des machines physiques distinctes, on va voir que virtualisation de serveur et de poste de travail.

eg on a un serveur qui a trois couches, hardware, os et application, virtualisation est d'utiliser le hardware pour faire tourner plusieurs systèmes comme si chaque système avait ses propres ressoures matérielle on a donc plusieurs OS ou tourne des applications ; c'set un eg de virtualisation et il en existe d'autre

![[Pasted image 20260731114506.png|444]]

couche de virtualisation on l'explique: l'hyperviseur est la pour controle le processuer et les resource de la machien hôte il alloue à chaque VM les resources dont il a besoin et s'assure que les VM n'interférent pas l'une avec la'utre.

hyperviseru 1: logiciel sur le hardware, OS invité ie OS secondaire => plus comune en entreprise
hyperviseur 2: 

![[Pasted image 20260731114642.png|243]]

eg y'en a plein ici poste de travail à distance

![[Pasted image 20260731114713.png|175]]


# docker

la virtualisation par conteneur se base sur la virtualisation de linux, LXC = linux containers, méthode de cloisnement au n iveau de l'os, on tourne des env linux différente mais partageant les même noyaux, donc un conteneur n'inclus pas d'OS car il utilise l'OS hôte

![[Pasted image 20260731114905.png|267]]

![[Pasted image 20260731114847.png|265]]

conteur virtualise l'env d'exécution comme processeur, mémoire vive, mais pas la machbine c pr ça qu'on dit conteneur et pas machine virtuel 

![[Pasted image 20260731115001.png|415]]

diff&érence entre contenruet et VM classique: une VM recrée intégralement un serveur ie OS complet, pilote , appli elle même, car le conteneur n'embarque pas d'OS il est plus facile à migrer, télécharger, sauvegarder restaurer.. la virtualisation par conteneur permet aussi au serveur d'héberger bcp plus de conteur que si c'était des VM. 

![[Pasted image 20260731115142.png|431]]

https://www.youtube.com/watch?v=caXHwYC3tq8

## systeme d'exploitation

premier programme exécuté lors du démarrage de la machine le OS, ce q'il faut afficher et premiers opérations etc  . Ensuite, l'oS assure le lien entre ressource matériel et les aplications qui tourne dessus. l'os c un guichet il a des demandes des application spour les ressources de la machine, stockage, mémoire, calcul du processeur. Le OS accepte ou refuse ses demande il est responsable de la bonne exécution des applications. Les users intéresse avec des applications qui interagit avec OS; 

![[Pasted image 20260731134827.png|217]]

néanmoins c aussi possible aux utilisateurs de demander directment des services à l'OS via ligne de commande ou interface de prog 

![[Pasted image 20260731134901.png|258]]

composantse les plus importants de l'OS:
* kernel ou noyau espace mémoire isolé regroiupant les fcts clefs de l'os come gestion mémoire, processus ou I/O principale
* shell (inteprréteur de commande) permet de communiquer avec l'OS
* le file system (système de fichier): permet d'erengistre les fichiers dans une arborescene pour les gérer et écriture et lecture des fichiers
* drivesr (pilots): permettent la gestion des périphériques. chaque périphérique a ses propres instructions avec lesquel il peut etre manipuler et le OS en prend comtpe


![[Pasted image 20260731142449.png]]