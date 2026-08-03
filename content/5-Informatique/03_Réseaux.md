## A. Vue d'ensemble

**Pourquoi des couches ?**

Deux machines qui veulent communiquer doivent parler le même "langage" — sinon elles n'échangent que des 0 et des 1 incompréhensibles. Plutôt que de tout gérer d'un bloc, le réseau découpe le problème en **couches empilées**, chacune avec un rôle précis, qui ne parle qu'à son équivalent sur la machine distante (la couche 4 de chez toi "discute" avec la couche 4 du serveur, sans se soucier de comment la couche 1 a physiquement transporté les bits).

> [!tip] 💡 Le lien avec ce qu'on a déjà vu
> Ce n'est pas nouveau : on avait déjà rencontré cette logique de couches empilées avec Application → OS → Hardware dans `02_Systèmes d'exploitation`. Le réseau applique le même principe, juste appliqué à la communication *entre* machines plutôt qu'*à l'intérieur* d'une seule.

**Modèle OSI vs modèle TCP/IP**

Le modèle **OSI** (théorique, 7 couches) est le plus rigoureux et sert de référence académique. Le modèle **TCP/IP** (5 couches, parfois 4) est plus proche de ce qui tourne réellement sur Internet — c'est celui qu'on va suivre dans ce fichier.

| # | Couche (TCP/IP) | Équivaut à (OSI) | Rôle | Exemple |
|---|---|---|---|---|
| 5 | Application | Application, Présentation, Session | Ce que voit l'utilisateur/le développeur | HTTP, DNS |
| 4 | Transport | Transport | Achemine les données au bon programme | TCP, UDP |
| 3 | Réseau | Réseau | Achemine les données à la bonne machine | IP |
| 2 | Liaison | Liaison | Achemine les données sur un même segment local | Ethernet, MAC |
| 1 | Physique | Physique | Transport brut des bits (câble, ondes) | Câbles, Wi-Fi |

comparaison des deux modes OSI et TCP/IP

![[Pasted image 20260731161347.png|474]]



**Encapsulation** -> en réalité j'aurais bien mis un encadré pour parler de définition d'encapsulation

Chaque couche ajoute son propre en-tête (adressage, contrôle) autour de ce que la couche du dessus lui a donné — comme une lettre glissée dans une enveloppe, elle-même mise dans un colis. Le nom de "l'objet" transporté change à chaque couche :

| Couche | Nom des données |
|---|---|
| Application | Message |
| Transport | Segment |
| Réseau | Paquet / Datagramme |
| Liaison | Trame |
| Physique | Bits |

À l'émission, on descend les couches (encapsulation) ; à la réception, on les remonte en retirant chaque en-tête un par un (désencapsulation).

![[Pasted image 20260731161202.png|608]]

---

## B. Couche physique

Le rôle de cette couche : transporter des bits bruts (0/1) via un câble ou une onde, en modulant une tension électrique ou un signal lumineux/radio.

![[cable-2.png|414]]

Trois appareils, de plus en plus "intelligents", connectent les machines entre elles — chacun opérant à une couche plus haute que le précédent :

> [!info] Hub → Switch → Routeur
> - **Hub** (couche 1, physique) : renvoie tout ce qu'il reçoit à tous les appareils connectés, sans distinction. Crée un **domaine de collision** unique (un seul appareil peut parler à la fois) — devenu rare aujourd'hui.
> - **Switch** (couche 2, liaison) : inspecte l'adresse MAC de destination et n'envoie la donnée qu'au bon appareil — plus efficace, élimine les collisions.
> - **Routeur** (couche 3, réseau) : relie des réseaux **différents** entre eux (ton réseau local ↔ Internet), en inspectant l'adresse IP plutôt que MAC.

Le **modem** (modulateur/démodulateur) fait le pont entre le signal analogique de la ligne internet (câble coaxial ou téléphonique) et le signal numérique de tes appareils.

---

## C. Couche liaison de données

### C.1 L'adresse MAC

Chaque interface réseau (carte réseau, Wi-Fi) a une adresse **MAC** : un identifiant physique unique de 48 bits, gravé par le fabricant, qui ne change jamais (contrairement à l'adresse IP, vue en D, qui dépend du réseau où tu te connectes).

### C.2 Unicast, Multicast, Broadcast

- **Unicast** : message pour un seul destinataire.
- **Multicast** : message pour un groupe de destinataires abonnés.
- **Broadcast** : message pour tous les appareils du réseau local (adresse spéciale, tous les bits à 1).

### C.3 ARP — le pont entre MAC et IP

Le problème : les applications raisonnent en adresses IP (couche réseau), mais l'envoi effectif sur le câble a besoin d'une adresse MAC (couche liaison). **ARP** (*Address Resolution Protocol*) fait la traduction : la machine envoie un broadcast "qui a l'IP X ?", le propriétaire répond avec sa MAC, et l'association est mise en cache (table ARP) pour ne pas avoir à redemander à chaque fois.

---

## D. Couche réseau

### D.1 L'adresse IP

Une adresse IP (v4) est un nombre de 32 bits, écrit en 4 blocs de 8 bits (0-255) séparés par des points — la **notation décimale pointée**. Exemple valide : `142.250.184.14`. Une IP n'appartient pas à un appareil mais à un **réseau** : ton ordinateur a une IP différente à la maison et au bureau, alors que son adresse MAC ne change jamais.

![[IP-1.png|324]]



> [!warning] IP publique vs IP privée
> **IP publique** : unique au monde entier, visible sur Internet. C'est ce que voit un site web quand tu t'y connectes.
> **IP privée** : valable seulement à l'intérieur d'un réseau local (ex. `192.168.1.10`), non joignable depuis Internet — évite de gaspiller des IP publiques et expose moins chaque appareil.
>
> Ta box internet a les deux : une IP privée (pour parler à tes appareils) et une IP publique (fournie par ton FAI, pour parler à Internet). Elle joue le rôle de **routeur**, faisant le pont entre les deux mondes — depuis Internet, on ne voit que l'IP publique de la box, jamais celles de tes appareils individuels.

**Le datagramme IP** encapsule les données de la couche transport, avec un en-tête contenant notamment :

| Champ | Rôle |
|---|---|
| Version | IPv4 ou IPv6 |
| TTL (*Time To Live*) | Nombre de routeurs que le paquet peut encore traverser avant d'être jeté (évite les boucles infinies de routage) |
| Protocole | Indique si le contenu est du TCP ou de l'UDP (couche transport) |
| Checksum | Vérifie l'intégrité de l'en-tête |
| IP source / IP destination | Les deux adresses, chacune sur 32 bits |

### D.2 DHCP

Configurer une IP à la main sur chaque appareil (IP **statique**) était la méthode d'origine. Aujourd'hui, le **DHCP** (*Dynamic Host Configuration Protocol*) attribue automatiquement une IP **dynamique** à tout appareil qui rejoint le réseau — c'est ce qui se passe silencieusement quand tu te connectes à un nouveau Wi-Fi.

### D.3 Routage

Un routeur choisit le meilleur chemin pour faire voyager un paquet entre réseaux, en consultant une **table de routage**. À l'échelle d'Internet, les routeurs (notamment ceux des FAI) échangent leurs tables via un protocole appelé **BGP** (*Border Gateway Protocol*) pour connaître les chemins optimaux — une page web traverse en général des dizaines de routeurs différents avant d'arriver.

---

## E. Couche transport

### E.1 TCP vs UDP

| | TCP | UDP |
|---|---|---|
| Fiabilité | Garantit l'ordre et l'absence de perte (accusés de réception) | Aucune garantie |
| Vitesse | Plus lent (overhead de contrôle) | Rapide |
| Connexion | Établit une connexion avant d'échanger (voir E.3) | Envoie directement, sans connexion préalable |
| Usage typique | Web (HTTP), email, transfert de fichiers | Streaming vidéo/audio, jeux en ligne, DNS |

### E.2 Ports et sockets

Une IP identifie une **machine**, pas une application précise — or plusieurs programmes tournent en même temps sur un même ordinateur (navigateur, mail...). Le **port** (nombre 16 bits) identifie l'application visée. La combinaison IP + port s'appelle un **socket** (ex. `10.1.1.100:80`).

| Port | Protocole |
|---|---|
| 80 / 443 | HTTP / HTTPS |
| 21 | FTP |
| 53 | DNS |
| 22 | SSH |
| 25 | SMTP (email) |

### E.3 Le three-way handshake

TCP établit une connexion avant d'échanger des données, via 3 messages :

1. **SYN** — "je veux établir une connexion" (émetteur → récepteur)
2. **SYN-ACK** — "d'accord, moi aussi" (récepteur → émetteur)
3. **ACK** — "confirmé, on peut commencer" (émetteur → récepteur)

C'est ce mécanisme (déclenché à chaque nouvelle connexion TCP, par exemple à chaque fois que tu charges une page web) qui garantit que les deux machines sont bien prêtes à communiquer avant d'échanger la moindre vraie donnée.

### E.4 Firewalls

Un **firewall** bloque ou autorise du trafic selon des règles (souvent par port et par IP). Exemple typique : autoriser tout le monde sur le port 80 (site web public), bloquer l'accès au port du serveur de fichiers interne depuis l'extérieur.

---

## F. Couche application

### F.1 DNS — traduire un nom en adresse IP

Ton navigateur ne comprend que des adresses IP, pas des noms comme `fr.wikipedia.org`. Le **DNS** (*Domain Name System*) fait la traduction, en suivant une hiérarchie :

1. Le navigateur interroge un **résolveur DNS récursif** (ton FAI, ou un service comme `8.8.8.8`).
2. Ce résolveur demande aux **serveurs racine** : "qui gère `.org` ?"
3. Puis aux serveurs de `.org` : "qui gère `wikipedia.org` ?"
4. Puis aux serveurs de `wikipedia.org` : "quelle est l'IP de `fr.wikipedia.org` ?"
5. L'IP obtenue est renvoyée au navigateur, qui peut enfin se connecter au bon serveur.

![[Pasted image 20260731160825.png|506]]


### F.2 HTTP et HTTPS

**HTTP** est le protocole de communication entre navigateur (client) et serveur web.

> [!warning] HTTP est stateless
> HTTP ne garde aucun historique entre deux requêtes — chacune est traitée comme la première fois. C'est pour ça qu'on a besoin de **cookies** ou de **tokens** (JWT) pour qu'un site "se souvienne" de toi (panier d'achat, connexion...).

**Méthodes courantes** : `GET` (récupérer), `POST` (envoyer/créer), `PUT` (mettre à jour), `DELETE` (supprimer).

**Codes de statut** : `200 OK`, `301 Moved Permanently`, `404 Not Found`, `500 Internal Server Error`.

**HTTPS** = HTTP + chiffrement (SSL/TLS, détaillé en G.2) — confidentialité, authentification du serveur, intégrité des données.

### F.3 Serveurs web : statique vs dynamique

- **Statique** : envoie des fichiers tout prêts (HTML/CSS), tels quels.
- **Dynamique** : génère la page à la volée, en interrogeant une base de données et un langage serveur (Python/Flask, PHP...) — ex. une page "mon compte" personnalisée.

### F.4 API et REST

Une **API** (*Application Programming Interface*) est un intermédiaire qui permet à deux applications d'échanger des données ou des fonctionnalités, sans que l'une ait besoin de connaître les détails internes de l'autre.

> [!info] 💡 Exemple : Uber
> Quand tu commandes un trajet, Uber appelle l'**API de Google Maps** pour calculer l'itinéraire, puis l'**API Apple Pay** pour le paiement — sans avoir eu à recréer ces technologies elle-même. Certaines entreprises (Google, Stripe) vendent même l'accès à leur API comme un vrai produit commercial.

**REST** est une façon standardisée de concevoir une API : basée sur HTTP (mêmes méthodes GET/POST/PUT/DELETE), stateless (comme HTTP), données échangées en **JSON**. L'API, c'est *ce qu'on peut faire* ; REST, c'est *la manière* de le faire.

---

## G. Sécurité transverse

### G.1 Chiffrement symétrique vs asymétrique

**Chiffrement symétrique** : une seule clé, partagée, sert à chiffrer et déchiffrer. Problème : si un attaquant intercepte la clé pendant son envoi, il peut tout lire (*man in the middle*).

**Chiffrement asymétrique** : deux clés — une **publique** (transmise à tout le monde) et une **privée** (jamais transmise). On chiffre avec la clé publique du destinataire, seul son détenteur peut déchiffrer avec sa clé privée.

> [!warning] La limite : toujours le man in the middle
> Un attaquant M peut se faire passer pour le destinataire légitime, en donnant sa **propre** clé publique à l'émetteur. Ce dernier chiffre alors pour M sans le savoir — M déchiffre, lit, re-chiffre avec la vraie clé publique du destinataire, et personne ne voit la différence.

### G.2 Certificats et HTTPS

Le **certificat** résout ce problème : il associe une clé publique à une identité vérifiée (nom, domaine...), signée par une **autorité de certification** de confiance. Quand tu te connectes en HTTPS à ta banque :

1. Le serveur envoie son certificat (avec sa clé publique).
2. Ton navigateur vérifie la signature — il possède déjà les clés publiques des autorités de certification reconnues.
3. Si la signature est valide, la connexion est de confiance ; les deux parties définissent alors une clé secrète commune (chiffrement symétrique, plus rapide) pour le reste de l'échange.

### G.3 Proxy, reverse proxy, load balancing

**Proxy** : intermédiaire entre un utilisateur et Internet — filtre l'accès (whitelist/blacklist), masque l'IP de l'utilisateur (anonymisation), met en cache les pages fréquentes.

**Reverse proxy** : intermédiaire dans l'autre sens — protège des serveurs internes des utilisateurs externes. Gère souvent le chiffrement HTTPS à leur place, et répartit les requêtes entre plusieurs serveurs (**load balancing**) pour éviter d'en surcharger un seul.

### G.4 VPN

Un **VPN** (*Virtual Private Network*) crée un tunnel chiffré entre ton appareil et un serveur distant — comme si tu rejoignais un réseau privé (D.1) à distance, par-dessus le réseau public. Ton trafic sort avec l'IP du serveur VPN, pas la tienne — même principe d'anonymisation qu'un proxy, mais chiffré de bout en bout.

---

## H. Cas pratique — relier la théorie à un vrai exemple de travail

Un scénario concret pour ancrer tout ce qui précède : un service Python (FastAPI) qui envoie des données de risque vers un système de reporting interne (GRS — Global Risk Store), pour alimenter des dashboards.

> [!info] Le rôle de chaque couche dans ce scénario
> - **Application (F)** : le service FastAPI joue le rôle de **client**, GRS le rôle de **serveur**. La requête est une **HTTP POST** (on envoie/crée de la donnée, pas juste GET). Le corps de la requête — le **payload** — contient les données au format **JSON**. C'est le même mot "payload" qu'en C (trame Ethernet) et D (datagramme IP) : à chaque couche, le payload est la vraie donnée utile, distincte des en-têtes de contrôle qui l'entourent.
> - **Transport (E)** : la requête part sur le port 443 (HTTPS) en TCP — logique, on veut que les données de risque arrivent complètes et dans l'ordre, pas de perte tolérée comme avec UDP.
> - **Sécurité (G)** : HTTPS chiffre l'échange — le certificat de GRS est vérifié avant l'envoi.
> - **Réseau (D)** : l'IP de GRS est résolue au préalable (DNS, F.1) si on l'appelle par nom de domaine plutôt que par IP directe.

**L'onglet "Network" des outils développeur du navigateur** (ou un outil comme Postman) rend tout ça visible concrètement : la méthode (POST), l'URL cible, le **code de statut** de la réponse (F.2 — `200` si GRS a bien reçu et traité, `4xx` si le problème vient de la requête envoyée, `5xx` si le problème vient de GRS), et l'onglet **Payload/Request** qui montre exactement le JSON transmis.

**Pour troubleshooter un problème réseau dans ce genre de contexte**, la théorie de ce fichier donne une grille de lecture directe :
- Rien ne part du tout → problème DNS (résolution du nom de GRS) ou de connectivité de base.
- Connexion refusée → mauvais port, ou firewall (E.4) qui bloque le trafic vers GRS.
- Erreur de certificat → problème côté G.2 (certificat expiré ou non reconnu).
- Timeout → la connexion TCP ne se termine jamais (E.3), souvent un problème réseau intermédiaire plutôt qu'applicatif.
- Réponse reçue mais code 4xx/5xx → le réseau a fonctionné, le problème est dans le contenu de la requête ou côté serveur GRS, pas dans la tuyauterie réseau elle-même.

Cette dernière distinction est la plus utile à retenir : **savoir si un bug vient du réseau (rien n'arrive) ou de l'application (ça arrive, mais mal traité)** — exactement la question à se poser, et à savoir expliquer, en entretien.
