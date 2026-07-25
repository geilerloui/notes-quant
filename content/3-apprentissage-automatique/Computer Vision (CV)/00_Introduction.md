---
title: Introduction
description: Pourquoi la computer vision est un problème dur, bref historique, et panorama des tâches couvertes dans ce dossier
weight: 0
---

# Introduction à la Computer Vision

## 1. Pourquoi c'est un problème dur

La donnée visuelle domine le trafic internet (environ 80% en excluant même la vidéo, selon une étude Cisco de 2015 ; 5h de vidéo uploadées sur YouTube chaque seconde). On l'appelle parfois le "dark matter" d'internet : elle représente la majorité des bits, mais reste difficile à interpréter algorithmiquement.

La CV est un domaine intrinsèquement multidisciplinaire : physique (comment se forme une image), biologie (comment le cerveau des animaux traite l'information visuelle), ingénierie (comment construire des systèmes qui voient).

## 2. Un pipeline concret : du filtre à l'application

Un **pipeline**, ici, c'est juste une chaîne d'étapes où la sortie de l'une sert d'entrée à la suivante, jusqu'à obtenir le résultat final utile. Avant le deep learning, ces étapes étaient chacune un filtre/algorithme conçu à la main (cf. [[01_Filtres classiques]] et [[02_Detection_algorithms]]) — aucune des briques ne "comprend" l'image individuellement, c'est l'enchaînement qui produit un résultat exploitable.

Exemple concret — **le déverrouillage de téléphone par reconnaissance faciale**, qui enchaîne plusieurs tâches *distinctes*, chacune correspondant à un chapitre différent de ce dossier :

```mermaid
flowchart LR
    A["Image caméra"] --> B["Tâche 1 : Détection de visage (où est le visage dans l'image ?)"]
    B --> C["Tâche 2 : Landmark detection (où sont les yeux, nez, bouche ?)"]
    C --> D["Tâche 3 : Face recognition (encoder le visage, comparer au propriétaire)"]
    D --> E["Déverrouillé / Refusé"]
```

Ici chaque boîte est une **tâche à part entière**, avec sa propre littérature et ses propres algorithmes (pas juste des sous-étapes d'une même technique comme dans mon premier exemple raté) : la tâche 1 localise le visage dans l'image (une région, un rectangle), la tâche 2 trouve des points précis sur ce visage pour le recadrer/aligner correctement, la tâche 3 encode le visage aligné en un vecteur et le compare à celui du propriétaire enregistré (Siamese network + triplet loss, cf. [[05_Face recognition]]). Aucune étape ne "sait" faire du Face ID toute seule — c'est l'enchaînement de tâches indépendantes qui produit le résultat final.

Même logique pour l'OCR (binarisation → segmentation en caractères → reconnaissance de forme) ou la voiture autonome (détection de voie → détection d'objets → tracking → décision) : une chaîne de tâches indépendantes, assemblées à la main par un ingénieur, pour arriver à une application haut niveau.

## 3. Bref historique

- **1959** — Hubel & Wiesel étudient le cortex visuel du chat (proche de celui de l'humain) : ils découvrent les "simple cells", qui répondent à des bords orientés en mouvement. La vision biologique commence par des features très simples, dont la complexité s'accumule ensuite le long du traitement.
- **1963** — "Block World" (Larry Roberts), première thèse de doctorat en CV : le monde visuel est simplifié en formes géométriques simples à reconstruire.
- **1966** — MIT Summer Vision Project : tentative (trop) ambitieuse de résoudre la vision par ordinateur en un été.
- **David Marr** (MIT) formalise une théorie du traitement visuel en étapes : primal sketch → 2.5D sketch → modèle 3D. Cette vision en pipeline a dominé la CV pendant longtemps.
- **Années 70** — Generalized cylinders et pictorial structures : réduire des objets complexes à des primitives géométriques simples, motivé par la faible puissance de calcul de l'époque.
- **Années 80** — David Lowe tente de reconstruire/reconnaître des objets (ex rasoirs) à partir de lignes et de bords droits. Peu de progrès concrets sur la reconnaissance d'objets complète.
- Faute de résoudre la reconnaissance, le champ se tourne vers la **segmentation** : regrouper les pixels en régions cohérentes sans forcément les identifier (ex algorithmes de graphes à Berkeley).
- **Fin 1990s** — SIFT : reconnaissance d'objets par features locales invariantes (échelle, point de vue), plutôt que matching de l'objet entier.
- **1999-2001** — Détection de visage : montée du ML (SVM, boosting...), AdaBoost permet la détection de visage en temps réel malgré des puces lentes. 2006 : premier appareil photo Fujifilm avec détection de visage intégrée.
- **~2005** — PASCAL VOC, premier dataset de référence pour la détection d'objets (20 classes, ~10 000 images) : permet enfin de mesurer et comparer les progrès du champ.
- **2006** — Features de type "scene descriptor" pour classifier des scènes entières (autoroute, montagne...).
- **2009** — Reconnaissance de personnes par assemblage de parties du corps.
- **Fin 2000s** — motivés à la fois par l'envie de couvrir tous les objets du monde et par la nécessité de combattre l'overfitting (données visuelles = très haute dimension, peu de données = overfitting rapide), un groupe de Stanford lance **ImageNet** : 3 ans de construction à partir de la hiérarchie WordNet, des millions d'images labellisées via crowdsourcing.
- **2010** — Lancement du challenge ImageNet (ILSVRC) : ~1.4M images, succès si le bon label est dans le top 5 des prédictions.
- **2012** — Rupture : un CNN (AlexNet, 7 couches) bat tous les autres algorithmes au challenge ImageNet. Puis 2014 : 19 couches (VGG), 2015 : ResNet à 152 couches (jusqu'à 200+, limité par la mémoire).

> [!note]- Pourquoi 2012 et pas avant ?
> Les CNN existent depuis 1998 (LeNet, Bell Labs, reconnaissance de chiffres manuscrits) — l'architecture d'AlexNet leur ressemble beaucoup. Le déclic en 2012 vient de deux facteurs : (1) la puissance de calcul (loi de Moore + GPU massivement parallélisables, parfaits pour les CNN), (2) la disponibilité de données massives (ImageNet), qui manquait cruellement dans les années 90.

## 4. Au-delà de la reconnaissance d'objets : problèmes ouverts

- **Segmentation sémantique / perceptual grouping** : classifier chaque pixel individuellement
- **Compréhension 3D** de la scène
- **Activity recognition** : reconnaître une activité à partir d'une vidéo
- **Réalité virtuelle/augmentée**
- **Visual genome** : décrire une image comme un graphe de concepts sémantiques (objets, attributs, relations) — approche non standard mais riche
- Le graal ultime : comprendre une image aussi vite et aussi profondément qu'un humain (Fei-Fei Li montrait des images 0.5s à des sujets, qui pouvaient ensuite en raconter toute une histoire) — et gérer les biais d'interprétation contextuels (ex : une image ambiguë peut raconter des histoires très différentes selon le contexte social qu'on y projette).

## 5. Panorama des tâches couvertes dans ce dossier

| Tâche | Fichier | Contenu |
|---|---|---|
| Filtrage classique | [[01_Filtres classiques]] | Box, gaussien, Sobel, Laplacien, médian, bilatéral, template matching |
| Detection algorithms (CV classique) | [[02_Detection_algorithms]] | Boundary fitting (lignes/courbes, snakes, Hough), SIFT (blobs, scale space, descripteur) |
| Convolution & CNN | [[03_CNN]] | Convolution 1D/2D, intuition des filtres appris, symétries (Mallat) |
| Object Detection (deep learning) | [[04_Object Detection]] | Localisation, sliding window, YOLO, anchor boxes, R-CNN family, landmarks |
| Face recognition | [[05_Face recognition]] | One-shot learning, Siamese network, triplet loss |
| Self-supervised learning | [[06_Self supervised learning]] | Pretext tasks, DeepCluster, contrastive learning (SimCLR, PIRL) |

**Tâches pas encore couvertes** (à ajouter au fur et à mesure) :
- Image classification (bases : AlexNet/VGG/ResNet en détail)
- Semantic / instance segmentation
- Image captioning
- Generative models pour la vision (GAN, VAE, diffusion)
- Pose estimation / activity recognition
- 3D vision (depth estimation, point clouds, NeRF...)
- Video understanding (au-delà des images fixes)

> [!warning] Brouillon
> Historique condensé depuis une transcription de cours (probablement CS231n intro, Fei-Fei Li). Le tableau des tâches est à mettre à jour à chaque nouveau fichier ajouté au dossier.
