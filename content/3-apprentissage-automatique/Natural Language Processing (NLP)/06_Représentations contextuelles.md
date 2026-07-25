---
title: "Représentations Contextuelles"
description: "Transformers et mécanismes d'attention"
weight: 5
---

# Représentations Contextuelles

![[timeline.png]]

# I - Prérequis : Les Composants Fondamentaux

## a - Positional Encoding

Le self-attention ne tient pas compte de l'ordre des mots. **"Chat mange souris"** et **"Souris mange chat"** auraient les mêmes représentations ! Pour résoudre ce problème, on ajoute des **encodages positionnels** aux embeddings.

### **Notre exemple : "Le chat mange la petite souris"**

Gardons les mots : **"chat", "mange", "petite", "souris"** (positions 0, 1, 2, 3)

**Matrice d'embeddings de départ** :
$$X = \begin{pmatrix}
0.2 & 0.8 & 0.1 & 0.9 \\
0.7 & 0.3 & 0.8 & 0.2 \\
0.1 & 0.6 & 0.4 & 0.7 \\
0.5 & 0.2 & 0.9 & 0.3
\end{pmatrix}
\begin{matrix}
\text{chat} \\
\text{mange} \\
\text{petite} \\
\text{souris}
\end{matrix}$$

### **Encodage Positionnel Absolu**

**Problème naïf** : Ajouter directement le numéro de position.

```
Position 0 → "chat"   + [0, 0, 0, 0]
Position 1 → "mange"  + [1, 1, 1, 1]  
Position 2 → "petite" + [2, 2, 2, 2]
Position 3 → "souris" + [3, 3, 3, 3]
```

**Résultat final** :
$$X + PE_{abs} = \begin{pmatrix}
0.2+0 & 0.8+0 & 0.1+0 & 0.9+0 \\
0.7+1 & 0.3+1 & 0.8+1 & 0.2+1 \\
0.1+2 & 0.6+2 & 0.4+2 & 0.7+2 \\
0.5+3 & 0.2+3 & 0.9+3 & 0.3+3
\end{pmatrix} = \begin{pmatrix}
0.2 & 0.8 & 0.1 & 0.9 \\
1.7 & 1.3 & 1.8 & 1.2 \\
2.1 & 2.6 & 2.4 & 2.7 \\
3.5 & 3.2 & 3.9 & 3.3
\end{pmatrix}$$

**Inconvénients** :
1. **Échelle explosive** : Avec 500 mots → [500, 500, 500, 500] !
2. **Pas normalisé** : Les réseaux préfèrent des valeurs autour de 0

### **Première Solution : Normalisation**

**Idée** : Diviser par la longueur de séquence.

```
Position 0 → "chat"   + [0/4, 0/4, 0/4, 0/4] = [0.0, 0.0, 0.0, 0.0]
Position 1 → "mange"  + [1/4, 1/4, 1/4, 1/4] = [0.25, 0.25, 0.25, 0.25]
Position 2 → "petite" + [2/4, 2/4, 2/4, 2/4] = [0.5, 0.5, 0.5, 0.5]
Position 3 → "souris" + [3/4, 3/4, 3/4, 3/4] = [0.75, 0.75, 0.75, 0.75]
```

**Résultat final** :
$$X + PE_{norm} = \begin{pmatrix}
0.2+0.0 & 0.8+0.0 & 0.1+0.0 & 0.9+0.0 \\
0.7+0.25 & 0.3+0.25 & 0.8+0.25 & 0.2+0.25 \\
0.1+0.5 & 0.6+0.5 & 0.4+0.5 & 0.7+0.5 \\
0.5+0.75 & 0.2+0.75 & 0.9+0.75 & 0.3+0.75
\end{pmatrix} = \begin{pmatrix}
0.2 & 0.8 & 0.1 & 0.9 \\
0.95 & 0.55 & 1.05 & 0.45 \\
0.6 & 1.1 & 0.9 & 1.2 \\
1.25 & 0.95 & 1.65 & 1.05
\end{pmatrix}$$

**Problème fatal** : **0.25 dans une phrase de 4 mots ≠ 0.25 dans une phrase de 8 mots !**
- Phrase courte : position 1/4 = "je suis proche du début"
- Phrase longue : position 1/8 = "je suis au tout début"
→ **Impossible de généraliser** entre longueurs différentes !

### **Deuxième Solution : Bits Rotatifs**

**Idée** : Représenter les positions en binaire, puis transformer avec f(x) = 2x-1.

```
Position 0 → binaire 00 → [0,0,0,0] → f(x)=2x-1 → [-1,-1,-1,-1]
Position 1 → binaire 01 → [0,0,1,1] → f(x)=2x-1 → [-1,-1,1,1]
Position 2 → binaire 10 → [0,1,0,1] → f(x)=2x-1 → [-1,1,-1,1]  
Position 3 → binaire 11 → [1,1,1,1] → f(x)=2x-1 → [1,1,1,1]
```

**Résultat final** :
$$X + PE_{bit} = \begin{pmatrix}
0.2-1 & 0.8-1 & 0.1-1 & 0.9-1 \\
0.7-1 & 0.3-1 & 0.8+1 & 0.2+1 \\
0.1-1 & 0.6+1 & 0.4-1 & 0.7+1 \\
0.5+1 & 0.2+1 & 0.9+1 & 0.3+1
\end{pmatrix} = \begin{pmatrix}
-0.8 & -0.2 & -0.9 & -0.1 \\
-0.3 & -0.7 & 1.8 & 1.2 \\
-0.9 & 1.6 & -0.6 & 1.7 \\
1.5 & 1.2 & 1.9 & 1.3
\end{pmatrix}$$

**Avantages** :
- ✅ Normalisé entre [-1, 1]
- ✅ Pas d'explosion d'échelle

**Inconvénient** : 
- ❌ **Discret** ! Pas d'interpolation lisse entre positions
- ❌ Difficile de capturer les relations de proximité

### **Troisième Solution (utilisée par les Transformers) : Sinus/Cosinus**

**Idée** : Fonctions trigonométriques = version **continue** des bits rotatifs.

**Formule** :
$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \quad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**Application sur notre exemple** (4 dimensions) :

```
Position 0 → "chat"   + [sin(0/1), cos(0/1), sin(0/100), cos(0/100)] = [0.0, 1.0, 0.0, 1.0]
Position 1 → "mange"  + [sin(1/1), cos(1/1), sin(1/100), cos(1/100)] = [0.84, 0.54, 0.01, 1.0]
Position 2 → "petite" + [sin(2/1), cos(2/1), sin(2/100), cos(2/100)] = [0.91, -0.42, 0.02, 1.0]
Position 3 → "souris" + [sin(3/1), cos(3/1), sin(3/100), cos(3/100)] = [0.14, -0.99, 0.03, 1.0]
```

**Résultat final** :
$$X + PE_{sin} = \begin{pmatrix}
0.2+0.0 & 0.8+1.0 & 0.1+0.0 & 0.9+1.0 \\
0.7+0.84 & 0.3+0.54 & 0.8+0.01 & 0.2+1.0 \\
0.1+0.91 & 0.6-0.42 & 0.4+0.02 & 0.7+1.0 \\
0.5+0.14 & 0.2-0.99 & 0.9+0.03 & 0.3+1.0
\end{pmatrix} = \begin{pmatrix}
0.2 & 1.8 & 0.1 & 1.9 \\
1.54 & 0.84 & 0.81 & 1.2 \\
1.01 & 0.18 & 0.42 & 1.7 \\
0.64 & -0.79 & 0.93 & 1.3
\end{pmatrix}$$

### **Pourquoi ça marche ?**

**Propriétés magiques des sinus/cosinus** :
1. **Valeurs bornées** : [-1, 1] → pas d'explosion d'échelle
2. **Périodiques** : Capturent des patterns de distance
3. **Continues** : Interpolation lisse entre positions
4. **Généralisables** : Fonctionnent pour toute longueur de séquence
5. **Relations linéaires** : PE(pos+k) peut s'exprimer en fonction de PE(pos)

## **🎯 L'insight clé**

Maintenant **"mange" en position 1 ≠ "mange" en position 3** !

Le self-attention peut distinguer :
- **"Chat mange souris"** (chat=pos0, mange=pos1, souris=pos2)  
- **"Souris mange chat"** (souris=pos0, mange=pos1, chat=pos2)

**Les positions sont encodées dans les vecteurs** → L'ordre des mots est préservé ! 🚀

## b - Self-Attention

Le **self-attention** permet à chaque mot d'une phrase de "faire attention" à tous les autres mots pour mieux se comprendre dans son contexte.

![[attention1.png]]

### **Formule complète (qu'on va décortiquer)**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Notre mission** : Comprendre chaque élément avec l'exemple **"Le chat mange la petite souris"** !

### **Roadmap : Les 5 étapes du Self-Attention**

1. **Query** : "Qu'est-ce que je cherche ?" (besoins du mot)
2. **Key** : "Qu'est-ce que je peux offrir ?" (rôle/information disponible)
3. **Attention** : Produit scalaire Query×Key → compatibilité → softmax
4. **Value** : "Quelle information utile je transmets ?" (contenu filtré)
5. **Résultat** : Combinaison pondérée des values → représentation enrichie

### **Point de départ : Matrice d'embeddings X + positional encoding**

Nous gardons uniquement les mots porteurs de sens : **"chat", "mange", "petite", "souris"**

$$X = \begin{pmatrix}
0.2 & 0.8 & 0.1 & 0.9 \\
0.7 & 0.3 & 0.8 & 0.2 \\
0.1 & 0.6 & 0.4 & 0.7 \\
0.5 & 0.2 & 0.9 & 0.3
\end{pmatrix}
\begin{matrix}
\text{chat} \\
\text{mange} \\
\text{petite} \\
\text{souris}
\end{matrix}$$

### **1. Query : "Que cherche le mot 'mange' ?"**

> 💡 **Pourquoi ?** Chaque mot a des "besoins" selon son rôle. Un verbe cherche un sujet/objet, un adjectif cherche un nom à qualifier.

Nous calculons le vecteur query pour **"mange"** :

**Matrice Query** $W^Q$ (apprise pendant l'entraînement) :
$$W^Q = \begin{pmatrix}
0.3 & 0.1 & 0.5 \\
0.8 & 0.2 & 0.7 \\
0.4 & 0.9 & 0.1 \\
0.6 & 0.3 & 0.8
\end{pmatrix}$$

**Calcul du query pour "mange"** :
$$Q_{\text{mange}} = X_{\text{mange}} \times W^Q = [0.7, 0.3, 0.8, 0.2] \times W^Q = [0.78, 0.61, 0.85]$$

Ce vecteur encode "ce que le verbe 'mange' cherche" : un sujet et un objet !

### **2. Keys : "Que peut offrir chaque mot ?"**

> 💡 **Pourquoi ?** Chaque mot a des "capacités" selon son type. Un nom peut être sujet/objet, un verbe peut être l'action principale.

Calculons les **keys** pour tous les mots avec la matrice $W^K$ :

**Matrice Key** $W^K$ :
$$W^K = \begin{pmatrix}
0.2 & 0.7 & 0.4 \\
0.5 & 0.1 & 0.8 \\
0.9 & 0.3 & 0.2 \\
0.1 & 0.6 & 0.7
\end{pmatrix}$$

**Calcul des keys pour tous les mots** :
$$K = X \times W^K = \begin{pmatrix}
0.73 & 0.71 & 0.91 \\
0.57 & 0.61 & 0.72 \\
0.41 & 0.49 & 0.63 \\
0.42 & 0.53 & 0.81
\end{pmatrix}
\begin{matrix}
\text{chat} \\
\text{mange} \\
\text{petite} \\
\text{souris}
\end{matrix}$$

Chaque ligne représente "ce que ce mot peut offrir comme information".

### **3. Scores d'attention : "Compatibilité Query-Key"**

> 💡 **Pourquoi ?** Le produit scalaire mesure la "compatibilité" entre ce qu'un mot cherche et ce qu'un autre peut offrir.

Calculons les scores d'attention de "mange" vers tous les mots :

$$\text{Scores} = Q_{\text{mange}} \times K^T = [0.78, 0.61, 0.85] \times \begin{pmatrix}
0.73 & 0.57 & 0.41 & 0.42 \\
0.71 & 0.61 & 0.49 & 0.53 \\
0.91 & 0.72 & 0.63 & 0.81
\end{pmatrix} = [1.95, 1.55, 1.25, 1.64]$$

**Après softmax** :
$$\text{Attention}_{\text{mange}} = \text{softmax}([1.95, 1.55, 1.25, 1.64]) = [0.41, 0.28, 0.20, 0.31]$$

**Interprétation** :
- "mange" fait **41% d'attention à "chat"** (sujet principal !)
- "mange" fait **31% d'attention à "souris"** (objet principal !)  
- "mange" fait **28% d'attention à "mange"** (lui-même)
- "mange" fait **20% d'attention à "petite"** (moins important)

### **4. Values : "Quelle information transmettre ?"**

> 💡 **Pourquoi ?** Les poids d'attention disent "à qui faire attention", mais pas "quoi récupérer". Les values filtrent l'information utile.

**Matrice Value** $W^V$ :
$$W^V = \begin{pmatrix}
0.4 & 0.2 & 0.9 \\
0.7 & 0.5 & 0.1 \\
0.3 & 0.8 & 0.6 \\
0.6 & 0.1 & 0.4
\end{pmatrix}$$

**Calcul des values pour tous les mots** :
$$V = X \times W^V = \begin{pmatrix}
0.89 & 0.61 & 0.58 \\
0.67 & 0.53 & 0.71 \\
0.88 & 0.59 & 0.67 \\
0.64 & 0.39 & 0.69
\end{pmatrix}
\begin{matrix}
\text{chat} \\
\text{mange} \\
\text{petite} \\
\text{souris}
\end{matrix}$$

### **5. Nouvelle représentation de "mange"**

> 💡 **Pourquoi ?** On combine les values avec les poids d'attention pour créer une représentation enrichie par le contexte.

$$\text{mange}_{\text{nouveau}} = 0.41 \times V_{\text{chat}} + 0.28 \times V_{\text{mange}} + 0.20 \times V_{\text{petite}} + 0.31 \times V_{\text{souris}}$$

$$= 0.41 \times [0.89, 0.61, 0.58] + 0.28 \times [0.67, 0.53, 0.71] + 0.20 \times [0.88, 0.59, 0.67] + 0.31 \times [0.64, 0.39, 0.69]$$

$$= [0.78, 0.54, 0.64]$$

### **6. Matrice finale : Représentations enrichies**

Après avoir appliqué le self-attention à tous les mots, nous obtenons une nouvelle matrice où chaque mot a été enrichi par le contexte :

![[1776674920221_image.png]]

## **🧠 L'insight clé**

**"mange" n'est plus juste "mange"** ! Sa nouvelle représentation `[0.78, 0.54, 0.64]` contient maintenant :
- L'information de **"chat"** (son sujet)
- L'information de **"souris"** (son objet)  
- Un peu d'information de **"petite"** (contexte)

Le vecteur final de "mange" encode désormais **"mange-dans-ce-contexte-avec-chat-et-souris"** !

![[attention2.png]]

**La magie** : Les matrices $W^Q$, $W^K$, $W^V$ s'entraînent automatiquement pour que :
- Les verbes apprennent à "chercher" des sujets/objets  
- Les noms apprennent à "offrir" des rôles syntaxiques
- Les values transmettent l'information sémantique pertinente

C'est ainsi que le self-attention capture les **relations contextuelles** entre tous les mots d'une phrase ! 🚀

## c - Multi-Head Attention

L'attention multi-têtes permet au modèle de porter attention conjointement à l'information provenant de différents sous-espaces de représentation à différentes positions.

$$\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}\left(\text{head}_{1}, \ldots, \text{head}_{\mathrm{h}}\right) W^{O} \\
\text{où } \text{head}_{\mathrm{i}} &= \text{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}$$

Où les projections sont des matrices de paramètres $W_{i}^{Q} \in \mathbb{R}^{d_{\text{model}} \times d_{k}}, W_{i}^{K} \in \mathbb{R}^{d_{\text{model}} \times d_{k}}, W_{i}^{V} \in \mathbb{R}^{d_{\text{model}} \times d_{v}}$ et $W^{O} \in \mathbb{R}^{h d_{v} \times d_{\text{model}}}$

![[attention3.png]]

### **Exemple concret : 8 têtes d'attention**

Reprenons notre exemple **"chat mange petite souris"** avec 8 têtes d'attention et embeddings de dimension 4.

**Input** : Matrice `[4, 4]` (4 mots × 4 dimensions)

**Multi-Head Output** : Tenseur 3D `[4, 8, head_dim]` où :
- **4** = nombre de mots ("chat", "mange", "petite", "souris")
- **8** = nombre de têtes d'attention
- **head_dim** = 4 ÷ 8 = 0.5... → Dans la pratique, on utilisera head_dim = 64 avec embed_dim = 512

**Avec des dimensions réalistes** (embed_dim = 64, 8 têtes) :
```
Input:  [4, 64]     ← 4 mots × 64 dimensions
Output: [4, 8, 8]   ← 4 mots × 8 têtes × 8 dimensions par tête
Concat: [4, 64]     ← 4 mots × (8×8) = 64 dimensions
```

### **Spécialisation des têtes : Preuves empiriques**

**Contrairement au papier original "Attention is All You Need"**, les recherches ultérieures ont prouvé que chaque tête se spécialise spontanément :

#### **Voita et al. (2019)** - "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting"
Découverte que certaines têtes se spécialisent dans :
- **Relations syntaxiques** : détection sujet-verbe, verbe-objet
- **Dépendances à long terme** : résolution d'anaphores
- **Relations positionnelles** : mots adjacents vs distants

#### **Vig & Belinkov (2019)** - "Analyzing the Structure of Attention in a Transformer Language Model"  
Analyse de GPT-2 montrant que les têtes capturent :
- **Différentes parties du discours** selon la profondeur de la couche
- **Relations de dépendance syntaxique** principalement dans les couches moyennes
- **Relations à long terme** dans les couches profondes

### **Exemple de spécialisation sur "chat mange petite souris"**

**Tête 1 (syntaxe)** : Attention de "mange" vers :
- **"chat" (85%)** ← sujet du verbe
- **"souris" (10%)** ← objet du verbe  
- **"petite" (3%)** ← modificateur
- **"mange" (2%)** ← auto-attention

**Tête 2 (sémantique)** : Attention de "petite" vers :
- **"souris" (90%)** ← nom qualifié
- **"petite" (5%)** ← auto-attention
- **"mange" (3%)** ← contexte
- **"chat" (2%)** ← contexte distant

**Tête 3 (coréférence)** : Dans une phrase plus longue, détecterait les pronoms et leurs antécédents.

> [!info]- 🎯 Intuition Multi-Head Attention
> **Pourquoi ça marche ?** L'entraînement force naturellement la spécialisation :
> - Si toutes les têtes faisaient la même chose → redondance  
> - L'optimisation pousse chaque tête vers sa "niche" pour minimiser la loss
> - **Résultat** : Division automatique du travail linguistique !
> 
> **Parallélisme** : Toutes les têtes calculent en parallèle, puis leurs sorties sont concaténées et projetées.

## d - Layer Normalization

Layer Normalization est une technique de normalisation qui stabilise l'entraînement des réseaux profonds en normalisant les activations au sein de chaque exemple.

**Formule** :

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

où :
- $\mu$ : moyenne des activations pour cet exemple
- $\sigma$ : écart-type des activations pour cet exemple  
- $\gamma, \beta$ : paramètres apprenables (scale et shift)

### **Exemple concret avec un batch**

Imaginons qu'on ait un **batch de 2 phrases** après self-attention :

$$\text{Batch} = \begin{pmatrix}
\begin{bmatrix}
0.75 & 0.58 & 0.69 & 0.82 \\
0.78 & 0.54 & 0.64 & 0.71 \\
0.61 & 0.73 & 0.55 & 0.68 \\
0.69 & 0.48 & 0.77 & 0.59
\end{bmatrix} \text{ (phrase 1)} \\
\\
\begin{bmatrix}
1.2 & 0.3 & 0.8 & 0.5 \\
0.9 & 1.1 & 0.4 & 0.7 \\
0.6 & 0.8 & 1.3 & 0.2 \\
1.0 & 0.2 & 0.9 & 0.6
\end{bmatrix} \text{ (phrase 2)}
\end{pmatrix}$$

**Layer Norm normalise chaque token indépendamment** :

- **Token "chat"** `[0.75, 0.58, 0.69, 0.82]` → μ = 0.71, σ = 0.10 → normalisé
- **Token "mange"** `[0.78, 0.54, 0.64, 0.71]` → μ = 0.67, σ = 0.09 → normalisé  
- **Chaque phrase** traitée **séparément** et **indépendamment**
- **Chaque token** normalisé sur ses propres dimensions uniquement

![[Pasted image 20260420104812.png|618]]

**Différence avec Batch Normalization** :
- **Batch Norm** : Normalise à travers les exemples du batch (colonnes)
- **Layer Norm** : Normalise à travers les features d'un token (lignes)

**Pourquoi crucial dans les Transformers** :
- Stabilise l'entraînement avec residual connections
- Permet des gradients plus stables  
- **Indépendant de la taille du batch** ← crucial pour l'inférence !

# II - Architecture du Transformer (2017)
### Language model

LM probability distribution over sequence of tokens/words $p(x_1,..,x_L)$ 

$$
p(the, mouse, ate, the, cheese) = 0.02 
$$

LM are generative models: $x_{1:L} \sim p(x_1, .., x_L)$

Autoregressive (AR) language models : 

$$
p\left(x_1, \ldots, x_L\right)=p\left(x_1\right) p\left(x_2 \mid x_1\right) p\left(x_3 \mid x_2, x_1\right) \ldots =\prod_i p\left(x_i \mid x_{1: i-1}\right)
$$
### Algorithme

```
Task: Traduction (séquence → séquence)

TRAINING
├── Pretraining (= foundational model)
│     forward(x, y) → ŷ
│     x = phrase source                     (entrée encoder)
│     y = phrase cible, décalée à droite     (entrée decoder, teacher forcing)
│     ŷ = distribution sur le vocab, position par position
│     loss = cross-entropy(ŷ, y_cible)
│
└── Fine-tuning
      aucun — pretraining = entraînement complet sur la tâche cible

INFERENCE (autoregressif, mot par mot)
  y_0 = <BOS>
  pour t = 1, 2, ... jusqu'à <EOS> ou max_len :
      y_t = argmax( decoder( encoder(x), y_{0:t-1} ) )
      append y_t
```

## a - Bloc Encoder

Le bloc encoder traite toute la séquence source d'un coup et produit une représentation enrichie de chaque position. Contrairement au decoder, il voit **toute la phrase en même temps** (pas de masque causal) : chaque mot peut regarder tous les autres, y compris ceux qui le suivent.

### 1. Tokenization et Embedding Lookup

Prenons un exemple filé sur tout le bloc : on veut traduire **"Le chat mange"** (source) vers **"The cat eats"** (cible).

**Étape 1 — Tokenization** : chaque mot de la phrase source est mappé vers un identifiant du vocabulaire.

```
Vocabulaire (extrait) : {..., "le": 42, "chat": 17, "mange": 89, ...}

"Le chat mange" → [42, 17, 89]
```

**Étape 2 — Embedding lookup** : le modèle possède une matrice d'embeddings $X_{emb} \in \mathbb{R}^{V \times d}$, apprise pendant l'entraînement, où $V$ = taille du vocabulaire entier et $d$ = dimension des embeddings. Cette matrice contient **une ligne par mot du vocabulaire** — pas seulement les mots de la phrase traitée.

Pour cette phrase, on ne calcule rien : on **cherche (lookup)** les 3 lignes correspondant aux ids [42, 17, 89] dans $X_{emb}$ :

$$X = \begin{pmatrix}
0.1 & 0.4 & 0.2 & 0.7 \\
0.3 & 0.8 & 0.1 & 0.5 \\
0.6 & 0.2 & 0.9 & 0.3
\end{pmatrix}
\begin{matrix}
\text{le} \\
\text{chat} \\
\text{mange}
\end{matrix}$$

Une ligne = un mot, une colonne = une dimension de l'embedding — exactement comme un DataFrame à 3 lignes (les mots) et $d=4$ colonnes (les features apprises).

### 2. Positional Encoding

On additionne à $X$ une matrice $PE$ de **même dimension** $(3, 4)$, calculée avec les formules sinus/cosinus vues en I.a — une ligne par position (0, 1, 2) :

$$PE = \begin{pmatrix}
0.0 & 1.0 & 0.0 & 1.0 \\
0.84 & 0.54 & 0.01 & 1.0 \\
0.91 & -0.42 & 0.02 & 1.0
\end{pmatrix}
\begin{matrix}
\text{pos 0 (le)} \\
\text{pos 1 (chat)} \\
\text{pos 2 (mange)}
\end{matrix}$$

$$X_0 = X + PE = \begin{pmatrix}
0.1 & 1.4 & 0.2 & 1.7 \\
1.14 & 1.34 & 0.11 & 1.5 \\
1.51 & -0.22 & 0.92 & 1.3
\end{pmatrix}$$

$X_0$ est l'entrée du premier bloc encoder.

### 3. Multi-Head Self-Attention (bidirectionnelle)

Comme en I.b, on calcule $Q$, $K$, $V$ à partir de $X_0$, puis les scores $\frac{QK^T}{\sqrt{d_k}}$ passés au softmax. La différence avec le decoder : **aucun masque**, donc "le" peut regarder "chat" et "mange", "chat" peut regarder "le" et "mange", etc. — la matrice d'attention est pleine, pas triangulaire.

Pour 3 mots, la matrice d'attention est $3\times3$ : une ligne par mot-qui-regarde, une colonne par mot-regardé, chaque ligne somme à 1 (softmax) :

$$ Attention= \begin{array}{cc} & \begin{matrix} \text{le} & \text{chat} & \text{mange} \end{matrix} \\ \begin{matrix} \text{le} \\ \text{chat} \\ \text{mange} \end{matrix} & \begin{pmatrix} 0.6 & 0.3 & 0.1 \\ 0.1 & 0.7 & 0.2 \\ 0.1 & 0.2 & 0.7 \end{pmatrix} \end{array} $$

**Lecture ligne par ligne** : la ligne "chat" dit que "chat" porte 10% d'attention sur "le", 70% sur lui-même, 20% sur "mange".

**Concrètement** : la nouvelle représentation de "chat" est une **combinaison linéaire** des values des autres mots, pondérée par cette ligne :

$$\text{chat}_{\text{new}} = 0.1 \times \text{le}_{\text{old}} + 0.7 \times \text{chat}_{\text{old}} + 0.2 \times \text{mange}_{\text{old}}$$

(où "old" = value du mot avant attention, "new" = sortie enrichie par le contexte). On répète ce calcul pour **chaque ligne** de la matrice — donc pour chaque mot de la phrase — et on obtient une nouvelle matrice de même taille que $X_0$, où chaque mot a "absorbé" un peu des autres mots selon les poids d'attention.

Multi-head = on répète ce calcul $h$ fois en parallèle avec des matrices $W^Q, W^K, W^V$ différentes (cf I.c), puis on concatène les $h$ sorties.

### 4. Add & Norm, Feed-Forward, Add & Norm

**Architecture mathématique** :

$$\begin{aligned}
\text{Attention}_{out} &= \text{MultiHead}(X_0) \\
\text{Add\&Norm}_1 &= \text{LayerNorm}(X_0 + \text{Attention}_{out}) \\
\text{FFN}_{out} &= \text{FFN}(\text{Add\&Norm}_1) \\
\text{Add\&Norm}_2 &= \text{LayerNorm}(\text{Add\&Norm}_1 + \text{FFN}_{out})
\end{aligned}$$

$\text{Add\&Norm}_2$ est la sortie d'un bloc encoder : elle a exactement la même taille que $X_0$ ($3 \times 4$ ici) et sert d'entrée au bloc encoder suivant — on empile $N$ blocs identiques ($N=6$ dans l'article original).

## b - Bloc Decoder

Le bloc decoder génère la séquence de sortie de manière autorégressive. On continue l'exemple : on veut produire la traduction **"The cat eats"** de la phrase source "Le chat mange".

### 1. Tokenization (phrase cible, teacher forcing)

Point important : ce n'est **pas** "eats" qui est absent de l'entrée du decoder — c'est le token de fin **`<EOS>`** (End Of Sequence). Ce qui rentre dans le decoder pendant l'entraînement (teacher forcing), c'est la phrase cible **décalée à droite**, précédée d'un token spécial **`<SOS>`** (Start Of Sequence) :

```
Phrase cible complète : <SOS>  The  cat  eats  <EOS>

Decoder INPUT  (on retire le dernier token, <EOS>) : <SOS>, The, cat, eats
Decoder LABELS (on retire le premier token, <SOS>) : The, cat, eats, <EOS>
```

À chaque position, le decoder voit un mot et doit prédire le **suivant** :

| Position | Input (ce qu'il voit) | Label (ce qu'il doit prédire) |
|---|---|---|
| 0 | `<SOS>` | The |
| 1 | The | cat |
| 2 | cat | eats |
| 3 | eats | `<EOS>` |

"eats" est donc bien dans l'input, en position 3 — c'est ce mot qui sert à prédire `<EOS>`, la fin de la phrase.

Comme pour l'encoder, on fait l'**embedding lookup** des 4 tokens `[<SOS>, The, cat, eats]` dans la table $Y_{emb}$, puis on ajoute le positional encoding correspondant, pour obtenir $Y_0 \in \mathbb{R}^{4 \times d}$.

### 2. Masked Multi-Head Self-Attention

Même calcul que l'encoder ($Q$, $K$, $V$, puis $\frac{QK^T}{\sqrt{d_k}}$), **sauf qu'on ajoute un masque avant le softmax** : on met $-\infty$ sur toutes les cases "futures" — un mot ne doit jamais voir ce qui vient après lui.

**Masque** (0 = autorisé, $-\infty$ = interdit) pour les 4 tokens `<SOS>, The, cat, eats` :

$$\text{Masque} = \begin{array}{cc} & \begin{matrix} \text{<SOS>} & \text{The} & \text{cat} & \text{eats} \end{matrix} \\ \begin{matrix} \text{<SOS>} \\ \text{The} \\ \text{cat} \\ \text{eats} \end{matrix} & \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix} \end{array}$$

On calcule $\text{Scores}_{\text{masked}} = \frac{QK^T}{\sqrt{d_k}} + \text{Masque}$, puis softmax ligne par ligne. Comme $e^{-\infty} = 0$, les cases masquées deviennent des probabilités nulles :

$$\text{Attention}_{\text{masked}} = \begin{array}{cc} & \begin{matrix} \text{<SOS>} & \text{The} & \text{cat} & \text{eats} \end{matrix} \\ \begin{matrix} \text{<SOS>} \\ \text{The} \\ \text{cat} \\ \text{eats} \end{matrix} & \begin{pmatrix} 1.0 & 0 & 0 & 0 \\ 0.3 & 0.7 & 0 & 0 \\ 0.2 & 0.3 & 0.5 & 0 \\ 0.1 & 0.2 & 0.3 & 0.4 \end{pmatrix} \end{array}$$

**Lecture** : la ligne `<SOS>` ne peut regarder qu'elle-même (100%) — logique, c'est le tout premier token, rien avant lui. La ligne "eats" regarde les 4 tokens (elle est en dernière position visible) mais jamais `<EOS>` (qui n'est pas encore généré).

Exactement comme dans l'encoder, la nouvelle représentation de chaque mot est la combinaison linéaire des *values* pondérée par sa ligne — mais uniquement sur les mots **visibles** (non masqués) :

$$\text{cat}_{\text{new}} = 0.2 \times \text{<SOS>}_{\text{old}} + 0.3 \times \text{The}_{\text{old}} + 0.5 \times \text{cat}_{\text{old}}$$

(pas de terme pour "eats" — "cat" ne peut pas encore le voir, il vient après).

### 3. Multi-Head Cross-Attention

C'est ici que la phrase source et la phrase cible se rencontrent. Contrairement aux deux étapes précédentes, les $Q$ et les $(K, V)$ viennent de **deux endroits différents** :

- $Q$ vient de la sortie du decoder à l'étape 2 (masked self-attention + Add&Norm)
- $K$ et $V$ viennent de la sortie finale de l'**encoder** — la représentation de "le chat mange" obtenue en II.a ($\text{Add\&Norm}_2$)

La matrice d'attention n'est donc plus carrée : **une ligne par mot cible** (decoder), **une colonne par mot source** (encoder). Pas de masque ici : l'encoder a vu toute la phrase source, donc chaque mot cible peut regarder n'importe quel mot source.

$$\text{Cross-Attention} = \begin{array}{cc} & \begin{matrix} \text{le} & \text{chat} & \text{mange} \end{matrix} \\ \begin{matrix} \text{<SOS>} \\ \text{The} \\ \text{cat} \\ \text{eats} \end{matrix} & \begin{pmatrix} 0.34 & 0.33 & 0.33 \\ 0.70 & 0.20 & 0.10 \\ 0.10 & 0.80 & 0.10 \\ 0.05 & 0.15 & 0.80 \end{pmatrix} \end{array}$$

**Lecture** : la ligne "cat" porte 80% de son attention sur "chat" — logique, "cat" cherche sa contrepartie française pour décider quoi générer ensuite. La ligne "eats" porte 80% sur "mange", pour la même raison.

Comme toujours, la nouvelle représentation est une combinaison linéaire — mais cette fois des **values de l'encoder**, pas du decoder :

$$\text{cat}_{\text{new}} = 0.1 \times \text{le}_{\text{enc}} + 0.8 \times \text{chat}_{\text{enc}} + 0.1 \times \text{mange}_{\text{enc}}$$

(où $\text{le}_{\text{enc}}, \text{chat}_{\text{enc}}, \text{mange}_{\text{enc}}$ sont les sorties finales de l'encoder, pas les embeddings de départ). C'est ce mécanisme qui permet au decoder de "regarder" la phrase source à chaque étape de génération — le seul point de contact entre les deux stacks.

### 4. Add & Norm, Feed-Forward, Add & Norm

Rien de nouveau ici — exactement les mêmes formules que dans l'encoder (II.a, point 4), appliquées à la sortie de la cross-attention :

$$\begin{aligned}
\text{Add\&Norm}_2 &= \text{LayerNorm}(\text{Add\&Norm}_1 + \text{CrossAttention}_{out}) \\
\text{FFN}_{out} &= \text{FFN}(\text{Add\&Norm}_2) \\
\text{Add\&Norm}_3 &= \text{LayerNorm}(\text{Add\&Norm}_2 + \text{FFN}_{out})
\end{aligned}$$

$\text{Add\&Norm}_3$ est la sortie d'un bloc decoder complet — même taille que $Y_0$ ($4 \times d$) — et sert d'entrée au bloc decoder suivant. On empile $N$ blocs decoder identiques.

⚠️ **Remarque importante : la projection vers le vocabulaire, le softmax et la cross-entropy détaillés ci-dessous ne sont pas ce point 4 — c'est une étape complètement différente, qui n'arrive qu'une seule fois, après que les $N$ blocs decoder aient tous fini de tourner.**

### Après le N-ième bloc decoder : Projection finale, Softmax, Cross-Entropy

Après le dernier bloc decoder, on a une matrice $Z \in \mathbb{R}^{4 \times d}$ — toujours 4 lignes (les tokens `<SOS>, The, cat, eats`), $d$ colonnes (dimension du modèle).

**Projection linéaire** : on multiplie $Z$ par une matrice apprise $W_{out} \in \mathbb{R}^{d \times V}$ (+ biais), où $V$ = taille du **vocabulaire cible** (anglais ici) — pas juste les 4 mots, tout le vocabulaire :

$$\text{Logits} = Z W_{out} + b \in \mathbb{R}^{4 \times V}$$

On obtient une matrice avec toujours les 4 mots en ligne, mais en colonne **tout le vocabulaire** :

$$\text{Logits} = \begin{array}{cc} & \begin{matrix} \text{a} & \text{cat} & \text{dog} & \cdots & \text{zebra} \end{matrix} \\ \begin{matrix} \text{<SOS>} \\ \text{The} \\ \text{cat} \\ \text{eats} \end{matrix} & \begin{pmatrix} 1.2 & 0.3 & 0.1 & \cdots & -0.5 \\ 0.4 & 3.1 & 1.8 & \cdots & -1.2 \\ -0.3 & 0.5 & 2.9 & \cdots & 0.1 \\ 0.8 & 1.1 & 0.4 & \cdots & 3.5 \end{pmatrix} \end{array}$$

**Softmax ligne par ligne** : chaque ligne devient une distribution de probabilité sur tout le vocabulaire (somme = 1) :

$$\hat{y}_t = \text{softmax}(\text{Logits}_t)$$

Pour la ligne "The" (qui doit prédire "cat", cf. le tableau labels de la section 1) :

$$\hat{y}_{\text{The}} = \left(\; \text{a}: 0.02, \;\; \text{cat}: 0.62, \;\; \text{dog}: 0.19, \;\; \cdots, \;\; \text{zebra}: 0.001 \;\right)$$

**Cross-entropy loss** : pour chaque position, on ne regarde que la probabilité assignée au mot **correct** (le label), et on prend son log négatif :

$$\mathcal{L}_t = -\log\left(\hat{y}_t[\text{label}_t]\right)$$

Pour la position "The" → label "cat" avec proba 0.62 :

$$\mathcal{L}_{\text{The}} = -\log(0.62) \approx 0.478$$

On fait ça pour les 4 positions (`<SOS>`→The, The→cat, cat→eats, eats→`<EOS>`) et on moyenne :

$$\mathcal{L} = \frac{1}{4}\sum_{t=0}^{3} \mathcal{L}_t$$

**Lien avec la définition du language model** (tout en haut de la section II) : ce n'est pas juste une analogie, c'est la même identité mathématique (chain rule) appliquée dans l'autre sens. La **somme** (pas la moyenne) des cross-entropy par position est exactement égale à $-\log$ de la probabilité **jointe** de la séquence cible :

$$\sum_{t=0}^{3} \mathcal{L}_t = \sum_{t=0}^{3} -\log\big(p(x_t \mid x_{<t})\big) = -\log\left(\prod_{t=0}^{3} p(x_t \mid x_{<t})\right) = -\log\big(p(\text{The, cat, eats, }\langle\text{EOS}\rangle)\big)$$

Minimiser cette somme de cross-entropy revient donc littéralement à **maximiser** $p(\text{"The cat eats"} \mid \text{"le chat mange"})$ — la proba jointe visée depuis le début avec `forward(x, y) → ŷ` dans l'algo box. Le $\frac{1}{4}$ (moyenne plutôt que somme) est juste une normalisation pratique pour que la loss ne dépende pas de la longueur de la séquence — ça ne casse pas le lien, ça le rescale par $L$.

C'est **ce $\mathcal{L}$** qui sert de point de départ au backward : le gradient remonte à travers le softmax → la projection $W_{out}$ → tous les blocs decoder (cross-attention comprise, donc jusque dans l'encoder) → tous les blocs encoder → jusqu'aux embeddings. Tous les poids ($W^Q, W^K, W^V$ de chaque tête, chaque FFN, $W_{out}$, les tables d'embeddings...) sont mis à jour en une seule passe de backprop.

## c - Inférence

À l'inférence, contrairement à l'entraînement, on n'a **pas** la phrase cible sous la main — le teacher forcing est un luxe réservé à l'entraînement, où on connaît déjà la traduction. Il faut **générer** la sortie un token à la fois. Cinq étapes qui bouclent :

1. **Tokenize** (le prompt / ce qui a déjà été généré)
2. **Forward**
3. **Predict** la proba du prochain token
4. **Sample**
5. **Detokenize** (une fois la génération terminée)

**Sur le sampling** : $x_{1:L} \sim p(x_1, ..., x_L)$ (vu dans la définition du language model plus haut) est bien l'objectif — tirer toute la séquence selon sa distribution jointe. Mais le modèle ne donne accès qu'aux conditionnelles $p(x_i \mid x_{1:i-1})$ via la factorisation autorégressive. On ne peut donc pas tirer $x_{1:L}$ d'un coup : on tire token par token, en enchaînant les conditionnelles — à chaque étape $i$, on tire $x_i \sim p(\cdot \mid x_{1:i-1})$, on l'ajoute à la séquence, et on recommence pour $i+1$.

**Reprenons l'exemple "le chat mange" → "The cat eats"**, avec une précision importante sur ce qui tourne à chaque étape.

L'**encoder ne tourne qu'une seule fois** : on encode "le chat mange" une bonne fois pour toutes → $\text{Add\&Norm}_2$ figé, réutilisé à chaque étape de génération via cross-attention. Pas besoin de le refaire tourner.

Le **decoder tourne à chaque étape**, avec un input qui grandit :

| Étape | Decoder input | Nb lignes du softmax final | Ligne qui compte | Token tiré |
|---|---|---|---|---|
| 1 | `<SOS>` | 1 | la seule | "The" |
| 2 | `<SOS>`, The | 2 | la 2e (dernière) | "cat" |
| 3 | `<SOS>`, The, cat | 3 | la 3e (dernière) | "eats" |
| 4 | `<SOS>`, The, cat, eats | 4 | la 4e (dernière) | `<EOS>` → stop |

À l'étape 1, le softmax final n'a **qu'une ligne** (un seul token en input). Mais à partir de l'étape 2, le forward pass recalcule bien une matrice softmax à plusieurs lignes, parce que le masked self-attention traite toute la séquence input d'un coup — sauf qu'on **ignore toutes les lignes sauf la dernière**. Les lignes précédentes servaient à calculer la loss pendant l'entraînement (teacher forcing : toutes les prédictions comptent en même temps), mais à l'inférence on ne veut que "le prochain token", donc seule la dernière ligne (celle du dernier token généré) compte.

**Sample (étape 4)** : une fois la distribution de proba obtenue sur le vocabulaire pour la dernière position, on tire dedans — greedy (argmax, comme dans l'algo box tout en haut de la section II), ou avec de la température / top-k / nucleus sampling pour plus de diversité.

## d - Architecture Complète

![[Pasted image 20260419213034.png|328]]

**Composants principaux** :

1. **Embeddings d'entrée** + **Positional Encoding**
2. **Stack d'encoders** (N = 6 dans l'article original)
3. **Stack de decoders** (N = 6 dans l'article original)
4. **Couche de sortie linéaire + Softmax**

**Innovation clé** : "Attention is All You Need" - Pas de RNN ni CNN, que de l'attention !

# III - Variants et Applications

### Encoder-only (BERT family)

**Architecture** : Stack d'encodeurs uniquement (comme II.a, mais sans decoder — pas de génération).

#### Algorithme

```
Task: Compréhension / classification de texte (pas de génération)

TRAINING
├── Pretraining : Masked Language Model (MLM)
│     forward(x_masked) → ŷ (distribution sur le vocab)
│     loss = cross-entropy(ŷ, labels)   UNIQUEMENT aux positions masquées
│     (+ Next Sentence Prediction dans le papier original — tâche annexe sur [CLS], pas détaillée ici)
│
└── Fine-tuning
      brancher une tête de classification sur [CLS] (ou sur chaque token pour NER/QA)
      peu de données → transfer learning : fine-tuner tout le réseau avec un petit LR
      (ou geler le backbone et n'entraîner que la tête — alternative moins courante)

INFERENCE
  un seul forward pass, pas de génération autorégressive — dépend de la tâche fine-tunée
```

⚠️ **Deux "masques" totalement différents cohabitent dans ce fichier, à ne pas confondre** :
- Le masque **causal** du decoder (II.b.2) : un artifice de *calcul*, des $-\infty$ ajoutés aux scores d'attention, pour empêcher de voir le futur. Il ne touche jamais l'input.
- Le masque **BERT** (ci-dessous) : on remplace carrément des mots de l'**input** par le token `[MASK]` — une corruption des *données*, pas un artifice d'attention. BERT n'a **aucun** masque d'attention causal : c'est un encoder bidirectionnel pur (comme II.a), chaque token voit toute la phrase, y compris ce qui suit.

#### Pretraining — Masked Language Model, en détail

Prenons "Le chat est content".

**1. Tokenization + tokens spéciaux** : `[CLS]` en tête de séquence (son vecteur de sortie servira de résumé de toute la phrase, utilisé en fine-tuning), `[SEP]` en fin (sépare deux phrases quand il y en a deux — QA, NLI...) :

```
[CLS], le, chat, est, content, [SEP]
```

**2. Masking** : ~15% des tokens sont remplacés par `[MASK]` (dans le papier c'est un peu plus subtil — 80% du temps `[MASK]`, 10% un mot aléatoire, 10% le mot original inchangé, pour éviter que le modèle ne se repose que sur la présence du token `[MASK]` — mais gardons `[MASK]` simple ici). Masquons "chat" :

```
[CLS], le, [MASK], est, content, [SEP]
```

**3. Forward** : embedding lookup + positional encoding, puis $N$ blocs encoder bidirectionnels (comme II.a — self-attention pleine, aucun masque d'attention). Sortie : une matrice $Z \in \mathbb{R}^{6 \times d}$, une ligne par token (`[CLS], le, [MASK], est, content, [SEP]`).

**4. Projection + softmax** : $Z W_{pred} + b \in \mathbb{R}^{6 \times V}$, softmax ligne par ligne — même mécanique que la projection finale du Transformer (II.b). Le calcul produit techniquement une ligne de proba pour **chaque** token, `[CLS]` compris — mais toutes ne sont pas utilisées :

$$\text{Logits} = \begin{array}{cc} & \begin{matrix} \text{chat} & \text{chien} & \cdots & \text{zebra} \end{matrix} \\ \begin{matrix} \text{[CLS]} \\ \text{le} \\ \text{[MASK]} \\ \text{est} \\ \text{content} \\ \text{[SEP]} \end{matrix} & \begin{pmatrix} \cdot & \cdot & \cdots & \cdot \\ \cdot & \cdot & \cdots & \cdot \\ 0.75 & 0.15 & \cdots & 0.001 \\ \cdot & \cdot & \cdots & \cdot \\ \cdot & \cdot & \cdots & \cdot \\ \cdot & \cdot & \cdots & \cdot \end{pmatrix} \end{array}$$

**5. Cross-entropy — uniquement sur les positions masquées** : on calcule la loss **seulement** sur les tokens remplacés par `[MASK]` (ici, une seule position : "chat"), avec le vrai mot comme label :

$$\mathcal{L} = -\log\big(\hat{y}_{\text{[MASK]}}[\text{chat}]\big) = -\log(0.75) \approx 0.288$$

`le`, `est`, `content`, `[CLS]`, `[SEP]` **ne participent pas à la loss MLM** — ils sont déjà visibles tels quels dans l'input, prédire ce qu'on voit déjà n'apprendrait rien au modèle. Si plusieurs tokens sont masqués (le cas normal sur une phrase longue, ~15% des tokens), on moyenne les cross-entropy sur ces positions masquées uniquement.

#### Fine-tuning

On branche une petite couche linéaire sur la sortie de `[CLS]` (le vecteur qui a "absorbé" le contexte de toute la phrase via le self-attention — même principe qu'en II.a.3) :

```
[CLS]_output (d dimensions) → Linear(d, nb_classes) → softmax → proba par classe
```

Avec peu de données (transfer learning), deux stratégies :
- **Fine-tuning complet** : on réentraîne tout le réseau (backbone + tête) avec un **petit learning rate**, pour ne pas détruire ce que BERT a appris au pretraining. C'est l'approche standard du papier original.
- **Feature extraction** : on gèle le backbone, on n'entraîne que la tête de classification. Moins courant pour BERT, utile surtout avec très peu de données ou peu de compute.

#### Inférence

Un simple `forward()` : on tokenize la phrase, un seul passage dans le réseau fine-tuné, et la tête branchée sur `[CLS]` donne directement la proba de chaque classe (ex. sentiment positif/négatif) — pas de boucle, pas de génération.

**Applications** :
- Classification de texte
- Question-answering  
- Named Entity Recognition
- Analyse de sentiment

**Exemples** : BERT, RoBERTa, DistilBERT, ALBERT

### Decoder-only (GPT family)

**Architecture** : Stack de decoders uniquement — mais **sans cross-attention du tout**, pas simplement "remplacée par du self-attention non masqué". Il n'y a pas d'alternance masqué/non-masqué entre les couches : **toutes** les couches de self-attention sont masquées (causales), du début à la fin du réseau. La cross-attention disparaît simplement parce qu'il n'y a plus d'encoder séparé à consulter — prompt et génération vivent dans un seul flux causal.

#### Algorithme

```
Task: Prédire le token suivant (causal)

TRAINING
├── Pretraining (= foundational model)
│     forward(x) → ŷ  (toute la séquence, en parallèle)
│     x = séquence complète (le "prompt" et la "suite" ne sont pas distingués)
│     à la position t : xt sert d'input ET x_{t+1} sert de label (next-token prediction)
│     loss = cross-entropy(ŷ, x) à CHAQUE position (contrairement à BERT — pas de sélection)
│
└── Fine-tuning
      aucun dans le GPT original (2018) — pretraining pur, next-token prediction sur texte brut

INFERENCE (autorégressif, mot par mot — identique à II.c, sans encoder)
  y_0 = <SOS>
  pour t = 1, 2, ... jusqu'à <EOS> ou max_len :
      y_t = sample( decoder( y_{0:t-1} ) )   ← pas de encoder(x) à fournir, il n'y en a pas
      append y_t
```

#### Self-attention — un seul type, toujours masqué

Exactement le mécanisme de II.b.2, appliqué à toute la séquence (prompt + génération confondus, sans distinction). Exemple avec `<SOS>, the, cat` :

$$\text{Attention} = \begin{array}{cc} & \begin{matrix} \text{<SOS>} & \text{the} & \text{cat} \end{matrix} \\ \begin{matrix} \text{<SOS>} \\ \text{the} \\ \text{cat} \end{matrix} & \begin{pmatrix} 1.0 & 0 & 0 \\ 0.4 & 0.6 & 0 \\ 0.2 & 0.3 & 0.5 \end{pmatrix} \end{array}$$

`<SOS>` ne voit que lui-même (100%), exactement comme en II.b.2 — même logique, même matrice triangulaire. Un bloc GPT = Masked Self-Attention → Add&Norm → FFN → Add&Norm, empilé $N$ fois. Pas de deuxième type de couche : c'est le même bloc qui se répète, contrairement au decoder du Transformer original qui alternait self-attention masquée et cross-attention.

#### Sortie finale et cross-entropy

Même mécanique que II.b (projection $\times W_{out}$ + softmax ligne par ligne), puis cross-entropy — mais ici à **chaque** position, sans exception : contrairement à BERT (où seules les positions `[MASK]` comptent), ici l'objectif "prédire le mot suivant" s'applique à toutes les positions de la séquence, puisqu'il n'y a pas de trous artificiels à combler, juste la suite naturelle du texte.

#### RNN vs Transformer : l'argument clé du papier

Le tableau suivant résume le point le plus important de "Attention is All You Need" :

| | RNN (avant) | Transformer (GPT) |
|---|---|---|
| **Training** | Séquentiel — l'état caché à $t$ dépend du calcul à $t-1$ | **Parallèle** — la séquence entière est traitée en une seule passe matricielle |
| **Inference** | Séquentiel (inévitable) | **Séquentiel** — même contrainte, un token à la fois |

Le gain n'est **pas** à l'inférence (toujours séquentielle, comme en II.c) — il est à l'**entraînement** : au lieu de dérouler token par token comme un RNN, le masked self-attention calcule en une seule opération matricielle les représentations de toute la séquence (la matrice d'attention $L \times L$ se calcule d'un coup), donc toutes les positions "s'entraînent en même temps" grâce au teacher forcing. C'est ce qui a rendu l'entraînement sur des milliards de tokens tractable.

**Applications** :
- Génération de texte
- Modèles de langage
- Complétion de code
- Dialogue

**Exemples** : GPT-1/2/3/4, GPT-J, PaLM, LLaMA

### Encoder-Decoder (T5 family)

**Architecture** : Encoder + Decoder complets
**Applications** :
- Traduction automatique
- Résumé de texte
- Génération conditionnelle

**Exemples** : T5, BART, mT5, UL2

> [!success]- 🚀 Impact Révolutionnaire
> **2017** : Transformer révolutionne le NLP avec "Attention is All You Need"
> 
> **2018-2019** : BERT et GPT émergent, dominance encoder-only vs decoder-only
> 
> **2020+** : GPT-3 démontre l'émergence à grande échelle
> 
> **Aujourd'hui** : ChatGPT, GPT-4, Claude sont tous des Transformers !

Les Transformers ont **unifié l'IA moderne** - de la compréhension à la génération, de la vision à la robotique ! 🧠✨
