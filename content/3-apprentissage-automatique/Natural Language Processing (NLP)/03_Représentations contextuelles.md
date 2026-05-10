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

## a - Bloc Encoder

Le bloc encoder traite la séquence d'entrée et produit une représentation enrichie de chaque position.

**Composants d'un bloc encoder** :

1. **Multi-Head Self-Attention** : Chaque position peut porter attention à toutes les positions
2. **Residual Connection** : `output = input + attention(input)`
3. **Layer Normalization** : Normalisation des activations
4. **Feed-Forward Network** : Transformation non-linéaire position par position
5. **Residual Connection + LayerNorm** : Deuxième connexion résiduelle

**Architecture mathématique** :

$$\begin{aligned}
\text{Attention}_{out} &= \text{MultiHead}(\text{input}) \\
\text{Add\&Norm}_1 &= \text{LayerNorm}(\text{input} + \text{Attention}_{out}) \\
\text{FFN}_{out} &= \text{FFN}(\text{Add\&Norm}_1) \\
\text{Add\&Norm}_2 &= \text{LayerNorm}(\text{Add\&Norm}_1 + \text{FFN}_{out})
\end{aligned}$$

## b - Bloc Decoder

Le bloc decoder génère la séquence de sortie de manière autorégressive.

**Composants d'un bloc decoder** :

1. **Masked Multi-Head Self-Attention** : Attention causale (ne voit pas le futur)
2. **Residual Connection + LayerNorm**
3. **Multi-Head Cross-Attention** : Attention entre decoder et encoder
4. **Residual Connection + LayerNorm**  
5. **Feed-Forward Network**
6. **Residual Connection + LayerNorm**

**Masking** : Le decoder utilise un masque triangulaire pour empêcher l'accès aux positions futures pendant l'entraînement.

## c - Architecture Complète

![[Pasted image 20260419213034.png|328]]

**Composants principaux** :

1. **Embeddings d'entrée** + **Positional Encoding**
2. **Stack d'encoders** (N = 6 dans l'article original)
3. **Stack de decoders** (N = 6 dans l'article original)
4. **Couche de sortie linéaire + Softmax**

**Innovation clé** : "Attention is All You Need" - Pas de RNN ni CNN, que de l'attention !

## d - Variants et Applications

### Encoder-only (BERT family)

**Architecture** : Stack d'encodeurs uniquement
**Applications** :
- Classification de texte
- Question-answering  
- Named Entity Recognition
- Analyse de sentiment

**Exemples** : BERT, RoBERTa, DistilBERT, ALBERT

### Decoder-only (GPT family)

**Architecture** : Stack de decoders uniquement (sans cross-attention)
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

# III - Large Language Models & Alignment (2019-2024)

## a - De GPT-3 à InstructGPT : Le Problème d'Alignement

**2020** : GPT-3 marque une rupture avec **175 milliards de paramètres**, mais révèle un problème fondamental.

### **GPT-3 : Puissant mais pas utile**

**Entraînement** : Prédiction du token suivant sur des milliards de pages web
```
Input:  "The weather today is"
Output: "quite unpredictable given the recent climate patterns in the region..."
```

**Problèmes** :
1. **Completion vs Assistance** : GPT-3 "complète" au lieu d'aider
2. **Toxicité** : Reproduit les biais des données d'entraînement  
3. **Hallucinations** : Invente des "faits" plausibles mais faux
4. **Prompt Engineering** : Nécessite des prompts très spécifiques

**Exemple typique** :
```
Human: "How do I bake a cake?"
GPT-3: "How do I bake a cake? This is a question many people ask when they 
       first start cooking. Baking has been around for centuries..."
```
→ **Pas une réponse utile !**

### **Le Problème d'Alignement**

**Objectif réel** : Un assistant utile, inoffensif, honnête
**Training objectif** : Prédire le token suivant sur internet

**Cette divergence** crée des modèles qui :
- Optimisent pour la vraisemblance, pas l'utilité
- Reproduisent le "style internet" au lieu du "style assistance"
- N'apprennent pas les préférences humaines

## b - InstructGPT : Pipeline RLHF

**OpenAI (2022)** introduit **"Training language models to follow instructions with human feedback"** - la solution révolutionnaire.

### **Pipeline en 3 étapes**

#### **Étape 1 : Supervised Fine-Tuning (SFT)**

**Objectif** : Apprendre le format "instruction-following"

**Données** : ~13k exemples de haute qualité créés par des humains
```
Input:  "How do I bake a chocolate cake?"
Output: "Here's a step-by-step recipe for chocolate cake:
         1. Preheat oven to 350°F
         2. Mix 2 cups flour, 1.5 cups sugar..."
```

**Résultat** : GPT-3.5 (modèle de base pour ChatGPT)

#### **Étape 2 : Reward Model Training**

**Objectif** : Apprendre les préférences humaines

**Processus** :
1. Générer plusieurs réponses pour une même question
2. Humains classent les réponses (A > B > C > D)
3. Entraîner un **Reward Model** à prédire ces classements

**Exemple** :
```
Question: "Explain quantum computing"

Réponse A: "Quantum computing uses quantum mechanics..."
Réponse B: "Quantum computers are magic machines..."
Réponse C: "I don't know about quantum computing"

Human ranking: A > C > B
```

#### **Étape 3 : Reinforcement Learning (PPO)**

**Objectif** : Optimiser pour maximiser le reward humain

**Algorithme** : Proximal Policy Optimization (PPO)
- Le modèle génère des réponses
- Le Reward Model évalue chaque réponse  
- PPO ajuste les poids pour maximiser les rewards
- **Constraint** : Ne pas trop s'écarter du modèle SFT

### **Architecture du Reward Model**

$$R_\phi(x, y) = \text{score de qualité de la réponse } y \text{ à la question } x$$

Le Reward Model prédit un score scalaire pour chaque paire (prompt, completion).

### **Objectif PPO avec contraintes**

$$\text{maximize } \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\phi}^{RL}}[R_\phi(x, y)] - \beta \log\left(\frac{\pi_{\phi}^{RL}(y|x)}{\pi^{SFT}(y|x)}\right)$$

où :
- $R_\phi(x, y)$ : reward du Reward Model
- $\beta$ : coefficient de régularisation KL
- Le terme KL empêche le modèle de devenir "trop différent" du modèle SFT

## c - Résultats & Révolution ChatGPT

### **Métriques de succès**

**Helpful, Harmless, Honest** (3H Framework) :

1. **Helpfulness** : Répond de manière utile et informationnelle
2. **Harmlessness** : Évite le contenu toxique, biaisé, ou dangereux
3. **Honesty** : Admet quand il ne sait pas quelque chose

### **Résultats InstructGPT**

**Avec seulement 1.3B paramètres**, InstructGPT surpasse GPT-3 **175B** sur :
- ✅ **Truthfulness** : 2× moins d'hallucinations
- ✅ **Toxicity** : 3× moins de sorties toxiques  
- ✅ **Helpfulness** : Préféré par les humains dans 85% des cas

**L'insight clé** : **La qualité > quantité de paramètres**

### **ChatGPT : InstructGPT + Dialogue**

**Novembre 2022** : OpenAI lance ChatGPT
- **Base** : InstructGPT avec fine-tuning additionnel sur des conversations
- **Interface** : Chat conversationnel au lieu d'API
- **Résultat** : **100 millions d'utilisateurs en 2 mois** 🚀

**Exemple transformation** :
```
GPT-3: "Write a poem" → "Here are some poems from famous poets..."
ChatGPT: "Write a poem" → "Here's an original poem I wrote for you:
                          Roses are red, violets are blue..."
```

## d - Post-ChatGPT : L'écosystème moderne

### **GPT-4 (Mars 2023) : Multimodalité & Reasoning**

**"GPT-4 Technical Report"** introduit :
- **Vision** : Compréhension d'images
- **Reasoning amélioré** : Meilleur sur les problèmes complexes
- **Réduction des hallucinations** : Plus factuel que GPT-3.5
- **Emergent abilities** : Capacités qui émergent à grande échelle

### **Claude (Anthropic) : Constitutional AI**

**"Constitutional AI: Harmlessness from AI Feedback"** :
- **Self-improvement** : Le modèle s'améliore via sa propre critique
- **Constitution** : Ensemble de principes éthiques codifiés
- **AI Feedback** : Moins de dépendance aux annotations humaines

### **LLaMA (Meta) : Open Source**

**"Open and Efficient Foundation Language Models"** :
- Modèles **open source** compétitifs 
- **Efficacité** : Performance similaire avec moins de paramètres
- **Démocratisation** : Recherche accessible à tous

### **Scaling Laws & Emergent Abilities**

**Découvertes clés** :
1. **Scaling Laws** : Performance ∝ log(paramètres)
2. **Emergent abilities** : Capacités qui apparaissent soudainement à certaines échelles
3. **Data quality** >> data quantity pour l'alignement

## **🔥 L'Impact Révolutionnaire**

**RLHF a changé l'IA** :
- ✅ **Utilisabilité** : De "compléteur de texte" à "assistant intelligent"  
- ✅ **Sécurité** : Modèles plus alignés avec les valeurs humaines
- ✅ **Adoption** : Millions d'utilisateurs quotidiens
- ✅ **Industrie** : Toutes les Big Tech adoptent RLHF

**Le passage GPT-3 → ChatGPT** n'est pas juste une amélioration technique, c'est **une révolution d'usage** ! 

L'IA est passée d'un **outil de laboratoire** à un **assistant quotidien** grâce à l'alignement par feedback humain ! 🚀

> [!success]- 🎯 L'Insight Final
> **InstructGPT prouve** que l'alignement avec les préférences humaines est plus important que la taille du modèle.
> 
> **RLHF est devenu** le standard pour tous les LLMs modernes : GPT-4, Claude, Gemini, LLaMA-2.
> 
> **L'avenir de l'IA** se joue sur l'alignement, pas seulement sur le scaling ! 🧠✨