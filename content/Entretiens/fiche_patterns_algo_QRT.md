# Fiche récap — Patterns algo pour screen technique

> Objectif : **reconnaître** le bon pattern en 10 secondes à la lecture de l'énoncé.
> Le tableau du haut sert au scan rapide ; les détails suivent.

---

## Tableau de reconnaissance rapide

| # | Le déclencheur dans l'énoncé | Pattern | Complexité |
|---|------------------------------|---------|------------|
| 1 | « trouve le max / min / somme / compte en parcourant » | Accumulateur d'état | O(n) |
| 2 | « k-ème plus grand / petit », « top k » | Top-k en une passe | O(n) |
| 3 | « deux éléments qui… », « somme = cible », « déjà vu ? », « compter les paires/sous-tableaux » | Hashmap / complément | O(n) |
| 4 | « sous-tableau / sous-chaîne **contigu** », « fenêtre », « k éléments consécutifs » | Sliding window | O(n) |
| 5 | tableau **trié** + « paire », « deux éléments », « palindrome » | Two pointers | O(n) |
| 6 | « parenthèses / équilibré », « le plus récent », « annuler / backspace », « imbrication » | Pile (stack) | O(n) |
| 7 | tableau **trié** + « chercher / position », ou « O(log n) demandé » | Binary search | O(log n) |
| 8 | « intervalles », « chevauchement », « fusionner », le problème **dépend de l'ordre** | Tri + balayage | O(n log n) |
| 9 | « **toutes** les combinaisons / permutations / sous-ensembles », « générer toutes les façons » | Backtracking | exponentiel |
| 10 | « arbre », « graphe », « nœuds reliés », « chemin », « niveaux » | DFS / BFS | O(n+arêtes) |

**Réflexe transversal n°1 :** le mot **« trié »** débloque presque toujours two pointers ou binary search. Le mot **« contigu »** = sliding window. Le mot **« toutes les »** = backtracking.

---

## Détail des 10 patterns

### 1. Accumulateur d'état
Garder une variable « meilleur jusqu'ici » et la mettre à jour en parcourant.
```python
best = nums[0]                 # JAMAIS 0 : penser aux négatifs
for x in nums[1:]:
    if x > best:
        best = x
```

### 2. Top-k en une passe
Maintenir plusieurs variables d'état. L'ordre de mise à jour est tout l'enjeu.
```python
premier = deuxieme = float('-inf')
for x in nums:
    if x > premier:
        deuxieme = premier     # l'ancien max DESCEND avant
        premier = x
    elif x > deuxieme:         # elif, pas if
        deuxieme = x
```

### 3. Hashmap / complément
« Ai-je déjà vu X ? » en O(1) grâce à un dico construit au fil de l'eau.
```python
seen = {}
for i, x in enumerate(nums):
    if target - x in seen:     # 'in dico' teste les CLÉS en O(1)
        return [seen[target - x], i]
    seen[x] = i                # enregistrer APRÈS le test
```
Variante « compter » (sous-tableaux de somme k) : stocker `{somme_cumulée : nb de fois vue}`, partir de `{0: 1}`, faire `count += seen.get(somme - k, 0)`.

### 4. Sliding window (fenêtre fixe)
Ne pas resommer : l'entrant entre, le sortant sort.
```python
somme = sum(nums[:w])
best = somme
for i in range(w, len(nums)):
    somme += nums[i] - nums[i - w]   # +entrant  -sortant
    best = max(best, somme)
```

### 5. Two pointers (tableau trié)
Deux curseurs qui se resserrent ; le tri remplace le dico (zéro mémoire en plus).
```python
g, d = 0, len(nums) - 1
while g < d:                   # borne l'espace : pas "tant que pas trouvé"
    s = nums[g] + nums[d]
    if s == target:   return [g, d]
    elif s < target:  g += 1   # trop petit → pousse à gauche
    else:             d -= 1   # trop grand → recule à droite
```

### 6. Pile (stack)
« Dernier ouvert = premier fermé ». Empiler les ouvrantes, vérifier le sommet sur les fermantes.
```python
paires = {")": "(", "]": "[", "}": "{"}
pile = []
for c in s:
    if c in "([{":
        pile.append(c)
    elif not pile or pile.pop() != paires[c]:
        return False
return len(pile) == 0          # tout a été refermé
```

### 7. Binary search
Diviser l'espace de recherche par deux. Deux bornes, pas de slicing (le slicing copie en O(n) et tue le O(log n)).
```python
g, d = 0, len(nums) - 1
while g <= d:                  # <= obligatoire (cas un seul élément)
    mid = (g + d) // 2
    if nums[mid] == cible:   return mid
    elif nums[mid] < cible:  g = mid + 1   # +1/-1 : exclut mid déjà testé
    else:                    d = mid - 1
return -1
```

### 8. Tri + balayage (intervalles)
Trier selon le bon critère fait apparaître la structure. Comparer au **dernier élément construit**, pas au précédent brut.
```python
intervalles.sort(key=lambda x: x[0])         # ÉTAPE 1 obligatoire
res = [intervalles[0]]
for deb, fin in intervalles[1:]:
    if deb <= res[-1][1]:                    # chevauchement
        res[-1][1] = max(res[-1][1], fin)    # max : cas intervalle contenu
    else:
        res.append([deb, fin])
```

### 9. Backtracking
**CHOISIR → EXPLORER → DÉFAIRE.** « Défaire » est l'inverse exact de « choisir » (Petit Poucet : pose le caillou, explore, ramasse-le). Régénérer le squelette via deux questions : *mes choix ?* / *quand est-ce complet ?*
```python
def backtrack(courant):
    if complet:                        # condition d'arrêt
        resultat.append(courant[:])    # copie !
        return
    for choix in choix_possibles:
        courant.append(choix)          # CHOISIR
        backtrack(courant)             # EXPLORER
        courant.pop()                  # DÉFAIRE
```

### 10. Arbres & graphes (DFS / BFS)
**Saut de foi récursif** : suppose que la fonction marche déjà sur les sous-arbres, et combine. (Manager paresseux : délègue aux enfants, fais confiance, combine.)
```python
# DFS récursif — profondeur d'un arbre
def dfs(noeud):
    if noeud is None:
        return 0
    return 1 + max(dfs(noeud.gauche), dfs(noeud.droite))

# BFS — parcours par niveaux, avec une FILE
from collections import deque
file = deque([racine])
while file:
    noeud = file.popleft()
    # traiter noeud
    file.extend(n for n in (noeud.gauche, noeud.droite) if n)
```
Repère : DFS → récursion ou pile (profondeur). BFS → file / `deque` (largeur, niveaux, plus court chemin non pondéré).

---

## Réflexes transversaux (aussi importants que les patterns)

- **Initialisation** : jamais `max = 0`. Partir de `nums[0]` ou `float('-inf')`. (Le piège des négatifs revient sans arrêt.)
- **Edge cases à énoncer à voix haute** : liste vide, négatifs, doublons, un seul élément. Les dire = points gagnés même sans les coder.
- **Condition de boucle** : toujours *borner l'espace de recherche* (`g <= d`, `g < d`), jamais « tant que pas trouvé » → boucle infinie sur le cas sans solution.
- **Dico** : `in dico` teste les clés en O(1) ; `dico.get(clé, défaut)` évite le `KeyError`.
- **Clarifier avant de coder** : les ambiguïtés (« deuxième plus grand » = distinct ou pas ? trié ou non ?) se posent à l'examinateur — ça montre ta rigueur.
- **Tracer son code à la main** sur un petit exemple : ton meilleur debug, et ça impressionne en live.

---

## Méthode en 5 étapes (à dérouler à voix haute en entretien)

1. **Reformuler + clarifier** — répéter l'énoncé, poser les ambiguïtés (doublons ? vide ? trié ?).
2. **Brute force annoncée** — « naïvement, c'est du O(n²) » : on pose une solution qui marche.
3. **Optimiser** — quel déclencheur matche ? quel pattern applique-t-on ?
4. **Coder proprement** — noms clairs, pas de variable `max` qui masque la built-in.
5. **Tester + complexité** — dérouler un edge case, puis énoncer la complexité temps/espace finale.

> Le screen ne récompense pas la solution parfaite du premier coup : il récompense une démarche **claire, structurée et verbalisée**. Une brute force annoncée + une optimisation expliquée vaut mieux qu'un code optimal sorti sans un mot.
