import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définir les données d'exemple du réseau
X = np.array([[3, 5], [5, 1], [10, 2]])  # 3x2
print("Données d'entrée X:")
print(X)

# Initialisation aléatoire des paramètres (exemple)
np.random.seed(42)  # Pour reproductibilité
W1 = np.random.normal(0, 1, (2, 3))  # 2x3
b1 = np.random.normal(0, 0.1, (1, 3))  # 1x3
W2 = np.random.normal(0, 1, (3, 1))  # 3x1  
b2 = np.random.normal(0, 0.1, (1, 1))  # 1x1

print("\nParamètres du réseau:")
print(f"W1 (2x3):\n{W1}")
print(f"b1 (1x3): {b1}")
print(f"W2 (3x1):\n{W2}")
print(f"b2 (1x1): {b2}")

# Fonctions d'activation
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass du réseau
def neural_network(x, W1, b1, W2, b2):
    """
    Forward pass: x (1x2) -> y_pred (1x1)
    """
    # Étape 2: Couche cachée
    z2 = np.dot(x.reshape(1, -1), W1) + b1  # 1x3
    a2 = relu(z2)  # ReLU pour couche cachée
    
    # Étape 3: Couche de sortie  
    z3 = np.dot(a2, W2) + b2  # 1x1
    y_pred = z3  # Identité pour régression
    
    return y_pred[0, 0]

# Test sur les données d'origine
print("\nPrédictions sur les données d'origine:")
for i, point in enumerate(X):
    pred = neural_network(point, W1, b1, W2, b2)
    print(f"Point {point} -> Prédiction: {pred:.3f}")

# Créer une grille pour la visualisation 3D
x_range = np.linspace(0, 12, 30)
y_range = np.linspace(0, 6, 30)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Calculer la surface Z = f(X, Y) 
Z_grid = np.zeros_like(X_grid)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        input_point = np.array([X_grid[i, j], Y_grid[i, j]])
        Z_grid[i, j] = neural_network(input_point, W1, b1, W2, b2)

print(f"\nGrille créée: {X_grid.shape[0]}x{X_grid.shape[1]} points")
print(f"Valeurs Z min: {Z_grid.min():.3f}, max: {Z_grid.max():.3f}")

# Générer le code LaTeX pour pgfplots
latex_code = f"""\\documentclass{{article}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\usepackage{{filecontents}}

% Données des points d'origine
\\begin{{filecontents}}{{points.dat}}
x y z
{X[0,0]} {X[0,1]} {neural_network(X[0], W1, b1, W2, b2):.6f}
{X[1,0]} {X[1,1]} {neural_network(X[1], W1, b1, W2, b2):.6f}  
{X[2,0]} {X[2,1]} {neural_network(X[2], W1, b1, W2, b2):.6f}
\\end{{filecontents}}

% Données de la surface (échantillonnée)
\\begin{{filecontents}}{{surface.dat}}
"""

# Ajouter les données de surface (sous-échantillonnage pour LaTeX)
step = 3  # Prendre 1 point sur 3 pour éviter trop de données
for i in range(0, X_grid.shape[0], step):
    for j in range(0, X_grid.shape[1], step):
        latex_code += f"{X_grid[i,j]:.3f} {Y_grid[i,j]:.3f} {Z_grid[i,j]:.6f}\n"

latex_code += """\\end{filecontents}

\\begin{document}

\\begin{figure}[htbp]
\\centering
\\begin{tikzpicture}
\\begin{axis}[
    width=12cm,
    height=8cm,
    xlabel={$x_1$},
    ylabel={$x_2$}, 
    zlabel={$\\hat{y}$},
    title={Surface 3D apprise par le réseau de neurones},
    colormap/viridis,
    view={45}{30},
    grid=major,
]

% Surface du réseau de neurones
\\addplot3[
    surf,
    opacity=0.7,
    mesh/ordering=y varies,
    samples=10,
] table {surface.dat};

% Points d'entraînement originaux
\\addplot3[
    only marks,
    mark=*,
    mark size=4pt,
    mark options={color=red, fill=red},
] table {points.dat};

\\legend{Surface apprise, Points d'entraînement}

\\end{axis}
\\end{tikzpicture}
\\caption{Visualisation 3D de la fonction $\\hat{y} = f(x_1, x_2)$ apprise par le réseau de neurones à 1 couche cachée (2→3→1) avec activation ReLU.}
\\end{figure}

\\end{document}"""

print("\n" + "="*50)
print("CODE LATEX GÉNÉRÉ:")
print("="*50)
print(latex_code)
