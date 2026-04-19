import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuration pour de beaux plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# Définir les données d'exemple du réseau
X = np.array([[3, 5], [5, 1], [10, 2]])  # 3x2
print("Données d'entrée X:")
print(X)

# Initialisation des paramètres du réseau (reproductible)
np.random.seed(42)
W1 = np.random.normal(0, 1, (2, 3))  # 2x3
b1 = np.random.normal(0, 0.1, (1, 3))  # 1x3
W2 = np.random.normal(0, 1, (3, 1))  # 3x1  
b2 = np.random.normal(0, 0.1, (1, 1))  # 1x1

print(f"\nParamètres du réseau:")
print(f"W1 shape: {W1.shape}")
print(f"W2 shape: {W2.shape}")

# Fonctions d'activation
def relu(x):
    return np.maximum(0, x)

def neural_network(x, W1, b1, W2, b2):
    """
    Forward pass: x (1x2) -> y_pred (scalaire)
    """
    # Étape 2: Couche cachée
    z2 = np.dot(x.reshape(1, -1), W1) + b1  # 1x3
    a2 = relu(z2)  # ReLU pour couche cachée
    
    # Étape 3: Couche de sortie  
    z3 = np.dot(a2, W2) + b2  # 1x1
    y_pred = z3[0, 0]  # Identité pour régression
    
    return y_pred

# Test sur les données d'origine
print("\nPrédictions sur les données d'origine:")
predictions_original = []
for i, point in enumerate(X):
    pred = neural_network(point, W1, b1, W2, b2)
    predictions_original.append(pred)
    print(f"Point {point} -> Prédiction: {pred:.3f}")

# Créer une grille pour visualisation 3D
x_min, x_max = 0, 12
y_min, y_max = 0, 6
resolution = 50

x_range = np.linspace(x_min, x_max, resolution)
y_range = np.linspace(y_min, y_max, resolution)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Calculer la surface Z = f(X, Y)
print(f"\nCalcul de la surface {resolution}x{resolution}...")
Z_grid = np.zeros_like(X_grid)
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        input_point = np.array([X_grid[i, j], Y_grid[i, j]])
        Z_grid[i, j] = neural_network(input_point, W1, b1, W2, b2)

print(f"Surface calculée. Valeurs Z: min={Z_grid.min():.3f}, max={Z_grid.max():.3f}")

# Créer la visualisation
fig = plt.figure(figsize=(15, 12))

# Plot 1: Surface 3D
ax1 = fig.add_subplot(221, projection='3d')
surface = ax1.plot_surface(X_grid, Y_grid, Z_grid, 
                          cmap='viridis', alpha=0.7,
                          linewidth=0, antialiased=True)

# Ajouter les points d'entraînement
ax1.scatter(X[:, 0], X[:, 1], predictions_original, 
           color='red', s=100, alpha=1, label='Points d\'entraînement')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$') 
ax1.set_zlabel('$\\hat{y}$')
ax1.set_title('Surface 3D du réseau de neurones')
ax1.legend()

# Plot 2: Vue de dessus (contour)
ax2 = fig.add_subplot(222)
contour = ax2.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap='viridis', alpha=0.8)
ax2.contour(X_grid, Y_grid, Z_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
ax2.scatter(X[:, 0], X[:, 1], color='red', s=100, zorder=5, 
           edgecolors='white', linewidth=2, label='Points d\'entraînement')

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Vue de dessus (contours)')
ax2.legend()
plt.colorbar(contour, ax=ax2, label='$\\hat{y}$')

# Plot 3: Coupe selon x_2 = 3
ax3 = fig.add_subplot(223)
y_slice = 3
x_slice = np.linspace(x_min, x_max, 100)
z_slice = [neural_network(np.array([x, y_slice]), W1, b1, W2, b2) for x in x_slice]

ax3.plot(x_slice, z_slice, 'b-', linewidth=2, label=f'$x_2 = {y_slice}$')

# Ajouter les points d'entraînement sur cette coupe
for i, point in enumerate(X):
    if abs(point[1] - y_slice) < 1:  # Points proches de la coupe
        pred = predictions_original[i]
        ax3.plot(point[0], pred, 'ro', markersize=8, label=f'Point ({point[0]}, {point[1]})')

ax3.set_xlabel('$x_1$')
ax3.set_ylabel('$\\hat{y}$')
ax3.set_title(f'Coupe de la fonction à $x_2 = {y_slice}$')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Erreur d'interpolation
ax4 = fig.add_subplot(224)

# Créer des points test pour évaluer l'interpolation
n_test = 20
x_test = np.random.uniform(x_min, x_max, n_test)
y_test = np.random.uniform(y_min, y_max, n_test)
z_test = [neural_network(np.array([x_test[i], y_test[i]]), W1, b1, W2, b2) 
          for i in range(n_test)]

# Scatter plot des prédictions
scatter = ax4.scatter(x_test, y_test, c=z_test, cmap='viridis', s=50, alpha=0.7)
ax4.scatter(X[:, 0], X[:, 1], color='red', s=100, zorder=5,
           edgecolors='white', linewidth=2, label='Points d\'entraînement')

ax4.set_xlabel('$x_1$')
ax4.set_ylabel('$x_2$')
ax4.set_title('Points test avec prédictions')
ax4.legend()
plt.colorbar(scatter, ax=ax4, label='$\\hat{y}$')

plt.tight_layout()
plt.show()

# Statistiques du réseau
print("\n" + "="*50)
print("RÉSUMÉ DU RÉSEAU DE NEURONES")
print("="*50)
print(f"Architecture: 2 → 3 → 1")
print(f"Activation cachée: ReLU")
print(f"Activation sortie: Identité (régression)")
print(f"\nPoints d'entraînement:")
for i, point in enumerate(X):
    print(f"  ({point[0]}, {point[1]}) → {predictions_original[i]:.3f}")
print(f"\nPlage de sortie sur la grille: [{Z_grid.min():.3f}, {Z_grid.max():.3f}]")
print(f"Nombre de paramètres: {W1.size + b1.size + W2.size + b2.size}")
