import shutil
import os

# Chemins
source_dir = r"C:\Users\geile\mon-site\content"
target_dir = r"C:\Users\geile\mon-site\content\images\regression-logistique"

# Images odds/log-odds
odds_images = [
    "im1.png",
    "im2.png", 
    "im3 (1).png",
    "im4.png",
    "im5.png"
]

# Images des graphiques générés pour la régression logistique
regression_images = [
    "Pasted image 20260418145222.png",
    "Pasted image 20260418151041.png",
    "Pasted image 20260418151313.png", 
    "Pasted image 20260418152205.png",
    "Pasted image 20260418152730.png",
    "Pasted image 20260418154035.png",
    "Pasted image 20260418155040.png",
    "Pasted image 20260418155550.png",
    "Pasted image 20260418155807.png",
    "Pasted image 20260418160136.png",
    "Pasted image 20260418175327.png",
    "Pasted image 20260418185333.png"
]

# Combiner toutes les images
all_images = odds_images + regression_images

print("Déplacement des images de régression logistique...")

for image in all_images:
    source_path = os.path.join(source_dir, image)
    target_path = os.path.join(target_dir, image)
    
    if os.path.exists(source_path):
        try:
            shutil.move(source_path, target_path)
            print(f"✓ Déplacé: {image}")
        except Exception as e:
            print(f"✗ Erreur lors du déplacement de {image}: {e}")
    else:
        print(f"⚠ Image non trouvée: {image}")

print(f"\nTerminé! Les images sont maintenant dans: {target_dir}")
