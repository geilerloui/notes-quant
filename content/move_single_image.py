
import base64
import os

def read_and_write_image(source_path, dest_path):
    try:
        with open(source_path, 'rb') as src:
            content = src.read()
        with open(dest_path, 'wb') as dst:
            dst.write(content)
        os.remove(source_path)  # Supprimer l'original
        return True
    except Exception as e:
        print(f"Erreur: {e}")
        return False

# Déplacer attention1.png
source = r"C:\Users\geile\mon-site\content\attention1.png"
dest = r"C:\Users\geile\mon-site\content\images\reseaux_sequentielles\attention1.png"

if read_and_write_image(source, dest):
    print("✅ attention1.png déplacé")
else:
    print("❌ Échec attention1.png")
