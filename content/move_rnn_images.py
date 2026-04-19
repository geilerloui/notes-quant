import os
import shutil

# Fonction pour déplacer une image si elle existe
def move_image(source_dir, dest_dir, filename):
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)
    
    if os.path.exists(source_path):
        try:
            # Si le fichier de destination existe déjà, le supprimer d'abord
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(source_path, dest_path)
            print(f"✅ Déplacé: {filename}")
            return True
        except Exception as e:
            print(f"❌ Erreur pour {filename}: {e}")
            return False
    else:
        print(f"⚠️  Non trouvé: {filename}")
        return False

# Dossiers
source_dir = r"C:\Users\geile\mon-site\content"
dest_dir = r"C:\Users\geile\mon-site\content\images\reseaux_sequentielles"

# Images liées aux réseaux séquentiels
rnn_images = [
    # Images attention
    "attention1.png",
    "attention2.png", 
    "attention3.png",
    "attention3 1.png",
    
    # Images générales RNN 
    "im2.png",
    "im2 (1).png",
    "im3.png", 
    "im3 (1).png",
    "im4.png",
    "im5.png",
    "im5 (1).png",
    "im6.png",
    "im7.png",
    "im7 (1).png",
    "im8.png",
    "im10.png",
    "im10 (1).png",
    "im11.png",
    "im12.png",
    "im13.png", 
    "im14.png",
    
    # Images Pasted du 19 avril (jour des RNN)
    "Pasted image 20260419170349.png",
    "Pasted image 20260419192535.png", 
    "Pasted image 20260419194100.png",
    "Pasted image 20260419194151.png",
    "Pasted image 20260419194323.png"
]

# Déplacer les images
moved_count = 0
total_count = len(rnn_images)

print(f"🚀 Déplacement de {total_count} images vers {dest_dir}")
print("=" * 60)

for image in rnn_images:
    if move_image(source_dir, dest_dir, image):
        moved_count += 1

print("=" * 60)
print(f"📊 Résumé: {moved_count}/{total_count} images déplacées avec succès")
