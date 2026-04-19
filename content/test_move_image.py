import base64

def move_image(source_path, dest_path, filesystem_funcs):
    """Déplace une image en la lisant et la recréant dans le nouveau dossier"""
    try:
        # Lire l'image source
        source_data = filesystem_funcs['read_media'](source_path)
        
        if source_data and 'data' in source_data:
            # Décode les données base64
            image_data = base64.b64decode(source_data['data'])
            
            # Écrire dans le nouveau dossier (on simule en Python mais on ferait avec filesystem:write_file pour du binaire)
            # Note: Avec les outils filesystem, on devra faire ça différemment
            
            print(f"✅ Image {source_path} lue avec succès ({len(image_data)} bytes)")
            return True
        else:
            print(f"❌ Échec lecture {source_path}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur pour {source_path}: {e}")
        return False

# Test avec attention1.png
print("Test de lecture d'image...")
