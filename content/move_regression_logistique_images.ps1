# Script PowerShell pour déplacer les images de régression logistique

$sourceDir = "C:\Users\geile\mon-site\content"
$targetDir = "C:\Users\geile\mon-site\content\images\regression-logistique"

# Images odds/log-odds
$oddsImages = @(
    "im1.png",
    "im2.png", 
    "im3 (1).png",
    "im4.png",
    "im5.png"
)

# Images des graphiques générés pour la régression logistique
$regressionImages = @(
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
)

# Combiner toutes les images
$allImages = $oddsImages + $regressionImages

Write-Host "Déplacement des images de régression logistique..." -ForegroundColor Green

foreach ($image in $allImages) {
    $sourcePath = Join-Path $sourceDir $image
    $targetPath = Join-Path $targetDir $image
    
    if (Test-Path $sourcePath) {
        try {
            Move-Item $sourcePath $targetPath -Force
            Write-Host "✓ Déplacé: $image" -ForegroundColor Green
        } catch {
            Write-Host "✗ Erreur lors du déplacement de $image : $_" -ForegroundColor Red
        }
    } else {
        Write-Host "⚠ Image non trouvée: $image" -ForegroundColor Yellow
    }
}

Write-Host "`nTerminé! Les images sont maintenant dans: $targetDir" -ForegroundColor Cyan
