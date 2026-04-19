# Script PowerShell pour déplacer les images RNN
$sourceDir = "C:\Users\geile\mon-site\content"
$destDir = "C:\Users\geile\mon-site\content\images\reseaux_sequentielles"

# Liste des images à déplacer
$rnnImages = @(
    "attention1.png",
    "attention2.png", 
    "attention3.png",
    "attention3 1.png",
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
    "Pasted image 20260419170349.png",
    "Pasted image 20260419192535.png", 
    "Pasted image 20260419194100.png",
    "Pasted image 20260419194151.png",
    "Pasted image 20260419194323.png"
)

$moved = 0
$total = $rnnImages.Length

Write-Host "🚀 Déplacement de $total images vers le dossier reseaux_sequentielles" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

foreach ($image in $rnnImages) {
    $sourcePath = Join-Path $sourceDir $image
    $destPath = Join-Path $destDir $image
    
    if (Test-Path $sourcePath) {
        try {
            Move-Item -Path $sourcePath -Destination $destPath -Force
            Write-Host "✅ Déplacé: $image" -ForegroundColor Green
            $moved++
        }
        catch {
            Write-Host "❌ Erreur pour $image : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    else {
        Write-Host "⚠️  Non trouvé: $image" -ForegroundColor Yellow
    }
}

Write-Host "=" * 60 -ForegroundColor Gray
Write-Host "📊 Résumé: $moved/$total images déplacées avec succès" -ForegroundColor Cyan