"""
Script om een schone zip te maken van het opdracht_transport project,
zonder cache bestanden en development directories.
"""
import os
import zipfile
from pathlib import Path

# Configuratie
PROJECT_DIR = Path(__file__).parent
OUTPUT_ZIP = PROJECT_DIR.parent / "learning_preferences.zip"

# Exclude patterns
EXCLUDE_PATTERNS = {
    '__pycache__',
    '.git',
    '.vscode',
    '.ipynb_checkpoints',
    'venv',
    'env',
    '.pytest_cache',
    '.gitkeep',
    '.gitignore',
    '.pyc',
    '.DS_Store',
    'Thumbs.db',
    'data',  # Data directory
    'create_zip.py'     # Dit script ook
}

def should_exclude(path: Path, project_root: Path) -> bool:
    """Check of een bestand/directory moet worden uitgesloten."""
    relative_path = path.relative_to(project_root)
    parts = relative_path.parts
    
    # Check alle parts van het pad
    for part in parts:
        if part in EXCLUDE_PATTERNS:
            return True
    
    # Check bestandsextensies
    if path.is_file() and path.suffix in {'.pyc'}:
        return True
        
    # Check bestandsnamen
    if path.name in EXCLUDE_PATTERNS:
        return True
    
    return False

def create_clean_zip():
    """Maak een zip bestand zonder cache en development bestanden."""
    # Verwijder oude zip indien aanwezig
    if OUTPUT_ZIP.exists():
        try:
            OUTPUT_ZIP.unlink()
            print(f"Oude zip verwijderd: {OUTPUT_ZIP}")
        except PermissionError:
            print(f"⚠ Waarschuwing: Kon oude zip niet verwijderen (mogelijk in gebruik)")
            print(f"   Probeer het bestand te sluiten en opnieuw uit te voeren")
            # Try alternatieve naam
            alt_name = OUTPUT_ZIP.parent / f"{OUTPUT_ZIP.stem}_new{OUTPUT_ZIP.suffix}"
            OUTPUT_ZIP_PATH = alt_name
            print(f"   Gebruik alternatieve naam: {alt_name.name}")
        else:
            OUTPUT_ZIP_PATH = OUTPUT_ZIP
    else:
        OUTPUT_ZIP_PATH = OUTPUT_ZIP
    
    # Verzamel alle bestanden
    all_files = []
    excluded_count = 0
    
    for root, dirs, files in os.walk(PROJECT_DIR):
        root_path = Path(root)
        
        # Filter directories (in-place modify om recursie te stoppen)
        dirs[:] = [d for d in dirs if not should_exclude(root_path / d, PROJECT_DIR)]
        
        # Voeg bestanden toe die niet excluded zijn
        for file in files:
            file_path = root_path / file
            if should_exclude(file_path, PROJECT_DIR):
                excluded_count += 1
            else:
                all_files.append(file_path)
    
    print(f"Verzameld: {len(all_files)} bestanden")
    print(f"Uitgesloten: {excluded_count} bestanden")
    
    # Maak de zip
    print(f"\nZip bestand aanmaken: {OUTPUT_ZIP_PATH.name}")
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in all_files:
            # Arcname is het pad in de zip (relatief tot project parent)
            arcname = Path('opdracht_transport') / file_path.relative_to(PROJECT_DIR)
            zipf.write(file_path, arcname)
            
    # Resultaat
    zip_size_mb = OUTPUT_ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✓ Zip succesvol aangemaakt!")
    print(f"  Locatie: {OUTPUT_ZIP_PATH}")
    print(f"  Grootte: {zip_size_mb:.2f} MB")
    print(f"  Bestanden: {len(all_files)}")
    
if __name__ == "__main__":
    create_clean_zip()
