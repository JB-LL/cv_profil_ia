import subprocess
import sys
import os

DEPENDANCES = [
    "pymupdf",
    "pytesseract",
    "Pillow",
    "spacy",
    "scikit-learn",
    "python-docx",
    "tkinterdnd2",
]

def installer_dependances():
    print("🔧 Vérification des dépendances...")
    for dep in DEPENDANCES:
        try:
            __import__(dep.replace("-", "_").split("[")[0])
        except ImportError:
            print(f"  📦 Installation de {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

    # Modèle spaCy français
    try:
        import spacy
        spacy.load("fr_core_news_sm")
    except OSError:
        print("  📦 Installation du modèle de langue français...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])

    print("✅ Toutes les dépendances sont prêtes.")

if __name__ == "__main__":
    installer_dependances()
    print("🚀 Lancement de l'application...")
    os.system(f"{sys.executable} app.py")