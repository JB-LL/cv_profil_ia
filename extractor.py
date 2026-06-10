import os
import platform
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Gestion de la portabilité de Tesseract
if platform.system() == "Windows":
    # Chemin par défaut standard sur Windows
    chemin_win = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(chemin_win):
        pytesseract.pytesseract.tesseract_cmd = chemin_win
# Sur Mac/Linux, Tesseract est généralement directement dans le PATH global

def extraire_texte(chemin_fichier):
    """Extrait le texte de manière hybride (Texte natif ou OCR de secours)"""
    ext = chemin_fichier.lower().split('.')[-1]
    
    if ext == 'txt':
        with open(chemin_fichier, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
            
    elif ext == 'pdf':
        texte_complet = ""
        try:
            doc = fitz.open(chemin_fichier)
            for page in doc:
                texte_page = page.get_text()
                # Si la page contient du vrai texte numérique
                if len(texte_page.strip()) > 50:
                    texte_complet += texte_page + "\n"
                # Si la page est vide (probablement un scan/image)
                else:
                    zoom = 2.0
                    matrice = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrice)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    texte_complet += pytesseract.image_to_string(img, lang='fra') + "\n"
            return texte_complet
        except Exception as e:
            print(f"❌ Erreur PDF ({os.path.basename(chemin_fichier)}) : {e}")
            return ""
            
    elif ext in ['png', 'jpg', 'jpeg']:
        try:
            return pytesseract.image_to_string(Image.open(chemin_fichier), lang='fra')
        except Exception as e:
            print(f"❌ Erreur Image ({os.path.basename(chemin_fichier)}) : {e}")
            return ""
            
    return ""