import os
import platform
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from docx import Document

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

    elif ext == 'docx':
        try:
            doc = Document(chemin_fichier)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"❌ Erreur DOCX ({os.path.basename(chemin_fichier)}) : {e}")
            return "" 
        
    return ""

def extraire_contacts(texte):
    """Extrait email, téléphone et LinkedIn depuis le texte brut d'un CV"""
    
    # Email
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', texte)
    
    # Téléphone (formats FR : 06, 07, +33, avec espaces/tirets/points)
    telephones = re.findall(
        r'(?:(?:\+|00)33[\s.-]?|0)[1-9](?:[\s.-]?\d{2}){4}', texte
    )
    
    # LinkedIn
    linkedin = re.findall(
        r'(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9\-]+)', texte
    )
    
    return {
        "email":     emails[0] if emails else None,
        "telephone": telephones[0] if telephones else None,
        "linkedin":  linkedin[0] if linkedin else None,
    }