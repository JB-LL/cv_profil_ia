import os
import fitz
import easyocr

# Initialisation unique du lecteur OCR (chargé une seule fois)
print("🔄 Chargement du moteur OCR...")
lecteur_ocr = easyocr.Reader(['fr'], gpu=False)
print("✅ Moteur OCR prêt.")

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
                if len(texte_page.strip()) > 50:
                    texte_complet += texte_page + "\n"
                else:
                    zoom = 2.0
                    matrice = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrice)
                    img_bytes = pix.tobytes("png")
                    resultats = lecteur_ocr.readtext(img_bytes, detail=0)
                    texte_complet += " ".join(resultats) + "\n"
            return texte_complet
        except Exception as e:
            print(f"❌ Erreur PDF ({os.path.basename(chemin_fichier)}) : {e}")
            return ""

    elif ext in ['png', 'jpg', 'jpeg']:
        try:
            resultats = lecteur_ocr.readtext(chemin_fichier, detail=0)
            return " ".join(resultats)
        except Exception as e:
            print(f"❌ Erreur Image ({os.path.basename(chemin_fichier)}) : {e}")
            return ""

    elif ext == 'docx':
        try:
            from docx import Document
            doc = Document(chemin_fichier)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"❌ Erreur DOCX ({os.path.basename(chemin_fichier)}) : {e}")
            return ""

    return ""

import re

def extraire_contacts(texte):
    """Extrait email, téléphone et LinkedIn depuis le texte brut d'un CV"""

    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', texte)

    telephones = re.findall(
        r'(?:(?:\+|00)33[\s.-]?|0)[1-9](?:[\s.-]?\d{2}){4}', texte
    )

    linkedin = re.findall(
        r'(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9\-]+)', texte
    )

    return {
        "email":     emails[0] if emails else None,
        "telephone": telephones[0] if telephones else None,
        "linkedin":  linkedin[0] if linkedin else None,
    }