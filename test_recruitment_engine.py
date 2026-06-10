import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from extractor import extraire_texte
from search_engine import nettoyer_texte

class TestRecruitmentEngine(unittest.TestCase):

    def test_nettoyer_texte(self):
        """Test basic French text cleaning and key word retention."""
        texte_brut = "Un developpeur Python tres experimente avec Django."
        texte_propre = nettoyer_texte(texte_brut)
        
        # Check that common stopwords are removed and core keywords are lowercased and retained
        self.assertNotIn("un", texte_propre.split())
        self.assertNotIn("avec", texte_propre.split())
        self.assertTrue("python" in texte_propre or "developpeur" in texte_propre or "django" in texte_propre)

    def test_extraire_texte_txt(self):
        """Test text extraction from a raw text file."""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write("Ceci est un CV de test pour Python.")
            temp_file_path = temp_file.name

        try:
            extracted = extraire_texte(temp_file_path)
            self.assertEqual(extracted.strip(), "Ceci est un CV de test pour Python.")
        finally:
            os.remove(temp_file_path)

    @patch('extractor.fitz.open')
    def test_extraire_texte_pdf_native(self, mock_fitz_open):
        """Test native PDF text extraction when text is found."""
        # Setup mock document and pages
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Compétences : Python, Flask, SQL, Django, Angular, Cyber, Machine Learning, Data Science, et beaucoup d'autres compétences techniques nécessaires pour ce test."
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        extracted = extraire_texte("fake_cv.pdf")
        self.assertIn("Python", extracted)
        self.assertIn("Django", extracted)

    @patch('extractor.fitz.open')
    @patch('extractor.pytesseract.image_to_string')
    @patch('extractor.Image.open')
    def test_extraire_texte_pdf_ocr(self, mock_image_open, mock_ocr, mock_fitz_open):
        """Test OCR-based PDF text extraction when no native text is found."""
        # Setup mock page with no native text
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "  # Empty native text
        
        # Mock rendering of the page as image
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b"fake_png_bytes"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        # Mock OCR output
        mock_ocr.return_value = "Texte extrait par OCR"

        extracted = extraire_texte("scanned_cv.pdf")
        self.assertEqual(extracted.strip(), "Texte extrait par OCR")

if __name__ == '__main__':
    unittest.main()
