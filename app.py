import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

# Import du moteur existant
from extractor import extraire_texte, extraire_contacts
from search_engine import nettoyer_texte, nlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Moteur de Recrutement Local")
        self.geometry("1000x700")
        self.resizable(True, True)
        self.configure(bg="#f5f5f5")

        self.dossier_cv = None
        self.fichiers_deposes = []
        self.noms_candidats = []
        self.cv_nettoyes = []
        self.contacts_candidats = []
        self.vectoriseur = None
        self.matrice_cv = None

        self._build_ui()
        self._activer_drag_drop()

    def _build_ui(self):
        # ── Titre
        tk.Label(self, text="🔍 Moteur de Recrutement Local",
                 font=("Helvetica", 18, "bold"), bg="#f5f5f5", fg="#2c2c2c"
                 ).pack(pady=(20, 5))

        # ── Zone CV
        frame_cv = tk.LabelFrame(self, text="  CV  ", font=("Helvetica", 11),
                                  bg="#f5f5f5", fg="#444", padx=10, pady=10)
        frame_cv.pack(fill="x", padx=20, pady=(5, 0))

        # Bouton dossier
        tk.Button(frame_cv, text="📁 Sélectionner un dossier",
                  command=self._choisir_dossier,
                  bg="#4a90d9", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=10, pady=6, cursor="hand2"
                  ).pack(side="left", padx=(0, 15))

        # Zone glisser-déposer
        self.drop_zone = tk.Label(
            frame_cv,
            text="🗂️  Glissez vos fichiers CV ici\n(PDF, DOCX, TXT, PNG, JPG)",
            bg="#e8f0fe", fg="#555", font=("Helvetica", 9),
            relief="groove", width=45, height=3, cursor="hand2"
        )
        self.drop_zone.pack(side="left", padx=(0, 15))

        # Statut chargement
        self.label_statut_cv = tk.Label(frame_cv, text="Aucun CV chargé",
                                         font=("Helvetica", 9), bg="#f5f5f5", fg="#888")
        self.label_statut_cv.pack(side="left")

        # ── Barre de progression
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=400)
        self.progress.pack(pady=(8, 0))

        # ── Recherche
        frame_recherche = tk.LabelFrame(self, text="  Recherche  ", font=("Helvetica", 11),
                                         bg="#f5f5f5", fg="#444", padx=10, pady=10)
        frame_recherche.pack(fill="x", padx=20, pady=(10, 0))

        tk.Label(frame_recherche, text="Mots-clés :",
                 font=("Helvetica", 10), bg="#f5f5f5").pack(side="left")

        self.entry_requete = tk.Entry(frame_recherche, font=("Helvetica", 11),
                                      width=40, relief="solid")
        self.entry_requete.pack(side="left", padx=10)
        self.entry_requete.bind("<Return>", lambda e: self._lancer_recherche())

        tk.Label(frame_recherche, text="Nb résultats :",
                 font=("Helvetica", 10), bg="#f5f5f5").pack(side="left", padx=(10, 5))

        self.entry_nb = tk.Entry(frame_recherche, font=("Helvetica", 11),
                                  width=5, relief="solid")
        self.entry_nb.insert(0, "20")
        self.entry_nb.pack(side="left", padx=(0, 10))

        tk.Button(frame_recherche, text="🔍 Rechercher",
                  command=self._lancer_recherche,
                  bg="#27ae60", fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", padx=10, pady=6, cursor="hand2"
                  ).pack(side="left")

        self.label_statut_recherche = tk.Label(frame_recherche, text="",
                                                font=("Helvetica", 9),
                                                bg="#f5f5f5", fg="#888")
        self.label_statut_recherche.pack(side="left", padx=10)

        # ── Tableau résultats
        frame_tableau = tk.Frame(self, bg="#f5f5f5")
        frame_tableau.pack(fill="both", expand=True, padx=20, pady=10)

        colonnes = ("rang", "candidat", "correspondance", "contact")
        self.tableau = ttk.Treeview(frame_tableau, columns=colonnes,
                                     show="headings", height=20)

        self.tableau.heading("rang",          text="Rang")
        self.tableau.heading("candidat",      text="Candidat")
        self.tableau.heading("correspondance",text="Correspondance %")
        self.tableau.heading("contact",       text="Contact")

        self.tableau.column("rang",          width=60,  anchor="center")
        self.tableau.column("candidat",      width=200, anchor="w")
        self.tableau.column("correspondance",width=150, anchor="center")
        self.tableau.column("contact",       width=400, anchor="w")

        # Scrollbar
        scrollbar = ttk.Scrollbar(frame_tableau, orient="vertical",
                                   command=self.tableau.yview)
        self.tableau.configure(yscrollcommand=scrollbar.set)

        self.tableau.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Couleurs alternées
        self.tableau.tag_configure("pair",   background="#ffffff")
        self.tableau.tag_configure("impair", background="#f0f4ff")

    def _activer_drag_drop(self):
        """Active le glisser-déposer si tkinterdnd2 est disponible"""
        try:
            from tkinterdnd2 import DND_FILES
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_zone.config(text="🗂️  Glissez vos fichiers CV ici\n(PDF, DOCX, TXT, PNG, JPG)")
        except Exception:
            self.drop_zone.config(text="🗂️  Glisser-déposer non disponible\nUtilisez le bouton dossier",
                                   fg="#aaa")

    def _on_drop(self, event):
        """Récupère les fichiers glissés-déposés"""
        fichiers = self.tk.splitlist(event.data)
        extensions = ('.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg')
        valides = [f for f in fichiers if f.lower().endswith(extensions)]
        if valides:
            self.fichiers_deposes.extend(valides)
            self.label_statut_cv.config(
                text=f"{len(self.fichiers_deposes)} fichier(s) déposé(s)", fg="#27ae60")
            threading.Thread(target=self._indexer, daemon=True).start()

    def _choisir_dossier(self):
        dossier = filedialog.askdirectory(title="Sélectionner le dossier des CV")
        if dossier:
            self.dossier_cv = dossier
            self.fichiers_deposes = []
            self.label_statut_cv.config(text=f"Chargement de {dossier}...", fg="#e67e22")
            threading.Thread(target=self._indexer, daemon=True).start()

    def _indexer(self):
        """Indexation des CV dans un thread séparé pour ne pas bloquer l'UI"""
        self.progress.start()
        self.label_statut_cv.config(text="Indexation en cours...", fg="#e67e22")

        fichiers = []

        if self.dossier_cv:
            for ext in ('*.txt', '*.pdf', '*.png', '*.jpg', '*.jpeg', '*.docx'):
                fichiers.extend(glob.glob(os.path.join(self.dossier_cv, ext)))
        if self.fichiers_deposes:
            fichiers.extend(self.fichiers_deposes)

        # Dédoublonnage
        fichiers = list(set(fichiers))

        if not fichiers:
            self.progress.stop()
            self.label_statut_cv.config(text="Aucun CV trouvé", fg="#e74c3c")
            return

        self.noms_candidats = []
        self.cv_nettoyes = []
        self.contacts_candidats = []

        for chemin in fichiers:
            brut = extraire_texte(chemin)
            if brut.strip():
                self.cv_nettoyes.append(nettoyer_texte(brut))
                self.noms_candidats.append(os.path.basename(chemin))
                self.contacts_candidats.append(extraire_contacts(brut))

        if self.cv_nettoyes:
            self.vectoriseur = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
            self.matrice_cv = self.vectoriseur.fit_transform(self.cv_nettoyes)

        self.progress.stop()
        self.label_statut_cv.config(
            text=f"✅ {len(self.cv_nettoyes)} CV indexés", fg="#27ae60")

    def _lancer_recherche(self):
        if not self.cv_nettoyes:
            messagebox.showwarning("Attention", "Aucun CV indexé. Chargez un dossier d'abord.")
            return

        requete = self.entry_requete.get().strip()
        if not requete:
            messagebox.showwarning("Attention", "Entrez des mots-clés.")
            return

        try:
            nb_max = int(self.entry_nb.get())
        except ValueError:
            nb_max = 20

        requete_propre = nettoyer_texte(requete)
        if not requete_propre.strip():
            messagebox.showwarning("Attention", "Mots-clés trop génériques.")
            return

        vecteur = self.vectoriseur.transform([requete_propre])
        scores  = cosine_similarity(vecteur, self.matrice_cv).flatten()

        classement = sorted(
            zip(self.noms_candidats, scores, self.contacts_candidats),
            key=lambda x: x[1], reverse=True
        )

        resultats = [
            (rang, nom, round(score * 100, 1), contacts)
            for rang, (nom, score, contacts) in enumerate(classement, 1)
            if score > 0
        ][:nb_max]

        # Vider le tableau
        for row in self.tableau.get_children():
            self.tableau.delete(row)

        if not resultats:
            self.label_statut_recherche.config(text="Aucun résultat", fg="#e74c3c")
            return

        for rang, nom, score, contacts in resultats:
            email = contacts.get("email") or ""
            tel   = contacts.get("telephone") or ""
            contact_str = " | ".join(filter(None, [email, tel])) or "—"
            tag = "pair" if rang % 2 == 0 else "impair"
            self.tableau.insert("", "end",
                                 values=(rang, nom, f"{score}%", contact_str),
                                 tags=(tag,))

        self.label_statut_recherche.config(
            text=f"{len(resultats)} résultat(s) trouvé(s)", fg="#27ae60")

if __name__ == "__main__":
    app = App()
    app.mainloop()