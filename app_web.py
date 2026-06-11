import os
import glob
import threading
import tkinter as tk
from tkinter import filedialog
from flask import Flask, render_template, request, jsonify
from extractor import extraire_texte, extraire_contacts
from search_engine import nettoyer_texte
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

app = Flask(__name__)

etat = {
    "noms": [],
    "cv_nettoyes": [],
    "contacts": [],
    "vectoriseur": None,
    "matrice": None,
    "statut": "En attente de CV"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/choisir-dossier", methods=["GET"])
def choisir_dossier():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', True)
    dossier = filedialog.askdirectory(title="Sélectionner le dossier des CV")
    root.destroy()
    return jsonify({"dossier": dossier})

@app.route("/indexer", methods=["POST"])
def indexer():
    dossier = request.json.get("dossier")
    if not os.path.exists(dossier):
        return jsonify({"erreur": "Dossier introuvable"}), 400

    fichiers = []
    for ext in ('*.txt', '*.pdf', '*.png', '*.jpg', '*.jpeg', '*.docx'):
        fichiers.extend(glob.glob(os.path.join(dossier, ext)))

    if not fichiers:
        return jsonify({"erreur": "Aucun CV trouvé dans ce dossier"}), 400

    etat["noms"] = []
    etat["cv_nettoyes"] = []
    etat["contacts"] = []

    for chemin in fichiers:
        brut = extraire_texte(chemin)
        if brut.strip():
            etat["cv_nettoyes"].append(nettoyer_texte(brut))
            etat["noms"].append(os.path.basename(chemin))
            etat["contacts"].append(extraire_contacts(brut))

    if not etat["cv_nettoyes"]:
        return jsonify({"erreur": "Aucun texte extrait"}), 400

    etat["vectoriseur"] = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    etat["matrice"] = etat["vectoriseur"].fit_transform(etat["cv_nettoyes"])
    etat["statut"] = f"{len(etat['cv_nettoyes'])} CV indexés"

    return jsonify({"message": etat["statut"]})

@app.route("/rechercher", methods=["POST"])
def rechercher():
    if not etat["cv_nettoyes"]:
        return jsonify({"erreur": "Aucun CV indexé"}), 400

    data = request.json
    requete = data.get("requete", "").strip()
    nb_max  = int(data.get("nb_max", 20))

    if not requete:
        return jsonify({"erreur": "Requête vide"}), 400

    requete_propre = nettoyer_texte(requete)
    if not requete_propre.strip():
        return jsonify({"erreur": "Mots-clés trop génériques"}), 400

    vecteur = etat["vectoriseur"].transform([requete_propre])
    scores  = cosine_similarity(vecteur, etat["matrice"]).flatten()

    classement = sorted(
        zip(etat["noms"], scores, etat["contacts"]),
        key=lambda x: x[1], reverse=True
    )

    resultats = []
    for rang, (nom, score, contacts) in enumerate(classement, 1):
        if score <= 0 or rang > nb_max:
            break
        email = contacts.get("email") or ""
        tel   = contacts.get("telephone") or ""
        contact_str = " | ".join(filter(None, [email, tel])) or "—"
        resultats.append({
            "rang": rang,
            "candidat": nom,
            "score": round(score * 100, 1),
            "contact": contact_str
        })

    return jsonify({"resultats": resultats})

def ouvrir_navigateur():
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    threading.Timer(1.5, ouvrir_navigateur).start()
    app.run(debug=False, port=5000)