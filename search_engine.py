import os
import glob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from extractor import extraire_texte, extraire_contacts  # ← ajout extraire_contacts

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("📥 Téléchargement du modèle de langue français requis...")
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

def nettoyer_texte(texte):
    """Lemmatise et nettoie le texte pour le moteur de correspondance"""
    doc = nlp(texte.lower())
    mots = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) > 1]
    return " ".join(mots)

def main():
    DOSSIER_CV = "./cv_folder"

    if not os.path.exists(DOSSIER_CV):
        os.makedirs(DOSSIER_CV)
        print(f"📁 Dossier '{DOSSIER_CV}' créé. Ajoute tes CV dedans et relance.")
        return

    fichiers = []
    for ext in ('*.txt', '*.pdf', '*.png', '*.jpg', '*.jpeg'):
        fichiers.extend(glob.glob(os.path.join(DOSSIER_CV, ext)))

    if not fichiers:
        print(f"⚠️ Aucun CV trouvé dans le dossier '{DOSSIER_CV}'.")
        return

    print(f"📚 Ingestion locale de {len(fichiers)} documents...")

    noms_candidats = []
    cv_nettoyes = []
    contacts_candidats = []  # ← nouveau

    for chemin in fichiers:
        nom = os.path.basename(chemin)
        brut = extraire_texte(chemin)
        if brut.strip():
            cv_nettoyes.append(nettoyer_texte(brut))
            noms_candidats.append(nom)
            contacts_candidats.append(extraire_contacts(brut))  # ← nouveau

    if not cv_nettoyes:
        print("❌ Aucun texte n'a pu être extrait des fichiers trouvés ou tous les documents sont vides.")
        return

    print("✅ Indexation terminée. Prêt pour les requêtes dynamiques.")

    while True:
        print("\n" + "="*60)
        requete = input("🕵️ Entrez vos mots-clés (ex: 'Stage Cyber Python') [ou 'q' pour quitter] : ")
        print("="*60)

        if requete.lower() == 'q':
            print("Fermeture du moteur de recrutement.")
            break

        if not requete.strip():
            continue

        requete_propre = nettoyer_texte(requete)
        if not requete_propre.strip():
            print("⚠️ Tous vos mots-clés sont des mots vides (stop words) en français. Saisissez des termes plus spécifiques.")
            continue

        vectoriseur = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        matrice_cv = vectoriseur.fit_transform(cv_nettoyes)
        vecteur_requete = vectoriseur.transform([requete_propre])

        scores = cosine_similarity(vecteur_requete, matrice_cv).flatten()

        # ← on zippe les contacts avec le reste
        classement = list(zip(noms_candidats, scores, contacts_candidats))
        classement.sort(key=lambda x: x[1], reverse=True)

        resultats_valides = [
            (rang, candidat, round(score * 100, 1), contacts)
            for rang, (candidat, score, contacts) in enumerate(classement, 1)
            if score > 0
        ]

        if resultats_valides:
            print(f"\n📊 --- CLASSEMENT DES CANDIDATS POUR : '{requete}' ---")

            col_rang     = 6
            col_candidat = max(len(c) for _, c, _, _ in resultats_valides)
            col_candidat = max(col_candidat, 12)
            col_score    = 16
            col_contact  = 35  # ← nouveau

            sep = f"+{'-'*(col_rang+2)}+{'-'*(col_candidat+2)}+{'-'*(col_score+2)}+{'-'*(col_contact+2)}+"
            print(sep)
            print(f"| {'Rang'.ljust(col_rang)} | {'Candidat'.ljust(col_candidat)} | {'Correspondance'.ljust(col_score)} | {'Contact'.ljust(col_contact)} |")
            print(sep)

            for rang, candidat, score, contacts in resultats_valides:  # ← ajout contacts
                str_rang    = str(rang)
                str_score   = f"{score}%"
                email = contacts.get("email") or ""
                tel   = contacts.get("telephone") or ""
                contact_str = " | ".join(filter(None, [email, tel])) or "—"
                print(f"| {str_rang.ljust(col_rang)} | {candidat.ljust(col_candidat)} | {str_score.rjust(col_score)} | {contact_str.ljust(col_contact)} |")

            print(sep)
        else:
            print("❌ Aucun CV ne correspond à ces critères.")

if __name__ == "__main__":
    main()