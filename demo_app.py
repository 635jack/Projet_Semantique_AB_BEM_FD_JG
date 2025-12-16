import streamlit as st
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import re
import pickle
import altair as alt

# Page Configuration
st.set_page_config(
    page_title="Knowledge Extraction Demo",
    page_icon="🧠",
    layout="wide"
)

# Load Data (Cached)
@st.cache_data
def load_data():
    if os.path.exists('data_preprocessed.csv'):
        return pd.read_csv('data_preprocessed.csv')
    elif os.path.exists('data.csv'):
        return pd.read_csv('data.csv')
    return None

# Load NLP Model (Cached)
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Load TF-IDF Vectorizer (Cached)
@st.cache_resource
def load_tfidf():
    if os.path.exists('tfidf_vectorizer.pkl'):
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            return pickle.load(f)
    return None

nlp = load_nlp()
vectorizer = load_tfidf()
df = load_data()

# Helper Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text(text):
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    lemmatized = " ".join([token.lemma_ for token in doc])
    pos_tags = [(token.text, token.pos_) for token in doc]
    return cleaned, lemmatized, pos_tags, doc

# UI Layout
st.title("🧠 Projet Semantique: Knowledge Extraction")
st.markdown("**Partie A: Preprocessing & Analyse Statistique**")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Explorateur de Données", "📈 Statistiques & Visualisations", "🧪 Démo Interactive NLP"])

# Tab 1: Data Explorer
with tab1:
    st.header("Exploration du Corpus")
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", len(df))
        
        st.subheader("Aperçu des Données")
        st.dataframe(df.head(10), width="stretch")
        
        st.subheader("Recherche")
        search_query = st.text_input("Rechercher un mot dans le corpus")
        if search_query:
            results = df[df['text'].str.contains(search_query, case=False, na=False)]
            st.write(f"{len(results)} résultats trouvés.")
            st.dataframe(results[['id', 'text', 'lemmatized_text']].head(5), width="stretch")
    else:
        st.error("Fichier de données non trouvé.")

# Tab 2: Statistics & Visualizations
with tab2:
    st.header("Analyse Statistique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Nuage de Mots")
        if os.path.exists('stats_wordcloud.png'):
            st.image('stats_wordcloud.png', width="stretch")
        else:
            st.info("Image stats_wordcloud.png non trouvée.")
            
        st.subheader("Distribution POS")
        if os.path.exists('pos_distribution.png'):
            st.image('pos_distribution.png', width="stretch")
        else:
            st.info("Image pos_distribution.png non trouvée.")

    with col2:
        st.subheader("Impact du Preprocessing")
        if os.path.exists('stats_preprocessing_impact.png'):
            st.image('stats_preprocessing_impact.png', width="stretch")
        else:
            st.info("Image stats_preprocessing_impact.png non trouvée.")
            
        st.subheader("Top Termes TF-IDF")
        if os.path.exists('tfidf_top_terms.png'):
            st.image('tfidf_top_terms.png', width="stretch")
        else:
            st.info("Image tfidf_top_terms.png non trouvée.")

# Tab 3: Interactive Demo
with tab3:
    st.header("Pipeline de Preprocessing en Temps Réel")
    st.markdown("Testez le pipeline sur votre propre texte.")
    
    user_input = st.text_area("Entrez une phrase en anglais:", "Quantum computers utilize quantum bits to perform quantum calculations much faster than classical computers can perform classical calculations.")
    
    if st.button("Traiter"):
        cleaned, lemmatized, pos_tags, doc = process_text(user_input)
        
        st.subheader("1. Nettoyage")
        st.code(cleaned, language="text")
        
        st.subheader("2. Lemmatization (Mots modifiés en évidence)")
        
        # Highlight lemmatization changes
        lemma_html = ""
        for token in doc:
            if token.text != token.lemma_:
                # Change detected: Highlight
                lemma_html += f'<span style="background-color: #FFF9C4; color: #F57F17; padding: 2px 4px; border-radius: 4px; margin-right: 4px; border: 1px solid #FBC02D;" title="{token.text} → {token.lemma_}">{token.lemma_}</span>'
            else:
                # No change
                lemma_html += f'<span style="padding: 2px 4px; margin-right: 4px; color: #424242;">{token.lemma_}</span>'
        
        st.markdown(lemma_html, unsafe_allow_html=True)
        st.caption("Légende : Les mots surlignés en jaune ont été modifiés par la lemmatisation (ex: 'theories' → 'theory').")
        
        st.subheader("3. POS Tagging")
        # Visualisation colorée des POS tags (Palette optimisée pour Vidéoprojecteur - Haut Contraste)
        pos_html = ""
        colors = {
            "NOUN": "#1565C0",   # Bleu foncé
            "VERB": "#2E7D32",   # Vert foncé
            "ADJ": "#EF6C00",    # Orange vif
            "ADV": "#7B1FA2",    # Violet
            "PRON": "#00838F",   # Cyan foncé
            "DET": "#424242",    # Gris foncé
            "ADP": "#4E342E",    # Marron
            "PROPN": "#C62828"   # Rouge
        }
        
        for word, tag in pos_tags:
            color_bg = colors.get(tag, "#616161")
            pos_html += f'<span style="background-color: {color_bg}; color: white; padding: 4px 8px; border-radius: 4px; margin-right: 6px; font-weight: bold; display: inline-block; margin-bottom: 4px;">{word} <span style="font-size: 0.75em; opacity: 0.8; margin-left: 4px; text-transform: uppercase;">{tag}</span></span>'
        
        st.markdown(pos_html, unsafe_allow_html=True)
        
        # Legend
        st.markdown("---")
        st.caption("Légende: " + ", ".join([f"{k}" for k in colors.keys()]))

        # TF-IDF Analysis
        if vectorizer:
            st.markdown("---")
            st.subheader("4. Analyse TF-IDF (Mots Clés)")
            
            try:
                # Transform input
                if not lemmatized.strip():
                     st.warning("Le texte est vide après nettoyage.")
                else:
                    tfidf_vector = vectorizer.transform([lemmatized])
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # Extract non-zero values
                    coo_matrix = tfidf_vector.tocoo()
                    tuples = zip(coo_matrix.col, coo_matrix.data)
                    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
                    
                    # Display results
                    if sorted_items:
                        st.write("Termes les plus significatifs (selon le modèle entraîné) :")
                        
                        top_terms = []
                        for idx, score in sorted_items[:5]: # Top 5 terms
                            term = feature_names[idx]
                            top_terms.append({"Terme": term, "Score TF-IDF": round(score, 4)})
                        
                        df_tfidf = pd.DataFrame(top_terms)
                        
                        # Création d'un graphique à barres horizontal avec Altair
                        chart = alt.Chart(df_tfidf).mark_bar().encode(
                            x=alt.X('Score TF-IDF', title='Score TF-IDF'),
                            y=alt.Y('Terme', sort='-x', title='Terme'),
                            color=alt.Color('Score TF-IDF', scale=alt.Scale(scheme='blues'), legend=None),
                            tooltip=['Terme', 'Score TF-IDF']
                        ).properties(
                            title='Top 5 Mots-Clés TF-IDF',
                            height=300
                        ).configure_axis(
                            labelFontSize=14,
                            titleFontSize=16
                        ).configure_title(
                            fontSize=18
                        )
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Optionnel : Afficher aussi les données brutes dans un expander
                        with st.expander("Voir les données brutes"):
                            st.table(df_tfidf)
                    else:
                        st.warning("Aucun terme connu du vocabulaire TF-IDF n'a été trouvé dans cette phrase.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse TF-IDF: {str(e)}")
        else:
            st.info("Modèle TF-IDF non chargé (fichier 'tfidf_vectorizer.pkl' manquant).")

# Sidebar
st.sidebar.title("Info Projet")
st.sidebar.info("Master 2 VMI - Université Paris Cité\n\nPartie A: Analyse et Preprocessing d'un corpus NER.")
if df is not None and 'text_length' in df.columns:
     st.sidebar.metric("Longueur Moyenne (chars)", int(df['text_length'].mean()))
