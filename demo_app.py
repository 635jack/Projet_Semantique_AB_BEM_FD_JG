import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import re
import altair as alt
from collections import Counter
import streamlit.components.v1 as components
import json
import pickle

# Import NLTK for NLP processing (pure Python, works on cloud)
import nltk

# Download required NLTK data (cached)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except:
        return False

NLTK_AVAILABLE = download_nltk_data()

# Import NLTK components after download
if NLTK_AVAILABLE:
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag

# Page Configuration
st.set_page_config(
    page_title="Knowledge Extraction Demo",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for hover zoom effect on images
st.markdown("""
<style>
    /* Hover zoom effect for images */
    img {
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        cursor: pointer;
    }
    img:hover {
        transform: scale(1.15);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        z-index: 100;
    }
    
    /* Hover effect for iframes (embedded charts) */
    iframe {
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    iframe:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load Data (Cached)
@st.cache_data
def load_data():
    if os.path.exists('data_preprocessed.csv'):
        return pd.read_csv('data_preprocessed.csv')
    elif os.path.exists('data.csv'):
        return pd.read_csv('data.csv')
    return None

# Load TF-IDF Vectorizer (Cached) - Optional
@st.cache_resource
def load_tfidf():
    if os.path.exists('tfidf_vectorizer.pkl'):
        try:
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

# Get Word Frequencies for Word Cloud
@st.cache_data
def get_word_frequencies(dataframe, top_n=40):
    if dataframe is None or 'lemmatized_text' not in dataframe.columns:
        return []
    all_words = ' '.join(dataframe['lemmatized_text'].dropna()).split()
    # Filter out short words and common stopwords
    stopwords = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'}
    filtered_words = [w for w in all_words if len(w) > 2 and w not in stopwords]
    word_counts = Counter(filtered_words).most_common(top_n)
    return [{"tag": word, "count": count} for word, count in word_counts]

vectorizer = load_tfidf()
df = load_data()

# Initialize NLTK lemmatizer
if NLTK_AVAILABLE:
    lemmatizer = WordNetLemmatizer()

# Helper Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text_nltk(text):
    """Process text using NLTK (cloud-compatible)"""
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(cleaned)
    
    # POS Tagging (NLTK uses Penn Treebank tags)
    pos_tags_raw = pos_tag(tokens)
    
    # Convert Penn Treebank to Universal POS tags (for display)
    penn_to_universal = {
        'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
        'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
        'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
        'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
        'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
        'DT': 'DET', 'WDT': 'DET',
        'IN': 'ADP', 'TO': 'ADP',
        'CC': 'CCONJ', 'CD': 'NUM'
    }
    pos_tags = [(word, penn_to_universal.get(tag, 'X')) for word, tag in pos_tags_raw]
    
    # Lemmatization
    lemmas = []
    for word, tag in pos_tags_raw:
        # Map POS tag to WordNet format
        if tag.startswith('V'):
            wn_tag = 'v'
        elif tag.startswith('J'):
            wn_tag = 'a'
        elif tag.startswith('R'):
            wn_tag = 'r'
        else:
            wn_tag = 'n'
        lemmas.append((word, lemmatizer.lemmatize(word, pos=wn_tag)))
    
    lemmatized = " ".join([lemma for _, lemma in lemmas])
    
    return cleaned, lemmatized, pos_tags, lemmas

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
        st.subheader("Nuage de Mots Interactif")
        word_data = get_word_frequencies(df)
        if word_data:
            word_data_json = json.dumps(word_data)
            wordcloud_html = f"""
            <div id="chartdiv" style="width: 100%; height: 400px;"></div>
            <script src="https://cdn.amcharts.com/lib/4/core.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/plugins/wordCloud.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>
            <script>
            am4core.useTheme(am4themes_animated);
            var chart = am4core.create("chartdiv", am4plugins_wordCloud.WordCloud);
            var series = chart.series.push(new am4plugins_wordCloud.WordCloudSeries());
            series.accuracy = 4;
            series.step = 15;
            series.rotationThreshold = 0.7;
            series.maxCount = 200;
            series.minWordLength = 2;
            series.labels.template.tooltipText = "{{word}}: {{value}}";
            series.fontFamily = "Courier New";
            series.maxFontSize = am4core.percent(30);
            series.dataFields.word = "tag";
            series.dataFields.value = "count";
            series.colors = new am4core.ColorSet();
            series.colors.passOptions = {{}};
            series.data = {word_data_json};
            </script>
            """
            components.html(wordcloud_html, height=420)
        else:
            st.info("Image stats_wordcloud.png non trouvée.")
            
        st.subheader("Distribution POS (Pie Chart)")
        # Load POS data from JSON
        pos_data = []
        if os.path.exists('pos_tfidf_statistics.json'):
            with open('pos_tfidf_statistics.json', 'r') as f:
                stats = json.load(f)
                pos_data = stats.get('pos_analysis', {}).get('top_5_pos', [])
        
        if pos_data:
            pos_data_json = json.dumps(pos_data)
            pie_chart_html = f"""
            <div id="piechart" style="width: 100%; height: 350px;"></div>
            <script src="https://cdn.amcharts.com/lib/4/core.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>
            <script>
            am4core.useTheme(am4themes_animated);
            var chart = am4core.create("piechart", am4charts.PieChart);
            chart.data = {pos_data_json};
            var pieSeries = chart.series.push(new am4charts.PieSeries());
            pieSeries.dataFields.value = "percentage";
            pieSeries.dataFields.category = "tag";
            pieSeries.slices.template.tooltipText = "{{category}}: {{value}}%";
            pieSeries.slices.template.stroke = am4core.color("#fff");
            pieSeries.slices.template.strokeWidth = 2;
            pieSeries.slices.template.strokeOpacity = 1;
            // Hover effect
            pieSeries.slices.template.states.getKey("hover").properties.scale = 1.1;
            pieSeries.slices.template.states.getKey("hover").properties.shiftRadius = 0.05;
            pieSeries.labels.template.text = "{{category}}";
            pieSeries.ticks.template.disabled = true;
            chart.legend = new am4charts.Legend();
            chart.legend.position = "right";
            </script>
            """
            components.html(pie_chart_html, height=380)
        else:
            st.info("Données POS non disponibles.")

    with col2:
        st.subheader("Top Termes TF-IDF (Bar Chart)")
        # Load TF-IDF data from JSON
        tfidf_data = []
        if os.path.exists('pos_tfidf_statistics.json'):
            with open('pos_tfidf_statistics.json', 'r') as f:
                stats = json.load(f)
                tfidf_data = stats.get('tfidf_analysis', {}).get('top_10_terms', [])
        
        if tfidf_data:
            tfidf_data_json = json.dumps(tfidf_data)
            bar_chart_html = f"""
            <div id="barchart" style="width: 100%; height: 350px;"></div>
            <script src="https://cdn.amcharts.com/lib/4/core.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/charts.js"></script>
            <script src="https://cdn.amcharts.com/lib/4/themes/animated.js"></script>
            <script>
            am4core.useTheme(am4themes_animated);
            var chart = am4core.create("barchart", am4charts.XYChart);
            chart.data = {tfidf_data_json};
            var categoryAxis = chart.yAxes.push(new am4charts.CategoryAxis());
            categoryAxis.dataFields.category = "term";
            categoryAxis.renderer.inversed = true;
            categoryAxis.renderer.grid.template.location = 0;
            categoryAxis.renderer.labels.template.fontSize = 12;
            var valueAxis = chart.xAxes.push(new am4charts.ValueAxis());
            valueAxis.min = 0;
            valueAxis.title.text = "Score TF-IDF";
            var series = chart.series.push(new am4charts.ColumnSeries());
            series.dataFields.valueX = "tfidf_score";
            series.dataFields.categoryY = "term";
            series.columns.template.tooltipText = "{{categoryY}}: {{valueX}}";
            series.columns.template.strokeOpacity = 0;
            // Hover effect
            series.columns.template.states.getKey("hover").properties.fillOpacity = 0.8;
            series.columns.template.adapter.add("fill", function(fill, target) {{
                return chart.colors.getIndex(target.dataItem.index);
            }});
            chart.cursor = new am4charts.XYCursor();
            </script>
            """
            components.html(bar_chart_html, height=380)
        else:
            st.info("Données TF-IDF non disponibles.")
        
        st.subheader("Impact du Preprocessing (Boxplot)")
        
        # Create interactive boxplot with Plotly
        if df is not None:
            import plotly.graph_objects as go
            
            # Calculate text lengths for each column
            text_lengths = {
                'Texte Original': df['text'].str.len().dropna().tolist(),
                'Texte Nettoyé': df['cleaned_text'].str.len().dropna().tolist() if 'cleaned_text' in df.columns else [],
                'Texte Lemmatisé': df['lemmatized_text'].str.len().dropna().tolist() if 'lemmatized_text' in df.columns else []
            }
            
            fig = go.Figure()
            
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
            for i, (name, data) in enumerate(text_lengths.items()):
                if data:
                    fig.add_trace(go.Box(
                        y=data,
                        name=name,
                        marker_color=colors[i],
                        boxmean='sd',
                        hovertemplate='<b>%{x}</b><br>Valeur: %{y}<extra></extra>'
                    ))
            
            fig.update_layout(
                title='Impact du Preprocessing sur la Longueur des Textes',
                yaxis_title='Longueur (caractères)',
                showlegend=True,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données non disponibles pour le boxplot.")

# Tab 3: Interactive Demo
with tab3:
    st.header("Pipeline de Preprocessing en Temps Réel")
    
    if not NLTK_AVAILABLE:
        st.warning("⚠️ **Démo interactive temporairement indisponible**")
        st.info("Chargement de NLTK en cours... Veuillez rafraîchir la page.")
    else:
        st.markdown("Testez le pipeline sur votre propre texte.")
        
        user_input = st.text_area("Entrez une phrase en anglais:", "Quantum computers utilize quantum bits to perform quantum calculations much faster than classical computers can perform classical calculations.")
        
        if st.button("Traiter"):
            cleaned, lemmatized, pos_tags, lemmas = process_text_nltk(user_input)
            
            st.subheader("1. Nettoyage")
            st.code(cleaned, language="text")
            
            st.subheader("2. Lemmatization (Mots modifiés en évidence)")
            
            # Highlight lemmatization changes (using NLTK format: list of (word, lemma) tuples)
            lemma_html = ""
            for original, lemma in lemmas:
                if original != lemma:
                    # Change detected: Highlight
                    lemma_html += f'<span style="background-color: #FFF9C4; color: #F57F17; padding: 2px 4px; border-radius: 4px; margin-right: 4px; border: 1px solid #FBC02D;" title="{original} → {lemma}">{lemma}</span>'
                else:
                    # No change
                    lemma_html += f'<span style="padding: 2px 4px; margin-right: 4px; color: #424242;">{lemma}</span>'
            
            st.markdown(lemma_html, unsafe_allow_html=True)
            st.caption("Légende : Les mots surlignés en jaune ont été modifiés par la lemmatisation (ex: 'computers' → 'computer').")
            
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
