# Projet Knowledge Extraction from Unstructured Text

**Université Paris Cité - Master 2 VMI**  
**Cours :** IFLCE085 - Recherche et extraction sémantique à partir de texte  
**Professeur :** Salima Benbernou

## Équipe

- **Partie A (Preprocessing)** : Jacques Gastebois
- **Partie B** : Boutayna EL MOUJAOUID
- **Partie C** : Franz Dervis
- **Partie D** : Aya Benkabour

## Description du Projet

Ce projet vise à développer un système complet d'extraction de connaissances à partir de textes scientifiques non structurés, en utilisant le dataset **SciREX** (Scientific Information Extraction).

## Partie A : Preprocessing et Représentation Textuelle

### Pipeline Complet

1. **Nettoyage** : Lowercase, suppression caractères spéciaux, normalisation espaces
2. **Tokenization** : NLTK `word_tokenize`
3. **POS Tagging** : NLTK `pos_tag`
4. **Lemmatization** : spaCy `en_core_web_sm` (306 documents)
5. **Vectorisation TF-IDF** : scikit-learn (matrice 306×5000)

### Fichiers Principaux

- **`PartieA_Preprocessing.ipynb`** : Notebook Jupyter complet avec tous les résultats
- **`Rapport_PartieA.pdf`** : Rapport technique LaTeX (10 pages)
- **`Rapport_PartieA.tex`** : Source LaTeX du rapport
- **`preprocessed_data/`** : Données exportées (CSV, NPZ, Pickle, JSON)
- **`release_data/`** : Dataset SciREX original

### Données Exportées

| Fichier | Description | Taille |
|---------|-------------|--------|
| `train_preprocessed.csv` | Textes bruts, nettoyés et lemmatisés | 25.7 MB |
| `tfidf_matrix.npz` | Matrice TF-IDF (sparse) | 1.97 MB |
| `tfidf_vectorizer.pkl` | Vectoriseur TF-IDF entraîné | 0.19 MB |
| `tfidf_feature_names.npy` | Noms des 5000 features | 0.06 MB |
| `correspondence_dict.json` | Métadonnées et correspondances | 0.01 MB |

### Statistiques

- **Documents train** : 306 (tous lemmatisés)
- **Documents dev** : 66
- **Documents test** : 66
- **Features TF-IDF** : 5000
- **Densité matrice** : 23.7%

## Utilisation

### Sur Google Colab (Recommandé)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/635jack/Projet_Semantique_AB_BEM_FD_JG/blob/master/PartieA_Preprocessing.ipynb)

1. Cliquez sur le badge ci-dessus
2. Exécutez toutes les cellules : `Runtime → Run all`
3. Les données seront automatiquement téléchargées et traitées

### En Local

```bash
# Cloner le repository
git clone https://github.com/635jack/Projet_Semantique_AB_BEM_FD_JG

# Installer Jupyter
pip install jupyter

# Lancer le notebook
cd Projet_Semantique_AB_BEM_FD_JG
jupyter notebook PartieA_Preprocessing.ipynb
```

**Note** : Les dépendances sont installées automatiquement dans le notebook.

## Chargement des Données (Partie B)

```python
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz

# Charger les textes prétraités
df = pd.read_csv('preprocessed_data/train_preprocessed.csv')

# Charger la matrice TF-IDF
tfidf_matrix = load_npz('preprocessed_data/tfidf_matrix.npz')

# Charger le vectoriseur
with open('preprocessed_data/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Charger les noms de features
feature_names = np.load('preprocessed_data/tfidf_feature_names.npy')
```

## Liens

- **GitHub Repository** : [https://github.com/635jack/Projet_Semantique_AB_BEM_FD_JG](https://github.com/635jack/Projet_Semantique_AB_BEM_FD_JG)
- **Google Colab** : [Ouvrir le notebook](https://colab.research.google.com/github/635jack/Projet_Semantique_AB_BEM_FD_JG/blob/master/PartieA_Preprocessing.ipynb)
- **Dataset SciREX** : [https://github.com/allenai/SciREX](https://github.com/allenai/SciREX)

## Documentation

Consultez le **[Rapport_PartieA.pdf](Rapport_PartieA.pdf)** pour une documentation technique complète incluant :
- Méthodologie détaillée
- Justification des choix techniques
- Exemples de code
- Instructions d'utilisation pour la Partie B

## Licence

Projet académique - Université Paris Cité 2025
