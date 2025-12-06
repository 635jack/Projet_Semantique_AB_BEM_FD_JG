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

Ce projet vise à développer un système complet d'extraction de connaissances à partir de textes, en utilisant un dataset **NER (Named Entity Recognition)** de 2221 phrases annotées.

## Partie A : Preprocessing et Représentation Textuelle

### Dataset

- **Source** : NER Dataset (Named Entity Recognition)
- **Total** : 2221 phrases
- **Split** : Train (70%), Dev (15%), Test (15%)
- **Colonnes** : `id`, `words`, `ner_tags`, `text`

### Pipeline Complet

1. **Nettoyage** : Lowercase, suppression caractères spéciaux, normalisation espaces
2. **Lemmatization** : spaCy `en_core_web_sm`
3. **Vectorisation TF-IDF** : scikit-learn (3000 features, bigrammes)

### Fichiers Principaux

- **`PartieA_Preprocessing.ipynb`** : Notebook Jupyter complet
- **`data.csv`** : Dataset source (2221 phrases)
- **`preprocessed_data/`** : Données exportées (CSV, NPZ, Pickle, JSON)

### Données Exportées

| Fichier | Description | Taille estimée |
|---------|-------------|----------------|
| `train_preprocessed.csv` | Textes TRAIN prétraités (1554 phrases) | ~2 MB |
| `dev_preprocessed.csv` | Textes DEV prétraités (333 phrases) | ~0.5 MB |
| `test_preprocessed.csv` | Textes TEST prétraités (334 phrases) | ~0.5 MB |
| `tfidf_matrix.npz` | Matrice TF-IDF TRAIN (sparse) | ~0.5 MB |
| `tfidf_matrix_dev.npz` | Matrice TF-IDF DEV (sparse) | ~0.1 MB |
| `tfidf_matrix_test.npz` | Matrice TF-IDF TEST (sparse) | ~0.1 MB |
| `tfidf_vectorizer.pkl` | Vectoriseur TF-IDF entraîné | ~0.2 MB |
| `tfidf_feature_names.npy` | Noms des 3000 features | ~0.05 MB |
| `metadata.json` | Métadonnées du preprocessing | ~0.01 MB |

### Statistiques

- **Phrases train** : 1554 (70%)
- **Phrases dev** : 333 (15%)
- **Phrases test** : 334 (15%)
- **Features TF-IDF** : 3000
- **Pipeline** : Nettoyage → Lemmatization → TF-IDF

## Utilisation

### Sur Google Colab (Recommandé)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/635jack/Projet_Semantique_AB_BEM_FD_JG/blob/master/PartieA_Preprocessing.ipynb)

1. Cliquez sur le badge ci-dessus
2. Uploadez le fichier `data.csv` dans Colab
3. Exécutez toutes les cellules : `Runtime → Run all`
4. Téléchargez les fichiers exportés depuis `preprocessed_data/`

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
df_train = pd.read_csv('preprocessed_data/train_preprocessed.csv')
df_dev = pd.read_csv('preprocessed_data/dev_preprocessed.csv')

# Charger les matrices TF-IDF
tfidf_train = load_npz('preprocessed_data/tfidf_matrix.npz')
tfidf_dev = load_npz('preprocessed_data/tfidf_matrix_dev.npz')

# Charger le vectoriseur
with open('preprocessed_data/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Charger les noms de features
feature_names = np.load('preprocessed_data/tfidf_feature_names.npy')
```

## Liens

- **GitHub Repository** : [https://github.com/635jack/Projet_Semantique_AB_BEM_FD_JG](https://github.com/635jack/Projet_Semantique_AB_BEM_FD_JG)
- **Google Colab** : [Ouvrir le notebook](https://colab.research.google.com/github/635jack/Projet_Semantique_AB_BEM_FD_JG/blob/master/PartieA_Preprocessing.ipynb)

## Licence

Projet académique - Université Paris Cité 2025
