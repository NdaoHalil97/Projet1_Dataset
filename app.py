import streamlit as st
import numpy as np
import cv2
from descriptor import glcm, bitdesc
from distances import retrieve_similar_images
from PIL import Image

# Charger les signatures
signatures = np.load('signatures.npy')

# Titre de l'application
st.title('🖼️ Interface de Visualisation Avancée')

# Sélection des descripteurs
st.sidebar.header('🔍 Paramètres du Dataset')
descriptor = st.sidebar.radio(
    'Descriptor',
    ('GLCM', 'RFI')
)

# Sélection du type de distance
distance = st.sidebar.radio(
    'Distance',
    ('Manhattan', 'Euclidienne', 'Chebyshev', 'Autre')
)

# Nombre d'images
num_images = st.sidebar.number_input('Nombre d\'images', min_value=1, max_value=100, value=20)

# Section de téléchargement
st.header('📂 Télécharger les images')
uploaded_files = st.file_uploader("Choisissez des fichiers à télécharger", accept_multiple_files=True)

# Espace pour afficher les images téléchargées
if uploaded_files:
    st.header('🖼️ Images Téléchargées')
    cols = st.columns(4)  # Disposer les images en 4 colonnes
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 4]:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)

# Ajout de CSS personnalisé pour améliorer le design
st.markdown("""
    <style>
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    .css-12oz5g7 {
        padding: 2rem 1rem;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .css-16idsys p {
        color: #333333;
        font-size: 1.1rem;
    }
    .css-16idsys h1 {
        color: #5a5a5a;
    }
    .css-16idsys h2 {
        color: #4c4c4c;
        margin-bottom: 1rem;
    }
    .css-16idsys .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .css-16idsys .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Téléversement de l'image pour la recherche
uploaded_file = st.file_uploader("Téléverser une image pour la recherche", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Erreur lors du chargement de l'image. Veuillez réessayer avec une autre image.")
    else:
        st.image(image, caption='Image téléversée', use_column_width=True)
        
        # Sélection du nombre d'images similaires
        num_images_search = st.slider("Nombre d'images similaires à afficher", min_value=1, max_value=10, value=5)
        
        # Choix de la mesure de distance
        distance_measure = st.selectbox("Choisir la mesure de distance", ["euclidean", "manhattan", "chebyshev", "canberra"])
        
        # Choix du descripteur
        descriptor = st.selectbox("Choisir le descripteur", ["GLCM", "BiT"])

        # Extraction des caractéristiques de l'image requête
        if descriptor == "GLCM":
            query_features = glcm(image)
        else:
            query_features = bitdesc(image)
        
        # Rechercher les images similaires lorsque l'utilisateur clique sur le bouton
        if st.button("Rechercher des images similaires"):
            results = retrieve_similar_images(signatures, query_features, distance_measure, num_images_search)
            if results:
                st.header('🔍 Résultats des images similaires')
                cols = st.columns(5)
                for i, (img_path, score) in enumerate(results):
                    with cols[i % 5]:
                        img = cv2.imread(img_path)
                        if img is not None:
                            st.image(img, caption=f'Image similaire {i+1}', use_column_width=True)
                        else:
                            st.error(f"Erreur lors du chargement de l'image {img_path}")
            else:
                st.error("Aucune image similaire trouvée.")




