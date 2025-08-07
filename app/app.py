import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger ton dataset et ton modèle
data = pd.read_csv("datas/DataSet_Emails.csv")
pipeline = joblib.load("models/pipeline_vect_model.pkl")

st.title("Application d'analyse et de prédiction de texte")

# === Section EDA ===
st.header("Exploration des données")

if st.checkbox("Afficher les 5 premières lignes du dataset"):
    st.write(data.head())

# Exemple : distribution d'une colonne 'label'
fig, ax = plt.subplots()
sns.countplot(x="label", data=data, ax=ax)
st.pyplot(fig)
data = data.dropna(subset=["text"])
# WordCloud sur la colonne 'text'
st.subheader("Nuage de mots")
texte = " ".join(data["text"].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(texte)
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

email_text = st.text_area("Saisissez le contenu de l'email :", height=200)

if st.button("Analyser"):
    if email_text.strip():
        def nettoyer(txt):
            return re.sub(r"[^a-zA-Z0-9À-ÖØ-öø-ÿ\s]", "", txt.lower())
        email_clean = nettoyer(email_text)

        prediction = pipeline.predict([email_clean])[0]

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([email_clean])[0]
            st.info(f"Ham: {round(proba[0]*100,2)}% | Spam: {round(proba[1]*100,2)}%")

        if prediction == "spam":
            st.error("SPAM détecté !")
        else:
            st.success("HAM (non-spam)")









# Optionnel : générer un word cloud à partir du texte saisi
if email_text.strip() != "":
    st.subheader("Word Cloud de votre email")
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(email_text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)