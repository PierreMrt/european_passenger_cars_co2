import streamlit as st

st.title("Analyse des émissions de CO₂ des voitures européennes")

exploration = st.Page("pages/exploration.py", title="Exploration des données", icon="📊")
results = st.Page("pages/results.py", title="Analyse des modèles", icon="📈")
predict = st.Page("pages/predict.py", title="Prédiction du modèle", icon="⭐")


pg = st.navigation([exploration, results, predict])
pg.run()
