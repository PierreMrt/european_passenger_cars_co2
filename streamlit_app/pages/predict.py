import streamlit as st
import pandas as pd
from utils.data_loaders import load_model, get_input_features
from utils.model_utils import predict_emission, explain_prediction, prettify_feature_name
from utils.viz_tools import plot_shap_values

def run_predict_page():
    """
    Affiche la page de prédiction des émissions de CO₂ dans l'application Streamlit.
    Présente un formulaire pour saisir les caractéristiques du véhicule, effectue la prédiction,
    et affiche les importances des variables avec un graphique SHAP.
    """
    feature_labels = {
        "ec (cm3)": "Cylindrée (cm³)",
        "ep (KW)": "Puissance moteur (kW)",
        "m (kg)": "Masse du véhicule (kg)",
        "age_months": "Âge du véhicule (mois)",
        "Ft": "Type de carburant"
    }
    fuel_types = ['essence', 'essence/électrique', 'diesel', 'diesel/électrique']

    st.header("Prédiction de l'émission de CO₂ pour un véhicule")

    input_features = get_input_features()

    user_inputs = {}
    for feat, default in input_features.items():
        if feat == "Ft":
            user_inputs[feat] = st.selectbox(feature_labels[feat], options=fuel_types, index=fuel_types.index(default) if default in fuel_types else 0)
        else:
            user_inputs[feat] = st.number_input(feature_labels[feat], value=default)

    model = load_model()
    df = pd.DataFrame([user_inputs])

    if st.button("Prédire"):
        prediction = predict_emission(model, df)
        st.success(f"Émission de CO₂ prédite : {prediction:.2f} g/km")

        explanation = explain_prediction(model, df)
        if isinstance(explanation, dict) and 'dict' in explanation:
            st.write("Importances des variables (SHAP, top 5) :")
            for feat, imp in explanation['dict'].items():
                st.write(f"- {prettify_feature_name(feat, feature_labels)} : {float(imp):.3f}")
            fig = plot_shap_values(explanation['shap_values'], explanation['feature_names'], feature_labels)
            st.pyplot(fig)
        else:
            st.write(f"Importances des variables : {explanation}")
