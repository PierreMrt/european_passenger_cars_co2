import streamlit as st
import pandas as pd
from utils.data_loaders import load_model, get_input_features, load_processed_data
from utils.model_utils import predict_emission, explain_prediction, prettify_feature_name,  calculate_emission_percentile
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

    # Mapping depuis le français vers l'anglais
    fuel_type_mapping = {
    "essence": "petrol",
    "essence/électrique": "petrol/electric",
    "diesel": "diesel",
    "diesel/électrique": "diesel/electric"
    }

    st.header("Prédiction de l'émission de CO₂ pour un véhicule")

    input_features = get_input_features()

    user_inputs = {}
    for feat, default in input_features.items():
        if feat == "Ft":
            user_inputs[feat] = st.selectbox(feature_labels[feat], options=fuel_types, index=fuel_types.index(default) if default in fuel_types else 0)
        else:
            user_inputs[feat] = st.number_input(feature_labels[feat], value=default)

    # Convertir le type de carburant en anglais
    user_inputs["Ft"] = fuel_type_mapping[user_inputs["Ft"]]
    model = load_model()

    df = pd.DataFrame([user_inputs])

    if st.button("Prédire"):
        prediction = predict_emission(model, df)
        st.success(f"Émission de CO₂ prédite : {prediction:.2f} g/km")

        # Charge le jeu de données prétraité pour le calcul du percentile
        data = load_processed_data()
        percentile = calculate_emission_percentile(prediction, data)
        
        # Montre un message basé sur le percentile
        if percentile < 25:
            message = "🟢 Très faible ! Votre véhicule émet moins que la majorité des véhicules."
        elif percentile < 50:
            message = "🟡 Assez faible. Votre véhicule est en dessous de la moyenne."
        elif percentile < 75:
            message = "🟠 Au-dessus de la moyenne. Considérez des alternatives plus écologiques."
        else:
            message = "🔴 Très élevé ! Votre véhicule fait partie des plus polluants."
        
        st.info(f"📊 Percentile : {percentile:.1f}% - {message}")

        explanation = explain_prediction(model, df)
        if isinstance(explanation, dict) and 'dict' in explanation:
            top_items = sorted(explanation["dict"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            # Construction du tableau markdown
            markdown_table = "| Variable | Valeur SHAP |\n|---|---|\n"
            for feat, imp in top_items:
                label = prettify_feature_name(feat, feature_labels)
                markdown_table += f"| {label} | {imp:.3f} |\n"

            st.markdown("🔍 **Importances des variables (SHAP, top 5):**")
            st.markdown(markdown_table)

            # Visualisation des valeurs SHAP
            fig = plot_shap_values(explanation['shap_values'], explanation['feature_names'], feature_labels)
            st.pyplot(fig)
        else:
            st.write(f"Importances des variables : {explanation}")
