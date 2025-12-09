import streamlit as st
import pandas as pd
from utils.data_loaders import load_model, get_input_features, load_processed_data
from utils.model_utils import (
    predict_emission, 
    explain_prediction, 
    prettify_feature_name,  
    calculate_emission_percentile,
    find_similar_less_polluting_cars  # Add this import
)
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

    # ===== SECTION POUR LE FORMULAIRE D'ENTRÉE =====
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

        # ===== SECTION POUR LA PRÉDICTION ET L'AFFICHAGE DU PERCENTILE =====
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


        # ===== SECTION POUR LES VÉHICULES SIMILAIRES MOINS POLLUANTS =====
        st.markdown("---")
        st.subheader("🌱 Top 3 des alternatives moins polluantes")
        
        # Mapping du type de carburant
        fuel_display = {
            'petrol': '⛽ Essence',
            'petrol/electric': '🔋 Essence/Électrique',
            'diesel': '⛽ Diesel',
            'diesel/electric': '🔋 Diesel/Électrique'
        }
        
        similar_cars = find_similar_less_polluting_cars(user_inputs, data, prediction, top_n=3)
        
        if len(similar_cars) > 0:
            medals = ["🥇", "🥈", "🥉"]
            
            for idx, (_, car) in enumerate(similar_cars.iterrows()):
                # Calcul de la réduction
                reduction_value = prediction - car['Ewltp (g/km)']
                reduction_percent = (reduction_value / prediction) * 100
                
                # Construire le nom du véhicule (ne pas afficher "OTHERS")
                if car['Mk'] == 'OTHERS':
                    car_name = car['Cn']
                else:
                    car_name = f"{car['Mk']} {car['Cn']}"
                
                # Créer une section extensible pour chaque véhicule
                with st.expander(f"{medals[idx]} **{car_name}** - {car['Ewltp (g/km)']:.1f} g/km", expanded=(idx==0)):
                    
                    st.markdown("""
                    <style>
                    .small-metric {
                        font-size: 0.8rem;
                    }
                    .small-metric .metric-label {
                        font-size: 0.7rem;
                        color: #6b7280;
                    }
                    .small-metric .metric-value {
                        font-size: 0.9rem;
                        font-weight: 600;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Informations en colonnes
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='small-metric'>
                            <div class='metric-label'>Type de carburant</div>
                            <div class='metric-value'>{fuel_display.get(car['Ft'], car['Ft'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class='small-metric'>
                            <div class='metric-label'>Cylindrée</div>
                            <div class='metric-value'>{car['ec (cm3)']:.0f} cm³</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class='small-metric'>
                            <div class='metric-label'>Puissance</div>
                            <div class='metric-value'>{car['ep (KW)']:.0f} kW</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class='small-metric'>
                            <div class='metric-label'>Masse</div>
                            <div class='metric-value'>{car['m (kg)']:.0f} kg</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Émissions et réduction 
                    st.markdown("---")
                    col_em1, col_em2 = st.columns(2)
                    with col_em1:
                        st.markdown(f"""
                        <div class='small-metric'>
                            <div class='metric-label'>Émissions réelles CO₂</div>
                            <div class='metric-value' style='font-size: 1.2rem; color: #10b981;'>{car['Ewltp (g/km)']:.1f} g/km</div>
                            <div style='font-size: 0.75rem; color: #10b981;'>↓ -{reduction_value:.1f} g/km</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_em2:
                        st.markdown(f"""
                        <div class='small-metric'>
                            <div class='metric-label'>Réduction</div>
                            <div class='metric-value' style='font-size: 1.2rem; color: #10b981;'>{reduction_percent:.1f}%</div>
                            <div style='font-size: 0.75rem; color: #6b7280;'>vs prédiction</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.info("💡 Ces véhicules ont le même type de carburant et des caractéristiques similaires aux vôtres, mais émettent moins de CO₂.")
        else:
            st.warning(f"Aucun véhicule similaire moins polluant trouvé avec le type de carburant {fuel_display.get(user_inputs['Ft'], user_inputs['Ft'])} et des caractéristiques dans une marge de ±10%.")


        # ===== SECTION POUR L'EXPLICATION AVEC SHAP =====
        st.markdown("---")
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
            st.plotly_chart(fig, width='stretch')
        else:
            st.write(f"Importances des variables : {explanation}")

if __name__ == "__main__":
    run_predict_page()
