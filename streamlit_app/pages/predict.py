import streamlit as st
import pandas as pd
from utils.data_loaders import load_model, get_input_features, load_processed_data
from utils.model_utils import (
    predict_emission, 
    explain_prediction, 
    prettify_feature_name,  
    calculate_emission_percentile,
    find_similar_less_polluting_cars
)
from utils.viz_tools import plot_shap_values


def run_predict_page():
    """
    Affiche la page de prédiction des émissions de CO₂ dans l'application Streamlit.
    Permet à l'utilisateur de saisir manuellement les caractéristiques ou de sélectionner un modèle.
    """
    feature_labels = {
        "ec (cm3)": "Cylindrée (cm³)",
        "ep (KW)": "Puissance moteur (kW)",
        "m (kg)": "Masse du véhicule (kg)",
        "age_months": "Âge du véhicule (mois)",
        "Ft": "Type de carburant"
    }
    fuel_types = ['essence', 'essence/électrique', 'diesel', 'diesel/électrique']
    fuel_type_mapping = {
        "essence": "petrol",
        "essence/électrique": "petrol/electric",
        "diesel": "diesel",
        "diesel/électrique": "diesel/electric"
    }

    st.header("Prédiction de l'émission de CO₂ pour un véhicule")

    # Chargement des données pour la sélection de véhicule
    data = load_processed_data()
    
    # ===== SÉLECTION DE LA MÉTHODE DE SAISIE =====
    st.subheader("Mode de saisie")
    input_method = st.radio(
        "Choisissez votre méthode de saisie :",
        ["Saisie manuelle", "Sélection d'un modèle"],
        horizontal=True
    )
    
    st.markdown("---")
    
    user_inputs = {}
    
    if input_method == "Sélection d'un modèle":
        # ===== MODE SÉLECTION DE MODÈLE =====
        st.subheader("Sélection du véhicule")
        
        # Filtrer les entrées invalides
        valid_data = data[(data['Mk'].notna()) & (data['Cn'].notna())]
        
        # Obtenir les marques uniques triées alphabétiquement
        makes = sorted(valid_data['Mk'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_make = st.selectbox(
                "🚗 Marque (Mk)",
                options=[""] + makes,
                help="Sélectionnez la marque de votre véhicule"
            )
        
        with col2:
            if selected_make:
                # Filtrer les noms commerciaux par marque sélectionnée
                models = sorted(valid_data[valid_data['Mk'] == selected_make]['Cn'].unique())
                selected_model = st.selectbox(
                    "🏷️ Modèle commercial (Cn)",
                    options=[""] + models,
                    help="Sélectionnez le modèle commercial"
                )
            else:
                selected_model = st.selectbox(
                    "🏷️ Modèle commercial (Cn)",
                    options=[""],
                    disabled=True,
                    help="Sélectionnez d'abord une marque"
                )
        
        if selected_make and selected_model:
            # Récupérer les données du véhicule - prendre la version la plus récente (age_months le plus faible)
            car_data = valid_data[
                (valid_data['Mk'] == selected_make) & 
                (valid_data['Cn'] == selected_model)
            ].sort_values('age_months').iloc[0]
            
            # Afficher les spécifications du véhicule sélectionné
            st.success(f"✅ Véhicule sélectionné : {selected_make} {selected_model}")
            
            # Afficher les caractéristiques dans une section extensible
            with st.expander("📋 Caractéristiques du véhicule sélectionné", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Cylindrée", f"{car_data['ec (cm3)']:.0f} cm³")
                    st.metric("Puissance", f"{car_data['ep (KW)']:.0f} kW")
                with col_b:
                    st.metric("Masse", f"{car_data['m (kg)']:.0f} kg")
                with col_c:
                    # Mapping inversé pour l'affichage
                    fuel_display_rev = {v: k for k, v in fuel_type_mapping.items()}
                    st.metric("Carburant", fuel_display_rev.get(car_data['Ft'], car_data['Ft']))
            
            # Remplir user_inputs avec les données du véhicule sélectionné
            user_inputs = {
                "ec (cm3)": float(car_data['ec (cm3)']),
                "ep (KW)": float(car_data['ep (KW)']),
                "m (kg)": float(car_data['m (kg)']),
                "age_months": float(car_data['age_months']),
                "Ft": car_data['Ft']
            }
            
            # Permettre la modification manuelle
            st.markdown("---")
            if st.checkbox("🔧 Modifier les caractéristiques"):
                st.info("Vous pouvez ajuster les valeurs ci-dessous :")
                for feat in ["ec (cm3)", "ep (KW)", "m (kg)", "age_months"]:
                    user_inputs[feat] = st.number_input(
                        feature_labels[feat], 
                        value=user_inputs[feat],
                        key=f"override_{feat}"
                    )
                
                fuel_display_rev = {v: k for k, v in fuel_type_mapping.items()}
                current_fuel_fr = fuel_display_rev.get(user_inputs['Ft'], 'essence')
                selected_fuel = st.selectbox(
                    feature_labels["Ft"], 
                    options=fuel_types, 
                    index=fuel_types.index(current_fuel_fr),
                    key="override_fuel"
                )
                user_inputs["Ft"] = fuel_type_mapping[selected_fuel]
        else:
            st.warning("⚠️ Veuillez sélectionner une marque et un modèle pour continuer.")
            
    else:
        # ===== MODE SAISIE MANUELLE =====
        st.subheader("Saisie des caractéristiques")
        
        input_features = get_input_features()
        
        for feat, default in input_features.items():
            if feat == "Ft":
                user_inputs[feat] = st.selectbox(
                    feature_labels[feat], 
                    options=fuel_types, 
                    index=fuel_types.index(
                        {v: k for k, v in fuel_type_mapping.items()}.get(default, 'essence')
                    )
                )
            else:
                user_inputs[feat] = st.number_input(feature_labels[feat], value=default)
        
        # Convertir le type de carburant en anglais
        user_inputs["Ft"] = fuel_type_mapping[user_inputs["Ft"]]
    
    # ===== SECTION DE PRÉDICTION  =====
    if user_inputs:  
        model = load_model()
        df = pd.DataFrame([user_inputs])
        
        if st.button("🔮 Prédire", type="primary", use_container_width=True):
            # Prédiction
            prediction = predict_emission(model, df)
            st.success(f"Émission de CO₂ prédite : {prediction:.2f} g/km")
            
            # Calcul du percentile
            percentile = calculate_emission_percentile(prediction, data)
            
            if percentile < 25:
                message = "🟢 Très faible ! Votre véhicule émet moins que la majorité des véhicules."
            elif percentile < 50:
                message = "🟡 Assez faible. Votre véhicule est en dessous de la moyenne."
            elif percentile < 75:
                message = "🟠 Au-dessus de la moyenne. Considérez des alternatives plus écologiques."
            else:
                message = "🔴 Très élevé ! Votre véhicule fait partie des plus polluants."
            
            st.info(f"📊 Percentile : {percentile:.1f}% - {message}")
            
            # === SECTION DES ALTERNATIVES MOINS POLLUANTES ===
            st.markdown("---")
            st.subheader("🌱 Top 3 des alternatives moins polluantes")
            
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
            
            # === SECTION EXPLICATION DE LA PRÉDICTION ===
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
