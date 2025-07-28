import streamlit as st
from ui.components.model_filters import render_model_filters
from ui.components.model_selector import render_model_selector
from llm_client import query_llm  # Import existant
from response_evaluator import ResponseEvaluator  # Import existant

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def render_analytics_tab():
    # === History ===
    HISTORY_FILE = Path("data/prompt_history.json")
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    st.title("🔬 Advanced Analytics (alpha-version)")
    st.markdown("Analyses approfondies et comparaisons avancées des modèles LLM")
    
    # Vérifier s'il y a des données
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            analytics_data = json.load(f)
        
        if len(analytics_data) > 0:
            df_analytics = pd.DataFrame(analytics_data)
            
            # Sélection de période
            st.subheader("📅 Sélection de Période")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'timestamp' in df_analytics.columns:
                    df_analytics['date'] = pd.to_datetime(df_analytics['timestamp']).dt.date
                    min_date = df_analytics['date'].min()
                    max_date = df_analytics['date'].max()
                    
                    date_range = st.date_input(
                        "Période d'analyse",
                        value=[min_date, max_date],
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        df_analytics = df_analytics[
                            (df_analytics['date'] >= date_range[0]) & 
                            (df_analytics['date'] <= date_range[1])
                        ]
            
            with col2:
                # Sélection de modèles pour l'analyse
                available_models_analytics = df_analytics['model_name'].unique().tolist()
                selected_models_analytics = st.multiselect(
                    "Modèles à analyser",
                    options=available_models_analytics,
                    default=available_models_analytics[:5] if len(available_models_analytics) > 5 else available_models_analytics
                )
                
                if selected_models_analytics:
                    df_analytics = df_analytics[df_analytics['model_name'].isin(selected_models_analytics)]
            
            if len(df_analytics) > 0:
                # Analyse de Performance Détaillée
                st.subheader("📊 Analyse de Performance Détaillée")
                
                # Métriques avancées
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    consistency_scores = df_analytics.groupby('model_name')['evaluation_score'].std().fillna(0)
                    most_consistent = consistency_scores.idxmin()
                    consistency_value = 1 - (consistency_scores.min() / 10)  # Normaliser
                    st.metric("🎯 Plus Consistant", most_consistent, f"{consistency_value:.2f}")
                
                with col2:
                    improvement_rates = {}
                    for model in selected_models_analytics:
                        model_data = df_analytics[df_analytics['model_name'] == model].sort_values('timestamp')
                        if len(model_data) > 1:
                            recent = model_data.tail(len(model_data)//2)['evaluation_score'].mean()
                            older = model_data.head(len(model_data)//2)['evaluation_score'].mean()
                            improvement_rates[model] = recent - older
                    
                    if improvement_rates:
                        best_improving = max(improvement_rates.items(), key=lambda x: x[1])
                        st.metric("📈 Plus d'Amélioration", best_improving[0], f"+{best_improving[1]:.1f}")
                
                with col3:
                    specialized_scores = {}
                    for model in selected_models_analytics:
                        model_data = df_analytics[df_analytics['model_name'] == model]
                        if 'sources_score' in model_data.columns:
                            specialized_scores[model] = model_data['sources_score'].mean()
                    
                    if specialized_scores:
                        best_sources = max(specialized_scores.items(), key=lambda x: x[1])
                        st.metric("🔗 Meilleur Sources", best_sources[0], f"{best_sources[1]:.1f}/10")
                
                with col4:
                    total_comparisons = len(df_analytics)
                    unique_prompts = df_analytics['prompt'].nunique() if 'prompt' in df_analytics.columns else 0
                    st.metric("🔄 Comparaisons", total_comparisons, f"{unique_prompts} prompts uniques")
                
                # Analyse de Corrélation
                st.subheader("🔗 Analyse de Corrélation")
                
                if all(col in df_analytics.columns for col in ['readability_score', 'structure_score', 'sources_score', 'completeness_score', 'relevance_score']):
                    correlation_cols = ['readability_score', 'structure_score', 'sources_score', 'completeness_score', 'relevance_score']
                    correlation_matrix = df_analytics[correlation_cols].corr()
                    
                    import plotly.graph_objects as go
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence'],
                        y=['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence'],
                        colorscale='RdBu',
                        zmid=0
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Matrice de Corrélation des Critères d'Évaluation",
                        height=500
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Insights de corrélation
                    st.markdown("#### 💡 Insights de Corrélation")
                    
                    # Trouver les corrélations les plus fortes
                    correlations = []
                    for i in range(len(correlation_cols)):
                        for j in range(i+1, len(correlation_cols)):
                            corr_value = correlation_matrix.iloc[i, j]
                            correlations.append((correlation_cols[i], correlation_cols[j], corr_value))
                    
                    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    for i, (col1, col2, corr) in enumerate(correlations[:3]):
                        if abs(corr) > 0.3:  # Seulement les corrélations significatives
                            col1_name = col1.replace('_score', '').title()
                            col2_name = col2.replace('_score', '').title()
                            
                            if corr > 0:
                                st.write(f"📈 **Corrélation positive** entre {col1_name} et {col2_name} : {corr:.2f}")
                            else:
                                st.write(f"📉 **Corrélation négative** entre {col1_name} et {col2_name} : {corr:.2f}")
            
            else:
                st.info("Aucune donnée disponible pour la période et les modèles sélectionnés.")
        else:
            st.info("Aucune donnée d'évaluation disponible pour l'analyse avancée.")
    else:
        st.info("Aucun historique disponible. Effectuez quelques comparaisons d'abord.")