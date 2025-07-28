import streamlit as st
from ui.components.model_filters import render_model_filters
from ui.components.model_selector import render_model_selector
from llm_client import query_llm  # Import existant
from response_evaluator import ResponseEvaluator, display_comparative_dashboard  # Import existant
from services.history_service import get_history_service

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def render_history_tab(evaluator=None):

    st.title("📜 Prompt History")

    # Chargement de l'historique via le service
    history_service = get_history_service()
    history_data, data_source = history_service.load_history_with_source()
    

    if history_data:
        # Convertir les données en DataFrame
        df = pd.DataFrame(history_data)
        
        # Vérifier si les colonnes d'évaluation existent (nouvelles données)
        has_evaluation_data = 'evaluation_score' in df.columns
        
        if 'response' not in df.columns:
            df['response'] = ''
        
        # Traitement des sources et réponses principales
        df["sources"] = df["response"].apply(
            lambda r: r.split("=== SOURCES ===")[1].strip() 
            if isinstance(r, str) and "=== SOURCES ===" in r 
            else ""
        )
        df["main_response"] = df["response"].apply(
            lambda r: r.split("=== SOURCES ===")[0].strip() 
            if isinstance(r, str) and "=== SOURCES ===" in r 
            else str(r) if r else ""
        )

        # === ANALYTICS DASHBOARD ===
        if has_evaluation_data:
            st.markdown("## 📊 Analytics Dashboard")
            
            # Statistiques globales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = df['evaluation_score'].mean()
                st.metric("Score Moyen Global", f"{avg_score:.1f}/10")
            
            with col2:
                best_model = df.loc[df['evaluation_score'].idxmax(), 'model_name'].split(':')[0]
                st.metric("Meilleur Modèle", best_model)
            
            with col3:
                total_evaluations = len(df[df['evaluation_score'].notna()])
                st.metric("Évaluations", total_evaluations)
            
            with col4:
                avg_sources = df['sources_score'].mean() if 'sources_score' in df.columns else 0
                st.metric("Score Sources Moyen", f"{avg_sources:.1f}/10")
            
            # Graphiques d'analyse
            tab_analytics, tab_models, tab_trends = st.tabs(["📊 Vue d'ensemble", "🤖 Par Modèle", "📈 Tendances"])
            
            with tab_analytics:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Distribution des scores globaux
                    import plotly.express as px
                    fig_hist = px.histogram(
                        df, 
                        x='evaluation_score', 
                        title='Distribution des Scores Globaux',
                        nbins=20,
                        labels={'evaluation_score': 'Score Global', 'count': 'Nombre'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_chart2:
                    # Scores moyens par catégorie
                    if all(col in df.columns for col in ['readability_score', 'structure_score', 'sources_score', 'completeness_score', 'relevance_score']):
                        categories_scores = {
                            'Lisibilité': df['readability_score'].mean(),
                            'Structure': df['structure_score'].mean(),
                            'Sources': df['sources_score'].mean(),
                            'Complétude': df['completeness_score'].mean(),
                            'Pertinence': df['relevance_score'].mean()
                        }
                        
                        fig_bar = px.bar(
                            x=list(categories_scores.keys()),
                            y=list(categories_scores.values()),
                            title='Scores Moyens par Catégorie',
                            labels={'x': 'Catégorie', 'y': 'Score Moyen'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab_models:
                # Comparaison par modèle
                if len(df['model_name'].unique()) > 1:
                    model_stats = df.groupby('model_name').agg({
                        'evaluation_score': ['mean', 'count'],
                        'readability_score': 'mean',
                        'structure_score': 'mean',
                        'sources_score': 'mean',
                        'completeness_score': 'mean',
                        'relevance_score': 'mean'
                    }).round(1)
                    
                    model_stats.columns = ['Score Moyen', 'Nb Tests', 'Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence']
                    model_stats['Modèle'] = [name.split(':')[0] for name in model_stats.index]
                    model_stats = model_stats.reset_index(drop=True)
                    
                    st.dataframe(model_stats, use_container_width=True)
                    
                    # Graphique radar par modèle (top 3)
                    top_models = model_stats.nlargest(3, 'Score Moyen')
                    
                    import plotly.graph_objects as go
                    fig_radar = go.Figure()
                    
                    categories = ['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence']
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                    
                    for i, row in top_models.iterrows():
                        values = [row['Lisibilité'], row['Structure'], row['Sources'], row['Complétude'], row['Pertinence']]
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=row['Modèle'],
                            line_color=colors[i % len(colors)]
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                        showlegend=True,
                        title="Top 3 Modèles - Comparaison Radar"
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour la comparaison entre modèles")
            
            with tab_trends:
                # Évolution temporelle
                if 'timestamp' in df.columns and len(df) > 5:
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    daily_scores = df.groupby('date')['evaluation_score'].mean().reset_index()
                    
                    fig_trend = px.line(
                        daily_scores, 
                        x='date', 
                        y='evaluation_score',
                        title='Évolution du Score Moyen dans le Temps'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Tendance par modèle
                    model_trends = df.groupby(['date', 'model_name'])['evaluation_score'].mean().reset_index()
                    model_trends['model_short'] = model_trends['model_name'].str.split(':').str[0]
                    
                    fig_model_trend = px.line(
                        model_trends,
                        x='date',
                        y='evaluation_score',
                        color='model_short',
                        title='Évolution par Modèle'
                    )
                    st.plotly_chart(fig_model_trend, use_container_width=True)
                else:
                    st.info("Pas assez de données pour analyser les tendances temporelles")
        
        st.markdown("---")
        
        # === NOUVELLE SECTION : ANALYSE BATCH HISTORIQUE ===
        if has_evaluation_data and len(df) > 5:  # Au moins 6 entrées pour l'analyse
            st.markdown("---")
            st.markdown("## 🔬 Analyse Batch Historique")
            
            # Bouton pour lancer l'analyse avancée
            if st.button("🚀 Générer Analyse Comparative Historique"):
                # Vérifier si l'evaluator est disponible
                if evaluator is None:
                    # Créer un evaluator temporaire si nécessaire
                    evaluator = ResponseEvaluator()
                with st.spinner("Analyse en cours des données historiques..."):
                    # Préparer les données pour l'analyse batch
                    historical_data = []
                    for _, row in df.iterrows():
                        historical_data.append((
                            row['prompt'],
                            row['response'],
                            row['model_name']
                        ))
                    
                    # Grouper par modèle pour l'analyse
                    evaluations_by_model = {}
                    for model_name in df['model_name'].unique():
                        model_entries = df[df['model_name'] == model_name]
                        model_evaluations = []
                        
                        for _, entry in model_entries.iterrows():
                            # Recréer l'objet EvaluationResult depuis les données stockées
                            from response_evaluator import EvaluationResult
                            eval_result = EvaluationResult(
                                overall_score=entry.get('evaluation_score', 0),
                                readability_score=entry.get('readability_score', 0),
                                structure_score=entry.get('structure_score', 0),
                                sources_score=entry.get('sources_score', 0),
                                completeness_score=entry.get('completeness_score', 0),
                                relevance_score=entry.get('relevance_score', 0),
                                details={},
                                recommendations=[]
                            )
                            model_evaluations.append(eval_result)
                        
                        evaluations_by_model[model_name] = model_evaluations
                    
                    # Générer le rapport historique
                    historical_report = evaluator.generate_comparative_report(evaluations_by_model)
                    
                    # Afficher le dashboard historique
                    st.markdown("### 📈 Rapport Historique Complet")
                    display_comparative_dashboard(historical_report)
                    
                    # Insights spécifiques à l'historique
                    st.markdown("### 🔍 Insights Historiques Supplémentaires")
                    
                    # Évolution dans le temps
                    # Filtrer les données avec des scores valides
                    df_valid_scores = df[df['evaluation_score'].notna()]

                    if len(df_valid_scores) > 0:
                        daily_best = df_valid_scores.groupby('date').apply(
                            lambda x: x.loc[x['evaluation_score'].idxmax(), 'model_name'] if len(x) > 0 else None
                        ).dropna().value_counts()
                        
                        if len(daily_best) > 0:
                            st.write(f"🏆 Modèle le plus souvent classé premier : **{daily_best.index[0]}** ({daily_best.iloc[0]} jours)")
                        else:
                            st.write("📊 Données insuffisantes pour déterminer le modèle dominant")
                    else:
                        st.write("⚠️ Aucune donnée d'évaluation valide trouvée")


                    #if 'timestamp' in df.columns:
                    #    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    #    
                        # Évolution du meilleur modèle dans le temps
                    #    daily_best = df.groupby('date').apply(
                    #        lambda x: x.loc[x['evaluation_score'].idxmax(), 'model_name']
                    #    ).value_counts()
                    #    
                    #    if len(daily_best) > 0:
                    #        st.write(f"🏆 Modèle le plus souvent classé premier : **{daily_best.index[0]}** ({daily_best.iloc[0]} jours)")
                        
                        
                        # Tendance d'amélioration
                        df_sorted = df_valid_scores.sort_values('timestamp')
                        if len(df_sorted) >= 6:  # Au moins 6 entrées pour comparer
                            recent_avg = df_sorted.tail(min(10, len(df_sorted)//2))['evaluation_score'].mean()
                            older_avg = df_sorted.head(min(10, len(df_sorted)//2))['evaluation_score'].mean()
                        
                        # Tendance d'amélioration
                        #df_sorted = df.sort_values('timestamp')
                        #recent_avg = df_sorted.tail(10)['evaluation_score'].mean()
                        #older_avg = df_sorted.head(10)['evaluation_score'].mean()
                        
                        if len(df_sorted) >= 6 and not pd.isna(recent_avg) and not pd.isna(older_avg):
                            if recent_avg > older_avg:
                                improvement = ((recent_avg - older_avg) / older_avg) * 100
                                st.write(f"📈 Amélioration générale : +{improvement:.1f}% sur les récentes évaluations")
                            elif recent_avg < older_avg:
                                decline = ((older_avg - recent_avg) / older_avg) * 100
                                st.write(f"📉 Dégradation observée : -{decline:.1f}% sur les récentes évaluations")
                            else:
                                st.write("⚖️ Performance stable dans le temps")
                        else:
                            st.write("📊 Données insuffisantes pour analyser les tendances")
        
        # Filtres améliorés
        st.subheader("🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filtre par modèle
            available_models = df["model_name"].unique().tolist() if "model_name" in df.columns else []
            model_filter = st.multiselect("Filter by model", options=available_models)
            if model_filter:
                df = df[df["model_name"].isin(model_filter)]
        
        with col2:
            # Recherche dans les prompts
            prompt_search = st.text_input("Search in prompts:")
            if prompt_search:
                df = df[df["prompt"].str.contains(prompt_search, case=False, na=False)]
            # Recherche dans les réponses principales
            main_response_search = st.text_input("Search in main_response:")
            if main_response_search:
                df = df[df["main_response"].str.contains(main_response_search, case=False)]                

        with col3:
            # Filtre par score (si disponible)
            if has_evaluation_data:
                score_range = st.slider(
                    "Score Range",
                    min_value=0.0,
                    max_value=10.0,
                    value=(0.0, 10.0),
                    step=0.1
                )
                df = df[(df['evaluation_score'] >= score_range[0]) & (df['evaluation_score'] <= score_range[1])]

        # Affichage du tableau avec scores
        st.subheader("📋 History Table")
        
        if has_evaluation_data:
            display_columns = [
                "timestamp", "model_name", "prompt", "evaluation_score",
                "readability_score", "structure_score", "sources_score",
                "main_response", "sources"
            ]
        else:
            display_columns = ["timestamp", "model_name", "prompt", "main_response", "sources"]
        
        # Vérifier quelles colonnes existent réellement
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            # Reformater l'affichage pour de meilleures colonnes
            df_display = df[available_columns].copy()
            
            if has_evaluation_data:
                # Renommer les colonnes pour l'affichage
                df_display.columns = [
                    "Date", "Modèle", "Prompt", "Score Global",
                    "Lisibilité", "Structure", "Sources Score",
                    "Réponse", "Sources"
                ]
                
                # Formater les scores
                score_cols = ["Score Global", "Lisibilité", "Structure", "Sources Score"]
                for col in score_cols:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].round(1)
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Unable to display data: required columns not found")

        # Bouton de téléchargement Excel via le service
        if len(df) > 0:
            if st.button("📅 Generate Excel Export"):
                excel_data = history_service.export_to_excel(history_data)
                
                if excel_data:
                    st.download_button(
                        label="⬇️ Download Excel File",
                        data=excel_data,
                        file_name=f"prompt_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("❌ Échec de génération du fichier Excel")

    else:
        st.info("No prompt history available yet.")

    # === SECTION MAINTENANCE (OPTIONNELLE) ===
    st.markdown("---")
    st.markdown("## 🔧 Maintenance")
    
    col_maint1, col_maint2, col_maint3 = st.columns(3)
    
    with col_maint1:
        if st.button("🧹 Nettoyer Anciens (90j)"):
            removed = history_service.cleanup_old_entries(90)
            if removed > 0:
                st.success(f"✅ {removed} entrée(s) supprimée(s)")
            else:
                st.info("ℹ️ Aucune entrée à supprimer")
    
    with col_maint2:
        if st.button("✅ Valider Intégrité"):
            is_valid = history_service.validate_data_integrity()
            if is_valid:
                st.success("✅ Données intègres")
            else:
                st.warning("⚠️ Problèmes détectés")
    
    with col_maint3:
        file_size = history_service.get_file_size()
        st.metric("📁 Taille Fichier", file_size)