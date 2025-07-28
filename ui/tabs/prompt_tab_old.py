import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from datetime import datetime
from ui.components.model_filters import render_model_filters, get_filter_stats
from ui.components.model_selector import render_model_selector, render_selection_summary
from llm_client import query_llm  # Import existant
from response_evaluator import ResponseEvaluator  # Import existant
from services.history_service import get_history_service
from services.evaluation_service import get_evaluation_service, EvaluationRequest

def render_prompt_tab(model_service, evaluator):
    """Rendu complet du Tab 1 - Prompt & Compare"""
    st.title("🧠 LLM Comparator")
    prompt = st.text_area("💬 Your prompt:", height=150)
    
    # Utilisation des composants extraits
    with st.expander("🎯 Select Models", expanded=True):
        filtered_models = render_model_filters(model_service)
        selected = render_model_selector(filtered_models, model_service)

    render_selection_summary(selected, model_service)

    if st.button("🚀 Run", disabled=not (prompt and selected)) and prompt and selected:
        # Initialiser le service d'évaluation et créer une session
        evaluation_service = get_evaluation_service()
        session_id = evaluation_service.create_evaluation_session()
        
        # Stocker les réponses pour l'évaluation comparative
        responses_for_evaluation = {}
        models_dict = model_service.get_raw_models_dict()  # Pour compatibilité
        for name in selected:
            model = models_dict[name]
            model_id = model["id"]
            provider = model["provider"]

            # Affichage enrichi avec indicateurs
            indicators = []
            if model.get("free", False):
                indicators.append("🆓")
            else:
                indicators.append("💰")
            
            if model.get("web_search", False):
                indicators.append("🌐")
            
            header = f"### 🤖 {' '.join(indicators)} {name}"
            st.markdown(header)
            
            with st.spinner("Generating response..."):
                instruction = (
                    "\n\nÀ la fin de ta réponse, ajoute une section intitulée '=== SOURCES ===' "
                    "avec la liste des urls des sites web utilisées pour la réponse."
                )
                full_prompt = f"{prompt.strip()}{instruction}"

                try:
                    content = query_llm(provider, model_id, full_prompt)

                    if "=== SOURCES ===" in content:
                        answer_part, sources_part = content.split("=== SOURCES ===", 1)
                    else:
                        answer_part = content
                        sources_part = "*Aucune source identifiable fournie.*"

                    st.success("Response received.")
                    st.markdown("**📝 Answer:**")
                    st.write(answer_part.strip())
                    st.markdown("**🔗 Sources:**")
                    st.info(sources_part.strip())
                    
                    # === ÉVALUATION AUTOMATIQUE ===
                    st.markdown("---")
                    with st.spinner("Évaluation automatique en cours..."):
                        # Créer la demande d'évaluation
                        eval_request = EvaluationRequest(
                            prompt=prompt,
                            response=content,
                            model_name=name
                        )
                        
                        # Ajouter à la session (affichage automatique)
                        evaluation_service.add_to_session(session_id, eval_request, display_individual=True)
                        
                        # Récupérer l'évaluation pour compatibilité avec le code existant
                        session = evaluation_service.get_session(session_id)
                        evaluation = session.evaluations[name]
                    
                    # Stocker pour comparaison AVANCÉE
                    #responses_for_evaluation.append((prompt, content, name))
                    responses_for_evaluation[name] = {
                        'content': content,
                        'evaluation': evaluation
                    }
                    
                    # Save history avec évaluation via le service
                    history_service = get_history_service()
                    save_success = history_service.save_evaluation_entry(
                        prompt, name, model_id, content, evaluation
                    )                    
                    if not save_success:
                        st.warning("⚠️ Problème lors de la sauvegarde de l'historique")                    
                    

                except Exception as e:
                    st.error(f"Error: {e}")
        
        # === NOUVELLE SECTION : RAPPORT COMPARATIF AVANCÉ ===
        if len(responses_for_evaluation) > 1:
            st.markdown("---")
            st.markdown("## 🏆 Analyse Comparative Avancée")
            
            # Tableau comparatif
            comparison_data = []
            for model_name, data in responses_for_evaluation.items():
                eval_result = data['evaluation']
                comparison_data.append({
                    'Modèle': model_name.split(':')[0].strip(),
                    'Score Global': eval_result.overall_score,
                    '📖 Lisibilité': eval_result.readability_score,
                    '🏗️ Structure': eval_result.structure_score,
                    '🔗 Sources': eval_result.sources_score,
                    '📋 Complétude': eval_result.completeness_score,
                    '🎯 Pertinence': eval_result.relevance_score,
                    '📊 Mots': eval_result.details['word_count']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison = df_comparison.sort_values('Score Global', ascending=False)
            
            st.dataframe(df_comparison, use_container_width=True)
            
            # Graphique radar comparatif
            if len(comparison_data) <= 4:  # Éviter la surcharge visuelle
                
                categories = ['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence']
                
                fig = go.Figure()
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
                
                for i, (model_name, data) in enumerate(responses_for_evaluation.items()):
                    eval_result = data['evaluation']
                    values = [
                        eval_result.readability_score,
                        eval_result.structure_score,
                        eval_result.sources_score,
                        eval_result.completeness_score,
                        eval_result.relevance_score
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=model_name.split(':')[0].strip(),
                        line_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=True,
                    title="Comparaison Radar des Performances",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations du meilleur modèle
            best_model = max(responses_for_evaluation.items(), 
                            key=lambda x: x[1]['evaluation'].overall_score)
            
            st.markdown(f"### 🏆 Meilleur Modèle: {best_model[0].split(':')[0].strip()}")
            st.markdown(f"**Score: {best_model[1]['evaluation'].overall_score}/10**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🎯 Points forts:**")
                eval_details = best_model[1]['evaluation']
                strengths = []
                if eval_details.readability_score >= 8:
                    strengths.append("📖 Excellente lisibilité")
                if eval_details.structure_score >= 8:
                    strengths.append("🏗️ Très bien structuré")
                if eval_details.sources_score >= 8:
                    strengths.append("🔗 Sources de qualité")
                if eval_details.completeness_score >= 8:
                    strengths.append("📋 Réponse complète")
                if eval_details.relevance_score >= 8:
                    strengths.append("🎯 Très pertinent")
                
                for strength in strengths:
                    st.write(f"• {strength}")
            
            with col2:
                st.markdown("**⚠️ Points d'amélioration:**")
                weaknesses = []
                if eval_details.readability_score < 6:
                    weaknesses.append("📖 Lisibilité à améliorer")
                if eval_details.structure_score < 6:
                    weaknesses.append("🏗️ Structure à revoir")
                if eval_details.sources_score < 6:
                    weaknesses.append("🔗 Manque de sources")
                if eval_details.completeness_score < 6:
                    weaknesses.append("📋 Réponse incomplète")
                if eval_details.relevance_score < 6:
                    weaknesses.append("🎯 Pertinence à améliorer")
                
                if not weaknesses:
                    st.write("✅ Aucun point faible majeur identifié")
                else:
                    for weakness in weaknesses:
                        st.write(f"• {weakness}")





            # Génération du rapport comparatif
            with st.spinner("Génération du rapport comparatif..."):
                # Préparer les données pour le rapport
                evaluations_by_model = {}
                for model_name, data in responses_for_evaluation.items():
                    evaluations_by_model[model_name] = [data['evaluation']]
                
                # Générer le rapport
                comparative_report = evaluator.generate_comparative_report(evaluations_by_model)
                
                # Afficher le dashboard
                display_comparative_dashboard(comparative_report)
                
                # Option de sauvegarde du rapport
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Sauvegarder Rapport JSON", disabled=True):
                        report_filename = f"data/comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        try:
                            save_evaluation_report(comparative_report, report_filename)
                            st.success(f"Rapport sauvegardé : {report_filename}")
                        except Exception as e:
                            st.error(f"Erreur de sauvegarde : {e}")
                
                with col2:
                    # Export des données brutes du rapport
                    if st.button("📊 Export Données Brutes", disabled=True):
                        report_json = json.dumps(comparative_report, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="⬇️ Télécharger Rapport JSON",
                            data=report_json,
                            file_name=f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

