import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from datetime import datetime
from ui.components.model_filters import render_model_filters
from ui.components.model_selector import render_model_selector, render_selection_summary
from llm_client import query_llm
from response_evaluator import ResponseEvaluator
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
                    
                    # === ÉVALUATION AUTOMATIQUE VIA SERVICE ===
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
                        
                        # Récupérer l'évaluation pour la sauvegarde historique
                        session = evaluation_service.get_session(session_id)
                        evaluation = session.evaluations[name]
                    
                    # Save history avec évaluation via le service
                    history_service = get_history_service()
                    save_success = history_service.save_evaluation_entry(
                        prompt, name, model_id, content, evaluation
                    )
                    
                    if not save_success:
                        st.warning("⚠️ Problème lors de la sauvegarde de l'historique")

                except Exception as e:
                    st.error(f"Error: {e}")
        
'''        # === ANALYSE COMPARATIVE VIA SERVICE ===
        session = evaluation_service.get_session(session_id)
        if session and session.is_ready_for_comparison():
            evaluation_service.display_session_comparison_analysis(session_id)
            
            # === OPTIONS DE RAPPORT ===
            with st.spinner("Génération du rapport comparatif..."):
                comparative_report = evaluation_service.generate_comparison_report(session_id)
                
                if comparative_report:
                    # Option de sauvegarde du rapport
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Sauvegarder Rapport JSON", disabled = True):
                            report_filename = f"data/comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            try:
                                Path("data").mkdir(parents=True, exist_ok=True)
                                with open(report_filename, 'w', encoding='utf-8') as f:
                                    json.dump(comparative_report, f, ensure_ascii=False, indent=2)
                                st.success(f"Rapport sauvegardé : {report_filename}")
                            except Exception as e:
                                st.error(f"Erreur de sauvegarde : {e}")
                    
                    with col2:
                        if st.button("📊 Export Données Brutes", disabled = True):
                            report_json = json.dumps(comparative_report, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="⬇️ Télécharger Rapport JSON",
                                data=report_json,
                                file_name=f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )'''