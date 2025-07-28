import streamlit as st
from ui.components.model_filters import render_model_filters
from ui.components.model_selector import render_model_selector
from llm_client import query_llm  # Import existant
from response_evaluator import ResponseEvaluator  # Import existant


from services.history_service import get_history_service
from services.evaluation_service import get_evaluation_service


def render_settings_tab():
    st.title("⚙️ Settings")
    st.markdown("## 🔧 Configuration du fichier de paramétrage des modèles LLMs disponibles")
    st.markdown("""
    - Modifiez `config/llm_config.yaml` pour ajouter des modèles
    - Configurez vos clés API dans les secrets Streamlit
    - Vérifiez la disponibilité des providers
    """)
    
    if st.button("🔄 Recharger la configuration"):
        # Clear cache and reload
        st.cache_data.clear()
        st.rerun()
    

    # Informations sur la configuration
    st.markdown("## Informations de configuration des modèles")
    with st.expander("📋 Configuration actuelle", expanded = True):
        try:
            model_service = st.session_state.get('model_service')
            if model_service:
                stats = model_service.get_model_statistics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Modèles disponibles", stats.available_models)
                with col2:
                    st.metric("Total modèles", stats.total_models)
                with col3:
                    st.metric("Providers", len(stats.providers))
                
                st.markdown("**Providers configurés:**")
                for provider in sorted(stats.providers):
                    provider_models = model_service.get_models_by_provider(provider)
                    st.write(f"- **{provider}**: {len(provider_models)} modèle(s)")

        except Exception as e:
            st.error(f"Erreur lors de l'affichage des informations: {e}")


        # === SECTION MAINTENANCE ÉTENDUE ===
    # Chargement de l'historique via le service
    history_service = get_history_service()


    st.markdown("---")
    st.markdown("## 🔧 Maintenance")
    
    col_maint1, col_maint2, col_maint3, col_maint4 = st.columns(4)
    
    with col_maint1:
        if st.button("🧹 Nettoyer Anciens (90j)", disabled = True):
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
    
    with col_maint4:
        # Nouvelle section : Cache d'évaluation
        if st.button("🗄️ Vider Cache Éval"):
            evaluation_service = get_evaluation_service()
            evaluation_service.clear_cache()
            st.success("✅ Cache vidé")
    
    # Affichage des stats de cache d'évaluation
    st.markdown("### 📊 Statistiques Cache d'Évaluation")
    evaluation_service = get_evaluation_service()
    evaluation_service.display_cache_info()