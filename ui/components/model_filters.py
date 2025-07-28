import streamlit as st
from services.model_config_service import FilterCriteria

def render_model_filters(model_service):
    """
    Composant réutilisable pour les filtres de modèles
    
    Args:
        model_service: Instance du ModelConfigService
        
    Returns:
        dict: Modèles filtrés selon les critères sélectionnés
    """
    # Récupérer les providers via le service
    providers = sorted(list(model_service.get_all_providers()))

    st.subheader("🔍 Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        pricing_filter = st.selectbox(
            "💰 Pricing",
            options=["All", "Free Only", "Paid Only"],
            key="pricing_filter"
        )
    
    with filter_col2:
        web_search_filter = st.selectbox(
            "🌐 Web Search Capability",
            options=["All", "With Web Search", "Without Web Search"],
            key="web_search_filter"
        )
    
    with filter_col3:
        provider_filter = st.selectbox(
            "🔧 Provider",
            options=["All"] + providers,
            key="provider_filter"
        )
    
    # Créer les critères
    criteria = FilterCriteria(
        pricing=pricing_filter,
        web_search=web_search_filter,
        provider=provider_filter,
        available_only=True
    )

    # Appliquer les filtres via le service
    filtered_models = model_service.filter_models(criteria)
    
    # Affichage du nombre de modèles après filtrage
    available_models = model_service.get_available_models()
    st.info(f"📊 {len(filtered_models)} model(s) available after filtering "
            f"(from {len(available_models)} total available)")
    
    if not filtered_models:
        st.warning("⚠️ No models match the current filters. Please adjust your selection.")
    
    return filtered_models


def get_filter_stats(selected_models, model_service):
    """
    Génère les statistiques des modèles sélectionnés
    
    Args:
        selected_models (list): Liste des noms de modèles sélectionnés
        model_service: Instance du ModelConfigService
        
    Returns:
        str: Texte formaté avec les statistiques
    """
    if not selected_models:
        return None
    
    stats = model_service.get_selection_stats(selected_models)
    
    # Générer le texte des statistiques
    stats_text = f"🎯 **{stats['count']} model(s) selected:** {', '.join(stats['short_names'])}"
    
    if stats['free_count'] > 0 and stats['paid_count'] > 0:
        stats_text += f" | 🆓 {stats['free_count']} free, 💰 {stats['paid_count']} paid"
    elif stats['free_count'] > 0:
        stats_text += f" | 🆓 All free"
    else:
        stats_text += f" | 💰 All paid"
    
    if stats['web_search_count'] == stats['count']:
        stats_text += " | 🌐 All with web search"
    elif stats['web_search_count'] > 0:
        stats_text += f" | 🌐 {stats['web_search_count']} with web search"
    
    return stats_text