import streamlit as st

def render_model_selector(filtered_models, model_service):
    """
    Composant pour la sélection des modèles avec checkboxes
    
    Args:
        filtered_models (dict): Modèles filtrés à afficher
        model_service: Instance du ModelConfigService
        
    Returns:
        list: Liste des noms de modèles sélectionnés
    """
    if not filtered_models:
        return []
    
    st.divider()
    
    # Récupérer le dictionnaire pour compatibilité
    models_dict = model_service.get_raw_models_dict()
    
    # === SÉLECTION DES MODÈLES ===
    # Option "Select All" / "Deselect All" pour les modèles filtrés
    col_all, col_none = st.columns(2)
    with col_all:
        if st.button("✅ Select All Filtered"):
            # Déselectionner tous d'abord
            for i in range(len(models_dict)):
                st.session_state[f"model_{i}"] = False
            # Puis sélectionner seulement les filtrés
            for name in filtered_models.keys():
                original_index = list(models_dict.keys()).index(name)
                st.session_state[f"model_{original_index}"] = True
    
    with col_none:
        if st.button("❌ Deselect All"):
            for i in range(len(models_dict)):
                st.session_state[f"model_{i}"] = False
    
    st.divider()

    # Organiser les modèles filtrés par provider
    providers_filtered = {}
    for name, model_info in filtered_models.items():
        # Convertir ModelInfo en dict si nécessaire
        if hasattr(model_info, 'to_dict'):
            model_data = model_info.to_dict()
        else:
            model_data = model_info
            
        provider = model_data.get("provider_to_display", "Unknown")
        if provider not in providers_filtered:
            providers_filtered[provider] = []
        providers_filtered[provider].append((name, model_data))
    
    selected = []
    
    # Affichage par provider
    for provider, models in providers_filtered.items():
        st.subheader(f"🔧 {provider.title()}")
        cols = st.columns(min(3, len(models)))
        
        for i, (name, model_data) in enumerate(models):
            # Trouver l'index original dans models_dict
            original_index = list(models_dict.keys()).index(name)
            col_index = i % len(cols)
            
            with cols[col_index]:
                # Valeurs par défaut : les 2 premiers modèles disponibles
                default_value = len(selected) < 2
                checkbox_key = f"model_{original_index}"
                
                # Initialiser la session state si elle n'existe pas
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = default_value
                
                # Créer le label avec indicateurs
                label = _create_model_label(name, model_data)
                help_text = _create_help_text(model_data)
                
                if st.checkbox(
                    label, 
                    value=st.session_state[checkbox_key], 
                    key=checkbox_key,
                    help=help_text
                ):
                    selected.append(name)
    
    return selected


def _create_model_label(name, model_data):
    """
    Crée le label d'affichage pour un modèle avec indicateurs visuels
    
    Args:
        name (str): Nom du modèle
        model_data (dict): Informations du modèle (format dict)
        
    Returns:
        str: Label formaté avec émojis indicateurs
    """
    indicators = []
    if model_data.get("free", False):
        indicators.append("🆓")
    else:
        indicators.append("💰")
    
    if model_data.get("web_search", False):
        indicators.append("🌐")
    
    return f"{' '.join(indicators)} {name}"


def _create_help_text(model_data):
    """
    Crée le texte d'aide (tooltip) pour un modèle
    
    Args:
        model_data (dict): Informations du modèle (format dict)
        
    Returns:
        str: Texte d'aide formaté
    """
    return (f"Provider: {model_data.get('provider_to_display', 'Unknown')} | "
            f"Free: {'Yes' if model_data.get('free', False) else 'No'} | "
            f"Web Search: {'Yes' if model_data.get('web_search', False) else 'No'}")


def render_selection_summary(selected, model_service):
    """
    Affiche un résumé des modèles sélectionnés
    
    Args:
        selected (list): Liste des modèles sélectionnés
        model_service: Instance du ModelConfigService
    """
    if selected:
        from ui.components.model_filters import get_filter_stats
        stats_text = get_filter_stats(selected, model_service)
        if stats_text:
            st.info(stats_text)
    else:
        st.warning("⚠️ Please select at least one model to continue.")


def reset_model_selection():
    """
    Utilitaire pour réinitialiser toutes les sélections de modèles
    """
    # Obtenir toutes les clés de session qui correspondent aux modèles
    model_keys = [key for key in st.session_state.keys() if key.startswith("model_")]
    for key in model_keys:
        st.session_state[key] = False


def select_default_models(model_service, count=2):
    """
    Sélectionne automatiquement les premiers modèles disponibles
    
    Args:
        model_service: Instance du ModelConfigService
        count (int): Nombre de modèles à sélectionner par défaut
    """
    models_dict = model_service.get_raw_models_dict()
    available_models = model_service.get_available_models()
    
    # Réinitialiser d'abord
    reset_model_selection()
    
    # Sélectionner les premiers modèles disponibles
    selected_count = 0
    for i, (name, model_info) in enumerate(models_dict.items()):
        if name in available_models and selected_count < count:
            st.session_state[f"model_{i}"] = True
            selected_count += 1