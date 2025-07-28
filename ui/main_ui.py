"""
Interface utilisateur principale de l'application LLM Comparator
Remplace l'ancien app.py monolithique par une architecture modulaire
"""

import streamlit as st
import yaml
from pathlib import Path

# Imports des composants UI
from ui.tabs.prompt_tab import render_prompt_tab
from ui.tabs.history_tab import render_history_tab  
from ui.tabs.analytics_tab import render_analytics_tab
from ui.tabs.settings_tab import render_settings_tab
from ui.tabs.userguide_tab import render_userguide_tab

# Imports des services existants (non modifiés)
from response_evaluator import ResponseEvaluator

from services.model_config_service import get_model_config_service, ModelStats


'''def load_configuration():
    """
    Charge la configuration des modèles depuis le fichier YAML
    
    Returns:
        dict: Configuration des modèles
    """
    config_path = Path("config/llm_config.yaml")
    
    if not config_path.exists():
        st.error(f"❌ Fichier de configuration non trouvé: {config_path}")
        st.error("🔍 Vérifiez que le fichier config/llm_config.yaml existe")
        st.stop()
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        return {m["name"]: m for m in model_config["models"]}
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la configuration: {e}")
        st.stop()'''


def setup_page_config():
    """Configure les paramètres de la page Streamlit"""
    st.set_page_config(
        page_title="LLM Comparator",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://github.com/votre-repo/llm-comparator',
            'Report a bug': "https://github.com/votre-repo/llm-comparator/issues",
            'About': "# LLM Comparator\nComparez facilement différents modèles LLM !"
        }
    )


def initialize_services():
    """
    Initialise les services nécessaires à l'application
    
    Returns:
        tuple: (model_service, evaluator)
    """
    try:
        # Initialisation du service de configuration des modèles
        model_service = get_model_config_service()
        model_service.load_models_config()  # Charge et valide la config
        
        # Initialisation de l'évaluateur
        evaluator = ResponseEvaluator()
        
        return model_service, evaluator
        
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation des services: {e}")
        st.stop()


def render_header():
    """Affiche l'en-tête de l'application"""
    st.markdown("""
    # 🧠 LLM Comparator
    
    Comparez facilement des prompts sur différents modèles de langage (LLM) et analysez leurs performances !
    """)
    



def render_tabs(model_service, evaluator):
    """
    Rendu des onglets principaux de l'application
    
    Args:
        model_service (dict): Configuration des modèles
        evaluator (ResponseEvaluator): Instance de l'évaluateur
    """
    # Créer les onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Prompt & Compare", 
        "📜 History", 
        "🔬 Advanced Analytics",
        "⚙️ Settings",
        "📖 User Guide"
    ])
    
    # Déléguer le rendu à chaque module spécialisé
    with tab1:
        try:
            render_prompt_tab(model_service, evaluator)
        except Exception as e:
            st.error(f"❌ Erreur dans l'onglet Prompt & Compare: {e}")
            st.error("🔄 Essayez de recharger la page")
    
    with tab2:
        try:
            render_history_tab(evaluator)
        except Exception as e:
            st.error(f"❌ Erreur dans l'onglet History: {e}")
            st.error("🔄 Essayez de recharger la page")
    
    with tab3:
        try:
            render_analytics_tab()
        except Exception as e:
            st.error(f"❌ Erreur dans l'onglet Advanced Analytics: {e}")
            st.error("🔄 Essayez de recharger la page")

    with tab4:
        try:
            render_settings_tab()
        except Exception as e:
            st.error(f"❌ Erreur dans l'onglet Settings: {e}")
            st.error("🔄 Essayez de recharger la page")

    with tab5:
        try:
            render_userguide_tab()
        except Exception as e:
            st.error(f"❌ Erreur dans l'onglet User Guide: {e}")
            st.error("🔄 Essayez de recharger la page")


def setup_session_state(model_service, evaluator):
    """
    Configure l'état de session pour l'application
    
    Args:
        model_service: Instance du ModelConfigService
        evaluator (ResponseEvaluator): Instance de l'évaluateur
    """
    # Stocker les objets globaux en session state pour accès dans les composants
    if 'model_service' not in st.session_state:
        st.session_state.model_service = model_service
    
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = evaluator
    
    # Initialiser d'autres variables de session si nécessaire
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True


def display_sidebar_info():
    """Affiche des informations utiles dans la sidebar"""
    with st.sidebar:
        st.markdown("### ℹ️ Informations")
        st.markdown("""
        **LLM Comparator** vous permet de :
        - 🔄 Comparer plusieurs modèles simultanément
        - 📊 Évaluer automatiquement les réponses
        - 📈 Analyser les performances dans le temps
        - 💾 Sauvegarder l'historique des comparaisons
        """)
        


def main():
    """
    Fonction principale de l'application
    Point d'entrée appelé par app.py
    """
    # Configuration de la page
    setup_page_config()
    
    # Affichage de la sidebar
    display_sidebar_info()
    
    # Initialisation des services
    try:
        model_service, evaluator = initialize_services()
        setup_session_state(model_service, evaluator)

    except Exception as e:
        st.error(f"❌ Erreur d'initialisation: {e}")
        st.error("🆘 Impossible de démarrer l'application")
        st.stop()
    
    # Affichage de l'en-tête
    render_header()
    
    # Séparateur
    st.markdown("---")
    
    # Rendu des onglets principaux
    render_tabs(model_service, evaluator)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🧠 LLM Comparator - Comparez intelligemment vos modèles de langage
    </div>
    """, unsafe_allow_html=True)


# Point d'entrée pour exécution directe
if __name__ == "__main__":
    main()