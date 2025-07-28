import streamlit as st



def render_userguide_tab():
    st.title("📖 User Guide")
    st.markdown("## ℹ️ Informations")
    st.markdown("""
    **LLM Comparator** vous permet de :
    - 🔄 Comparer plusieurs modèles simultanément
    - 📊 Évaluer automatiquement les réponses
    - 📈 Analyser les performances dans le temps
    - 💾 Sauvegarder l'historique des comparaisons
    """)
