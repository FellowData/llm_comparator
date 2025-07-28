import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
from supabase_client import insert_llm_result

from llm_client import query_llm
from response_evaluator import ResponseEvaluator, display_evaluation_results


# === Load model config ===
with open("config/llm_config.yaml", "r", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)
models_dict = {m["name"]: m for m in model_config["models"]}

# === History ===
HISTORY_FILE = Path("data/prompt_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="LLM Comparator",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialiser l'évaluateur (ajouter après les autres initialisations)
evaluator = ResponseEvaluator()

# === Tabs ===
tab1, tab2 = st.tabs(["🧠 Prompt & Compare", "📜 History"])

with tab1:
    st.title("🧠 LLM Comparator")
    prompt = st.text_area("💬 Your prompt:", height=150)
    
    # Filtrer les modèles disponibles uniquement
    available_models = {name: info for name, info in models_dict.items() if info.get("available", True)}
    
    # Sélection multiple avec checkboxes dans un expander
    with st.expander("🎯 Select Models", expanded=True):
        # === FILTRES ===
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
                options=["All"] + sorted(list(set(info.get("provider_to_display", "Unknown") for info in available_models.values()))),
                key="provider_filter"
            )
        
        # Appliquer les filtres
        filtered_models = available_models.copy()
        
        # Filtre par prix
        if pricing_filter == "Free Only":
            filtered_models = {name: info for name, info in filtered_models.items() if info.get("free", False)}
        elif pricing_filter == "Paid Only":
            filtered_models = {name: info for name, info in filtered_models.items() if not info.get("free", True)}
        
        # Filtre par web search
        if web_search_filter == "With Web Search":
            filtered_models = {name: info for name, info in filtered_models.items() if info.get("web_search", False)}
        elif web_search_filter == "Without Web Search":
            filtered_models = {name: info for name, info in filtered_models.items() if not info.get("web_search", True)}
        
        # Filtre par provider
        if provider_filter != "All":
            filtered_models = {name: info for name, info in filtered_models.items() if info.get("provider_to_display") == provider_filter}
        
        # Affichage du nombre de modèles après filtrage
        st.info(f"📊 {len(filtered_models)} model(s) available after filtering (from {len(available_models)} total available)")
        
        if not filtered_models:
            st.warning("⚠️ No models match the current filters. Please adjust your selection.")
        else:
            st.divider()
            
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
                provider = model_info.get("provider_to_display", "Unknown")
                if provider not in providers_filtered:
                    providers_filtered[provider] = []
                providers_filtered[provider].append((name, model_info))
            
            selected = []
            
            # Affichage par provider
            for provider, models in providers_filtered.items():
                st.subheader(f"🔧 {provider.title()}")
                cols = st.columns(min(3, len(models)))
                
                for i, (name, model_info) in enumerate(models):
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
                        label_parts = [name.split(':')[0].strip()]
                        
                        # Ajouter des indicateurs visuels
                        indicators = []
                        if model_info.get("free", False):
                            indicators.append("🆓")
                        else:
                            indicators.append("💰")
                        
                        if model_info.get("web_search", False):
                            indicators.append("🌐")
                        
                        label = f"{' '.join(indicators)} {name}"
                        
                        if st.checkbox(
                            label, 
                            value=st.session_state[checkbox_key], 
                            key=checkbox_key,
                            help=f"Provider: {model_info.get('provider_to_display', 'Unknown')} | "
                                 f"Free: {'Yes' if model_info.get('free', False) else 'No'} | "
                                 f"Web Search: {'Yes' if model_info.get('web_search', False) else 'No'}"
                        ):
                            selected.append(name)
    
    # Affichage des modèles sélectionnés
    if selected:
        selected_short = []
        free_count = 0
        paid_count = 0
        web_search_count = 0
        
        for name in selected:
            model_info = models_dict[name]
            short_name = name.split(':')[0].strip()
            
            # Compter les types
            if model_info.get("free", False):
                free_count += 1
            else:
                paid_count += 1
                
            if model_info.get("web_search", False):
                web_search_count += 1
            
            selected_short.append(short_name)
        
        # Affichage enrichi des statistiques
        stats_text = f"🎯 **{len(selected)} model(s) selected:** {', '.join(selected_short)}"
        
        if free_count > 0 and paid_count > 0:
            stats_text += f" | 🆓 {free_count} free, 💰 {paid_count} paid"
        elif free_count > 0:
            stats_text += f" | 🆓 All free"
        else:
            stats_text += f" | 💰 All paid"
        
        if web_search_count == len(selected):
            stats_text += " | 🌐 All with web search"
        elif web_search_count > 0:
            stats_text += f" | 🌐 {web_search_count} with web search"
        
        st.info(stats_text)
    else:
        st.warning("⚠️ Please select at least one model to continue.")

    if st.button("🚀 Run", disabled=not (prompt and selected)) and prompt and selected:
        # Stocker les réponses pour l'évaluation comparative
        responses_for_evaluation = {}
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
                        evaluation = evaluator.evaluate_response(prompt, content, name)
                        display_evaluation_results(evaluation, name.split(':')[0].strip())
                    
                    # Stocker pour comparaison
                    responses_for_evaluation[name] = {
                        'content': content,
                        'evaluation': evaluation
                    }



                    # Save history avec évaluation
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt,
                        "model_name": name,
                        "model_id": model_id,
                        "response": content,
                        "evaluation_score": evaluation.overall_score,
                        "readability_score": evaluation.readability_score,
                        "structure_score": evaluation.structure_score,
                        "sources_score": evaluation.sources_score,
                        "completeness_score": evaluation.completeness_score,
                        "relevance_score": evaluation.relevance_score
                    }

                    if HISTORY_FILE.exists():
                        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                            history_data = json.load(f)
                    else:
                        history_data = []
                    history_data.append(entry)
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(history_data, f, ensure_ascii=False, indent=2)

                    # Insert into Supabase
                    try:
                        st.info("Saving to Supabase...")
                        insert_llm_result("prompt_history", entry)
                        st.success("Saved to Supabase...")
                    except Exception as e:
                        st.warning(f"⚠️ Supabase save failed : {e}")

                except Exception as e:
                    st.error(f"Error: {e}")
        # === COMPARAISON FINALE ===
        if len(responses_for_evaluation) > 1:
            st.markdown("---")
            st.markdown("## 🏆 Comparaison Finale des Modèles")
            
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
            
            import pandas as pd
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison = df_comparison.sort_values('Score Global', ascending=False)
            
            st.dataframe(df_comparison, use_container_width=True)
            
            # Graphique radar comparatif
            if len(comparison_data) <= 4:  # Éviter la surcharge visuelle
                import plotly.graph_objects as go
                
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
with tab2:
    st.title("📜 Prompt History")

    # Tentative de chargement depuis Supabase d'abord
    history_data = []
    data_source = ""
    
    try:
        from supabase_client import get_llm_history
        st.info("🔄 Loading data from Supabase...")
        history_data = get_llm_history("prompt_history")
        data_source = "Supabase"
        
        # Si le chargement Supabase réussit, synchroniser le fichier JSON local
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            st.success(f"✅ Data loaded from {data_source} ({len(history_data)} entries) • Local backup updated")
        except Exception as sync_error:
            st.success(f"✅ Data loaded from {data_source} ({len(history_data)} entries)")
            st.warning(f"⚠️ Failed to sync local backup: {sync_error}")
            
    except Exception as e:
        st.warning(f"⚠️ Supabase load failed: {e}")
        st.info("🔄 Falling back to local JSON file...")
        
        # Fallback vers le fichier JSON local
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                data_source = "Local JSON"
                st.success(f"✅ Data loaded from {data_source} ({len(history_data)} entries)")
            except Exception as json_error:
                st.error(f"❌ Failed to load from JSON file: {json_error}")
                history_data = []
        else:
            st.info("📝 No local history file found.")

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
                df = df[df["main_response"].str.contains(main_response_search, case=False)]m                

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

        # Bouton de téléchargement Excel enrichi
        if len(df) > 0:
            try:
                excel_file = "data/prompt_history_with_evaluation.xlsx"
                Path("data").mkdir(parents=True, exist_ok=True)
                df.to_excel(excel_file, index=False)
                with open(excel_file, "rb") as f:
                    st.download_button(
                        "📅 Download history with evaluations (Excel)", 
                        f, 
                        file_name="prompt_history_with_evaluation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as excel_error:
                st.warning(f"⚠️ Excel export failed: {excel_error}")

    else:
        st.info("No prompt history available yet.")
