import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
from supabase_client import insert_llm_result

from llm_client import query_llm

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

                    # Save history
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt,
                        "model_name": name,
                        "model_id": model_id,
                        "response": content
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
        st.success(f"✅ Data loaded from {data_source} ({len(history_data)} entries)")
        # Si le chargement Supabase réussit, synchroniser le fichier JSON local
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            st.info("💾 Local JSON file synchronized with Supabase data")
        except Exception as sync_error:
            st.warning(f"⚠️ Failed to sync local JSON file: {sync_error}")
    except Exception as e:
        st.warning(f"⚠️ Supabase load failed: {e}")
        st.info("🔄 Falling back to local JSON file...")
        # Si l'échec de Supabase, on charge le fichier JSON local
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
        # Convertir les données en DataFrame pour une manipulation facile
        st.info(f"📊 Displaying {len(history_data)} entries from {data_source}")
        df = pd.DataFrame(history_data)
        if 'response' not in df.columns:
            df['response'] = ''
        # Extraire les sources et la réponse principale
        df["sources"] = df["response"].apply(lambda r: r.split("=== SOURCES ===")[1].strip() if "=== SOURCES ===" in r else "")
        df["main_response"] = df["response"].apply(lambda r: r.split("=== SOURCES ===")[0].strip() if "=== SOURCES ===" in r else r)

        # Affichage de la source des données
        st.info(f"📊 **Data source:** {data_source}")
        
        # Filtres
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

        with col3:
            # Recherche dans les réponses principales
            main_response_search = st.text_input("Search in main_response:")
            if main_response_search:
                df = df[df["main_response"].str.contains(main_response_search, case=False)]


        # Affichage du tableau
        st.subheader("📋 History Table")
        display_columns = ["timestamp", "model_name", "prompt", "main_response", "sources"]
        # Vérifier quelles colonnes existent réellement
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            st.dataframe(
                df[available_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("⚠️ Unable to display data: required columns not found")


        # Bouton de téléchargement Excel
        if len(df) > 0:
            try:
                excel_file = "data/prompt_history.xlsx"
                # Créer le dossier data s'il n'existe pas
                Path("data").mkdir(parents=True, exist_ok=True)
                df.to_excel(excel_file, index=False)
                with open(excel_file, "rb") as f:
                    st.download_button(
                        "📅 Download history (Excel)", 
                        f, 
                        file_name="prompt_history.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as excel_error:
                st.warning(f"⚠️ Excel export failed: {excel_error}")

    else:
        st.info("No prompt history available yet.")
