
import streamlit as st
import requests
from openai import OpenAI
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

# 🔐 Clé API OpenAI (direct)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}

AVAILABLE_MODELS = {
    "Mistral: Mistral Small 3.2 24B (free) 🟢": "mistralai/mistral-small-3.2-24b-instruct:free",
    "Meta: Llama 3.3 70B Instruct (free) 🔵": "meta-llama/llama-3.3-70b-instruct:free",
    "Google: Gemma 3n 2B (free) 🟣": "google/gemma-3n-e2b-it:free",
    "DeepSeek: R1 0528 (free) 🟠": "deepseek/deepseek-r1-0528:free",
    "OpenAI: gpt-4.1-mini 🧠": "gpt-4.1-nano",  
}

# Fichier historique
HISTORY_FILE = Path("data/prompt_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="LLM Comparator",
    layout="wide",  # 🟢 'wide' = pleine largeur | 'centered' = par défaut
    initial_sidebar_state="auto"
)

# Tabs
tab1, tab2 = st.tabs(["🧠 Prompt & Compare", "📜 History"])

with tab1:
    st.title("🧠 LLM Comparator (OpenRouter + OpenAI)")
    prompt = st.text_area("💬 Your prompt:", height=150)
    selected = st.multiselect("Select one or more models:", list(AVAILABLE_MODELS.keys()), default=list(AVAILABLE_MODELS.keys())[:2])

    if st.button("🚀 Run") and prompt and selected:
        for name in selected:
            model_id = AVAILABLE_MODELS[name]
            st.markdown(f"### 🤖 {name}")
            with st.spinner("Generating response..."):
                instruction = (
                "À la fin de ta réponse, ajoute une section intitulée '=== SOURCES ===' "
                "avec la liste des urls des sites web utilisées pour la réponse."
                )
                full_prompt = f"{prompt.strip()}\n\n{instruction}"

                try:
                    if model_id.startswith("gpt-"):
                        if not client.api_key:
                            st.error("Clé OPENAI_API_KEY manquante dans les secrets Streamlit.")
                            continue

                        response = client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.7,
                            max_tokens=1024,
                        )
                        content = response.choices[0].message.content
                    else:
                        payload = {
                            "model": model_id,
                            "messages": [{"role": "user", "content": full_prompt}]
                        }
                        response = requests.post(API_URL, headers=HEADERS, json=payload)
                        if response.status_code != 200:
                            st.error(f"HTTP Error {response.status_code}: {response.text}")
                            continue
                        data = response.json()
                        if "choices" not in data:
                            st.error(f"API OpenRouter Error: {data}")
                            continue
                        content = data["choices"][0]["message"]["content"]

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

                    # 🔸 Save to history
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

                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.title("📜 Prompt History")

    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        df = pd.DataFrame(history_data)

        # Extraire les sources si elles existent
        df["sources"] = df["response"].apply(lambda r: r.split("=== SOURCES ===")[1].strip() if "=== SOURCES ===" in r else "")

        # Raccourcir la réponse principale pour affichage
        df["main_response"] = df["response"].apply(lambda r: r.split("=== SOURCES ===")[0].strip() if "=== SOURCES ===" in r else r)

        # Filtres
        model_filter = st.multiselect("Filter by model", options=df["model_name"].unique().tolist())
        if model_filter:
            df = df[df["model_name"].isin(model_filter)]

        prompt_search = st.text_input("Search in prompts:")
        if prompt_search:
            df = df[df["prompt"].str.contains(prompt_search, case=False)]

        st.dataframe(
            df[["timestamp", "model_name", "prompt", "main_response", "sources"]],
            use_container_width=True,
            hide_index=True
        )

        # Export Excel
        excel_file = "data/prompt_history.xlsx"
        df.to_excel(excel_file, index=False)
        with open(excel_file, "rb") as f:
            st.download_button("📥 Download history (Excel)", f, file_name=excel_file)

    else:
        st.info("No prompt history available yet.")
