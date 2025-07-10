import streamlit as st
import requests

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
}

st.title("🧠 Comparateur de LLMs via OpenRouter.ai")
prompt = st.text_area("💬 Ton prompt ici :", height=150)

selected = st.multiselect("Sélectionne un ou plusieurs modèles :", list(AVAILABLE_MODELS.keys()), default=list(AVAILABLE_MODELS.keys())[:2])

if st.button("🚀 Lancer la requête") and prompt and selected:
    for name in selected:
        model_id = AVAILABLE_MODELS[name]
        st.markdown(f"### 🤖 {name}")
        with st.spinner("Réponse en cours..."):
            instruction = (
                "À la fin de ta réponse, ajoute une section intitulée '=== SOURCES ===' "
                "avec les sites web ou les sources dont tu t'es inspiré."
            )

            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": f"{prompt.strip()}\n\n{instruction}"}
                ]
            }

            try:
                response = requests.post(API_URL, headers=HEADERS, json=payload)
                if response.status_code != 200:
                    st.error(f"Erreur HTTP {response.status_code} : {response.text}")
                    continue

                data = response.json()
                if "choices" not in data:
                    st.error(f"Erreur API OpenRouter : {data}")
                    continue

                content = data["choices"][0]["message"]["content"]

                st.success("Réponse reçue")
                if "=== SOURCES ===" in content:
                    answer_part, sources_part = content.split("=== SOURCES ===", 1)
                else:
                    answer_part = content
                    sources_part = "*Aucune source identifiable fournie.*"

                st.markdown("**📝 Réponse :**")
                st.write(answer_part.strip())

                st.markdown("**🔗 Sources :**")
                st.info(sources_part.strip())
            except Exception as e:
                st.error(f"Erreur : {e}")


