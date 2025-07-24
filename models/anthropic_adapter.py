import anthropic
import streamlit as st

client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def query(model_id, prompt, **kwargs):
    # Paramètres par défaut
    params = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 1024),
    }
    
    # Ajouter d'autres paramètres optionnels si présents
    if "top_p" in kwargs:
        params["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs:
        params["top_k"] = kwargs["top_k"]
    
    response = client.messages.create(**params)
    return response.content[0].text