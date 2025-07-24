from openai import OpenAI
import streamlit as st


def query(model_id, prompt, **kwargs):
    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        #temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 1024),
        stream=False,
    )
    return response.choices[0].message.content
