from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")



def query(model_id, prompt, **kwargs):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=kwargs.get("max_tokens", 1024),
        #temperature=kwargs.get("temperature", 0.7),
    )
    return response.choices[0].message.content

