from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

'''def query(model_id, prompt, **kwargs):
    response = client.responses.create(
        model=model_id,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": "low",
        }],
        input="prompt",
    )

    return response.output_text'''



def query(model_id, prompt, **kwargs):
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        #temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 1024),
    )
    return response.choices[0].message.content

'''OLD function
def query(model_id, prompt, **kwargs):
    response = client.chat.completions.create(
        model=model_id,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": "low",
        }],
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 1024),
    )
    return response.choices[0].message.content'''