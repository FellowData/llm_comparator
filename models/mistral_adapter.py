#import os
from mistralai import Mistral
import streamlit as st

# Initialize the Mistral client with the API key from Streamlit secrets
# Mistral adapter for Streamlit
# This adapter uses the Mistral client to interact with Mistral models.
# It supports chat completions and returns the response content.
# The `query` function takes a model ID and a prompt, and returns the response content.
def query(model_id, prompt, **kwargs):
    client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
    # Make the chat completion request
    # The model_id is used to specify which Mistral model to use
    # The prompt is sent as a message from the user
    # Additional parameters can be passed via kwargs, such as temperature and max_tokens
    chat_response = client.chat.complete(
        model = model_id,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    return chat_response.choices[0].message.content