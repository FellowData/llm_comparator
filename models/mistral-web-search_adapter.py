#import os
from mistralai import Mistral
import streamlit as st

# Initialize the Mistral client with the API key from Streamlit secrets
# Mistral adapter for Streamlit
# This adapter uses the Mistral client to interact with Mistral models.
# It supports chat completions and returns the response content.
# The client gets the API key from the environment variable `MISTRAL_API_KEY`.
# The `query` function takes a model ID and a prompt, and returns the response content. 
def query(model_id, prompt, **kwargs):
    client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])


    websearch_agent = client.beta.agents.create(
        model=model_id,
        description="Agent able to search information over the web",
        name="Websearch Agent",
        instructions="You have the ability to perform web searches with `web_search` to find up-to-date information.",
        tools=[{"type": "web_search"}],
        completion_args={
            "temperature": 0.3,
            "top_p": 0.95,
        }
    )

    response = client.beta.conversations.start(
        agent_id=websearch_agent.id,
        inputs=prompt
    )
    
    # Accéder au texte de la réponse selon la documentation Mistral
    # La réponse contient un champ 'outputs' avec des entrées de type 'message.output'
    if hasattr(response, 'outputs') and response.outputs:
        # Chercher l'entrée de type 'message.output'
        for output in response.outputs:
            if hasattr(output, 'type') and output.type == 'message.output':
                if hasattr(output, 'content') and output.content:
                    # Extraire le texte des chunks de type 'text'
                    text_parts = []
                    for chunk in output.content:
                        if hasattr(chunk, 'type') and chunk.type == 'text':
                            text_parts.append(chunk.text)
                    return ''.join(text_parts)
    
    # Fallback - afficher la structure pour debug
    print("Structure de la réponse:", dir(response))
    if hasattr(response, 'outputs'):
        print("Nombre d'outputs:", len(response.outputs))
        for i, output in enumerate(response.outputs):
            print(f"Output {i} - type: {getattr(output, 'type', 'N/A')}")
    return str(response)


