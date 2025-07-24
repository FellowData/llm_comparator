from google import genai
from google.genai import types
import streamlit as st

# Google Gemini adapter for Streamlit
# This adapter uses the Google GenAI client to interact with Gemini models.
# It supports both standard and web search models, allowing for grounded responses.
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
# The `query` function takes a model ID and a prompt, and returns the grounded response.
def query(model_id, prompt, **kwargs):
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # Define the grounding tool
    grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
    )

    # Configure generation settings
    config = types.GenerateContentConfig(
    tools=[grounding_tool]
    )

    # Make the request
    response = client.models.generate_content(
    model=model_id,
    contents=prompt,
    config=config,
    )

    # ReturnPrint the grounded response
    return response.text



'''def temp_query(model_id, prompt, **kwargs):
    from google import genai
    from google.genai import types

    # Configure the client
    client = genai.Client()

    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    # Configure generation settings
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    # Make the request
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=config,
    )

    # Print the grounded response
    print(response.text)



print (temp_query("gemini-2.5-flash", "qui a gagné la coupe du monde des clubs 2025 ?\n\n\u00c0 la fin de ta réponse, ajoute une section intitulée '=== SOURCES ===' avec la liste des urls des sites web utilisées pour la réponse."))'''

