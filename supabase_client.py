# supabase_client.py
from supabase import create_client
import streamlit as st



SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_llm_result(table: str, entry: dict):
    try:
        response = supabase.table(table).insert(entry).execute()
        # Vérifier si l'insertion a réussi
        if hasattr(response, 'data') and response.data:
            return response.data
        else:
            raise Exception("No data returned from Supabase")
    except Exception as e:
        # Log plus détaillé pour le debugging
        error_msg = f"Supabase insertion failed: {str(e)}"
        if hasattr(e, 'details'):
            error_msg += f" - Details: {e.details}"
        raise Exception(error_msg)

# Get LLM history from Supabase
def get_llm_history(table: str):
    """
    Récupère l'historique depuis Supabase
    Retourne les données triées par timestamp décroissant
    """
    try:
        response = supabase.table(table).select("*").order("timestamp", desc=True).execute()
        if hasattr(response, 'data') and response.data:
            return response.data
        else:
            raise Exception("No data returned from Supabase")
    except Exception as e:
        error_msg = f"Supabase read failed: {str(e)}"
        if hasattr(e, 'details'):
            error_msg += f" - Details: {e.details}"
        raise Exception(error_msg)
