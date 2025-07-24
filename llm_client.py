import importlib


# This module provides a generic function to query a language model using a specified provider and model ID.
def query_llm(provider: str, model_id: str, prompt: str, **kwargs):
    try:
        # Dynamically import the appropriate adapter module based on the provider from the models package
        module = importlib.import_module(f"models.{provider}_adapter")
        return module.query(model_id, prompt, **kwargs)
    except ModuleNotFoundError:
        raise ValueError(f"Provider '{provider}' not supported (no adapter found).")
    except Exception as e:
        raise RuntimeError(f"Error in provider '{provider}': {str(e)}")
