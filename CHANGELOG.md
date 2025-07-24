# 📦 Changelog

## [0.1.0] - 2025-07-10
### Added
- Streamlit UI
- Multi-model comparison
- Initial release with support for:
  - Mistral (via OpenRouter)
  - LLaMA (via OpenRouter)
  - Gemma (via OpenRouter)
  - DeepSeek (via OpenRouter)

## [0.2.0] - 2025-07-10
### Added
- Prompt enrichment with request for source attribution
- Response display split into answer and sources
- Error handling for API failures (ex : 503 and invalid keys) and model unavailability


## [0.3.0] - 2025-07-11
### Added
- Add support GPT-4o (via native OpenAI API)

## [0.4.0] - 2025-07-11
### Added
- Prompt history / log (JSON)
- Download answers (Excel)
- Visualize prompts' history directly in the app
- Wide screen 

## [0.5.0] - 2025-07-24
### Added
- Separated configuration file (YAML) for listing models
- Streamlit user interface : 
  - selection of LLMs models now using checkboxes instead of multi-selection
  - button to select / deselect all models in one click
- Refactoring of modules to call LLMs :
  - Dynamically import the appropriate adapter module based on the provider from the models 
  - models modules in dedicated folders / files
- Add / Adapt support to LLM :
  - Add support Claude (via Anthropic)
  - Add support Perplexity (via Perplexity) with integration to search on the web
  - Add support Gemini (via Gemini) with integration to search on the web
  - Add support Mistral (via Mistral)
  - Add support Deepseek (via Deepseek)
  - Review OpenAI integration to search on the web
  - Review Claude integration to search on the web
- Add support to online Database
- Interface loads prompts history from Database, JSON used as fallback
- Result from prompts saved to Database as well as JSON
- Synchronizes Database to JSON when loading prompt history user interface
- Filters available models based on pricing, web search capability, provider
- Enrich llm_config.yaml configuration