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


## [0.6.0] - 2025-07-25 & 28
### Added
- requirements updated
- Database (table Supabase prompt_history) updated with :
  - evaluation_score : Score global /10
  - readability_score : Score de lisibilité /10
  - structure_score : Score de structure /10
  - sources_score : Score des sources /10
  - completeness_score : Score de complétude /10
  - relevance_score : Score de pertinence /10

- Automatic evaluation of the answer provided by LLM :
  1. Évaluation Multi-Critères
    📖 Lisibilité : Score Flesch-Kincaid, longueur des phrases/mots
    🏗️ Structure : Organisation, paragraphes, listes, titres
    🔗 Sources : Qualité, diversité, fiabilité des URLs
    📋 Complétude : Couverture des éléments du prompt
    🎯 Pertinence : Adéquation réponse/question

  2. Interface Visuelle

    Scores individuels avec barres colorées
    Graphique radar pour comparaison multi-modèles
    Recommandations personnalisées d'amélioration
    Ranking automatique du meilleur modèle

  3. Analytics Avancés

    Dashboard avec métriques globales
    Tendances temporelles des performances
    Comparaison entre modèles avec statistiques
    Export enrichi avec scores d'évaluation

### Refactoring
- Separate files for each tab
- Specific files for components
- Services (3) : ModelConfig + History + Evaluation
- Modular UI : Tabs + separate Components
- Caching : Optimized performance
- Duplicated code : 0 line VS 270 before