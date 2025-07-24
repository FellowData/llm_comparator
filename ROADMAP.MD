# 🛣️ Roadmap

## 🚧 URGENT / NEXT
### TODO


### FIXME
- [ ] Review Gemini integration to display original sources and not redirection URLs from Google grounding API

## Points d'amélioration
- Structure des Fichiers
- Refactorisation

### Phase 1 - Architecture & Stabilité (1-2 semaines)
Refactorisation structurelle selon la nouvelle organisation
Tests unitaires pour les services critiques
Gestion d'erreurs améliorée avec retry et circuit breaker
Logging structuré avec différents niveaux

### Phase 2 - Expérience Utilisateur (2-3 semaines)
Dashboard de comparaison avec métriques visuelles
Filtres avancés dans l'historique
Export multi-format (MD, JSON, PDF)
Mode batch pour tests automatisés
Templates de prompts prédéfinis

### Phase 3 - Fonctionnalités Avancées (3-4 semaines)
Évaluation automatique des réponses
A/B Testing entre modèles
Cache intelligent des réponses
API REST pour usage programmatique
Intégration Webhook pour notifications

### Phase 4 - Intelligence & Analytics (4-6 semaines)
Analyse de sentiment des réponses
Détection de similarité entre réponses
Recommandations de modèles basées sur l'usage
Métriques de performance et coûts
Tableaux de bord exécutifs

## Recommandations Techniques Immédiates
### Sécurité
Validation des entrées utilisateur
Rate limiting sur les requêtes LLM
Chiffrement des données sensibles en base

### Performance
Mise en cache des configurations
Requêtes asynchrones pour les comparaisons
Pagination de l'historique

### Monitoring
Health checks pour les services externes
Métriques d'usage et de performance
Alertes sur les échecs répétés



## 🔜 Next Features
- [ ] Retrieve token counts
- [ ] Review possibilities from LLMs to retrieve URLs
- [ ] Custom temperature / max tokens sliders
- [ ] Display response metadata (token count)

## 🧪 Ideas
- [ ] Batch prompt evaluation
- [ ] Model scoring UI (user or auto-eval)
- [ ] Dark/light theme toggle
- [ ] Switchable system prompts
- [ ] Side-by-side column layout for responses
- [ ] Add support Meta AI, via Meta AI (once made available in Europe). Library on pypi.org is not updated anymore
- [ ] Add support Deepseek (via Deepseek) with integration to search on the web (feature not yet available through API)